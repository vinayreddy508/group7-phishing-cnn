"""Dataset ingestion, duplicate cleanup, domain-disjoint splitting."""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from .urls import domain_group


def fetch_uci(output: str | Path) -> Path:
    """Retrieve UCI dataset 967; write ONLY URL and original label locally."""
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=967)
    urls = ds.data.features["URL"]
    target = ds.data.targets.iloc[:, 0]
    out = pd.DataFrame({"URL": urls, "Label": target})
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return path


def load_and_clean(path: str | Path, sample: int | None = None, seed: int = 42):
    header = pd.read_csv(path, nrows=0).columns.tolist()
    columns = {col.lower().strip(): col for col in header}
    if "url" not in columns or "label" not in columns:
        raise ValueError("Expected URL and label columns in the PhiUSIIL CSV")
    df = pd.read_csv(path, usecols=[columns["url"], columns["label"]], low_memory=False)
    df = df.rename(columns={columns["url"]: "URL", columns["label"]: "Label"})
    initial = len(df)
    df = df.dropna(subset=["URL", "Label"]).copy()
    df["URL"] = df["URL"].astype(str).str.strip()
    df["Label"] = pd.to_numeric(df["Label"], errors="coerce")
    df = df[(df["URL"] != "") & (df["Label"].isin([0, 1]))].copy()

    # Remove contradictory duplicates entirely, rather than choosing one label.
    conflicts = df.groupby("URL")["Label"].nunique()
    conflict_urls = set(conflicts.index[conflicts > 1])
    if conflict_urls:
        df = df.loc[~df["URL"].isin(conflict_urls)].copy()
    after_conflicts = len(df)
    df = df.drop_duplicates(subset=["URL"], keep="first").reset_index(drop=True)
    after_dedup = len(df)

    # Invert UCI labels: phishing=1, legitimate=0.
    df["label"] = 1 - df["Label"].astype("int8")
    df["domain_group"] = df["URL"].map(domain_group)
    invalid_count = int((df["domain_group"] == "").sum())
    df = df[df["domain_group"] != ""].reset_index(drop=True)

    # Small development sample; NEVER use quick-mode scores in a final report.
    if sample is not None and 0 < sample < len(df):
        df = df.sample(n=sample, random_state=seed).reset_index(drop=True)
    stats = {
        "initial_rows": initial,
        "conflicting_url_count": len(conflict_urls),
        "after_conflicts": after_conflicts,
        "after_exact_url_dedup": after_dedup,
        "invalid_host_removed": invalid_count,
        "final_rows": len(df),
        "unique_registrable_domains": int(df["domain_group"].nunique()),
        "phishing_rate": float(df["label"].mean()),
        "quick_sample": sample,
    }
    return df, stats


def _two_stage_group_split(df, seed):
    splitter = GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=seed)
    itr, irest = next(splitter.split(df, groups=df["domain_group"]))
    tr, rest = df.iloc[itr].copy(), df.iloc[irest].copy()
    splitter2 = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=seed + 10007)
    iv, it = next(splitter2.split(rest, groups=rest["domain_group"]))
    return tr, rest.iloc[iv].copy(), rest.iloc[it].copy()


def domain_disjoint_split(df: pd.DataFrame, seed=42, candidates=20):
    """Pick deterministic group split by label balance, never model scores."""
    if df["domain_group"].nunique() < 10:
        raise ValueError("Not enough unique domains for a reliable three-way split")
    overall_rate = float(df["label"].mean())
    best, best_score = None, float("inf")
    for offset in range(candidates):
        parts = _two_stage_group_split(df, seed + offset)
        if any(len(p) == 0 or p["label"].nunique() != 2 for p in parts):
            continue
        target_size = [0.7, 0.15, 0.15]
        score = sum(
            abs(len(part) / len(df) - desired) * 2.0 +
            abs(float(part["label"].mean()) - overall_rate)
            for part, desired in zip(parts, target_size)
        )
        if score < best_score:
            best, best_score = parts, score
    if best is None:
        raise ValueError("Unable to produce group split with both classes in each partition")
    train, val, test = [p.reset_index(drop=True) for p in best]
    groups = [set(p["domain_group"]) for p in (train, val, test)]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    return train, val, test


def random_split(df: pd.DataFrame, seed=42):
    """Optional non-primary reference: random URL split (may leak domains)."""
    tr, rest = train_test_split(df, test_size=0.3, random_state=seed, stratify=df["label"])
    va, te = train_test_split(rest, test_size=0.5, random_state=seed, stratify=rest["label"])
    return tuple(part.reset_index(drop=True) for part in (tr, va, te))


def split_diagnostics(train, val, test):
    diag = {}
    for name, part in [("train", train), ("validation", val), ("test", test)]:
        diag[name] = {
            "rows": int(len(part)),
            "domains": int(part["domain_group"].nunique()),
            "phishing_count": int(part["label"].sum()),
            "legitimate_count": int((part["label"] == 0).sum()),
            "phishing_rate": float(part["label"].mean()),
        }
    domains = [set(p["domain_group"]) for p in (train, val, test)]
    diag["domain_overlap"] = {
        "train_validation": len(domains[0] & domains[1]),
        "train_test": len(domains[0] & domains[2]),
        "validation_test": len(domains[1] & domains[2]),
    }
    return diag
