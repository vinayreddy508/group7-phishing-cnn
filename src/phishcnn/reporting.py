"""Deterministic tables/plots for project report and slides."""

from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def save_json(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def plot_shortcut_audit(audit, out):
    columns = ["starts_https_www", "starts_https", "host_starts_www", "has_path", "has_query"]
    subset = audit.loc[[s for s in columns if s in audit.index], ["legitimate", "phishing"]]
    ax = subset.plot(kind="bar", figsize=(9, 5))
    ax.set(title="Dataset shortcut rates by original class", ylabel="Mean (rate for binary features)")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def plot_f1(results, out):
    df = pd.DataFrame(results)
    df = df[df["model"].str.startswith("cnn_")]
    if df.empty:
        return
    summary = df.groupby("model", as_index=True)["f1"].agg(["mean", "std"]).fillna(0)
    ax = summary["mean"].plot(kind="barh", figsize=(10, 5), xerr=summary["std"])
    ax.set(xlim=(0, 1), xlabel="Phishing-class F1", title="CNN comparison (domain-disjoint split)")
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
