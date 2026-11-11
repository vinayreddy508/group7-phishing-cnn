#!/usr/bin/env python3
"""End-to-end, leakage-aware PhiUSIIL experiments. Run after pip install -r requirements.txt."""

import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from phishcnn.baselines import baseline_results, shortcut_audit
from phishcnn.data import load_and_clean, domain_disjoint_split, random_split, split_diagnostics
from phishcnn.metrics import evaluate, robustness, domain_bootstrap_f1
from phishcnn.reporting import save_json, plot_shortcut_audit, plot_f1
from phishcnn.urls import url_representation, force_prefix


def checksum(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Group 7 PhiUSIIL reproducible CNN experiments")
    parser.add_argument("--data", default="data/phiusiil_urls.csv")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--output", default="runs/domain_disjoint")
    parser.add_argument("--split", choices=["domain", "random"], default="domain")
    parser.add_argument("--quick", action="store_true", help="Smoke test only; 12k rows, 2 epochs, one seed")
    parser.add_argument("--audit-only", action="store_true", help="Run audit/baselines, skip neural models")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        parser.error(f"No dataset at {data_path}. First run: python scripts/download_data.py")
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.quick:
        cfg["seeds"] = cfg["seeds"][:1]
        cfg["epochs"] = 2
        cfg["early_stopping_patience"] = 1
        cfg["tfidf_max_features"] = min(cfg["tfidf_max_features"], 12000)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "effective_config.json", cfg)

    import importlib.metadata as im
    versions = {}
    for package in ["numpy", "pandas", "scikit-learn", "tensorflow", "tldextract", "ucimlrepo"]:
        try:
            versions[package] = im.version(package)
        except im.PackageNotFoundError:
            versions[package] = "not-installed"
    save_json(out / "environment_versions.json", versions)

    print("Loading, cleaning and grouping dataset...")
    df, ds_stats = load_and_clean(data_path, sample=12000 if args.quick else None,
                                  seed=cfg["seed"])
    if args.split == "domain":
        train, val, test = domain_disjoint_split(df, seed=cfg["seed"],
                                                 candidates=cfg["group_split_candidates"])
    else:
        train, val, test = random_split(df, seed=cfg["seed"])
    diagnostics = split_diagnostics(train, val, test)
    print(json.dumps(diagnostics, indent=2))
    if args.split == "domain":
        assert sum(diagnostics["domain_overlap"].values()) == 0, "Domain leakage!"
    else:
        print("WARNING: random split is an intentionally leaky reference, not primary evaluation")

    save_json(out / "data_split_diagnostics.json", {
        "dataset": ds_stats, "split": diagnostics,
        "split_type": args.split,
        "source_file_sha256": checksum(data_path),
        "python_version": platform.python_version(),
    })
    audit = shortcut_audit(df)
    audit.to_csv(out / "shortcut_audit.csv")
    plot_shortcut_audit(audit, out / "shortcut_audit.png")
    print("Shortcut audit complete.")

    print("Running majority, simple formatting, shortcut-only and TF-IDF baselines...")
    all_results = baseline_results(train, test, cfg)
    pd.DataFrame(all_results).to_csv(out / "baselines.csv", index=False)

    if args.audit_only:
        print("Audit-only mode finished. Outputs:", out)
        return

    from phishcnn.model import build_cnn, fit_cnn, probabilities
    import gc
    import tensorflow as tf

    max_len = int(min(cfg["max_length"], max(16, np.percentile(
        train["URL"].map(lambda x: len(url_representation(x, "raw"))),
        cfg["length_train_percentile"]
    ))))
    print("MAX_LEN (derived only from training URLs):", max_len)

    reference_models = {}
    reference_probs = {}
    test_rows = []
    robustness_rows = []
    seed_list = cfg["seeds"]
    configurations = [
        ("raw", "multi_357", cfg["width_configs"]["multi_357"]),
        *[("no_prefix", name, widths) for name, widths in cfg["width_configs"].items()],
    ]

    for representation, config_name, widths in configurations:
        for seed in seed_list:
            name = f"cnn_{representation}_{config_name}"
            print(f"\n== {name} seed={seed} ==")
            tr_text = train["URL"].map(lambda x: url_representation(x, representation))
            model = build_cnn(tr_text, max_len, widths, cfg, seed=seed)
            history = fit_cnn(model, train, val, representation, cfg, seed=seed)
            p_test = probabilities(model, test["URL"], representation, cfg["batch_size"])
            row = evaluate(test["label"], p_test, name)
            row.update({"seed": seed, "widths": str(widths), "representation": representation})
            test_rows.append(row)
            print({k: row[k] for k in ["model", "seed", "f1", "pr_auc", "roc_auc"]})
            save_json(out / f"history_{name}_seed{seed}.json", history.history)

            if config_name == "multi_357" and seed == seed_list[0]:
                artifact_dir = Path("artifacts")
                artifact_dir.mkdir(parents=True, exist_ok=True)
                model_file = artifact_dir / f"{representation}_multi_357_seed{seed}.keras"
                model.save(model_file)
                save_json(model_file.with_suffix(".json"), {
                    "representation": representation, "widths": widths,
                    "max_length": max_len, "threshold": 0.5,
                    "seed": seed, "dataset_sha256": checksum(data_path),
                    "warning": "Research-only prediction; not a production security control",
                })
                reference_models[representation] = model_file
                reference_probs[representation] = p_test

            del model
            tf.keras.backend.clear_session()
            gc.collect()

    all_results.extend(test_rows)
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(out / "all_model_results.csv", index=False)
    pd.DataFrame(test_rows).groupby("model")[["accuracy", "precision", "recall", "f1", "pr_auc", "roc_auc"]].agg(
        ["mean", "std"]
    ).to_csv(out / "cnn_width_and_input_summary.csv")
    plot_f1(test_rows, out / "cnn_f1_comparison.png")

    for representation in ["raw", "no_prefix"]:
        if representation not in reference_models:
            continue
        model = tf.keras.models.load_model(reference_models[representation], compile=False)
        p_original = reference_probs[representation]
        for kind in ["https_www", "http_no_www"]:
            changed_urls = test["URL"].map(lambda url: force_prefix(url, kind))
            p_new = probabilities(model, changed_urls, representation, cfg["batch_size"])
            robust_row = robustness(test["label"], p_original, p_new,
                                    f"{representation}_force_{kind}")
            robust_row["representation"] = representation
            robust_row["perturbation"] = kind
            robustness_rows.append(robust_row)
        del model
        tf.keras.backend.clear_session()
        gc.collect()
    pd.DataFrame(robustness_rows).to_csv(out / "prefix_robustness.csv", index=False)

    if "no_prefix" in reference_probs:
        p = reference_probs["no_prefix"]
        ci = domain_bootstrap_f1(test, p, n_boot=500, seed=cfg["seed"])
        save_json(out / "domain_bootstrap_f1.json", ci)
        errors = test[["URL", "domain_group", "label"]].copy()
        errors["probability_phishing"] = p
        errors["predicted_phishing"] = (p >= 0.5).astype(int)
        errors["error"] = np.where(errors["predicted_phishing"] == errors["label"],
                                   "correct", np.where(errors["label"] == 1, "FN", "FP"))
        errors.to_csv(out / "error_analysis_urls.csv", index=False)
        print("No-prefix multi-width 95% domain-bootstrap F1 CI:", ci)

    print("\nFinished. Outputs (IGNORED by Git):", out)
    print("Best use of results: compare domain-disjoint, bias-controlled models and robustness.")
    if args.quick:
        print("WARNING: QUICK MODE IS ONLY A SMOKE TEST. DO NOT USE METRICS IN YOUR FINAL REPORT.")


if __name__ == "__main__":
    main()
