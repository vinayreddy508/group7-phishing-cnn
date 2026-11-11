"""All metrics use phishing=1 (positive class), legitimate=0."""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score, confusion_matrix,
)


def evaluate(y_true, prob_phishing, name: str, threshold=0.5):
    y = np.asarray(y_true, dtype=int).ravel()
    p = np.asarray(prob_phishing, dtype=float).ravel()
    pred = (p >= threshold).astype(int)
    if len(y) != len(p):
        raise ValueError("y_true and probability arrays differ in length")
    return {
        "model": name,
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else None,
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None,
        "tn": int(confusion_matrix(y, pred, labels=[0, 1])[0, 0]),
        "fp": int(confusion_matrix(y, pred, labels=[0, 1])[0, 1]),
        "fn": int(confusion_matrix(y, pred, labels=[0, 1])[1, 0]),
        "tp": int(confusion_matrix(y, pred, labels=[0, 1])[1, 1]),
    }


def robustness(y_true, orig_probs, changed_probs, name):
    a, b = np.asarray(orig_probs), np.asarray(changed_probs)
    row = evaluate(y_true, b, name)
    row["mean_abs_probability_shift"] = float(np.mean(np.abs(a - b)))
    row["prediction_flip_rate"] = float(np.mean((a >= 0.5) != (b >= 0.5)))
    return row


def domain_bootstrap_f1(test_df, probs, n_boot=500, seed=42):
    """Domain-cluster bootstrap: resample domains instead of individual URLs."""
    groups = test_df["domain_group"].to_numpy()
    labels = test_df["label"].to_numpy()
    p = np.asarray(probs)
    domain_indices = {}
    for i, domain in enumerate(groups):
        domain_indices.setdefault(domain, []).append(i)
    domains = np.array(list(domain_indices), dtype=object)
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_boot):
        drawn = rng.choice(domains, size=len(domains), replace=True)
        idx = np.concatenate([domain_indices[d] for d in drawn])
        if len(set(labels[idx])) == 2:
            scores.append(f1_score(labels[idx], p[idx] >= 0.5, zero_division=0))
    return {"f1_lower_95": float(np.percentile(scores, 2.5)),
            "f1_median": float(np.percentile(scores, 50)),
            "f1_upper_95": float(np.percentile(scores, 97.5)),
            "bootstrap_replicates": len(scores)} if scores else {}
