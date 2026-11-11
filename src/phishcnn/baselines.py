"""Simple formatting-rule, feature-only, and character TF-IDF baselines."""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .urls import shortcut_features, url_representation
from .metrics import evaluate


def baseline_results(train, test, cfg):
    rows = []
    y = test["label"].to_numpy()
    majority = int(train["label"].mean() >= 0.5)
    rows.append(evaluate(y, np.full(len(test), majority), "majority"))

    trivial = np.array([not u.lower().startswith("https://www.") for u in test["URL"]], dtype=int)
    rows.append(evaluate(y, trivial, "one_line_not_https_www"))

    features_train = pd.DataFrame([shortcut_features(u) for u in train["URL"]])
    features_test = pd.DataFrame([shortcut_features(u) for u in test["URL"]])
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, random_state=cfg["seed"]))
    model.fit(features_train, train["label"])
    p = model.predict_proba(features_test)[:, 1]
    rows.append(evaluate(y, p, "formatting_features_logreg"))

    for representation in ["raw", "no_prefix"]:
        vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 5),
                              max_features=cfg["tfidf_max_features"], min_df=2,
                              dtype=np.float32, sublinear_tf=True)
        tr = train["URL"].map(lambda u: url_representation(u, representation))
        te = test["URL"].map(lambda u: url_representation(u, representation))
        mat_tr = vec.fit_transform(tr)
        mat_te = vec.transform(te)
        clf = LogisticRegression(max_iter=300, solver="liblinear", random_state=cfg["seed"])
        clf.fit(mat_tr, train["label"])
        probs = clf.predict_proba(mat_te)[:, 1]
        rows.append(evaluate(y, probs, f"tfidf_logreg_{representation}"))
    return rows


def shortcut_audit(df):
    feats = pd.DataFrame([shortcut_features(u) for u in df["URL"]])
    feats["label"] = df["label"].to_numpy()
    means = feats.groupby("label").mean().T.rename(columns={0: "legitimate", 1: "phishing"})
    means["absolute_difference"] = abs(means["legitimate"] - means["phishing"])
    return means.sort_values("absolute_difference", ascending=False)
