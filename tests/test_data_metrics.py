import numpy as np
import pandas as pd
from phishcnn.data import domain_disjoint_split, split_diagnostics
from phishcnn.metrics import evaluate, robustness, domain_bootstrap_f1


def fake_data():
    rows = []
    for i in range(100):
        rows.append({"URL": f"https://a{i}.example{i}.com/p", "label": i % 2,
                     "domain_group": f"example{i}.com"})
    return pd.DataFrame(rows)


def test_domain_disjoint_reproducible():
    df = fake_data()
    a = domain_disjoint_split(df, candidates=4)
    b = domain_disjoint_split(df, candidates=4)
    assert [x["URL"].tolist() for x in a] == [x["URL"].tolist() for x in b]
    d = split_diagnostics(*a)
    assert all(v == 0 for v in d["domain_overlap"].values())
    assert sum(len(x) for x in a) == len(df)


def test_evaluate_phishing_positive():
    vals = evaluate([0, 0, 1, 1], [0.1, 0.9, 0.8, 0.2], "toy")
    assert vals["tp"] == vals["tn"] == vals["fp"] == vals["fn"] == 1
    assert vals["f1"] == 0.5
    d = robustness([0, 1], [.2, .8], [.8, .2], "flip")
    assert d["prediction_flip_rate"] == 1.0


def test_domain_cluster_ci():
    df = fake_data().iloc[:30]
    pred = np.where(df["label"] == 1, 0.9, 0.1)
    ci = domain_bootstrap_f1(df, pred, n_boot=25)
    assert ci["f1_lower_95"] == 1.0


def test_load_and_clean_accepts_original_header_case_and_labels(tmp_path, monkeypatch):
    import phishcnn.data as data_module
    monkeypatch.setattr(data_module, "domain_group", lambda url: url.split("/")[2])
    csv = tmp_path / "original.csv"
    csv.write_text("URL,label\nhttps://one.test/a,1\nhttps://two.test/p,0\nhttps://one.test/a,1\n")
    df, stats = data_module.load_and_clean(csv)
    assert len(df) == 2
    assert list(df["label"]) == [0, 1]
    assert stats["after_exact_url_dedup"] == 2
