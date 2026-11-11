# Group 7 — Bias-Aware Phishing URL Detection with Character-Level CNNs

**Team:** Vinay Kumar Reddy · Rithesh Reddy · Balaji

## Purpose

**Research question:** Do character-level CNNs learn phishing URL patterns, or do they exploit superficial URL formatting shortcuts in the PhiUSIIL dataset?

This is an independent, reproducible Python/TensorFlow implementation. It investigates dataset bias using a trivial formatting rule, feature-only and TF-IDF logistic-regression baselines, registrable-domain-disjoint splitting, and CNNs with convolution widths 3, 5, 7, and parallel 3+5+7. The project also tests prediction changes under controlled \`https://www.\` versus \`http://\` prefix edits.

**Research prototype only:** Do not interpret predictions as a guarantee that a website is safe.

## Files

- \`src/phishcnn/data.py\` — downloading, cleaning, labels, group-disjoint splitting
- \`src/phishcnn/urls.py\` — hostname/domain parsing, normalization, bias features and perturbations
- \`src/phishcnn/baselines.py\` — majority, trivial prefix rule, formatting-only logistic regression, TF-IDF
- \`src/phishcnn/model.py\` — independent character-level CNNs with configurable widths
- \`src/phishcnn/metrics.py\` — phishing-class F1, PR-AUC, ROC-AUC, confusion matrix, domain-cluster bootstrap
- \`src/phishcnn/reporting.py\` — CSV/JSON exports and visualizations
- \`scripts/download_data.py\` — fetch official UCI data
- \`scripts/run_experiments.py\` — run main evaluation end to end
- \`scripts/predict.py\` — inference from locally trained Keras model
- \`configs/default.json\` — seeds, architecture, training hyperparameters
- \`tests/\` — source-code tests
- \`reports/RESULTS_TEMPLATE.md\` — honest reporting checklist
- \`COLAB.md\` — Google Colab execution walkthrough

## Setup

Python **3.10–3.12** recommended. GPU strongly recommended for full training.

\`\`\`bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
\`\`\`

For Colab, clone this repo, change to its root, and install requirements there; see [COLAB.md](COLAB.md).

## Download dataset

Download **PhiUSIIL Phishing URL Dataset** from the official [UCI repository, dataset #967](https://archive.ics.uci.edu/dataset/967/phiusiil%2Bphishing%2Burl%2Bdataset). The scripts download the dataset automatically:

\`\`\`bash
python scripts/download_data.py --out data/phiusiil_urls.csv
\`\`\`

Only original \`URL\` and \`Label\` fields are saved locally. The original labels are **1=legitimate, 0=phishing**; internally we invert them to **1=phishing, 0=legitimate** so standard recall/F1 evaluate the phishing class.

Do not upload the raw dataset. It is excluded by \`.gitignore\`.

## Reproduce experiments

**Fast sanity check only (not final-report evidence):**

\`\`\`bash
python scripts/run_experiments.py --quick --output runs/smoke
\`\`\`

**Full domain-disjoint evaluation using all 3 configured seeds:**

\`\`\`bash
python scripts/run_experiments.py --config configs/default.json --output runs/domain_disjoint
\`\`\`

**Shortcut audit and baseline comparisons without CNN training:**

\`\`\`bash
python scripts/run_experiments.py --audit-only --output runs/audit
\`\`\`

**Optional random-URL split (a potentially leaky comparison, not primary results):**

\`\`\`bash
python scripts/run_experiments.py --split random --output runs/random_reference
\`\`\`

The main split holds registrable domains out of train/validation/test and checks that their overlap is zero. Approximate target fractions are 70% / 15% / 15%; group sizes and class proportions can make exact row counts differ. Validation PR-AUC is used for early stopping. Test metrics are for the final comparisons, not hyperparameter tuning.

Each full run trains raw multi-width and prefix-normalized multi-width CNNs, plus prefix-normalized single-width CNNs. Widths and seed list are in \`configs/default.json\`. The multi-width CNN has more parameters than single-width variants: discuss this in your limitations.

## Outputs (LOCAL, ignored by Git)

- \`data_split_diagnostics.json\` — dataset counts, grouping and overlap
- \`shortcut_audit.csv\`, \`shortcut_audit.png\` — class-wise pattern statistics
- \`baselines.csv\` — trivial formatting rule and logistic-regression baselines
- \`all_model_results.csv\` — CNN and baseline scores
- \`cnn_width_and_input_summary.csv\` — architecture means and standard deviations
- \`cnn_f1_comparison.png\` — F1 plot
- \`prefix_robustness.csv\` — prediction flips and probability shifts
- \`domain_bootstrap_f1.json\` — domain-cluster uncertainty
- \`error_analysis_urls.csv\` — per-URL error inspection (private/local)
- \`artifacts/*.keras\` — local, trained reference models for inference

**Metrics:** phishing precision, recall, F1 (primary), PR-AUC, ROC-AUC, accuracy, and confusion counts. Do not claim real-world deployment performance using only PhiUSIIL.

## Inference

Train the full or quick run first. Then:

\`\`\`bash
python scripts/predict.py --model artifacts/no_prefix_multi_357_seed42.keras --url "https://example.com/login"
\`\`\`

Or predict from a local text file (one URL per line): \`python scripts/predict.py --file urls_to_check.txt\`.

The inference script never visits the URLs; it only processes the input string.

## Tests

\`\`\`bash
python -m pytest -q
\`\`\`

## Interpretation and limitations

- A one-line rule predicting phishing when a URL does not start \`https://www.\` demonstrates whether the dataset contains easy class artifacts.
- Normalization intentionally removes scheme and leading \`www\` information; invariance of this normalized model to *these exact changes* is by design, not proof of semantic robustness.
- Forced \`http\` to \`https\` edits may change real-world network security. Report the test as *textual reliance*, not label-preserving security in deployment.
- The full-domain test is more resistant to memorizing domains than a random URL split, but does not represent an out-of-time phishing campaign test.
- Single-width/multi-width comparisons are not matched for parameter count.
- No performance numbers have been fabricated. Generate actual results before writing the report.

## Independent work and sources

The professor noted an existing character-level CNN implementation: [vihanlalan/phishing-prevention](https://github.com/vihanlalan/phishing-prevention). This repository's contribution is the independent implementation and **bias-aware, leakage-aware experimental protocol**, not a claim to have invented Char-CNN.

Dataset credit: Prasad, A., & Chandra, S. (2024). *PhiUSIIL: A diverse security profile empowered phishing URL detection framework based on similarity index and incremental learning*. *Computers & Security*. https://doi.org/10.1016/j.cose.2023.103545

## GitHub submission

Data, model weights, output runs, credentials and local environments are excluded by \`.gitignore\`. If this repository is private, give the instructor collaborator access. Submit the **repository URL**.
