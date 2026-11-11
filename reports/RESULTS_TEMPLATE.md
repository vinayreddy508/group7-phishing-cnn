# Group 7 — Final Results Template

Fill only after running the code. **Do not invent results.**

## Research questions

1. Which trivial URL formatting shortcuts correlate with phishing labels?
2. What does domain-disjoint evaluation show relative to optional random splitting?
3. Does prefix normalization change raw-CNN classification?
4. What is the impact of single convolution widths 3/5/7 versus parallel widths [3,5,7]?
5. How sensitive is the raw model to forced scheme and www edits?

## Dataset and controls

- Official dataset citation and download date:
- Original class distribution:
- Duplicate and contradictory URL count:
- Unique registrable domains:
- Train / validation / test rows, class rates:
- Domain intersection counts:
- Data and config hashes:

## Main results table

Copy numbers from runs/domain_disjoint/all_model_results.csv; for multi-seed CNNs report mean ± std.

| Model | F1 phishing | Precision | Recall | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Trivial formatting rule | TBD | TBD | TBD | TBD | TBD |
| Formatting-only logistic regression | TBD | TBD | TBD | TBD | TBD |
| Character TF-IDF + Logistic Regression | TBD | TBD | TBD | TBD | TBD |
| Raw multi-width CNN | TBD | TBD | TBD | TBD | TBD |
| No-prefix width-3 CNN | TBD | TBD | TBD | TBD | TBD |
| No-prefix width-5 CNN | TBD | TBD | TBD | TBD | TBD |
| No-prefix width-7 CNN | TBD | TBD | TBD | TBD | TBD |
| No-prefix multi-width CNN | TBD | TBD | TBD | TBD | TBD |

## Analysis checklist

- Explain the practical significance of the prefix-rule score.
- Compare raw vs no-prefix, not solely CNN vs logistic regression.
- Explain domain-disjoint validation and overlap zero.
- Discuss width comparison and its parameter-count confounder.
- Describe prediction flip rate and mean absolute probability shift.
- Show key false-positive/false-negative examples; redact sensitive URLs if publishing.
- Report domain-cluster bootstrap confidence interval.
- Acknowledge that removing a pattern mechanically guarantees invariance to that pattern.
- Acknowledge the dataset is not a live deployment/out-of-time test.
- Report exact hyperparameters, seeds, software versions, and limitations.
