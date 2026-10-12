# GSE46306 external cohort check

GSE99511 was used for feature prefiltering, feature selection, scaling, and
classifier tuning. GSE46306 labels were used only for final metrics.

The matrix values were converted from GEO's documented arcsine-square-root
transformation back to beta values with `sin(value) ** 2`. The main comparison
uses 20 Normal and 18 CIN3 samples; six cancer samples are excluded.

## Highest scoring evaluated configurations

These are descriptive external scores; they were not used to select or tune
the panel. Full metrics for all 16 evaluated configurations are in
`external_metrics.csv`.

| Model | Balanced accuracy | ROC-AUC | Sensitivity | Specificity |
|---|---:|---:|---:|---:|
| `mw_filter_k50_linear_svm` | 0.614 | 0.639 | 0.778 | 0.450 |
| `rf_importance_k5_linear_svm` | 0.506 | 0.439 | 0.111 | 0.900 |
| `mw_filter_k5_logreg` | 0.503 | 0.519 | 0.056 | 0.950 |
| `mw_filter_k20_logreg` | 0.503 | 0.589 | 0.056 | 0.950 |
| `mw_filter_k5_linear_svm` | 0.503 | 0.497 | 0.056 | 0.950 |

This is a small, cautious external check. GSE46306 normal controls are HPV
negative and CIN3 cases are HPV positive, so class and HPV status are
confounded. Its preprocessing also differs from GSE99511. Do not interpret
these scores as validation for HPV-positive triage or as clinical performance.

The `.joblib` files are candidate models trained on all GSE99511 Normal/CIN3
samples. They are research artifacts, not final clinical models. They expect
beta values in the listed CpG order.
