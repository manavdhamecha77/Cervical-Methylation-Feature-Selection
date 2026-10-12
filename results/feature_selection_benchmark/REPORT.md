# Feature Selection Benchmark Report
## Cervical Cancer DNA Methylation — GSE99511
### Leakage-Free Small Panel Discovery

*Generated: 2026-10-06 14:42*

---

## 1. Objective

Identify small panels of original CpG sites (k ∈ {5, 10, 20, 50}) that best distinguish
Normal cervical tissue from CIN3, for use in a low-cost biosensor device.
All selection steps are performed inside each CV fold — no test data leaks into selection.

---

## 2. Data

| Property | Value |
|----------|-------|
| Dataset | GSE99511 (GEO) |
| Platform | Illumina HumanMethylation450K |
| Primary samples | 64 (Normal=28, CIN3=36) |
| CpGs (post-QC) | 405,336 |
| Feature values | Beta values (0–1, bimodal) |

---

## 3. Methods

### Pre-filter (inside each training fold)
1. Mean imputation (training mean only)
2. Variance filter (threshold = 0.001)
3. Top 5000 by variance

### Selection Methods
| Name | Description |
|------|-------------|
| MW Filter | Mann-Whitney U p-value ranking (non-parametric, baseline) |
| Lasso (L1) | L1 logistic regression, rank by \|coef\| |
| Elastic Net | L1+L2 logistic regression (l1_ratio ∈ {0.3, 0.5, 0.7}) |
| Stability Sel. | Repeated stratified subsampling + L1 LR, rank by selection frequency |
| SVM-RFE | Linear SVM recursive elimination |
| mRMR | Greedy max-relevance min-redundancy (pure numpy) |
| RF Importance | Random Forest mean-decrease-impurity |

### Evaluation Protocol
- **Outer CV**: 5-fold × 2 repeats = 10 outer folds
- **Inner CV**: 3-fold for classifier hyperparameter selection
- **Classifiers**: Logistic Regression, Linear SVM
- **No step uses test-fold data** — pre-filter, selection, scaling, hyperparameter tuning
  are all fit on training folds only.
- Explicit assertion in code: `assert len(set(train_idx) & set(val_idx)) == 0`

### Panel Sizes
k ∈ { 5, 10, 20, 50 }

---

## 4. Results Summary

### Performance Table (Mean ± SD across 10 outer folds)

> **Disclaimer**: n=64, high-dimensional. Standard deviations are large (0.10–0.25).
> Small differences between methods are NOT meaningful. Focus on patterns, not exact numbers.

| Selector | k | LogReg AUC | LogReg Bal.Acc | LinSVM AUC | LinSVM Bal.Acc |
|------------|-----|--------------|------------------|--------------|------------------|
| MW Filter | 5 | 0.565 ± 0.150 | 0.570 ± 0.158 | 0.581 ± 0.176 | 0.561 ± 0.125 |
| MW Filter | 10 | 0.555 ± 0.188 | 0.532 ± 0.172 | 0.533 ± 0.157 | 0.514 ± 0.158 |
| MW Filter | 20 | 0.582 ± 0.192 | 0.542 ± 0.167 | 0.593 ± 0.206 | 0.578 ± 0.150 |
| MW Filter | 50 | 0.588 ± 0.175 | 0.561 ± 0.109 | 0.632 ± 0.179 | 0.559 ± 0.109 |
| RF Importance | 5 | 0.547 ± 0.191 | 0.550 ± 0.143 | 0.562 ± 0.206 | 0.543 ± 0.151 |
| RF Importance | 10 | 0.573 ± 0.152 | 0.549 ± 0.111 | 0.565 ± 0.237 | 0.576 ± 0.196 |
| RF Importance | 20 | 0.588 ± 0.205 | 0.592 ± 0.093 | 0.568 ± 0.242 | 0.566 ± 0.193 |
| RF Importance | 50 | 0.591 ± 0.259 | 0.603 ± 0.209 | 0.617 ± 0.257 | 0.594 ± 0.144 |

### Best Configuration
- Selector: **RF Importance**
- Panel size: **k = 50**
- Classifier: **logreg**
- Mean Balanced Accuracy: **0.603**

---

## 5. Baseline Reproduction

Original pipeline (5-fold, single repeat): Bal.Acc ≈ 0.640 ± 0.231  
Reproduced (5×10 repeated nested CV):       Bal.Acc = 0.591 ± 0.171  
                                            ROC-AUC = 0.555 ± 0.199  

Difference in means: +0.049 (positive = original was inflated)

---

## 6. Permutation Test

_Permutation test not run._

---

## 7. Sensitivity Check (Pre-filter Size)

Pre-filter=5000: best Bal.Acc = 0.603  
Pre-filter=2000: best Bal.Acc = 0.631  
Conclusion: Results are robust to pre-filter size.

---

## 8. Feature Stability

| Selector | k | Jaccard | Kuncheva Index | n_folds |
|----------|---|---------|----------------|---------|
| MW Filter | 5 | 0.257 | 0.377 | 20 |
| MW Filter | 10 | 0.219 | 0.326 | 20 |
| MW Filter | 20 | 0.228 | 0.336 | 20 |
| MW Filter | 50 | 0.262 | 0.381 | 20 |
| RF Importance | 5 | 0.093 | 0.124 | 20 |
| RF Importance | 10 | 0.100 | 0.139 | 20 |
| RF Importance | 20 | 0.097 | 0.136 | 20 |
| RF Importance | 50 | 0.104 | 0.149 | 20 |

**Interpretation**: Kuncheva Index > 0 means better-than-chance stability.
At n=64, instability is expected — no single fold has enough samples to reliably
identify the same CpGs. Stability should be treated as a secondary criterion.

---

## 9. Consensus Panels


**k = 5**

| Selector | Consensus Panel CpGs |
|----------|----------------------|
| MW Filter | cg06260815, cg06834912, cg06776703, cg09352518, cg07353006 |
| RF Importance | cg11685843, cg06260815, cg06834912, cg15361697, cg02025737 |

**k = 10**

| Selector | Consensus Panel CpGs |
|----------|----------------------|
| MW Filter | cg06260815, cg06776703, cg09352518, cg06834912, cg14263858, cg07353006, cg03616722, cg02558625, cg11041161, cg07455789 |
| RF Importance | cg06260815, cg06834912, cg11685843, cg16006841, cg17227257, cg17509989, cg02025737, cg04650789, cg10327440, cg15361697 |

**k = 20**

| Selector | Consensus Panel CpGs |
|----------|----------------------|
| MW Filter | cg06260815, cg14263858, cg06776703, cg09352518, cg11041161, cg14199455, cg16006841, cg15415945, cg06834912, cg03616722, cg02558625, cg23657099, cg07353006, cg16792302, cg20756026, cg17240976, cg07455789, cg00025044, cg15361697, cg01243371 |
| RF Importance | cg06260815, cg11685843, cg16006841, cg06834912, cg13307782, cg07285237, cg02025737, cg21873524, cg17227257, cg11445797, cg06776703, cg03395511, cg04650789, cg09352518, cg17509989, cg20756026, cg11041161, cg15361697, cg16602369, cg05573767 |

**k = 50**

| Selector | Consensus Panel CpGs |
|----------|----------------------|
| MW Filter | cg02558625, cg06260815, cg06776703, cg14263858, cg07455789, cg16792302, cg03616722, cg25528916, cg15415945, cg09352518, cg12011343, cg15644634, cg11685843, cg10846615, cg08188268, cg11041161, cg07031797, cg07353006, cg16006841, cg16551732, cg14746387, cg13356884, cg12466610, cg14199455, cg15361697, cg17509989, cg12435665, cg23657099, cg21873524, cg06834912, cg06060754, cg23348582, cg27134767, cg23440816, cg27064482, cg05111645, cg00025044, cg20756026, cg03188948, cg11445109, cg05194426, cg25923609, cg01543583, cg03748376, cg21548772, cg17240976, cg21092413, cg18759102, cg08068630, cg01243371 |
| RF Importance | cg11685843, cg06260815, cg02025737, cg17227257, cg16006841, cg21873524, cg06834912, cg17509989, cg18110333, cg10909185, cg15361697, cg00895174, cg02983911, cg16602369, cg05573767, cg00907810, cg09352518, cg02632490, cg15415945, cg16792302, cg14911689, cg05111645, cg20756026, cg07285237, cg17179923, cg03016934, cg13307782, cg06776703, cg03395511, cg04650789, cg26603632, cg01062942, cg16441068, cg10317314, cg25923609, cg27449836, cg16594913, cg15934776, cg15019001, cg13685129, cg00892742, cg10327440, cg03658690, cg23214755, cg03928812, cg01201512, cg05136804, cg02483735, cg02268561, cg14263858 |

---

## 10. Biological Annotation

### Known Cervical Methylation Marker Overlap

Checked against: FAM19A4, PAX1, ZNF582, EPB41L3, MIR124-2, JAM3, CADM1, MAL, DAPK1, RARB, TWIST1, RASSF1

No selected CpGs map to the known cervical marker gene list.

> **Important**: This is a sanity check only. Overlap with known markers supports
> plausibility but does not validate the CpGs. Absence of overlap does not mean
> the CpGs are wrong — they may be novel candidates.

---

## 11. Batch Covariate Warning

See `batch_report.txt` for full details.

Chip `9340996078` contains ONLY CIN3 samples. This creates a chip-confound where
any CpG correlated with methylation batch effects on that chip could appear discriminative.
**All candidate CpGs must be validated in an independent cohort before biological claims.**

---

## 12. Limitations

1. **Sample size**: n=64 — all results have wide confidence intervals.
2. **Batch effects**: Chip `9340996078` is 100% CIN3. Selected CpGs may reflect batch.
3. **No independent validation**: Results cannot be generalized without external replication.
4. **Beta values**: Bimodal distribution; parametric methods (Lasso, SVM) may be suboptimal.
5. **mRMR uses Pearson correlation** as a proxy for mutual information — this is an approximation.
6. **SVM-RFE instability** at small panel sizes is expected (wrapper methods need large n).
7. **No biological validation**: CpGs are computational candidates. Laboratory confirmation required.
8. **Stability selection C=0.05**: This is a fixed hyperparameter; optimal C may vary.

---

## 13. Recommendations for Next Steps

1. **Wet-lab validation**: bisulfite pyrosequencing on the top-k consensus panel CpGs.
2. **Independent cohort**: obtain and test on a second dataset (e.g., TCGA-CESC methylation).
3. **ComBat batch correction**: apply before running this pipeline to remove Sentrix chip effects.
4. **Targeted assay**: design a methylation-specific PCR or LAMP assay for the k=5 or k=10 panel.
5. **M-value transformation**: re-run with M-values (logit(beta)) which are more homoscedastic.

---

## 14. File Index

```
results/feature_selection_benchmark/
├── batch_report.txt
├── raw_results/
│   ├── fold_results_main.csv        # per-fold metrics
│   ├── aggregated_results_main.csv  # mean±std summary
│   ├── fold_results_pf2000.csv  # sensitivity run
│   ├── baseline_rf5000.csv          # baseline reproduction
│   └── permutation_test.json        # null distribution
├── stability/
│   ├── stability_metrics.csv        # Jaccard, Kuncheva per (sel, k)
│   └── frequency_*.csv              # CpG selection frequencies
├── annotation/
│   └── panel_annotations.csv        # gene, chromosome, CpG island
├── panels/
│   └── consensus_panels.csv         # recommended panels
└── plots/
    ├── performance_vs_k_*.png
    ├── stability_vs_k_*.png
    ├── selection_frequency_heatmap_k*.png
    ├── method_comparison_*.png
    ├── permutation_test.png
    ├── correlation_heatmap_*.png
    ├── baseline_comparison.png
    └── summary_*.png
```

---

*This report was auto-generated by `feature_selection/run_benchmark.py`.*
*No results are computed outside of CV folds. No annotations are fabricated.*
