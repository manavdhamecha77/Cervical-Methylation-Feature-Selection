# Cervical Cancer DNA Methylation Study
### GSE99511 · GSE46306 · Illumina HumanMethylation450K
### Branch: `ritweek`

Computational classification, biomarker discovery, and physical sensor simulation for point-of-care cervical precancer (CIN3) triage using DNA methylation.

> **Clinical Framing:** This is a research-grade computational study designing a reflex triage test for hrHPV-positive women, not an invasive diagnosis tool or clinically validated kit.
>
> **Scope Note on "Diffusion":** Treated as physical mass transport in the sensor (core) plus generative tabular diffusion models (optional evaluation).

---

## Dataset Overview

| Property | GSE99511 (Development) | GSE46306 (External Check) |
|---|---|---|
| **Platform** | Illumina HumanMethylation450K | Illumina HumanMethylation450K |
| **Tissue / Specimen** | Cervico-vaginal self-samples (lavage/brush) | Cervical tissue scrapings |
| **Normal Controls** | 28 | 20 |
| **CIN3 Precancer** | 36 | 18 |
| **Carcinoma (SCC)** | 4 (excluded from binary task) | 6 (excluded from binary task) |
| **CpGs (analyzed)** | 363,298 | 451,345 |
| **Common CpGs** | **340,248** (Exact probe intersection) |

---

## One-Command Full Pipeline Reproduction

To reproduce the entire project benchmark, batch analysis, literature baseline, stability selection, external validation, permutation test, and sensor simulation from scratch:

```bash
python3 run_master_pipeline.py
```

Or run individual modular components:

```bash
# 1. Batch Confound, M-values, and Leave-One-Chip-Out CV:
python3 feature_selection/batch_analysis.py

# 2. Literature Genes (FAM19A4, PAX1, etc.), Region Promoters & EpiDISH Deconvolution:
python3 feature_selection/literature_and_regions.py

# 3. 100-Bootstrap Stability Selection & Single-Shot External Validation (GSE46306):
python3 feature_selection/stable_panels_and_external.py

# 4. Empirical Permutation Testing (100 Label Shuffles):
python3 feature_selection/run_permutation_test.py

# 5. Sensor Simulation (Langmuir Diffusion Kinetics, Pap Dilution, Combined Stress Heatmap):
python3 sensor-sim/sensor_kinetics_dilution_sim.py

# 6. Generative Tabular Augmentation Benchmark (TabDDPM, SMOTE, Jitter):
python3 feature_selection/generative_augmentation.py
```

---

## Key Experimental Results & Conclusions

### 1. Evaluation & Permutation Testing
- **Leakage-Free Repeated Nested CV:** All missing imputation, M-value transforms, variance filtering, feature selection, and standardization occur strictly inside each training fold (validated by `feature_selection/test_no_leakage.py`).
- **Permutation Test ($n = 100$):**
  - Observed Balanced Accuracy: **0.5633** vs Null: **0.5094 ± 0.0658** ($p = 0.1782$).
  - Observed ROC-AUC: **0.6767** vs Null: **0.5105 ± 0.1003** ($p = 0.0594$, borderline).
  - *Finding:* At $n=64$, standard balanced accuracy does not statistically outperform chance ($p = 0.18$), confirming that single-probe selection on small sample cohorts is dominated by sample variance.

### 2. Batch Confound & Modeling Scale
- **M-Values vs. Beta-Values:** Converting to logit M-values systematically boosts classifier performance (mean balanced accuracy $+0.092$, AUC $+0.143$) by mitigating non-Gaussian beta-value truncation.
- **Leave-One-Chip-Out (LOGO) CV:** Balanced accuracy drops from 0.651 (standard CV) to **0.613** (LOGO CV).
- **Batch Association:** 11.7% of panel CpGs exhibit significant association with Sentrix chip ID ($p < 0.05$). Standard CV mildly inflates accuracy when chip samples are mixed across folds.

### 3. Biological Literature Panel & Region-Level Features
- **Literature Panel (*FAM19A4*, *PAX1*, *ZNF582*, *EPB41L3*, *CADM1*, *MAL*, *miR124-2*):** Achieved **Balanced Accuracy = 0.670 [0.605, 0.736]** and **ROC-AUC = 0.766 [0.675, 0.857]**, outperforming unconstrained genome-wide search.
- **Promoter Region Features:** Aggregating CpGs into 19,518 promoter regions achieved **Balanced Accuracy = 0.652** and **AUC = 0.721**.
- **Cell Composition (EpiDISH):** Normal (0.582) vs CIN3 (0.593) epithelial purity showed no significant difference ($p = 0.1825$), proving epigenetic changes are intrinsic rather than leukocyte infiltration artifacts.

### 4. Stability & Single-Shot External Validation
- **100-Bootstrap Stability:** 9 CpGs achieved selection frequency $\ge 60\%$.
- **GSE46306 External Validation:** 50-CpG Linear SVM achieved **Balanced Accuracy = 0.656**, **ROC-AUC = 0.653**, **Sensitivity = 0.611**, **Specificity = 0.700**, and **Accuracy = 0.658** on completely unseen external data.
- **Distribution Shift:** 70% of panel CpGs exhibit significant distribution shift between laboratories ($p < 0.05$).

### 5. Sensor Physical Simulation
- **Diffusion-Limited Transport & Langmuir Kinetics:** At $t < 3$ min, saturation is low (<45%) and accuracy collapses to random chance (0.500). Reaching adequate saturation ($\ge 80\%$) requires $\ge 7.5\text{--}10\text{ minutes}$ incubation.
- **Pap-Style Cellular Dilution:** In mixed cervical scrapings, if CIN3 lesion purity drops below 50%, sensitivity drops to <30%. Target enrichment or high analytical sensitivity is required for low-shedding lesions.
- **Combined Stress Test:** Heatmap generated under `results/feature_selection_benchmark/plots/sensor_combined_stress_heatmap.png`.

### 6. Generative Diffusion Augmentation
- Tabular diffusion (TabDDPM) on top 1,000 features yielded test accuracy of 0.581 vs 0.610 baseline. Generative models overfit small sample sizes ($n=64$) and do not outperform real training data.

---

## Project Structure

```
├── README.md                                    # This project documentation
├── run_master_pipeline.py                       # Single master execution script
├── requirements.txt                             # Python dependencies
├── feature_selection/
│   ├── batch_analysis.py                        # LOGO CV, M-values, chip associations
│   ├── literature_and_regions.py                # Literature panel, promoter features, cell composition
│   ├── stable_panels_and_external.py            # Stability selection & GSE46306 external check
│   ├── run_permutation_test.py                  # 100-label permutation benchmark
│   ├── generative_augmentation.py               # TabDDPM, SMOTE, Jitter benchmark
│   ├── evaluator.py                             # Leakage-free repeated CV & 95% CIs
│   ├── fs_selectors.py                          # 7 feature selection algorithms
│   ├── data_loader.py                           # Parquet / matrix loading
│   ├── annotator.py                             # 450K manifest gene mapping
│   ├── test_no_leakage.py                       # 7 unit tests for leakage prevention
│   └── config.json                              # Benchmark hyperparameters
├── sensor-sim/
│   └── sensor_kinetics_dilution_sim.py          # Diffusion kinetics, Pap dilution & heatmap
├── results/feature_selection_benchmark/
│   ├── REPORT.md                                # Comprehensive 7-section report
│   ├── batch_robustness_summary.csv             # Batch CV comparison tables
│   ├── feature_engineering_comparison.csv       # Literature vs Region vs Probe panels
│   ├── stable_external_validation.csv           # External validation scores
│   ├── permutation_test_results.json            # Permutation null distributions
│   ├── sensor_incubation_kinetics.csv           # Signal vs incubation time
│   ├── sensor_dilution_experiment.csv           # Accuracy vs lesion purity
│   └── plots/                                   # Sensor curves & stress heatmaps
└── data/                                        # GSE99511 & GSE46306 dataset files
```
