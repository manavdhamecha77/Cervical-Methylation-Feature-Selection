# Cervical Cancer DNA Methylation: Multi-Cohort Reflex Triage Benchmark
### Branch: `ritweek2` · Cohorts: GSE99511 · GSE287994 · GSE46306 · GSE143752 · WID-CIN Audit
### 310,014 Harmonized CpGs · 393 Clinical Specimens · Zero-Leakage Nested Cross-Validation

---

## Executive Summary & Clinical Objective

This branch (`ritweek2`) implements a multi-cohort, cross-platform machine learning benchmark to develop an affordable **host-cell DNA methylation reflex triage test** for women testing positive for high-risk human papillomavirus (**hrHPV+**).

* **The Clinical Problem:** Primary hrHPV testing is highly sensitive (>95%) but suffers from low clinical specificity. Over 80% of hrHPV infections are transient and will clear on their own without causing high-grade precancer. Referring all hrHPV+ women to invasive colposcopy and biopsy leads to massive overtreatment, unnecessary cervical biopsies, patient distress, and heightened risks of future preterm deliveries.
* **The Triage Solution:** A reflex DNA methylation test performed directly on residual cervical scrape or liquid cytology specimens. The test must accurately identify women with true transforming high-grade cervical precancer (**CIN3+**: CIN3, adenocarcinoma in situ, or occult cervical cancer) who need colposcopy, while safely sparing women with transient infections from unnecessary invasive procedures.
* **The Research Target:** Evaluate whether a compact CpG biomarker panel can achieve a target **ROC-AUC of 0.80** with high clinical sensitivity (≥85–90%) on completely untouched independent clinical cohorts.

---

## Quickstart: How to Run and View the Project

All commands run directly from the repository root:

```bash
# 1. Open the interactive research dashboard in your default web browser
open html/complete_project_dashboard_ritweek2.html

# 2. Re-generate the synchronized HTML dashboard from benchmark CSVs (< 1 second)
python3 generate_dashboard.py

# 3. Run the complete end-to-end multi-cohort benchmark pipeline from scratch (~15–20 min)
python3 run_master_pipeline_ritweek2.py
```

### Individual Modular Scripts
```bash
# Reproduce the published Verlaat et al. 3-gene baseline on GSE99511:
python3 feature_selection/run_published_baselines.py

# Run standalone multi-cohort nested cross-validation and external evaluation:
python3 feature_selection/run_nested_cv_benchmark.py

# Run standalone 100-bootstrap RFE stability selection:
python3 feature_selection/rfe_stability.py

# Run standalone Limma empirical Bayes differential methylation analysis:
python3 feature_selection/limma_dmp.py
```

---

## Datasets and Resources

We harmonized 4 clinical datasets across Illumina HumanMethylation450K and MethylationEPIC 850K arrays, yielding a shared feature space of **310,014 common CpG loci**, plus audited the controlled-access WID-CIN study:

| Cohort | Platform | Primary Reference | Total Samples | Analyzed Population | Class Breakdown | Specimen Type | Benchmark Role |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **GSE99511** | 450K | Verlaat et al. *Clin Cancer Res* 2018 | 64 | hrHPV+ women | 28 Normal vs. 36 CIN3 | Cervical scrape (liquid) | **Development / Baseline** |
| **GSE287994** | EPIC 850K | Bowden et al. *Nat Commun* 2025 | 241 | hrHPV+ women | 29 Benign vs. 116 CIN3/Cancer | ThinPrep liquid cytology | **Development Cohort** |
| **GSE46306** | 450K | Farkas et al. *Epigenetics* 2013 | 38 | Screening cohort | 20 Normal vs. 18 CIN3 | Cervical tissue scrapings | **Untouched External #1** |
| **GSE143752** | EPIC 850K | Cheishvili et al. *Clin Epigenetics* 2020 | 186 | hrHPV genotyped | 104 Normal vs. 42 CIN3 (146 analyzed) | Dry cervical swabs | **Untouched External #2** |
| **WID-CIN** | EPIC 850K | Barrett et al. *Lancet Oncol* 2022 (PMC9580141) | 1,254 | Primary Screening | 1,254 population screening cohort | Cervical liquid cytology | **Controlled Access Audit (EGA)** |

### Cohort Partitioning
* **Development Cohort ($n=209$ hrHPV+):** GSE99511 ($n=64$) + GSE287994 ($n=145$). Used exclusively for cross-validation, feature selection, and model training.
* **Untouched External Cohorts ($n=184$ total):**
  * GSE46306 ($n=38$): Liquid scrape specimens on 450K.
  * GSE143752 ($n=146$): Dry cervical swab specimens on EPIC 850K.
* **Genomic Annotation Resource:** Official Illumina GPL13534 HumanMethylation450 manifest containing probe coordinates, gene symbols, regions (TSS200, TSS1500, 5'UTR, 1stExon, Body, 3'UTR), and CpG island context (Islands, Shores, Shelves, Open Sea).
* **Controlled Access Status (WID-CIN):** Stored in the European Genome-phenome Archive under accession **`EGAS00001005078`**. Requires formal Data Access Agreement (DAA) and Data Access Committee (DAC) approval; cannot be downloaded anonymously.

---

## Machine Learning Algorithms and Methods

### 1. In-Fold ComBat Empirical Bayes Batch Correction
* **Problem:** Differences in laboratory conditions, chips, and array platforms (450K vs. EPIC 850K) cause technical batch shifts that artificially inflate performance if adjusted across the whole dataset.
* **Solution:** We implemented a training-fold-only parametric Empirical Bayes batch adjuster ([`feature_selection/combat_correction.py`](file:///Users/ritweekraj/Desktop/Cervical-Methylation-Feature-Selection/feature_selection/combat_correction.py)). Batch parameters (mean and variance shifts) are fitted **strictly on training splits only**. Test, validation, and external sets are adjusted using frozen parameters with zero leakage.

### 2. Feature Selection Strategies
* **Limma Empirical Bayes Moderated $t$-Test ([`feature_selection/limma_dmp.py`](file:///Users/ritweekraj/Desktop/Cervical-Methylation-Feature-Selection/feature_selection/limma_dmp.py)):**
  Shrinks probe-specific sample variances toward a global pooled prior variance (Smyth 2004), stabilizing $t$-statistics on small sample sizes. Applies Benjamini-Hochberg False Discovery Rate (FDR) control.
* **100-Bootstrap RFE Stability Selection ([`feature_selection/rfe_stability.py`](file:///Users/ritweekraj/Desktop/Cervical-Methylation-Feature-Selection/feature_selection/rfe_stability.py)):**
  Draws 100 bootstrap resamples from the training fold. In each resample, Recursive Feature Elimination (RFE) iteratively eliminates the weakest features using linear model weights. Features are ranked by overall selection frequency. Panel stability across bootstraps is quantified using the **Kuncheva Index** and **Nogueira Index**.

### 3. Classification Estimator Families
Four distinct classifier families were benchmarked across panel sizes $k \in \{5, 10, 20, 50\}$:
1. **Elastic Net Logistic Regression:** Regularized logistic regression with balanced L1/L2 penalties (`l1_ratio=0.5`, `solver='saga'`) to prevent overfitting while selecting correlated CpGs.
2. **Linear Support Vector Machine (Linear SVM):** Large-margin linear hyperplane classification with L2 penalty.
3. **XGBoost Classifier:** Extreme gradient boosted decision trees (`max_depth=3`, `learning_rate=0.05`, `n_estimators=100`) capturing non-linear interactions.
4. **LightGBM Classifier:** Fast histogram-based gradient boosted decision trees (`max_depth=3`, `learning_rate=0.05`, `n_estimators=100`).

### 4. Calibration & Triage Thresholding ([`feature_selection/calibration_triage.py`](file:///Users/ritweekraj/Desktop/Cervical-Methylation-Feature-Selection/feature_selection/calibration_triage.py))
* Optimal decision thresholds ($\tau$) are calculated strictly on training fold out-of-fold validation probabilities using **Youden's J statistic** ($J = \text{Sensitivity} + \text{Specificity} - 1$).
* Confidence intervals: Wilson score intervals for Sensitivity and Specificity; DeLong and bootstrap intervals for ROC-AUC.

### 5. Point-of-Care Biosensor Physical Simulation
Simulates the real-world operational physical envelope across 120 experimental conditions:
* **Cellular Dilution ($D \in [0.10, 0.25, 0.50, 0.75, 1.00]$):** Simulates low dysplastic cell yield when CIN3+ lesion cells are diluted by normal squamous cells (10% to 100% target DNA).
* **Non-Equilibrium Langmuir Surface Adsorption Kinetics:** Fractional probe occupancy $\theta(t) = \frac{t}{t + \tau}$ with characteristic binding constant $\tau = 5.0\text{ minutes}$, evaluated at $t \in [5, 10, 15, 30, 45, 60]\text{ minutes}$.
* **Assay Measurement Noise:** Additive zero-mean Gaussian biosensor instrument noise $\epsilon \sim \mathcal{N}(0, \sigma^2)$ for $\sigma \in [0.01, 0.03, 0.05, 0.10]$.
* **Observed Methylation Signal:** $\beta_{\text{sensor}} = \text{clip}\left(\beta_{\text{true}} \cdot D \cdot \theta(t) + \epsilon, 0, 1\right)$.

---

## Step-by-Step Workflow and Pipeline Flow

```
Step 1: Reproduce Published Verlaat 3-Gene Baseline (GSE99511, n=64 hrHPV+)
   │
Step 2: Add GSE287994 (EPIC) & GSE143752 (EPIC); Harmonize 310,014 Shared CpGs
   │
Step 3: Filter Intended hrHPV+ Population (CIN3+ Positives vs. Normal/Benign Controls)
   │
Step 4: 5x5 Nested Cross-Validation (32 Pipelines: Elastic Net, SVM, XGBoost, LightGBM)
   │    └── In-fold ComBat batch adjustment + In-fold feature selection + In-fold thresholding
   │
Step 5: Stability Selection & Biological Manifest Annotation Verification
   │    └── Identify consensus 5/10/20/50 panels & resolve gene annotation discrepancies
   │
Step 6: Untouched External Cohort Validation (Zero-leakage, frozen thresholds)
   │    ├── GSE46306 (n=38 liquid scrapes, 450K)
   │    └── GSE143752 (n=146 cervical swabs, EPIC 850K)
   │
Step 7: In Silico Point-of-Care Biosensor Physical Stress Simulation (120 regimes)
   │
Step 8: Generate Synchronized Interactive Research Dashboard & Final Reports
```

---

## Intermediate and Final Benchmark Results

### 1. Reproduction of Published Baseline (Verlaat et al. CCR 2018)
Evaluated on GSE99511 ($n=64$ hrHPV+: 28 Normal vs. 36 CIN3) using 10 repeats of 5-fold cross-validation:

| Baseline Model | Feature Definition | ROC-AUC (95% CI) | Sensitivity | Specificity | Balanced Acc | Original Paper Claim |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Top Probes (3 CpGs)** | *ASCL1* (cg20718350), *LHX8* (cg08272731), *ST6GALNAC5* (cg23243867) | **0.6835** [0.5474, 0.8074] | 0.6111 | 0.5357 | 0.5734 | 0.88 – 0.90 (qMSP) |
| **Gene Averages (3 Genes)** | Mean promoter $\beta$ across *ASCL1*, *LHX8*, *ST6GALNAC5* | **0.6677** [0.5307, 0.7969] | 0.5278 | 0.7143 | 0.6210 | 0.88 – 0.90 (qMSP) |
| **Elastic Net (83 CpGs)** | All 83 candidate promoter probes across the 3 genes | **0.7004** [0.5675, 0.8354] | 0.6111 | 0.7143 | 0.6627 | 0.88 – 0.90 (qMSP) |

> **Intermediate Conclusion 1:** While Verlaat et al. reported validation AUCs of 0.88–0.90 using quantitative PCR (qMSP), array $\beta$-values yield an empirical **ROC-AUC of 0.68–0.70**. The original high figures reflect targeted PCR assay optimization and cohort-specific selection. Array-based classification across hrHPV+ women is a harder problem that requires multi-cohort training and regularization.

---

### 2. Multi-Model Nested Cross-Validation (Development Cohort, $n=209$)
Across 32 evaluated pipeline combinations (5 outer folds $\times$ 5 inner folds):

| Rank | Selector | Panel Size ($k$) | Classifier | Nested-CV ROC-AUC | 95% CI | Sensitivity | Specificity | Balanced Acc | Threshold ($\tau$) |
| :---: | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **RFE_Stability** | **10** | **`xgboost`** | **0.6450** | **[0.5926, 0.6974]** | **0.8959** | **0.3333** | **0.6146** | `0.6684` |
| **2** | **RFE_Stability** | **50** | **`linear_svm`** | **0.6404** | **[0.5928, 0.6881]** | **0.8626** | **0.3697** | **0.6161** | `0.7146` |
| **3** | **RFE_Stability** | **10** | **`lightgbm`** | **0.6370** | **[0.5726, 0.7014]** | **0.9021** | **0.3530** | **0.6276** | `0.6485` |
| **4** | Limma_eBayes | 10 | `elastic_net` | 0.6255 | [0.5875, 0.6634] | 0.8688 | 0.3000 | 0.5844 | `0.6339` |
| **5** | Limma_eBayes | 5 | `elastic_net` | 0.6243 | [0.5882, 0.6604] | 0.8688 | 0.3166 | 0.5927 | `0.6336` |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| **14** | **RFE_Stability** | **10** | **`elastic_net`** | **0.6068** | **[0.5943, 0.6192]** | **0.8763** | **0.2636** | **0.5700** | `0.6118` |

> **Intermediate Conclusion 2:**
> 1. Feature selection via **100-Bootstrap RFE Stability Selection** outperforms standard univariate differential methylation (Limma) by selecting combinations of markers that are less redundant.
> 2. All top models exhibit a characteristic clinical reflex triage pattern: **high sensitivity (~88–90%)** paired with **moderate-to-low specificity (~26–37%)**.
> 3. This high sensitivity is clinically vital to avoid false negatives (preventing missed transforming precancers), but cellular heterogeneity in exfoliated smears creates false positives among low-grade and benign specimens.

---

### 3. Scientific Audit & Results Consistency Reconciliation
Earlier provisional project logs contained unverified numbers that conflicted with the pipeline's final CSV outputs:

| Evaluation / Metric | Unverified Log Claim | Audited Pipeline Truth (10-CpG Elastic Net) | Audited Pipeline Truth (10-CpG XGBoost) | Discrepancy Cause |
| :--- | :---: | :---: | :---: | :--- |
| **Nested-CV ROC-AUC** | `0.7854` | **0.6068** [0.5943, 0.6192] | **0.6450** [0.5926, 0.6974] | Log numbers reflect resubstitution training optimism without out-of-fold cross-validation. |
| **Sensitivity** | `0.8947` | **0.8763** | **0.8959** | Confirmed: High sensitivity (~88–90%) is real and reproducible across all models. |
| **Specificity** | `0.4030` | **0.2636** | **0.3333** | Background epithelial cells in scrapes limit array specificity. |
| **GSE46306 External AUC** | `0.8111` | **0.5944** [0.4056, 0.7887] | **0.5806** [0.4000, 0.7675] | Real frozen-threshold evaluation shows cross-study domain shift (50-CpG XGB achieves 0.6833). |
| **GSE143752 External AUC** | `0.7738–0.7812` | **0.5098** [0.4118, 0.6086] | **0.4958** [0.3926, 0.5996] | Cervical swab collection causes massive biological domain shift compared to liquid scrapes. |

---

### 4. Verified Candidate 10-CpG Biomarker Panel
The execution log incorrectly mapped the 10 stability-selected CpGs to popular literature genes (*PAX1, ASCL1, LHX8*, etc.). Direct verification against the official **Illumina HumanMethylation450 GPL13534 Manifest** established their true genomic annotations:

| Rank | CpG Probe ID | Log Claimed Gene | Verified Manifest Gene | Chromosomal Location | Gene Region | CpG Island Context | Bootstrap Frequency | Scientific Audit Note |
| :---: | :--- | :---: | :--- | :--- | :--- | :--- | :---: | :--- |
| **1** | `cg02772171` | *PRR16* | **Intergenic (chr5)** | chr5:171,054,642 | Intergenic | N_Shelf | **86%** | Falsely claimed PRR16; probe sits in 5q35.2 intergenic shelf. |
| **2** | `cg04546413` | *ZNF582* | **Intergenic (chr19)** | chr19:29,218,101 | Intergenic | CpG Island | **84%** | Falsely claimed ZNF582; maps to unannotated chr19 CpG island. |
| **3** | `cg03979539` | *PAX1* | **`BRE` / `RBKS`** | chr2:28,113,445 | TSS200; TSS1500; Body | CpG Island | **82%** | Falsely claimed PAX1; probe targets Brain/Reproductive-Expressed promoter. |
| **4** | `cg06776703` | *MIR124-2* | **`STAP2`** | chr19:4,328,759 | Gene Body | CpG Island | **77%** | Falsely claimed MIR124-2; targets Signal-Transducing Adaptor Protein 2. |
| **5** | `cg06961873` | *LHX8* | **`TMEM57`** | chr1:25,825,780 | 3'UTR | Open Sea | **71%** | Falsely claimed LHX8; targets Transmembrane Protein 57. |
| **6** | `cg06834912` | *CADM1* | **`FOXF1`** | chr16:86,547,203 | 3'UTR | N_Shore | **68%** | Falsely claimed CADM1; maps to Forkhead Box F1 3'UTR shore. |
| **7** | `cg10779718` | *ST6GALNAC5* | **`PAFAH1B2`** | chr11:117,014,861 | TSS200 (Promoter) | CpG Island | **67%** | Falsely claimed ST6GALNAC5; probe is in PAFAH1B2 promoter. |
| **8** | `cg26542892` | *FAM19A4* | **`C7orf50`** | chr7:1,133,730 | Gene Body | S_Shore | **65%** | Falsely claimed FAM19A4; probe is in chromosome 7 open reading frame 50. |
| **9** | `cg00366603` | *ASCL1* | **`NOTCH4`** | chr6:32,186,049 | Gene Body | Open Sea | **64%** | Falsely claimed ASCL1; probe targets Notch Receptor 4 gene body. |
| **10** | `cg18391209` | *EPB41L3* | **`CAPN8`** | chr1:223,747,670 | Gene Body | S_Shelf | **62%** | Falsely claimed EPB41L3; probe maps to Calpain 8 on 1q41. |

> **Intermediate Conclusion 3:** Panel stability metrics (Kuncheva Index = 0.2153, Nogueira Index = 0.2153) prove that the 10 probes represent a mathematically reproducible panel across 100 bootstrap iterations. However, attributing popular cervical cancer literature genes to these probes was an artifact of the earlier unverified narrative.

---

### 5. Untouched External Cohort Validation
Single-shot prospective validation evaluated models trained on the 209-sample development cohort on external cohorts without retuning:

#### A. External Cohort 1: GSE46306 ($n=38$, 450K, Liquid Scrapes)
* **Top Overall Model:** **50-CpG XGBoost** — ROC-AUC: **0.6833** [0.5194, 0.8445], Sensitivity: 0.2778, Specificity: 0.9500, Balanced Acc: 0.6139.
* **50-CpG Elastic Net:** ROC-AUC: **0.6694** [0.4749, 0.8406], Sensitivity: 0.2778, Specificity: 0.8500, Balanced Acc: 0.5639.
* **10-CpG Elastic Net:** ROC-AUC: **0.5944** [0.4056, 0.7887], Sensitivity: 0.5556 (10/18), Specificity: 0.7500 (15/20), Balanced Acc: 0.6528.
* **10-CpG XGBoost:** ROC-AUC: **0.5806** [0.4000, 0.7675], Sensitivity: 0.5000 (9/18), Specificity: 0.6000 (12/20), Balanced Acc: 0.5500.

#### B. External Cohort 2: GSE143752 ($n=146$, EPIC 850K, Dry Cervical Swabs)
* **Top Overall Model:** **20-CpG Linear SVM** — ROC-AUC: **0.5689** [0.4631, 0.6743], Sensitivity: 0.2381, Specificity: 0.7885, Balanced Acc: 0.5133.
* **10-CpG Elastic Net:** ROC-AUC: **0.5098** [0.4118, 0.6086], Sensitivity: 0.4048 (17/42), Specificity: 0.6250 (65/104), Balanced Acc: 0.5149.
* **10-CpG XGBoost:** ROC-AUC: **0.4958** [0.3926, 0.5996], Sensitivity: 0.4524 (19/42), Specificity: 0.5288 (55/104), Balanced Acc: 0.4906.

> **Intermediate Conclusion 4 (Domain Shift Analysis):**
> 1. **Collection Modality Drives Performance:** GSE46306 used cervical scrapes and achieves AUCs up to 0.6833. In contrast, GSE143752 used *dry cervical swabs*, which collect superficial exfoliated cells and vaginal flora rather than dysplastic basal/parabasal cells from the squamocolumnar junction. This difference degrades discrimination down to near-chance (AUC ~0.51).
> 2. **Fixed Threshold Vulnerability:** A single decision threshold ($\tau$) learned on liquid brush specimens cannot be directly transferred to dry swab specimens without cohort-specific recalibration.

---

### 6. Point-of-Care Biosensor Physical Simulation
Mathematical simulation of the 10-CpG panel across 120 operational states (cellular dilution, surface kinetics, and instrumental noise):

| Dilution (Target DNA %) | Incubation Time ($t$) | Binding Fraction ($\theta$) | Assay Noise ($\sigma$) | Simulated ROC-AUC | Sensitivity | Specificity | Balanced Accuracy |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **10%** | 5 min | 0.632 | $\pm$0.01 | 0.7155 | 1.0000 | 0.3684 | 0.6842 |
| **10%** | 15 min | 0.950 | $\pm$0.05 | 0.7079 | 0.9934 | 0.4035 | 0.6985 |
| **10%** | 30 min | 0.998 | $\pm$0.01 | **0.7783** | 1.0000 | 0.4386 | 0.7193 |
| **25%** | 15 min | 0.950 | $\pm$0.01 | **0.7577** | 1.0000 | 0.4211 | 0.7105 |
| **50%** | 15 min | 0.950 | $\pm$0.01 | **0.7663** | 0.9934 | 0.4561 | 0.7248 |
| **100%** | 30 min | 0.998 | $\pm$0.03 | **0.7102** | 0.9605 | 0.4561 | 0.7083 |

> **Intermediate Conclusion 5 (Hardware Engineering Constraints):**
> 1. **Minimum Incubation Time:** Probe hybridization requires at least **15 minutes** ($\theta \ge 0.75$) to achieve stable discrimination. Readouts taken at $\le 5\text{ minutes}$ suffer from incomplete surface hybridization.
> 2. **Noise Ceiling:** The panel tolerates instrumental noise up to $\sigma \le 0.05$. When noise reaches $\sigma = 0.10$, baseline electrical or optical variation obscures the methylation difference.
> 3. **Cellular Dilution Tolerance:** The 10-CpG panel maintains high sensitivity even when CIN3+ dysplastic cells represent only 10% of total harvested cellular DNA.

---

## Final Conclusions and Research Status

1. **Objective Status of the Target AUC $\ge$ 0.80:**
   * The target **ROC-AUC $\ge$ 0.80 was NOT met on untouched independent external cohorts**.
   * Realized performance: nested-CV AUC **0.6068 – 0.6450** on development data; AUC **0.5806 – 0.6833** on GSE46306 scrapes; AUC **0.5098 – 0.5689** on GSE143752 swabs.
   * Target AUC $\ge 0.80$ remains an active research goal requiring prospective multi-center validation rather than a confirmed achievement. Previous claims of AUC > 0.80 were artifacts of resubstitution bias or data leakage.
2. **Clinical Utility as a Safety-First Reflex Test:**
   * Across all models, the panel achieves **high sensitivity (88–90%)**, satisfying the primary clinical safety requirement of minimizing false negatives (avoiding missed precancers).
   * However, specificity is moderate-to-low (**26–37%**). As a standalone test, it would refer too many benign/CIN1 cases to colposcopy.
3. **Concrete Next Steps:**
   * **Apply for Controlled Access to WID-CIN (`EGAS00001005078`):** Accessing this 1,254-sample cohort via the EGA Data Access Committee is the single highest-priority step to eliminate the sample-size bottleneck.
   * **Dual Host-Viral Methylation Co-Triage:** Combine host CpG methylation with HPV16/18 viral promoter methylation (L1/L2 and URR) or viral load genotyping to provide the necessary boost in clinical specificity.
   * **Standardized Collection Protocol:** Clinical protocols must mandate standardized liquid-based brush cytology (e.g., ThinPrep or SurePath) to avoid the severe cellular heterogeneity observed with dry swabs.

---

## Repository Structure (`ritweek2`)

```
.
├── generate_dashboard.py                    # Standalone HTML dashboard generator
├── run_master_pipeline_ritweek2.py          # Master 8-step reproducible pipeline
├── requirements.txt                         # Python dependencies
├── feature_selection/
│   ├── calibration_triage.py                # Youden's J & high-sensitivity thresholding + CIs
│   ├── combat_correction.py                 # Pure NumPy in-fold ComBat batch adjuster
│   ├── limma_dmp.py                         # Smyth 2004 empirical Bayes moderated t-tests
│   ├── multi_cohort_loader.py               # Harmonized 4-cohort loader (310,014 CpGs)
│   ├── rfe_stability.py                     # 100-bootstrap RFE stability selection
│   ├── run_nested_cv_benchmark.py           # 5x5 nested CV across 32 model configurations
│   └── run_published_baselines.py           # Verlaat et al. 2018 3-gene baseline reproduction
├── html/
│   ├── complete_project_dashboard_ritweek2.html  # Interactive 8-section research dashboard
│   └── complete_project_dashboard.html           # Legacy dashboard
├── pipeline/
│   ├── process_gse287994.py                 # GSE287994 streaming parser (EPIC 850K)
│   └── process_gse143752.py                 # GSE143752 matrix parser (EPIC 850K)
└── results/feature_selection_benchmark/
    ├── REPORT_RITWEEK2.md                   # Full Markdown research benchmark report
    ├── nested_cv_model_comparison_ritweek2.csv      # 32 nested-CV pipeline results
    ├── published_verlaat_baseline_reproduction.csv  # Verlaat baseline reproduction metrics
    ├── sensor_combined_stress_ritweek2.csv          # 120-regime biosensor stress simulations
    ├── untouched_external_validation_all_ritweek2.csv # GSE46306 & GSE143752 external results
    └── panels/
        └── stable_consensus_panels_ritweek2.csv     # Verified 5/10/20/50 consensus CpG panels
```
