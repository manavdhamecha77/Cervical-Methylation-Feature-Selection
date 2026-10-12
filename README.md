# Cervical Cancer DNA Methylation Study
### GSE99511 · Illumina HumanMethylation450K

Computational classification and candidate biomarker discovery using
DNA methylation profiles of cervical tissue samples.

> **Framing:** This is a research-grade computational study, not a clinically
> validated diagnostic system. Candidate CpGs require biological validation
> before any clinical interpretation.

---

## Dataset

| Property       | Value                          |
|----------------|-------------------------------|
| GEO accession  | GSE99511                       |
| Platform       | Illumina HumanMethylation450K  |
| Samples        | 68 total                       |
| Normal         | 28                             |
| CIN3           | 36                             |
| SCC            | 4 (exploratory only)           |
| CpGs (raw)     | 485,512                        |
| CpGs (post-QC) | 405,336                        |

### What is in `data/`?

Each GEO study has its own folder. Tabular experiment outputs stay in
`results/`, and human-readable reports are in `html/`. The feature-selection
benchmark currently trains on GSE99511 and uses GSE46306 as a separate
external check.

#### `data/GSE99511/` — development cohort

| File or folder | Contents / role |
|---|---|
| `source/` | Original GEO SOFT file, series matrix, and raw archive. Keep as source records. |
| `raw/` | Extracted red/green IDAT files and 450K platform manifests. Inputs to processing and CpG annotation. |
| `metadata/metadata.csv` | GSM-to-diagnosis mapping parsed from GEO records. |
| `metadata/samplesheet.csv` | Illumina sample sheet generated from IDAT filenames. |
| `metadata/sample_metadata.csv` | GSM IDs, Sentrix identifiers, and diagnoses; used by QC and modeling. |
| `intermediate/` | Renamed IDAT copies, methylprep sample sheet, normalized beta values, and signal matrices. Recreated by `pipeline/02_process_idat.py`. |
| `processed/GSE99511_beta_values.parquet` | Beta-value matrix from IDAT processing. |
| `processed/labeled_beta_matrix.parquet` | Beta values joined to sample diagnoses. |
| `processed/filtered_beta_matrix.csv.gz` | QC-filtered beta values; main feature-selection input. |
| `processed/qc_report.txt` | Summary of filtering decisions. |
| `processed/qc_probe_details.csv`, `processed/qc_sample_details.csv` | Probe-level and sample-level QC details. |

#### `data/GSE46306/` — external cohort

| File or folder | Contents / role |
|---|---|
| `GSE46306_Matrix_processed2.txt` | Original GEO processed matrix with probe annotation and sample group codes. |
| `GSE46306_clean/GSE46306_beta_matrix.csv.gz` | Extracted probe-by-sample matrix. GEO describes these values as arcsine-square-root transformed beta values; the external-validation script reverses that transform before scoring. |
| `GSE46306_clean/GSE46306_sample_metadata.csv` | Sample IDs and simplified group codes: `g0` normal, `g1` CIN3, `g2` cancer. The source GEO records label cancer `g3`; the simplified `g2` is a local recoding. |

GSE46306 has 20 normal, 18 CIN3, and 6 cancer samples on the same 450K
platform. The normal samples are HPV-negative while the CIN3 samples are
HPV-positive, so this is a cautious cross-cohort check, not a clean validation
of triage specifically among HPV-positive women. The six cancer samples are
excluded from the main Normal-versus-CIN3 evaluation.

The benchmark configuration is `feature_selection/config.json`. It selects
original CpG probes (not PCA components) and compares panels of 5, 10, 20, and
50 markers. Results and model artifacts are written under
`results/feature_selection_benchmark/`. These are research candidates, not a
clinically validated kit.

---

## Pipeline

`pipeline/` is useful when you need to rebuild processed GSE99511 inputs from
the GEO records and IDAT files. Its first four scripts form the data
preparation path:

```
pipeline/
├── 01_parse_metadata.py    Parse SOFT file → metadata.csv + samplesheet.csv
├── 02_process_idat.py      Run methylprep (Noob) on IDAT files → beta_values.parquet
├── 03_merge_labels.py      Merge GSM labels with beta matrix → labeled_beta_matrix.parquet
├── 04_quality_control.py   QC + probe filtering → filtered_beta_matrix.csv.gz
└── 05_ml_pipeline.py       Full ML pipeline → all results, plots, reports
```

Steps 01–04 have already been run and their processed outputs are present.
Keep `pipeline/` if you want to regenerate those inputs. Current feature
selection and sensor-simulation scripts read the processed matrices directly.
To rerun the current feature-selection benchmark:

```powershell
\.venv\Scripts\python.exe feature_selection\run_all_dataset_benchmarks.py
```

---

## Results

The `html/` folder contains browser-readable reports:

- `html/result.html` — feature-selection benchmark results.
- `html/sensor_simulation_results.html` — hypothetical sensor-error scenarios.

The top-level `results/feature_selection_benchmark/` folder keeps the
feature-selection results needed to review the experiments:

```
results/feature_selection_benchmark/
├── REPORT.md                         internal cross-validation summary
├── panels/consensus_panels.csv       selected CpGs for 5/10/20/50-marker panels
├── annotation/panel_annotations.csv  gene/probe annotations for panels
└── external_validation_GSE46306/     external scores, predictions, panel list,
                                      cohort metadata, and validation notes
```

The report summarizes internal cross-validation scores; the external folder
contains the detailed external scores and predictions. Only two `.joblib`
candidate artifacts are retained to keep this directory focused on original-
CpG feature selection. They are research artifacts, not clinically validated
models. Temporary fold-level files, stability frequency dumps, exploratory
PCA/deep-learning results, and plot galleries were removed.

---

## Key Design: No Data Leakage

Every CV fold independently performs:
1. Missing-value imputation (training mean only)
2. Variance filtering (training variance only)
3. The configured feature selector (training data and labels only)
4. StandardScaler fitting (training data only)

Validation fold data never touches any fitting step.

---

## Hypothetical Sensor Error Simulation

The [sensor simulation report](html/sensor_simulation_results.html) shows how
the screened 5/10/20/50-CpG models respond to assumed random noise,
calibration offset and gain, drift, missing readings, and combined stress
scenarios. The reproducible code and assumptions are in `sensor-sim/`.

No measurements from the planned sensor are available. These scenarios are
hypothetical robustness checks, not estimates of the device's real error or
evidence of clinical accuracy. Replace the assumed error levels with
measurements from repeated reference samples when the assay/device is
available.

---

## Documents

Project documents and the paper bibliography are in `docs/`:

- `docs/CC_Kit.docx` — hardware-kit project document.
- `docs/Research.pdf` — project research document.
- `docs/referred_research_papers.txt` — research papers and datasets referred to.

---

## Limitations

- Small sample size (n=64 primary task)
- 405k CpGs vs 64 samples — results require independent validation
- Only 4 SCC samples — SCC classification is not reliable
- Cross-sectional study — no longitudinal cancer prediction
- GSE46306 is a small cross-cohort check; HPV status is confounded with diagnosis
- Candidate CpGs are not clinically validated biomarkers
