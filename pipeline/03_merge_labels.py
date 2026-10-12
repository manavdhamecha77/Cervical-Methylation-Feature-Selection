"""
03_merge_labels.py
------------------
Merge GSM → diagnosis labels with the beta matrix, producing a
fully-labelled sample × probe matrix ready for QC and ML.

Inputs:
    data/GSE99511/metadata/metadata.csv
    data/GSE99511/metadata/samplesheet.csv
    data/GSE99511/intermediate/beta_values.parquet

Outputs:
    data/GSE99511/metadata/sample_metadata.csv
    data/GSE99511/processed/labeled_beta_matrix.parquet   (samples × probes+metadata)
"""

import sys
from pathlib import Path

import pandas as pd

ROOT     = Path(__file__).resolve().parent.parent
DATASETS = ROOT / "data" / "GSE99511"

META_CSV    = DATASETS / "metadata" / "metadata.csv"
SHEET_CSV   = DATASETS / "metadata" / "samplesheet.csv"
BETA_PARQ   = DATASETS / "intermediate" / "beta_values.parquet"
OUT_META    = DATASETS / "metadata" / "sample_metadata.csv"
OUT_LABELED = DATASETS / "processed" / "labeled_beta_matrix.parquet"
OUT_LABELED.parent.mkdir(parents=True, exist_ok=True)


# ── 1. Load and merge metadata ───────────────────────────────────────────────

print("Loading metadata ...")
meta = (pd.read_csv(META_CSV)[["gsm_id", "disease"]]
          .rename(columns={"gsm_id": "GSM_ID", "disease": "Diagnosis"}))

sheet = (pd.read_csv(SHEET_CSV)
           .rename(columns={"Sample_Name": "GSM_ID"}))
sheet["GSM_ID"] = sheet["GSM_ID"].astype(str).str.strip()

sample_meta = pd.merge(sheet, meta, on="GSM_ID", how="inner")
sample_meta["Sentrix_Key"] = (sample_meta["Sentrix_ID"].astype(str)
                               + "_" + sample_meta["Sentrix_Position"].astype(str))
sample_meta = sample_meta[["GSM_ID", "Sentrix_ID", "Sentrix_Position",
                             "Sentrix_Key", "Diagnosis"]]

print(f"  {len(sample_meta)} samples merged")
print(f"  Diagnosis counts:\n{sample_meta['Diagnosis'].value_counts().to_string()}")

sample_meta.to_csv(OUT_META, index=False)
print(f"  Saved → {OUT_META}")


# ── 2. Load beta matrix and align columns ────────────────────────────────────

print("\nLoading beta matrix ...")
beta = pd.read_parquet(BETA_PARQ)
beta.columns = beta.columns.astype(str).str.strip()
print(f"  Shape (probes × samples): {beta.shape}")

expected = set(sample_meta["Sentrix_Key"])
matched  = expected & set(beta.columns)
print(f"  Matched Sentrix keys: {len(matched)} / {len(expected)}")

ordered_keys   = [k for k in sample_meta["Sentrix_Key"] if k in beta.columns]
beta_T         = beta[ordered_keys].T          # samples × probes
beta_T.index.name = "Sentrix_Key"

labeled = sample_meta.set_index("Sentrix_Key").join(beta_T, how="inner")
print(f"  Labeled matrix shape: {labeled.shape}")


# ── 3. Verify ────────────────────────────────────────────────────────────────

diag_counts = labeled["Diagnosis"].value_counts()
checks = {
    "Total samples == 68":    len(labeled) == 68,
    "Normal == 28":           diag_counts.get("Normal", 0) == 28,
    "CIN3 == 36":             diag_counts.get("CIN3", 0) == 36,
    "SCC == 4":               diag_counts.get("SCC", 0) == 4,
    "No missing Diagnosis":   labeled["Diagnosis"].isna().sum() == 0,
    "No duplicate GSM_IDs":   labeled["GSM_ID"].duplicated().sum() == 0,
}

all_ok = True
for msg, result in checks.items():
    status = "OK" if result else "FAIL"
    print(f"  [{status}] {msg}")
    if not result:
        all_ok = False

if not all_ok:
    print("\nWARNING: Some checks failed — review above before continuing.", file=sys.stderr)


# ── 4. Save ──────────────────────────────────────────────────────────────────

print(f"\nSaving labeled matrix ...")
labeled.to_parquet(OUT_LABELED)
print(f"  Saved → {OUT_LABELED}  ({labeled.shape[0]} samples × {labeled.shape[1]} columns)")
