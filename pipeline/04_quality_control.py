"""
04_quality_control.py
---------------------
QC and probe filtering on the raw beta matrix.

Rules applied:
  1. Remove probes with >25% missing values
  2. Remove non-CpG probes (ch*, rs*, control)

All 68 samples are retained regardless of missingness.

Inputs:
    data/GSE99511/metadata/sample_metadata.csv
    data/GSE99511/intermediate/beta_values.parquet

Outputs:
    data/GSE99511/processed/filtered_beta_matrix.csv.gz
    data/GSE99511/processed/qc_report.txt
    data/GSE99511/processed/qc_probe_details.csv
    data/GSE99511/processed/qc_sample_details.csv
"""

import gzip
import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT     = Path(__file__).resolve().parent.parent
DATASETS = ROOT / "data" / "GSE99511"

META_CSV     = DATASETS / "metadata" / "sample_metadata.csv"
BETA_PARQ    = DATASETS / "intermediate" / "beta_values.parquet"
FILTERED_OUT = DATASETS / "processed" / "filtered_beta_matrix.csv.gz"
QC_REPORT    = DATASETS / "processed" / "qc_report.txt"
PROBE_STATS  = DATASETS / "processed" / "qc_probe_details.csv"
SAMPLE_STATS = DATASETS / "processed" / "qc_sample_details.csv"
FILTERED_OUT.parent.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("QC + PROBE FILTERING")
print("=" * 70)


# ── 1. Load inputs ───────────────────────────────────────────────────────────

print("\n[1] Loading inputs ...")
sample_meta = pd.read_csv(META_CSV)
print(f"  Samples: {len(sample_meta)}")

beta = pd.read_parquet(BETA_PARQ)    # probes × samples
print(f"  Beta shape: {beta.shape}  (probes × samples)")

probe_ids  = list(beta.index)
sample_keys = list(beta.columns)
n_probes, n_samples = len(probe_ids), len(sample_keys)


# ── 2. Probe-level missing-value analysis ────────────────────────────────────

print("\n[2] Analysing probe-level missingness ...")
probe_missing_pct = 100 * beta.isna().sum(axis=1) / n_samples

probe_types = defaultdict(int)
for p in probe_ids:
    prefix = "".join(c for c in p if c.isalpha()) or "UNKNOWN"
    probe_types[prefix] += 1

for prefix, cnt in sorted(probe_types.items(), key=lambda x: -x[1]):
    print(f"  {prefix:10s}: {cnt:7,d}")


# ── 3. Build keep mask ───────────────────────────────────────────────────────

print("\n[3] Applying filter rules ...")
high_missing = probe_missing_pct > 25
non_cg       = pd.Series([not p.startswith("cg") for p in probe_ids], index=beta.index)
keep         = ~(high_missing | non_cg)
keep_ids     = keep[keep].index.tolist()

print(f"  Removed (>25% missing): {high_missing.sum():,}")
print(f"  Removed (non-CpG):      {non_cg.sum():,}")
print(f"  Retained:               {len(keep_ids):,}")


# ── 4. Sample-level QC on filtered probes ────────────────────────────────────

print("\n[4] Computing sample-level QC ...")
beta_filt = beta.loc[keep_ids]
meta_idx  = sample_meta.set_index("Sentrix_Key")

rows = []
for key in sample_keys:
    vals      = beta_filt[key]
    valid     = vals.dropna()
    n_missing = vals.isna().sum()
    gsm       = meta_idx.loc[key, "GSM_ID"]   if key in meta_idx.index else key
    diag      = meta_idx.loc[key, "Diagnosis"] if key in meta_idx.index else "Unknown"
    rows.append({
        "Sentrix_Key":         key,
        "GSM_ID":              gsm,
        "Diagnosis":           diag,
        "Missing_Count":       int(n_missing),
        "Missing_Pct":         round(100 * n_missing / len(vals), 4),
        "N_Valid":             len(valid),
        "Mean_Beta":           round(float(valid.mean()), 6) if len(valid) else np.nan,
        "Std_Beta":            round(float(valid.std()),  6) if len(valid) else np.nan,
    })

sample_qc = pd.DataFrame(rows)
print(f"  Missing % range: {sample_qc['Missing_Pct'].min():.2f}% – "
      f"{sample_qc['Missing_Pct'].max():.2f}%  "
      f"(median {sample_qc['Missing_Pct'].median():.2f}%)")
problem = sample_qc.nlargest(1, "Missing_Pct").iloc[0]
print(f"  Highest missingness: {problem['GSM_ID']} ({problem['Missing_Pct']:.2f}%) — retained")


# ── 5. Write filtered matrix as samples × CpGs + metadata ───────────────────

print("\n[5] Writing filtered matrix ...")
meta_ordered = (sample_meta.set_index("Sentrix_Key")
                            .loc[sample_keys]
                            .reset_index())

tmp_csv = str(FILTERED_OUT).replace(".csv.gz", ".csv")
with open(tmp_csv, "w") as fh:
    fh.write("GSM_ID,Sentrix_ID,Sentrix_Position,Diagnosis," + ",".join(keep_ids) + "\n")
    for _, row in meta_ordered.iterrows():
        key  = row["Sentrix_Key"]
        vals = beta.loc[keep_ids, key]
        vals_str = ",".join(str(v) if pd.notna(v) else "" for v in vals)
        fh.write(f"{row['GSM_ID']},{row['Sentrix_ID']},{row['Sentrix_Position']},"
                 f"{row['Diagnosis']},{vals_str}\n")
    print(f"  Written {len(meta_ordered)} sample rows")

print("  Compressing ...")
with open(tmp_csv, "rb") as f_in, gzip.open(str(FILTERED_OUT), "wb") as f_out:
    shutil.copyfileobj(f_in, f_out)
os.remove(tmp_csv)
print(f"  Saved → {FILTERED_OUT}")


# ── 6. Save QC outputs ───────────────────────────────────────────────────────

print("\n[6] Saving QC reports ...")

# Probe details
probe_details = pd.DataFrame({
    "Probe_ID":     probe_ids,
    "Missing_Pct":  probe_missing_pct.values.round(4),
    "Probe_Type":   ["".join(c for c in p if c.isalpha()) or "UNKNOWN" for p in probe_ids],
    "Kept":         keep.values,
})
probe_details.to_csv(PROBE_STATS, index=False)

# Sample details
sample_qc.to_csv(SAMPLE_STATS, index=False)

# Text report
diag_counts = sample_meta["Diagnosis"].value_counts()
with open(QC_REPORT, "w", encoding="utf-8") as fh:
    fh.write("=" * 80 + "\n")
    fh.write("QC + PROBE FILTERING REPORT\n")
    fh.write("=" * 80 + "\n\n")
    fh.write(f"Initial probes:  {n_probes:>10,}\n")
    fh.write(f"Initial samples: {n_samples:>10}\n\n")
    fh.write("Diagnosis distribution:\n")
    for d in ["Normal", "CIN3", "SCC"]:
        fh.write(f"  {d:<10}: {diag_counts.get(d, 0)}\n")
    fh.write("\nProbe filtering:\n")
    fh.write(f"  Removed (>25% missing): {high_missing.sum():,}\n")
    fh.write(f"  Removed (non-CpG):      {non_cg.sum():,}\n")
    fh.write(f"  Retained:               {len(keep_ids):,}\n")
    fh.write(f"  Retention rate:         {100*len(keep_ids)/n_probes:.1f}%\n\n")
    fh.write("Sample missingness (post-filter):\n")
    fh.write(f"  Min:    {sample_qc['Missing_Pct'].min():.2f}%\n")
    fh.write(f"  Median: {sample_qc['Missing_Pct'].median():.2f}%\n")
    fh.write(f"  Max:    {sample_qc['Missing_Pct'].max():.2f}%\n\n")
    fh.write(f"Output: {FILTERED_OUT}\n")

print(f"  {QC_REPORT.name}")
print(f"  {PROBE_STATS.name}")
print(f"  {SAMPLE_STATS.name}")

print(f"\n{'='*70}")
print(f"QC COMPLETE  |  {len(keep_ids):,} CpGs  |  {n_samples} samples retained")
print(f"{'='*70}")
