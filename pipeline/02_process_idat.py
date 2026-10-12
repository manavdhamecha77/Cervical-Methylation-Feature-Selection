"""
02_process_idat.py
------------------
Prepare IDAT files for methylprep (rename to Sentrix key format),
then run the methylprep pipeline to produce Noob-normalised beta values.

Inputs:
    data/GSE99511/raw/               (GSM-prefixed IDAT files)

Outputs:
    data/GSE99511/intermediate/        (renamed IDATs + samplesheet)
    data/GSE99511/processed/GSE99511_beta_values.parquet
"""

import shutil
from pathlib import Path

import pandas as pd
from methylprep import run_pipeline

ROOT         = Path(__file__).resolve().parent.parent
DATASET_DIR  = ROOT / "data" / "GSE99511"
RAW_DIR      = DATASET_DIR / "raw"
PROC_DIR     = DATASET_DIR / "intermediate"
OUTPUT_FILE  = DATASET_DIR / "processed" / "GSE99511_beta_values.parquet"

PROC_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


# ── 1. Copy and rename IDATs to Sentrix key format ──────────────────────────

print("Preparing IDAT files ...")
metadata = []

for grn in sorted(RAW_DIR.glob("*_Grn.idat.gz")):
    parts = grn.stem.replace("_Grn.idat", "").split("_")
    gsm, sentrix_id, position = parts[0], parts[1], parts[2]

    red = grn.parent / grn.name.replace("_Grn.idat.gz", "_Red.idat.gz")
    if not red.exists():
        print(f"  WARNING: missing Red file for {gsm}, skipping.")
        continue

    new_grn = PROC_DIR / f"{sentrix_id}_{position}_Grn.idat.gz"
    new_red = PROC_DIR / f"{sentrix_id}_{position}_Red.idat.gz"

    if not new_grn.exists():
        shutil.copy2(grn, new_grn)
    if not new_red.exists():
        shutil.copy2(red, new_red)

    metadata.append({"Sample_Name": gsm, "Sentrix_ID": sentrix_id,
                      "Sentrix_Position": position})

df = pd.DataFrame(metadata)
samplesheet = PROC_DIR / "samplesheet.csv"
df.to_csv(samplesheet, index=False)
print(f"  {len(df)} samples prepared → {PROC_DIR}")


# ── 2. Run methylprep pipeline ───────────────────────────────────────────────

print("\nRunning methylprep pipeline (Noob normalisation) ...")
print("  This may take 20–40 minutes on first run.\n")

beta = run_pipeline(
    data_dir=str(PROC_DIR),
    array_type="450k",
    sample_sheet_filepath=str(samplesheet),
    betas=True,
    export=False,
    bit="float32",
    file_format="parquet",
    sesame=True,
)

print(f"\n  Raw beta shape (probes × samples): {beta.shape}")

# Save as parquet (probes as index, samples as columns)
beta.to_parquet(OUTPUT_FILE, engine="pyarrow")
print(f"  Saved → {OUTPUT_FILE}")
print(f"\n  Probes: {beta.shape[0]:,}  |  Samples: {beta.shape[1]}")
