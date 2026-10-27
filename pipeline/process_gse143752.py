"""
pipeline/process_gse143752.py
-----------------------------
Parses the downloaded GSE143752 raw processed matrix and series matrix metadata.
Outputs:
- data/GSE143752/processed/GSE143752_sample_metadata.csv
- data/GSE143752/processed/GSE143752_beta_matrix.parquet (and optional csv.gz)
Filters to CpGs present in 450K / GSE99511 to ensure complete cross-platform harmonization.
"""

import os
import gzip
import urllib.request
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_MATRIX_PATH = os.path.join(ROOT, "data", "GSE143752", "GSE143752_Matrix_processed.csv.gz")
SERIES_MATRIX_PATH = os.path.join(ROOT, "data", "GSE143752", "GSE143752_series_matrix.txt.gz")
OUT_DIR = os.path.join(ROOT, "data", "GSE143752", "processed")
REF_GSE99511_PATH = os.path.join(ROOT, "data", "GSE99511", "processed", "filtered_beta_matrix.csv.gz")


def parse_metadata():
    print("[GSE143752] Parsing metadata from series matrix...")
    meta_dict = {}
    with gzip.open(SERIES_MATRIX_PATH, "rt", errors="ignore") as f:
        for line in f:
            if line.startswith("!Sample_title"):
                meta_dict["Sample_Title"] = [p.strip('"') for p in line.strip().split("\t")[1:]]
            elif line.startswith("!Sample_geo_accession"):
                meta_dict["GSM_ID"] = [p.strip('"') for p in line.strip().split("\t")[1:]]
            elif line.startswith("!Sample_characteristics_ch1") and "grade:" in line:
                grades = [p.strip('"').replace("grade: ", "").strip() for p in line.strip().split("\t")[1:]]
                meta_dict["Grade"] = grades
            elif line.startswith("!series_matrix_table_begin"):
                break

    meta_df = pd.DataFrame(meta_dict)
    
    # Standardize diagnosis and triage binary labels:
    # Healthy -> Normal
    # Grade 1 -> CIN1
    # Grade 2 -> CIN2
    # Grade 3 -> CIN3
    grade_map = {
        "Healthy": "Normal",
        "Grade 1": "CIN1",
        "Grade 2": "CIN2",
        "Grade 3": "CIN3",
    }
    meta_df["Diagnosis"] = meta_df["Grade"].map(grade_map).fillna(meta_df["Grade"])
    meta_df["Clean_Title"] = meta_df["Sample_Title"].apply(lambda t: t.split()[0].strip())
    # For cervical triage test (CIN3+ positive outcome)
    meta_df["Is_CIN3"] = (meta_df["Diagnosis"] == "CIN3").astype(int)
    meta_df["Is_Normal"] = (meta_df["Diagnosis"] == "Normal").astype(int)
    meta_df["Cohort"] = "GSE143752"
    meta_df["Platform"] = "EPIC_850K"

    os.makedirs(OUT_DIR, exist_ok=True)
    out_meta = os.path.join(OUT_DIR, "GSE143752_sample_metadata.csv")
    meta_df.to_csv(out_meta, index=False)
    print(f"[GSE143752] Saved metadata ({len(meta_df)} samples) to {out_meta}")
    print(meta_df["Diagnosis"].value_counts())
    return meta_df


def process_matrix(meta_df, keep_cpgs=None):
    if not os.path.exists(RAW_MATRIX_PATH):
        print(f"[GSE143752] Downloading raw matrix from NCBI GEO...")
        url = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE143nnn/GSE143752/suppl/GSE143752_Matrix_processed.csv.gz"
        urllib.request.urlretrieve(url, RAW_MATRIX_PATH)
        print(f"[GSE143752] Downloaded raw matrix.")

    print(f"[GSE143752] Reading matrix from {RAW_MATRIX_PATH}...")
    
    # Let's inspect header to find AVG_Beta columns
    with gzip.open(RAW_MATRIX_PATH, "rt") as f:
        header = f.readline().strip().split(",")
    
    c_indices = [0] # TargetID
    c_names = ["TargetID"]
    for i, col in enumerate(header):
        if col.endswith(".AVG_Beta"):
            sample_name = col.replace(".AVG_Beta", "").strip()
            c_indices.append(i)
            c_names.append(sample_name)
            
    print(f"[GSE143752] Found {len(c_names)-1} AVG_Beta sample columns")
    
    # Stream in chunks to conserve memory
    chunksize = 50000
    chunks = []
    cpg_target_set = set(keep_cpgs) if keep_cpgs is not None else None
    
    total_rows = 0
    kept_rows = 0
    for chunk in pd.read_csv(RAW_MATRIX_PATH, usecols=c_indices, chunksize=chunksize, low_memory=False):
        total_rows += len(chunk)
        chunk.rename(columns={"TargetID": "CpG"}, inplace=True)
        chunk.dropna(subset=["CpG"], inplace=True)
        
        if cpg_target_set is not None:
            chunk = chunk[chunk["CpG"].isin(cpg_target_set)]
        
        kept_rows += len(chunk)
        if len(chunk) > 0:
            chunks.append(chunk)
            
        if total_rows % 200000 == 0:
            print(f"  Processed {total_rows} probes, kept {kept_rows}...")

    df_betas = pd.concat(chunks, axis=0, ignore_index=True)
    df_betas.set_index("CpG", inplace=True)
    print(f"[GSE143752] Transposing matrix: {df_betas.shape} (probes x samples)")
    
    # Transpose so samples are rows, CpGs are columns (consistent with GSE99511 & GSE46306)
    df_transposed = df_betas.T
    df_transposed.index = df_transposed.index.astype(str).str.replace(r"\.AVG_Beta$", "", regex=True).str.strip()
    df_transposed.index.name = "Clean_Title"
    df_transposed.reset_index(inplace=True)
    
    # Merge GSM_ID and Diagnosis using Clean_Title
    merged = meta_df[["Clean_Title", "Sample_Title", "GSM_ID", "Diagnosis", "Is_CIN3"]].merge(df_transposed, on="Clean_Title", how="inner")
    
    out_parquet = os.path.join(OUT_DIR, "GSE143752_beta_matrix.parquet")
    table = pa.Table.from_pandas(merged)
    pq.write_table(table, out_parquet, compression="snappy")
    print(f"[GSE143752] Saved transposed matrix ({merged.shape}) to {out_parquet}")
    
    # Reclaim disk space by deleting the raw 663 MB download
    if os.path.exists(RAW_MATRIX_PATH) and len(merged) > 0:
        os.remove(RAW_MATRIX_PATH)
        print(f"[GSE143752] Reclaimed disk space: deleted {RAW_MATRIX_PATH}")
    
    return merged


if __name__ == "__main__":
    meta = parse_metadata()
    # Get GSE99511 cpg list for cross-platform harmonization
    print("[GSE143752] Getting reference GSE99511 CpG list for 450K/EPIC harmonization...")
    with gzip.open(REF_GSE99511_PATH, "rt") as f:
        gse99511_cpgs = [c for c in f.readline().strip().split(",") if c.startswith("cg")]
    print(f"Reference GSE99511 contains {len(gse99511_cpgs)} CpGs")
    
    df_final = process_matrix(meta, keep_cpgs=gse99511_cpgs)
    print("[GSE143752] Processing complete!")
