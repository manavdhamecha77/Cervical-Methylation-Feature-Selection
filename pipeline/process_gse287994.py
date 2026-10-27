"""
pipeline/process_gse287994.py
-----------------------------
Streams and parses GSE287994 (EPIC 850K cervical study, Bowden et al. 2025).
Extracts:
1. Sample metadata (241 women: Benign vs CIN3/Cancer; HPV+ vs HPV-)
2. Methylation beta values for CpGs harmonized with 450K (GSE99511 reference)
Saves directly as Parquet to conserve disk space.
"""

import os
import gzip
import urllib.request
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SERIES_MATRIX_PATH = os.path.join(ROOT, "data", "GSE287994", "GSE287994_series_matrix.txt.gz")
OUT_DIR = os.path.join(ROOT, "data", "GSE287994", "processed")
REF_GSE99511_PATH = os.path.join(ROOT, "data", "GSE99511", "processed", "filtered_beta_matrix.csv.gz")
BETAS_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE287nnn/GSE287994/suppl/GSE287994_ewas_betas_2.txt.gz"


def parse_metadata():
    print("[GSE287994] Parsing metadata from series matrix...")
    meta = {}
    char_lines = []
    with gzip.open(SERIES_MATRIX_PATH, "rt", errors="ignore") as f:
        for line in f:
            if line.startswith("!Sample_title"):
                meta["Sample_Title"] = [p.strip('"') for p in line.strip().split("\t")[1:]]
            elif line.startswith("!Sample_geo_accession"):
                meta["GSM_ID"] = [p.strip('"') for p in line.strip().split("\t")[1:]]
            elif line.startswith("!Sample_characteristics_ch1"):
                parts = [p.strip('"') for p in line.strip().split("\t")[1:]]
                char_lines.append(parts)
            elif line.startswith("!series_matrix_table_begin"):
                break

    df = pd.DataFrame(meta)
    # Line 1: case.status: 0 vs 1
    # Line 2: hpv.status: 0 vs 1
    # Line 3: cell type: Benign cervical sample vs CIN3 or Cancer sample
    df["Case_Status"] = [int(x.split(": ")[1]) for x in char_lines[1]]
    df["HPV_Status"] = [int(x.split(": ")[1]) for x in char_lines[2]]
    df["Cell_Type"] = [x.split(": ")[1] for x in char_lines[3]]
    
    # Standardize diagnosis:
    # Case_Status == 0 -> Benign (Control)
    # Case_Status == 1 -> CIN3_Cancer (Case)
    df["Diagnosis"] = df["Case_Status"].map({0: "Benign", 1: "CIN3_Cancer"})
    df["hrHPV"] = df["HPV_Status"] == 1
    df["Is_CIN3"] = df["Case_Status"]
    df["Cohort"] = "GSE287994"
    df["Platform"] = "EPIC_850K"

    os.makedirs(OUT_DIR, exist_ok=True)
    out_meta = os.path.join(OUT_DIR, "GSE287994_sample_metadata.csv")
    df.to_csv(out_meta, index=False)
    print(f"[GSE287994] Saved metadata ({len(df)} samples) to {out_meta}")
    print("Diagnosis cross-tabulation with hrHPV:")
    print(pd.crosstab(df["Diagnosis"], df["hrHPV"], margins=True))
    return df


def stream_and_filter_matrix(meta_df, keep_cpgs):
    print(f"[GSE287994] Streaming beta matrix from {BETAS_URL}...")
    cpg_target_set = set(keep_cpgs)
    
    req = urllib.request.Request(BETAS_URL, headers={"User-Agent": "Mozilla/5.0"})
    resp = urllib.request.urlopen(req, timeout=60)
    gz_file = gzip.GzipFile(fileobj=resp)
    
    # Read header line
    header_line = gz_file.readline().decode("utf-8", errors="ignore").strip()
    raw_cpg_cols = [c.strip('"') for c in header_line.split("\t")]
    # Note: column 0 is empty string "" representing the sample ID column
    print(f"[GSE287994] Total CpG columns in raw header: {len(raw_cpg_cols)-1}")
    
    # Find column indices to keep
    col_indices_to_keep = []
    kept_cpg_names = []
    for col_idx, col_name in enumerate(raw_cpg_cols[1:], start=1):
        if col_name in cpg_target_set:
            col_indices_to_keep.append(col_idx)
            kept_cpg_names.append(col_name)
            
    print(f"[GSE287994] Harmonized CpGs to extract: {len(kept_cpg_names)}")
    col_indices_arr = np.array(col_indices_to_keep, dtype=int)
    
    # Read sample rows
    sample_ids = []
    matrix_rows = []
    row_count = 0
    
    while True:
        line_bytes = gz_file.readline()
        if not line_bytes:
            break
        line_str = line_bytes.decode("utf-8", errors="ignore").strip()
        if not line_str:
            continue
        parts = line_str.split("\t")
        sample_id = parts[0].strip('"')
        sample_ids.append(sample_id)
        
        # Extract kept values
        row_vals = np.array([float(parts[i]) if parts[i] != "NA" and parts[i] != "" else np.nan for i in col_indices_arr], dtype=np.float32)
        matrix_rows.append(row_vals)
        row_count += 1
        if row_count % 25 == 0 or row_count == 241:
            print(f"  Streamed {row_count}/241 samples...")

    X = np.array(matrix_rows, dtype=np.float32)
    print(f"[GSE287994] Finished streaming matrix: shape={X.shape} (samples x harmonized CpGs)")
    
    # Build DataFrame
    df_mat = pd.DataFrame(X, columns=kept_cpg_names)
    df_mat["Sample_ID_Raw"] = sample_ids
    
    # Match sample IDs with meta_df
    # In GSE287994, sample_ids are integer/string numbers (e.g. 1, 2, ... 243)
    # Check sample_title in meta_df
    meta_df["Sample_Num"] = meta_df["Sample_Title"].str.extract(r'(\d+)')[0]
    matched_meta = meta_df.set_index("Sample_Num")
    
    df_mat["Sample_Num"] = df_mat["Sample_ID_Raw"].astype(str)
    merged = df_mat.merge(meta_df[["Sample_Num", "GSM_ID", "Sample_Title", "Diagnosis", "hrHPV", "Is_CIN3"]], on="Sample_Num", how="left")
    
    # Move metadata columns to front
    meta_cols = ["Sample_Num", "GSM_ID", "Sample_Title", "Diagnosis", "hrHPV", "Is_CIN3", "Sample_ID_Raw"]
    cpg_cols = [c for c in merged.columns if c not in meta_cols]
    final_cols = meta_cols + cpg_cols
    merged = merged[final_cols]
    
    out_parquet = os.path.join(OUT_DIR, "GSE287994_beta_matrix.parquet")
    table = pa.Table.from_pandas(merged)
    pq.write_table(table, out_parquet, compression="snappy")
    print(f"[GSE287994] Saved matrix ({merged.shape}) to {out_parquet}")
    return merged


if __name__ == "__main__":
    meta = parse_metadata()
    print("[GSE287994] Getting reference GSE99511 CpG list for 450K/EPIC harmonization...")
    with gzip.open(REF_GSE99511_PATH, "rt") as f:
        gse99511_cpgs = [c for c in f.readline().strip().split(",") if c.startswith("cg")]
    print(f"Reference GSE99511 contains {len(gse99511_cpgs)} CpGs")
    
    stream_and_filter_matrix(meta, gse99511_cpgs)
    print("[GSE287994] Complete!")
