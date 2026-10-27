"""
feature_selection/multi_cohort_loader.py
----------------------------------------
Harmonized multi-cohort data loader for cervical cancer methylation triage.
Integrates:
1. GSE99511 (Discovery 450K: 64 hrHPV+ samples, Normal vs CIN3, Verlaat et al.)
2. GSE287994 (EPIC 850K: 241 samples, 145 hrHPV+, Benign vs CIN3/Cancer, Bowden et al.)
3. GSE143752 (EPIC 850K: 186 samples, Normal vs CIN3, Cheishvili/El-Zein et al.)
4. GSE46306 (Untouched external validation 450K: 38 samples, Normal vs CIN3)

Harmonizes shared CpGs between 450K and EPIC 850K arrays.
Filters to the intended clinical triage population: hrHPV-positive women with CIN3+ outcome.
"""

import os
import gzip
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from typing import Dict, List, Tuple, Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GSE99511_MAT = os.path.join(ROOT, "data", "GSE99511", "processed", "filtered_beta_matrix.csv.gz")
GSE99511_META = os.path.join(ROOT, "data", "GSE99511", "metadata", "sample_metadata.csv")
GSE46306_MAT = os.path.join(ROOT, "data", "GSE46306", "GSE46306_clean", "GSE46306_beta_matrix.csv.gz")
GSE46306_META = os.path.join(ROOT, "data", "GSE46306", "GSE46306_clean", "GSE46306_sample_metadata.csv")
GSE287994_MAT = os.path.join(ROOT, "data", "GSE287994", "processed", "GSE287994_beta_matrix.parquet")
GSE287994_META = os.path.join(ROOT, "data", "GSE287994", "processed", "GSE287994_sample_metadata.csv")
GSE143752_MAT = os.path.join(ROOT, "data", "GSE143752", "processed", "GSE143752_beta_matrix.parquet")
GSE143752_META = os.path.join(ROOT, "data", "GSE143752", "processed", "GSE143752_sample_metadata.csv")


def load_gse99511(verbose: bool = True) -> pd.DataFrame:
    """Load GSE99511 Normal vs CIN3 hrHPV+ cohort."""
    if verbose:
        print("[Loader] Loading GSE99511...")
    df = pd.read_csv(GSE99511_MAT, index_col=0, low_memory=False)
    # Filter to Normal and CIN3 (exclude SCC cancer from this 64-sample cohort if matching Verlaat)
    meta = pd.read_csv(GSE99511_META)
    diag_map = meta.set_index("GSM_ID")["Diagnosis"].to_dict()
    df["Diagnosis"] = df.index.map(diag_map)
    df = df[df["Diagnosis"].isin(["Normal", "CIN3"])].copy()
    df["GSM_ID"] = df.index
    df["Cohort"] = "GSE99511"
    df["Platform"] = "450K"
    df["hrHPV"] = True  # All Verlaat self-samples are hrHPV-positive
    df["Is_CIN3"] = (df["Diagnosis"] == "CIN3").astype(int)
    if verbose:
        print(f"  GSE99511 loaded: {df.shape[0]} samples (Normal: {sum(df['Is_CIN3']==0)}, CIN3: {sum(df['Is_CIN3']==1)})")
    return df


def load_gse46306(verbose: bool = True) -> pd.DataFrame:
    """Load GSE46306 untouched external cohort (38 samples: 20 Normal, 18 CIN3)."""
    if not os.path.exists(GSE46306_MAT):
        return pd.DataFrame()
    if verbose:
        print("[Loader] Loading GSE46306...")
    meta = pd.read_csv(GSE46306_META, dtype={"sample_id": str, "group_label": str})
    label_map = {"g0": "Normal", "g1": "CIN3", "g2": "Cancer"}
    meta["Diagnosis"] = meta["group_label"].map(label_map)
    meta = meta[meta["Diagnosis"].isin(["Normal", "CIN3"])].copy()

    # Read matrix (probes as rows, samples as cols) and transpose
    df_raw = pd.read_csv(GSE46306_MAT, index_col=0, low_memory=False)
    # Subset cols to meta samples
    keep_samples = [s for s in meta["sample_id"] if s in df_raw.columns]
    df_sub = df_raw[keep_samples].T  # samples as rows, probes as cols
    df_sub.index.name = "sample_id"
    df_sub.reset_index(inplace=True)

    df = meta.merge(df_sub, on="sample_id", how="inner")
    df["GSM_ID"] = df["sample_id"]
    df["Cohort"] = "GSE46306"
    df["Platform"] = "450K"
    df["hrHPV"] = True  # Evaluation cohort for cervical triage
    df["Is_CIN3"] = (df["Diagnosis"] == "CIN3").astype(int)
    if verbose:
        print(f"  GSE46306 loaded: {df.shape[0]} samples (Normal: {sum(df['Is_CIN3']==0)}, CIN3: {sum(df['Is_CIN3']==1)})")
    return df


def load_gse287994(hrhpv_only: bool = True, verbose: bool = True) -> pd.DataFrame:
    """Load GSE287994 (Bowden et al. 2025, EPIC 850K)."""
    if not os.path.exists(GSE287994_MAT):
        return pd.DataFrame()
    if verbose:
        print(f"[Loader] Loading GSE287994 (hrHPV_only={hrhpv_only})...")
    df = pq.read_table(GSE287994_MAT).to_pandas()
    if hrhpv_only:
        df = df[df["hrHPV"] == True].copy()
    if verbose:
        print(f"  GSE287994 loaded: {df.shape[0]} samples (Benign: {sum(df['Is_CIN3']==0)}, CIN3+: {sum(df['Is_CIN3']==1)})")
    return df


def load_gse143752(triage_classes_only: bool = True, verbose: bool = True) -> pd.DataFrame:
    """Load GSE143752 (Cheishvili/El-Zein et al., EPIC 850K)."""
    if not os.path.exists(GSE143752_MAT):
        return pd.DataFrame()
    if verbose:
        print(f"[Loader] Loading GSE143752 (triage_classes_only={triage_classes_only})...")
    df = pq.read_table(GSE143752_MAT).to_pandas()
    if triage_classes_only:
        # Triage comparison: Normal / Benign controls vs CIN3 cases
        df = df[df["Diagnosis"].isin(["Normal", "CIN1", "CIN3"])].copy()
    if verbose:
        print(f"  GSE143752 loaded: {df.shape[0]} samples (Controls: {sum(df['Is_CIN3']==0)}, CIN3: {sum(df['Is_CIN3']==1)})")
    return df


def get_harmonized_cohorts(
    target_cpgs: Optional[List[str]] = None,
    hrhpv_only: bool = True,
    max_features: int = 10000,
    verbose: bool = True,
) -> Dict:
    """
    Load, harmonize, and return aligned datasets across GSE99511, GSE287994, GSE143752, GSE46306.
    
    Returns
    -------
    dict with:
        'discovery': dict(X, y, cohort, cpg_names, meta) [GSE99511 + GSE287994 hrHPV+]
        'gse99511': dict(X, y, cpg_names, meta)
        'gse287994': dict(X, y, cpg_names, meta)
        'gse143752': dict(X, y, cpg_names, meta)
        'untouched_external': dict(X, y, cohort, cpg_names, meta) [GSE46306 + GSE143752]
        'shared_cpgs': list of harmonized CpG names
    """
    df_99511 = load_gse99511(verbose=verbose)
    df_287994 = load_gse287994(hrhpv_only=hrhpv_only, verbose=verbose)
    df_143752 = load_gse143752(verbose=verbose)
    df_46306 = load_gse46306(verbose=verbose)

    # Find shared CpG columns
    cpg_sets = []
    for name, df in [("GSE99511", df_99511), ("GSE287994", df_287994),
                     ("GSE143752", df_143752), ("GSE46306", df_46306)]:
        if len(df) > 0:
            cpgs = [c for c in df.columns if c.startswith("cg")]
            cpg_sets.append(set(cpgs))
            if verbose:
                print(f"  {name} has {len(cpgs)} CpG columns")

    shared = set.intersection(*cpg_sets) if cpg_sets else set()
    shared_cpgs = sorted(list(shared))
    if verbose:
        print(f"[Loader] Common harmonized CpGs across all active cohorts: {len(shared_cpgs)}")

    if target_cpgs is not None:
        shared_cpgs = [c for c in shared_cpgs if c in set(target_cpgs)]
        if verbose:
            print(f"[Loader] Filtered to target CpG panel: {len(shared_cpgs)} CpGs")
    elif len(shared_cpgs) > max_features:
        # Pre-filter by pooled variance on GSE99511 to prevent memory bottlenecks
        var_series = df_99511[shared_cpgs].var(axis=0)
        top_var_cpgs = var_series.sort_values(ascending=False).index[:max_features]
        shared_cpgs = sorted(list(top_var_cpgs))
        if verbose:
            print(f"[Loader] Retained top {len(shared_cpgs)} high-variance CpGs")

    def build_dataset_dict(df, cohort_name):
        if len(df) == 0:
            return None
        sub_df = df[shared_cpgs].apply(pd.to_numeric, errors="coerce")
        X = sub_df.values.astype(np.float32)
        # Impute any missing values with feature median
        col_med = np.nanmedian(X, axis=0)
        col_med = np.nan_to_num(col_med, nan=0.5)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_med, inds[1])

        y = df["Is_CIN3"].values.astype(int)
        cohort = np.array([cohort_name] * len(y))
        meta_cols = [c for c in df.columns if not c.startswith("cg")]
        return {
            "X": X, "y": y, "cohort": cohort,
            "cpg_names": shared_cpgs,
            "meta": df[meta_cols].reset_index(drop=True),
        }

    d_99511 = build_dataset_dict(df_99511, "GSE99511")
    d_287994 = build_dataset_dict(df_287994, "GSE287994")
    d_143752 = build_dataset_dict(df_143752, "GSE143752")
    d_46306 = build_dataset_dict(df_46306, "GSE46306")

    # Combine GSE99511 and GSE287994 (hrHPV+) into multi-cohort development set
    dev_parts = [p for p in [d_99511, d_287994] if p is not None]
    if len(dev_parts) > 1:
        X_dev = np.vstack([p["X"] for p in dev_parts])
        y_dev = np.concatenate([p["y"] for p in dev_parts])
        c_dev = np.concatenate([p["cohort"] for p in dev_parts])
        m_dev = pd.concat([p["meta"] for p in dev_parts], ignore_index=True)
        discovery = {"X": X_dev, "y": y_dev, "cohort": c_dev, "cpg_names": shared_cpgs, "meta": m_dev}
    else:
        discovery = dev_parts[0] if dev_parts else None

    # Combine external validation sets
    ext_parts = [p for p in [d_46306, d_143752] if p is not None]
    if len(ext_parts) > 1:
        X_ext = np.vstack([p["X"] for p in ext_parts])
        y_ext = np.concatenate([p["y"] for p in ext_parts])
        c_ext = np.concatenate([p["cohort"] for p in ext_parts])
        m_ext = pd.concat([p["meta"] for p in ext_parts], ignore_index=True)
        untouched = {"X": X_ext, "y": y_ext, "cohort": c_ext, "cpg_names": shared_cpgs, "meta": m_ext}
    elif len(ext_parts) == 1:
        untouched = ext_parts[0]
    else:
        untouched = None

    return {
        "discovery": discovery,
        "gse99511": d_99511,
        "gse287994": d_287994,
        "gse143752": d_143752,
        "gse46306": d_46306,
        "untouched_external": untouched,
        "shared_cpgs": shared_cpgs,
    }
