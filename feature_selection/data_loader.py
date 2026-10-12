"""
data_loader.py
--------------
Data loading for the feature selection benchmark.
Reuses the same file paths and loading logic as pipeline/05_ml_pipeline.py.
Returns X (n_samples x n_features), y (binary), cpg_names, sample_meta.

Also flags Sentrix chip batch covariate confounding.
"""

import os
import warnings
import numpy as np
import pandas as pd


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_primary_data(config: dict, verbose: bool = True) -> dict:
    """
    Load Normal vs CIN3 primary task data.

    Returns
    -------
    dict with keys:
        X          : np.ndarray float32, shape (n_samples, n_cpgs)
        y          : np.ndarray int, shape (n_samples,), 0=Normal 1=CIN3
        cpg_names  : list[str], length n_cpgs
        gsm_ids    : np.ndarray[str]
        sentrix_ids: np.ndarray[str]  (for batch flag)
        diag_labels: np.ndarray[str]
        sample_meta: pd.DataFrame (full per-sample metadata)
        batch_report: str (human-readable batch confounding report)
    """
    data_cfg = config["data"]

    matrix_path = os.path.join(ROOT, data_cfg["filtered_matrix"])
    meta_path   = os.path.join(ROOT, data_cfg["metadata"])

    if verbose:
        print(f"[DataLoader] Loading filtered beta matrix: {matrix_path}")

    if not os.path.exists(matrix_path):
        raise FileNotFoundError(f"Filtered beta matrix not found: {matrix_path}")

    # ----- Load -----
    META_COLS = ["GSM_ID", "Sentrix_ID", "Sentrix_Position", "Sentrix_Key", "Diagnosis"]

    beta_full = pd.read_csv(matrix_path, index_col=0, low_memory=False)
    if verbose:
        print(f"[DataLoader] Raw shape: {beta_full.shape}")

    cpg_cols = [c for c in beta_full.columns if c.startswith("cg")]
    meta_in_beta = [c for c in META_COLS if c in beta_full.columns]

    # Build sample metadata
    sample_meta = pd.DataFrame()
    for col in meta_in_beta:
        sample_meta[col] = beta_full[col].values
    # If GSM_ID came from index
    if "GSM_ID" not in sample_meta.columns:
        if beta_full.index.astype(str).str.startswith("GSM").all():
            sample_meta["GSM_ID"] = beta_full.index.values
        else:
            sample_meta["GSM_ID"] = [f"S{i}" for i in range(len(beta_full))]
    if "Diagnosis" not in sample_meta.columns:
        # merge from external metadata
        ext_meta = pd.read_csv(meta_path)
        sample_meta = sample_meta.merge(
            ext_meta[["GSM_ID", "Diagnosis"]], on="GSM_ID", how="left"
        )

    sample_meta = sample_meta.reset_index(drop=True)

    # ----- Subset to primary classes -----
    primary_classes = data_cfg["primary_classes"]
    positive_class  = data_cfg["positive_class"]

    # Build a mask aligned with the beta_full row order
    diag_series = beta_full["Diagnosis"] if "Diagnosis" in beta_full.columns else None
    if diag_series is None:
        # fallback to sample_meta order
        diag_series = pd.Series(sample_meta["Diagnosis"].values, index=beta_full.index)

    mask = diag_series.isin(primary_classes).values  # numpy bool array, aligned to beta_full rows
    sample_meta_primary = sample_meta[mask].reset_index(drop=True)
    
    # Use positional indexing to avoid index misalignment
    X_raw = beta_full[cpg_cols].values[mask].astype(np.float32)
    X_raw = np.ascontiguousarray(X_raw)

    y = (sample_meta_primary["Diagnosis"].values == positive_class).astype(int)
    gsm_ids    = sample_meta_primary["GSM_ID"].values
    diag_labels = sample_meta_primary["Diagnosis"].values
    sentrix_ids = (
        sample_meta_primary["Sentrix_ID"].values
        if "Sentrix_ID" in sample_meta_primary.columns
        else np.array(["unknown"] * len(y))
    )

    if verbose:
        unique, counts = np.unique(y, return_counts=True)
        label_map = {0: "Normal", 1: positive_class}
        print(f"[DataLoader] Primary task: {', '.join(f'{label_map[u]}={c}' for u,c in zip(unique, counts))}")
        print(f"[DataLoader] X shape: {X_raw.shape}, dtype: {X_raw.dtype}")
        nan_pct = np.isnan(X_raw).mean() * 100
        print(f"[DataLoader] NaN rate: {nan_pct:.3f}%")

    # ----- Batch confounding report -----
    batch_report = _batch_report(sample_meta_primary, y, positive_class, verbose)

    return {
        "X": X_raw,
        "y": y,
        "cpg_names": cpg_cols,
        "gsm_ids": gsm_ids,
        "sentrix_ids": sentrix_ids,
        "diag_labels": diag_labels,
        "sample_meta": sample_meta_primary,
        "batch_report": batch_report,
        "n_samples": len(y),
        "n_cpgs": len(cpg_cols),
    }


def _batch_report(sample_meta: pd.DataFrame, y: np.ndarray,
                  positive_class: str, verbose: bool) -> str:
    """
    Check Sentrix chip (batch) confounding.
    Reports whether any chip is entirely one class (worst case for batch effects).
    """
    lines = [
        "=" * 70,
        "BATCH COVARIATE REPORT: Sentrix Chip vs Diagnosis",
        "=" * 70,
    ]

    if "Sentrix_ID" not in sample_meta.columns:
        lines.append("WARNING: Sentrix_ID not found in metadata — batch check skipped.")
        report = "\n".join(lines)
        if verbose:
            print(report)
        return report

    chip_diag = sample_meta.groupby("Sentrix_ID")["Diagnosis"].value_counts().unstack(fill_value=0)
    lines.append(f"\nChip x Diagnosis counts:\n{chip_diag.to_string()}")

    # Flag pure-class chips
    pure_chips = []
    for chip in chip_diag.index:
        row = chip_diag.loc[chip]
        nonzero = row[row > 0]
        if len(nonzero) == 1:
            pure_chips.append((chip, nonzero.index[0], nonzero.iloc[0]))

    if pure_chips:
        lines.append("\n[WARNING] CONFOUNDING: The following chips contain only one diagnosis class:")
        for chip, diag, n in pure_chips:
            lines.append(f"   Chip {chip}: all {n} samples are {diag}")
        lines.append(
            "\n   This means CpGs correlated with chip identity may be selected as discriminative features."
            "\n   Selected CpGs should be interpreted with caution until validated in an independent cohort."
            "\n   Recommended: run ComBat batch correction (not implemented here) and re-evaluate."
        )
    else:
        lines.append("\n[OK] No pure-class chips detected (each chip has samples from >1 class).")

    # Per-chip class fractions
    lines.append("\nPer-chip class fractions (CIN3 / total):")
    for chip in chip_diag.index:
        row   = chip_diag.loc[chip]
        total = row.sum()
        cin3  = row.get(positive_class, 0)
        lines.append(f"   {chip}: {cin3}/{total} = {cin3/total:.2f} {positive_class}")

    report = "\n".join(lines)
    if verbose:
        print(report)
    return report


def impute_fold(X_train: np.ndarray, X_val: np.ndarray):
    """
    Mean-impute using training statistics ONLY.
    Returns (X_train_imp, X_val_imp, col_means).
    Columns with all-NaN in training get mean=0.5 (mid-range for beta values).
    """
    col_means = np.nanmean(X_train, axis=0)
    col_means = np.where(np.isnan(col_means), 0.5, col_means)

    X_train_imp = X_train.copy()
    X_val_imp   = X_val.copy()

    nan_tr = np.isnan(X_train_imp)
    if nan_tr.any():
        row_idx, col_idx = np.where(nan_tr)
        X_train_imp[row_idx, col_idx] = col_means[col_idx]

    nan_va = np.isnan(X_val_imp)
    if nan_va.any():
        row_idx, col_idx = np.where(nan_va)
        X_val_imp[row_idx, col_idx] = col_means[col_idx]

    return X_train_imp, X_val_imp, col_means


def variance_prefilter(X_train: np.ndarray, X_val: np.ndarray,
                       cpg_names: list, threshold: float = 0.001,
                       top_n: int = 5000) -> tuple:
    """
    Two-stage pre-filter using TRAINING data only:
    1. Remove zero/near-zero variance probes (var < threshold)
    2. Keep top_n by variance

    Returns (X_train_f, X_val_f, selected_names, mask)
    """
    variances = np.var(X_train, axis=0)

    # Stage 1: variance threshold
    pass_thresh = variances >= threshold

    # Stage 2: top_n by variance within passing probes
    var_passing = variances.copy()
    var_passing[~pass_thresh] = -1.0
    n_keep = min(top_n, pass_thresh.sum())
    top_idx = np.argsort(var_passing)[::-1][:n_keep]
    top_idx = np.sort(top_idx)  # keep original order

    mask = np.zeros(X_train.shape[1], dtype=bool)
    mask[top_idx] = True

    selected_names = [cpg_names[i] for i in top_idx]
    return (
        np.ascontiguousarray(X_train[:, top_idx]),
        np.ascontiguousarray(X_val[:, top_idx]),
        selected_names,
        mask,
    )
