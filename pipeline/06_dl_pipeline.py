"""
06_dl_pipeline.py
-----------------
Deep Learning extension: MLP classifier + Autoencoder-based feature compression.
All implemented using scikit-learn (no PyTorch/TensorFlow required).

Experiments:
  1. MLP Classifier       — leakage-safe 5-fold CV, same framework as 05_ml_pipeline.py
  2. Sparse Autoencoder   — unsupervised compression of selected CpGs → latent representation
  3. AE + MLP             — Autoencoder encoder output as features → MLP classifier
  4. Comparison           — MLP / AE+MLP vs best classical model (RandomForest)

Inputs:
    data/GSE99511/processed/filtered_beta_matrix.csv.gz
    data/GSE99511/metadata/sample_metadata.csv
    results/e2e_cervical_methylation/cv_results/model_comparison.csv  (for comparison)

Outputs:
    results/e2e_cervical_methylation/dl_results/
        fold_results_mlp.csv
        fold_results_ae_mlp.csv
        autoencoder_reconstruction.csv
        dl_model_comparison.csv
        latent_space.csv
    plots/
        dl_roc_curve.png
        dl_confusion_matrix.png
        dl_performance_comparison.png
        autoencoder_reconstruction_error.png
        latent_space_pca.png
"""

import os
import sys
import json
import time
import logging
import warnings
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif, VarianceThreshold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
)

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
DATASETS    = ROOT / "data" / "GSE99511"
META_FILE   = DATASETS / "metadata" / "sample_metadata.csv"
MATRIX_FILE = DATASETS / "processed" / "filtered_beta_matrix.csv.gz"
OUT_BASE    = ROOT / "results" / "e2e_cervical_methylation"
OUT_DL      = OUT_BASE / "dl_results"
OUT_PLOTS   = OUT_BASE / "plots"

OUT_DL.mkdir(parents=True, exist_ok=True)
OUT_PLOTS.mkdir(parents=True, exist_ok=True)

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_FILE = OUT_BASE / "dl_pipeline.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(str(LOG_FILE), mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────
RANDOM_SEED       = 42
CV_SPLITS         = 5
VARIANCE_THRESHOLD = 0.001
FEATURE_BUDGETS   = [100, 500, 1000, 2000, 5000]
BEST_K            = 5000          # from 05_ml_pipeline results
LATENT_DIM        = 32            # autoencoder bottleneck size
np.random.seed(RANDOM_SEED)

DIAG_COLORS = {"Normal": "#2196F3", "CIN3": "#FF5722", "SCC": "#9C27B0"}
plt.rcParams.update({"figure.dpi": 150, "font.size": 10})

log.info("=" * 70)
log.info("06: MLP + AUTOENCODER DL PIPELINE (sklearn-based)")
log.info("=" * 70)

# ── Helper functions (same leakage-safe framework as 05_ml_pipeline.py) ──────

def impute_fold(X_train, X_val):
    means = np.nanmean(X_train, axis=0)
    means = np.where(np.isnan(means), 0.5, means)
    X_tr = X_train.copy()
    X_va = X_val.copy()
    for j in range(X_tr.shape[1]):
        nans = np.isnan(X_tr[:, j])
        if nans.any():
            X_tr[nans, j] = means[j]
    for j in range(X_va.shape[1]):
        nans = np.isnan(X_va[:, j])
        if nans.any():
            X_va[nans, j] = means[j]
    return X_tr, X_va

def variance_filter_fold(X_train, X_val, threshold=VARIANCE_THRESHOLD):
    variances = np.var(X_train, axis=0)
    mask = variances >= threshold
    return X_train[:, mask], X_val[:, mask], mask

def feature_select_fold(X_train, y_train, X_val, k):
    f_vals, _ = f_classif(X_train, y_train)
    f_vals = np.nan_to_num(f_vals, nan=0.0, posinf=0.0)
    k_actual = min(k, X_train.shape[1])
    top_idx = np.argsort(f_vals)[::-1][:k_actual]
    return X_train[:, top_idx], X_val[:, top_idx], top_idx

def compute_metrics(y_true, y_pred, y_prob=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Accuracy":          round(accuracy_score(y_true, y_pred), 4),
        "Balanced_Accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
        "Precision":         round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Sensitivity":       round(recall_score(y_true, y_pred, zero_division=0), 4),
        "Specificity":       round(tn / (tn + fp) if (tn + fp) > 0 else 0.0, 4),
        "F1":                round(f1_score(y_true, y_pred, zero_division=0), 4),
        "ROC_AUC":           round(roc_auc_score(y_true, y_prob), 4) if y_prob is not None else np.nan,
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }

# ── Load data ─────────────────────────────────────────────────────────────────
log.info("\n[DATA] Loading filtered beta matrix ...")
t0 = time.time()

meta     = pd.read_csv(META_FILE)
beta_full = pd.read_csv(MATRIX_FILE, index_col=0, low_memory=False)

META_COLS = ["GSM_ID", "Sentrix_ID", "Sentrix_Position", "Sentrix_Key", "Diagnosis"]
cpg_cols  = [c for c in beta_full.columns if c not in META_COLS and c.startswith("cg")]

if "Diagnosis" in beta_full.columns:
    diag_all = beta_full["Diagnosis"].values
    gsm_all  = beta_full["GSM_ID"].values if "GSM_ID" in beta_full.columns else beta_full.index.values
else:
    diag_all = meta["Diagnosis"].values
    gsm_all  = meta["GSM_ID"].values

beta_full = beta_full.reset_index(drop=True)
X_all     = beta_full[cpg_cols].values.astype(np.float32)
cpg_arr   = np.array(cpg_cols)

log.info(f"  Loaded {X_all.shape[0]} samples x {X_all.shape[1]:,} CpGs in {time.time()-t0:.1f}s")

# Primary task: Normal vs CIN3
mask_primary = np.isin(diag_all, ["Normal", "CIN3"])
X_primary    = X_all[mask_primary]
y_labels     = diag_all[mask_primary]
gsm_primary  = gsm_all[mask_primary]
y_primary    = (y_labels == "CIN3").astype(int)

assert len(y_primary) == 64
assert (y_primary == 0).sum() == 28
assert (y_primary == 1).sum() == 36
log.info(f"  Primary: {len(y_primary)} samples (Normal=28, CIN3=36)")


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: MLP CLASSIFIER — leakage-safe 5-fold CV
# ═══════════════════════════════════════════════════════════════════════════
log.info("\n" + "=" * 70)
log.info("EXPERIMENT 1: MLP Classifier (sklearn MLPClassifier)")
log.info("=" * 70)

# MLP architectures to try
MLP_CONFIGS = {
    "MLP_small":  dict(hidden_layer_sizes=(64,),         max_iter=500,  alpha=0.01),
    "MLP_medium": dict(hidden_layer_sizes=(128, 64),     max_iter=500,  alpha=0.01),
    "MLP_deep":   dict(hidden_layer_sizes=(256, 128, 64),max_iter=500,  alpha=0.01),
}

skf = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_SEED)
mlp_fold_results = []
mlp_oof          = []

t_mlp = time.time()
for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_primary, y_primary)):
    fold_num = fold_idx + 1
    log.info(f"  Fold {fold_num}/{CV_SPLITS} ...")

    X_tr_raw, X_va_raw = X_primary[tr_idx], X_primary[va_idx]
    y_tr, y_va         = y_primary[tr_idx], y_primary[va_idx]
    gsm_va             = gsm_primary[va_idx]
    diag_va            = y_labels[va_idx]

    # Leakage-safe preprocessing
    X_tr_imp, X_va_imp       = impute_fold(X_tr_raw, X_va_raw)
    X_tr_var, X_va_var, vmask = variance_filter_fold(X_tr_imp, X_va_imp)

    for K in FEATURE_BUDGETS:
        X_tr_k, X_va_k, _ = feature_select_fold(X_tr_var, y_tr, X_va_var, K)

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr_k)
        X_va_s = sc.transform(X_va_k)

        for cfg_name, cfg_params in MLP_CONFIGS.items():
            try:
                mlp = MLPClassifier(
                    **cfg_params,
                    solver="adam",
                    activation="relu",
                    early_stopping=True,
                    validation_fraction=0.15,
                    random_state=RANDOM_SEED,
                    verbose=False,
                )
                mlp.fit(X_tr_s, y_tr)
                y_pred = mlp.predict(X_va_s)
                y_prob = mlp.predict_proba(X_va_s)[:, 1]
                m = compute_metrics(y_va, y_pred, y_prob)
                mlp_fold_results.append({
                    "Fold": fold_num, "Model": cfg_name, "K": K, **m
                })
                for gsm, tl, pl, pp, dg in zip(gsm_va, y_va, y_pred, y_prob, diag_va):
                    mlp_oof.append({
                        "GSM_ID": gsm, "Diagnosis": dg,
                        "True_Label": int(tl), "Predicted_Label": int(pl),
                        "Predicted_Probability": float(pp),
                        "Fold": fold_num, "Model": cfg_name, "Feature_Budget": K,
                    })
            except Exception as e:
                log.warning(f"    {cfg_name} K={K} fold {fold_num}: {e}")

log.info(f"  MLP CV done in {time.time()-t_mlp:.1f}s")

mlp_fold_df = pd.DataFrame(mlp_fold_results)
mlp_oof_df  = pd.DataFrame(mlp_oof)
mlp_fold_df.to_csv(OUT_DL / "fold_results_mlp.csv", index=False)
mlp_oof_df.to_csv(OUT_DL / "oof_predictions_mlp.csv", index=False)
log.info("  Saved: dl_results/fold_results_mlp.csv")

# Aggregate MLP comparison
mlp_agg = []
for model_name in mlp_fold_df["Model"].unique():
    for K in FEATURE_BUDGETS:
        sub = mlp_fold_df[(mlp_fold_df["Model"] == model_name) & (mlp_fold_df["K"] == K)]
        if sub.empty:
            continue
        rec = {"Model": model_name, "Feature_Budget": K}
        for met in ["Accuracy","Balanced_Accuracy","Sensitivity","Specificity","F1","ROC_AUC"]:
            v = sub[met].dropna()
            rec[f"Mean_{met}"] = round(v.mean(), 4) if len(v) else np.nan
            rec[f"Std_{met}"]  = round(v.std(),  4) if len(v) else np.nan
        mlp_agg.append(rec)

mlp_comp_df = pd.DataFrame(mlp_agg)

log.info("\n  MLP Top configurations by Balanced Accuracy:")
top_mlp = mlp_comp_df.dropna(subset=["Mean_Balanced_Accuracy"]).nlargest(5, "Mean_Balanced_Accuracy")
log.info(top_mlp[["Model","Feature_Budget","Mean_Balanced_Accuracy","Mean_F1","Mean_ROC_AUC"]].to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════
# AUTOENCODER CLASS (numpy/sklearn implementation)
# ═══════════════════════════════════════════════════════════════════════════

class SklearnAutoencoder:
    """
    Autoencoder implemented as two MLPRegressors (encoder + decoder).
    The encoder compresses input → latent_dim.
    The decoder reconstructs latent → input.
    After training, encoder output is used as compressed features.
    """
    def __init__(self, input_dim, latent_dim=32, hidden_dim=128,
                 max_iter=300, alpha=0.001, random_state=42):
        self.input_dim    = input_dim
        self.latent_dim   = latent_dim
        self.hidden_dim   = hidden_dim
        self.max_iter     = max_iter
        self.alpha        = alpha
        self.random_state = random_state

        # Encoder: input_dim → hidden_dim → latent_dim
        self.encoder = MLPRegressor(
            hidden_layer_sizes=(hidden_dim, latent_dim),
            activation="relu", solver="adam",
            max_iter=max_iter, alpha=alpha,
            random_state=random_state, verbose=False,
        )
        # Decoder: latent_dim → hidden_dim → input_dim
        self.decoder = MLPRegressor(
            hidden_layer_sizes=(hidden_dim, input_dim),
            activation="relu", solver="adam",
            max_iter=max_iter, alpha=alpha,
            random_state=random_state, verbose=False,
        )
        self._trained = False

    def fit(self, X):
        """Train: encoder maps X → latent; decoder maps latent → X."""
        # Step 1: train encoder (X → latent approximation via PCA init)
        pca = PCA(n_components=self.latent_dim, random_state=self.random_state)
        Z_init = pca.fit_transform(X)   # initial latent target

        self.encoder.fit(X, Z_init)
        Z_encoded = self.encoder.predict(X)

        # Step 2: train decoder (Z → X reconstruction)
        self.decoder.fit(Z_encoded, X)
        self._trained = True
        return self

    def encode(self, X):
        """Return latent representation."""
        assert self._trained, "Call fit() first."
        return self.encoder.predict(X)

    def reconstruct(self, X):
        """Full forward pass: X → latent → X_hat."""
        return self.decoder.predict(self.encode(X))

    def reconstruction_error(self, X):
        """Mean squared reconstruction error per sample."""
        X_hat = self.reconstruct(X)
        return np.mean((X - X_hat) ** 2, axis=1)


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: AUTOENCODER + MLP — leakage-safe 5-fold CV
# ═══════════════════════════════════════════════════════════════════════════
log.info("\n" + "=" * 70)
log.info(f"EXPERIMENT 2: Autoencoder (latent_dim={LATENT_DIM}) + MLP Classifier")
log.info("=" * 70)
log.info("  Pipeline per fold: select K CpGs → train AE on train fold →")
log.info("  encode train+val → MLP on latent features")

ae_fold_results = []
ae_oof          = []
recon_errors    = []   # per-sample reconstruction error (full data, last fold)

t_ae = time.time()
for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_primary, y_primary)):
    fold_num = fold_idx + 1
    log.info(f"  Fold {fold_num}/{CV_SPLITS} ...")

    X_tr_raw, X_va_raw = X_primary[tr_idx], X_primary[va_idx]
    y_tr, y_va         = y_primary[tr_idx], y_primary[va_idx]
    gsm_va             = gsm_primary[va_idx]
    diag_va            = y_labels[va_idx]

    # Leakage-safe preprocessing
    X_tr_imp, X_va_imp       = impute_fold(X_tr_raw, X_va_raw)
    X_tr_var, X_va_var, vmask = variance_filter_fold(X_tr_imp, X_va_imp)

    # Use BEST_K from classical pipeline for AE experiments
    X_tr_k, X_va_k, _ = feature_select_fold(X_tr_var, y_tr, X_va_var, BEST_K)

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr_k)
    X_va_s = sc.transform(X_va_k)

    # Train autoencoder on TRAINING fold only
    ae = SklearnAutoencoder(
        input_dim=X_tr_s.shape[1],
        latent_dim=LATENT_DIM,
        hidden_dim=min(256, X_tr_s.shape[1]),
        max_iter=300,
        alpha=0.001,
        random_state=RANDOM_SEED,
    )
    ae.fit(X_tr_s)

    # Encode both folds
    Z_tr = ae.encode(X_tr_s)
    Z_va = ae.encode(X_va_s)

    # Record reconstruction error on validation fold
    recon_err_va = ae.reconstruction_error(X_va_s)
    for gsm, err, dg in zip(gsm_va, recon_err_va, diag_va):
        recon_errors.append({
            "GSM_ID": gsm, "Diagnosis": dg,
            "Reconstruction_MSE": float(err), "Fold": fold_num,
        })

    # MLP on latent features — no further scaling needed (AE output is bounded)
    for cfg_name, cfg_params in MLP_CONFIGS.items():
        try:
            mlp_ae = MLPClassifier(
                **cfg_params,
                solver="adam", activation="relu",
                early_stopping=True, validation_fraction=0.15,
                random_state=RANDOM_SEED, verbose=False,
            )
            mlp_ae.fit(Z_tr, y_tr)
            y_pred = mlp_ae.predict(Z_va)
            y_prob = mlp_ae.predict_proba(Z_va)[:, 1]
            m = compute_metrics(y_va, y_pred, y_prob)
            ae_fold_results.append({
                "Fold": fold_num, "Model": f"AE+{cfg_name}",
                "K": BEST_K, "Latent_Dim": LATENT_DIM, **m
            })
            for gsm, tl, pl, pp, dg in zip(gsm_va, y_va, y_pred, y_prob, diag_va):
                ae_oof.append({
                    "GSM_ID": gsm, "Diagnosis": dg,
                    "True_Label": int(tl), "Predicted_Label": int(pl),
                    "Predicted_Probability": float(pp),
                    "Fold": fold_num, "Model": f"AE+{cfg_name}",
                    "Feature_Budget": BEST_K, "Latent_Dim": LATENT_DIM,
                })
        except Exception as e:
            log.warning(f"    AE+{cfg_name} fold {fold_num}: {e}")

log.info(f"  AE+MLP CV done in {time.time()-t_ae:.1f}s")

ae_fold_df = pd.DataFrame(ae_fold_results)
ae_oof_df  = pd.DataFrame(ae_oof)
ae_fold_df.to_csv(OUT_DL / "fold_results_ae_mlp.csv", index=False)
ae_oof_df.to_csv(OUT_DL / "oof_predictions_ae_mlp.csv", index=False)
pd.DataFrame(recon_errors).to_csv(OUT_DL / "autoencoder_reconstruction.csv", index=False)
log.info("  Saved: dl_results/fold_results_ae_mlp.csv")

# AE agg
ae_agg = []
for model_name in ae_fold_df["Model"].unique():
    sub = ae_fold_df[ae_fold_df["Model"] == model_name]
    rec = {"Model": model_name, "Feature_Budget": BEST_K, "Latent_Dim": LATENT_DIM}
    for met in ["Accuracy","Balanced_Accuracy","Sensitivity","Specificity","F1","ROC_AUC"]:
        v = sub[met].dropna()
        rec[f"Mean_{met}"] = round(v.mean(), 4) if len(v) else np.nan
        rec[f"Std_{met}"]  = round(v.std(),  4) if len(v) else np.nan
    ae_agg.append(rec)
ae_comp_df = pd.DataFrame(ae_agg)

log.info("\n  AE+MLP Top configurations:")
log.info(ae_comp_df[["Model","Mean_Balanced_Accuracy","Mean_F1","Mean_ROC_AUC"]].to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════
# LATENT SPACE ANALYSIS (train AE on all 64 samples for visualization)
# ═══════════════════════════════════════════════════════════════════════════
log.info("\n[LATENT SPACE] Training AE on all 64 samples for visualization ...")

# Global imputation for descriptive analysis
X_means = np.nanmean(X_primary, axis=0)
X_imp_full = X_primary.copy()
for j in range(X_imp_full.shape[1]):
    nans = np.isnan(X_imp_full[:, j])
    if nans.any():
        X_imp_full[nans, j] = X_means[j]

# Variance filter + feature select on full data (descriptive only)
var_full  = np.var(X_imp_full, axis=0)
vmask_full = var_full >= VARIANCE_THRESHOLD
X_var_full = X_imp_full[:, vmask_full]
f_full, _  = f_classif(X_var_full, y_primary)
f_full     = np.nan_to_num(f_full, nan=0.0)
top_full   = np.argsort(f_full)[::-1][:BEST_K]
X_k_full   = X_var_full[:, top_full]

sc_full    = StandardScaler()
X_k_scaled = sc_full.fit_transform(X_k_full)

ae_full = SklearnAutoencoder(
    input_dim=X_k_scaled.shape[1],
    latent_dim=LATENT_DIM,
    hidden_dim=256,
    max_iter=300,
    random_state=RANDOM_SEED,
)
ae_full.fit(X_k_scaled)
Z_full = ae_full.encode(X_k_scaled)

# Save latent coordinates
latent_df = pd.DataFrame(Z_full, columns=[f"Z{i+1}" for i in range(LATENT_DIM)])
latent_df.insert(0, "GSM_ID", gsm_primary)
latent_df.insert(1, "Diagnosis", y_labels)
latent_df.to_csv(OUT_DL / "latent_space.csv", index=False)
log.info(f"  Saved: dl_results/latent_space.csv  ({Z_full.shape})")

# ═══════════════════════════════════════════════════════════════════════════
# COMBINED COMPARISON TABLE
# ═══════════════════════════════════════════════════════════════════════════
log.info("\n[COMPARISON] Building DL vs Classical comparison ...")

# Load classical best results
classical_csv = OUT_BASE / "cv_results" / "model_comparison.csv"
classical_df  = pd.read_csv(classical_csv)
# Best classical config
best_classical = classical_df.loc[classical_df["Mean_Balanced_Accuracy"].idxmax()].copy()
best_classical_row = {
    "Model":            f"{best_classical['Model']} (K={int(best_classical['Feature_Budget'])})",
    "Type":             "Classical ML",
    "Mean_Balanced_Accuracy": best_classical["Mean_Balanced_Accuracy"],
    "Std_Balanced_Accuracy":  best_classical["Std_Balanced_Accuracy"],
    "Mean_Sensitivity":       best_classical["Mean_Sensitivity"],
    "Mean_Specificity":       best_classical["Mean_Specificity"],
    "Mean_F1":                best_classical["Mean_F1"],
    "Mean_ROC_AUC":           best_classical["Mean_ROC_AUC"],
}

# Best MLP
best_mlp_row_data = mlp_comp_df.loc[mlp_comp_df["Mean_Balanced_Accuracy"].idxmax()]
best_mlp_row = {
    "Model":            f"{best_mlp_row_data['Model']} (K={int(best_mlp_row_data['Feature_Budget'])})",
    "Type":             "MLP",
    "Mean_Balanced_Accuracy": best_mlp_row_data["Mean_Balanced_Accuracy"],
    "Std_Balanced_Accuracy":  best_mlp_row_data["Std_Balanced_Accuracy"],
    "Mean_Sensitivity":       best_mlp_row_data["Mean_Sensitivity"],
    "Mean_Specificity":       best_mlp_row_data["Mean_Specificity"],
    "Mean_F1":                best_mlp_row_data["Mean_F1"],
    "Mean_ROC_AUC":           best_mlp_row_data["Mean_ROC_AUC"],
}

# Best AE+MLP
best_ae_row_data = ae_comp_df.loc[ae_comp_df["Mean_Balanced_Accuracy"].idxmax()]
best_ae_row = {
    "Model":            f"{best_ae_row_data['Model']} (latent={LATENT_DIM})",
    "Type":             "Autoencoder+MLP",
    "Mean_Balanced_Accuracy": best_ae_row_data["Mean_Balanced_Accuracy"],
    "Std_Balanced_Accuracy":  best_ae_row_data["Std_Balanced_Accuracy"],
    "Mean_Sensitivity":       best_ae_row_data["Mean_Sensitivity"],
    "Mean_Specificity":       best_ae_row_data["Mean_Specificity"],
    "Mean_F1":                best_ae_row_data["Mean_F1"],
    "Mean_ROC_AUC":           best_ae_row_data["Mean_ROC_AUC"],
}

# All MLP + AE rows
all_dl_rows = []
for _, r in mlp_comp_df.iterrows():
    all_dl_rows.append({**r.to_dict(), "Type": "MLP"})
for _, r in ae_comp_df.iterrows():
    all_dl_rows.append({**r.to_dict(), "Type": "Autoencoder+MLP"})

dl_comp_full = pd.DataFrame(all_dl_rows)
dl_comp_full.to_csv(OUT_DL / "dl_model_comparison.csv", index=False)

summary_comp = pd.DataFrame([best_classical_row, best_mlp_row, best_ae_row])
summary_comp.to_csv(OUT_DL / "dl_vs_classical_summary.csv", index=False)

log.info("\n  === DL vs Classical Summary ===")
log.info(summary_comp[["Model","Type","Mean_Balanced_Accuracy","Mean_F1","Mean_ROC_AUC"]].to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════════
log.info("\n[PLOTS] Generating DL visualizations ...")

# ── Plot 1: DL ROC curves (best MLP + best AE+MLP) ──────────────────────
try:
    fig, ax = plt.subplots(figsize=(6, 5))
    best_mlp_name = best_mlp_row_data["Model"]
    best_mlp_k    = int(best_mlp_row_data["Feature_Budget"])
    oof_best_mlp  = mlp_oof_df[(mlp_oof_df["Model"] == best_mlp_name) &
                                (mlp_oof_df["Feature_Budget"] == best_mlp_k)]
    fpr, tpr, _ = roc_curve(oof_best_mlp["True_Label"], oof_best_mlp["Predicted_Probability"])
    auc_mlp     = roc_auc_score(oof_best_mlp["True_Label"], oof_best_mlp["Predicted_Probability"])
    ax.plot(fpr, tpr, color="#E53935", lw=2, label=f"MLP (AUC={auc_mlp:.3f})")

    best_ae_name = best_ae_row_data["Model"]
    oof_best_ae  = ae_oof_df[ae_oof_df["Model"] == best_ae_name]
    fpr2, tpr2, _ = roc_curve(oof_best_ae["True_Label"], oof_best_ae["Predicted_Probability"])
    auc_ae        = roc_auc_score(oof_best_ae["True_Label"], oof_best_ae["Predicted_Probability"])
    ax.plot(fpr2, tpr2, color="#43A047", lw=2, label=f"AE+MLP (AUC={auc_ae:.3f})")

    ax.plot([0,1],[0,1],"k--",lw=1,alpha=0.5,label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves – MLP and AE+MLP\nNormal vs CIN3 (OOF, 5-fold CV)", fontsize=10)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUT_PLOTS / "dl_roc_curve.png", dpi=150)
    plt.close()
    log.info("  Saved: plots/dl_roc_curve.png")
except Exception as e:
    log.warning(f"  DL ROC plot failed: {e}")

# ── Plot 2: DL confusion matrix (best MLP) ──────────────────────────────
try:
    cm = confusion_matrix(oof_best_mlp["True_Label"], oof_best_mlp["Predicted_Label"])
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Normal","CIN3"], yticklabels=["Normal","CIN3"],
                cbar=False, linewidths=0.5)
    ax.set_title(f"Confusion Matrix – {best_mlp_name}\n(OOF, 5-fold CV)", fontsize=9)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(OUT_PLOTS / "dl_confusion_matrix.png", dpi=150)
    plt.close()
    log.info("  Saved: plots/dl_confusion_matrix.png")
except Exception as e:
    log.warning(f"  DL confusion matrix failed: {e}")

# ── Plot 3: Performance comparison bar chart ─────────────────────────────
try:
    models_cmp  = [best_classical_row["Model"], best_mlp_row["Model"], best_ae_row["Model"]]
    bal_accs    = [best_classical_row["Mean_Balanced_Accuracy"],
                   best_mlp_row["Mean_Balanced_Accuracy"],
                   best_ae_row["Mean_Balanced_Accuracy"]]
    stds        = [best_classical_row["Std_Balanced_Accuracy"],
                   best_mlp_row["Std_Balanced_Accuracy"],
                   best_ae_row["Std_Balanced_Accuracy"]]
    aucs        = [best_classical_row["Mean_ROC_AUC"],
                   best_mlp_row["Mean_ROC_AUC"],
                   best_ae_row["Mean_ROC_AUC"]]
    short_names = ["Best Classical\n(RF K=5000)", "Best MLP", "Best AE+MLP"]
    colors      = ["#1565C0", "#E53935", "#43A047"]

    x = np.arange(3)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(x, bal_accs, yerr=stds, capsize=5, color=colors, edgecolor="white", lw=0.8)
    axes[0].set_xticks(x); axes[0].set_xticklabels(short_names, fontsize=8)
    axes[0].set_ylim(0, 1); axes[0].set_ylabel("Mean Balanced Accuracy")
    axes[0].set_title("Balanced Accuracy (5-fold CV)", fontsize=10)
    axes[0].axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Random baseline")
    axes[0].legend(fontsize=8)
    for xi, v, s in zip(x, bal_accs, stds):
        axes[0].text(xi, v + s + 0.01, f"{v:.3f}", ha="center", fontsize=8, fontweight="bold")

    axes[1].bar(x, aucs, color=colors, edgecolor="white", lw=0.8)
    axes[1].set_xticks(x); axes[1].set_xticklabels(short_names, fontsize=8)
    axes[1].set_ylim(0, 1); axes[1].set_ylabel("Mean ROC-AUC")
    axes[1].set_title("ROC-AUC (5-fold CV)", fontsize=10)
    axes[1].axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    for xi, v in zip(x, aucs):
        axes[1].text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=8, fontweight="bold")

    plt.suptitle("Classical ML vs MLP vs Autoencoder+MLP\nNormal vs CIN3 (n=64)", fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT_PLOTS / "dl_performance_comparison.png", dpi=150)
    plt.close()
    log.info("  Saved: plots/dl_performance_comparison.png")
except Exception as e:
    log.warning(f"  Comparison bar chart failed: {e}")

# ── Plot 4: Autoencoder reconstruction error by diagnosis ────────────────
try:
    recon_df = pd.DataFrame(recon_errors)
    fig, ax  = plt.subplots(figsize=(6, 4))
    for diag, color in DIAG_COLORS.items():
        sub = recon_df[recon_df["Diagnosis"] == diag]
        if sub.empty: continue
        ax.scatter(range(len(sub)), sub["Reconstruction_MSE"],
                   c=color, label=diag, alpha=0.75, s=50, edgecolors="white", lw=0.4)
    ax.set_xlabel("Sample Index (validation fold)"); ax.set_ylabel("Reconstruction MSE")
    ax.set_title(f"Autoencoder Reconstruction Error by Diagnosis\n(K={BEST_K}, latent={LATENT_DIM})", fontsize=10)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_PLOTS / "autoencoder_reconstruction_error.png", dpi=150)
    plt.close()
    log.info("  Saved: plots/autoencoder_reconstruction_error.png")
except Exception as e:
    log.warning(f"  Reconstruction error plot failed: {e}")

# ── Plot 5: Latent space PCA (2D projection of 32-dim latent space) ──────
try:
    pca_lat   = PCA(n_components=2, random_state=RANDOM_SEED)
    Z_2d      = pca_lat.fit_transform(Z_full)
    ve_lat    = pca_lat.explained_variance_ratio_ * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: full data PCA on raw features (5000 CpGs)
    pca_raw  = PCA(n_components=2, random_state=RANDOM_SEED)
    Z_raw_2d = pca_raw.fit_transform(X_k_scaled)
    ve_raw   = pca_raw.explained_variance_ratio_ * 100
    for diag, color in DIAG_COLORS.items():
        m = y_labels == diag
        axes[0].scatter(Z_raw_2d[m, 0], Z_raw_2d[m, 1],
                        c=color, label=diag, alpha=0.8, s=55, edgecolors="white", lw=0.4)
    axes[0].set_xlabel(f"PC1 ({ve_raw[0]:.1f}%)"); axes[0].set_ylabel(f"PC2 ({ve_raw[1]:.1f}%)")
    axes[0].set_title(f"PCA of Input Features\n(K={BEST_K} CpGs, scaled)", fontsize=10)
    axes[0].legend(title="Diagnosis")

    # Right: latent space PCA
    for diag, color in DIAG_COLORS.items():
        m = y_labels == diag
        axes[1].scatter(Z_2d[m, 0], Z_2d[m, 1],
                        c=color, label=diag, alpha=0.8, s=55, edgecolors="white", lw=0.4)
    axes[1].set_xlabel(f"PC1 ({ve_lat[0]:.1f}%)"); axes[1].set_ylabel(f"PC2 ({ve_lat[1]:.1f}%)")
    axes[1].set_title(f"PCA of Autoencoder Latent Space\n(latent_dim={LATENT_DIM})", fontsize=10)
    axes[1].legend(title="Diagnosis")

    plt.suptitle("Input Space vs Autoencoder Latent Space [Exploratory]", fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT_PLOTS / "latent_space_pca.png", dpi=150)
    plt.close()
    log.info("  Saved: plots/latent_space_pca.png")
except Exception as e:
    log.warning(f"  Latent space PCA plot failed: {e}")

log.info("  All DL plots complete.")

# ═══════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
total_time = time.time() - t0
log.info("\n" + "=" * 70)
log.info("DL PIPELINE COMPLETE")
log.info("=" * 70)
log.info(f"  MLP architectures tested: {list(MLP_CONFIGS.keys())}")
log.info(f"  Feature budgets tested:   {FEATURE_BUDGETS}")
log.info(f"  Autoencoder latent dim:   {LATENT_DIM}")
log.info(f"  Total runtime:            {total_time/60:.1f} min")
log.info("")
log.info(f"  Best MLP:      {best_mlp_row['Model']}")
log.info(f"    Bal. Acc:    {best_mlp_row['Mean_Balanced_Accuracy']}")
log.info(f"    ROC-AUC:     {best_mlp_row['Mean_ROC_AUC']}")
log.info("")
log.info(f"  Best AE+MLP:   {best_ae_row['Model']}")
log.info(f"    Bal. Acc:    {best_ae_row['Mean_Balanced_Accuracy']}")
log.info(f"    ROC-AUC:     {best_ae_row['Mean_ROC_AUC']}")
log.info("")
log.info(f"  Best Classical:{best_classical_row['Model']}")
log.info(f"    Bal. Acc:    {best_classical_row['Mean_Balanced_Accuracy']}")
log.info(f"    ROC-AUC:     {best_classical_row['Mean_ROC_AUC']}")
log.info("")
log.info(f"  Outputs: {OUT_DL}")
log.info("=" * 70)
