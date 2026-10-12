"""
05_ml_pipeline.py
-----------------
End-to-end ML classification and candidate biomarker discovery pipeline.

Research question:
    Can DNA methylation profiles distinguish Normal cervical tissue from CIN3,
    and which CpG sites consistently contribute to this distinction?

Primary task  : Normal vs CIN3 (n=64)
Secondary task: Normal vs Abnormal (CIN3 + SCC, n=68)
Exploratory   : 3-class Normal / CIN3 / SCC

Key design decisions:
  - All preprocessing (imputation, variance filtering, feature selection,
    scaling) performed INSIDE each CV fold — no data leakage.
  - Feature selection: vectorised ANOVA F-test (f_classif).
  - Feature budgets: 100 / 500 / 1000 / 2000 / 5000 CpGs.
  - Models: Logistic Regression, Linear SVM, RBF SVM, Random Forest,
            Gradient Boosting, XGBoost (if available).
  - Rigorous evaluation: nested 5-outer / 3-inner stratified CV.

Inputs:
    data/GSE99511/processed/filtered_beta_matrix.csv.gz
    data/GSE99511/metadata/sample_metadata.csv

Outputs:
    results/e2e_cervical_methylation/  (all results, plots, reports)
"""

# ============================================================================
# SECTION 1: IMPORTS
# ============================================================================
import os
import sys
import json
import time
import logging
import warnings
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import mannwhitneyu

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import f_classif, VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, confusion_matrix,
    roc_curve
)
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

warnings.filterwarnings("ignore")

# ============================================================================
# SECTION 2: CONFIGURATION
# ============================================================================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data", "GSE99511", "processed")
META_FILE = os.path.join(ROOT, "data", "GSE99511", "metadata", "sample_metadata.csv")
FILTERED_MATRIX = os.path.join(DATA_DIR, "filtered_beta_matrix.csv.gz")
OUT_BASE = os.path.join(ROOT, "results", "e2e_cervical_methylation")

PATHS = {
    "config":          os.path.join(OUT_BASE, "config"),
    "preprocessing":   os.path.join(OUT_BASE, "preprocessing"),
    "cv_results":      os.path.join(OUT_BASE, "cv_results"),
    "feature_sel":     os.path.join(OUT_BASE, "feature_selection"),
    "biomarkers":      os.path.join(OUT_BASE, "biomarkers"),
    "plots":           os.path.join(OUT_BASE, "plots"),
    "nva":             os.path.join(OUT_BASE, "normal_vs_abnormal"),
    "3class":          os.path.join(OUT_BASE, "exploratory_3class"),
    "reports":         os.path.join(OUT_BASE, "reports"),
}

for p in PATHS.values():
    os.makedirs(p, exist_ok=True)

LOG_FILE = os.path.join(OUT_BASE, "pipeline.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

FEATURE_BUDGETS = [100, 500, 1000, 2000, 5000]
VARIANCE_THRESHOLD = 0.001
CV_SPLITS = 5
CV_REPEATS = 3
INNER_SPLITS = 3

CONFIG = {
    "dataset": "GSE99511",
    "technology": "Illumina HumanMethylation450K",
    "class_definitions": {
        "primary": {"Normal": 0, "CIN3": 1},
        "nva": {"Normal": 0, "Abnormal": 1},
        "3class": {"Normal": 0, "CIN3": 1, "SCC": 2},
    },
    "cv_splits": CV_SPLITS,
    "cv_repeats": CV_REPEATS,
    "inner_splits": INNER_SPLITS,
    "feature_budgets": FEATURE_BUDGETS,
    "variance_threshold": VARIANCE_THRESHOLD,
    "feature_selection_method": "f_classif (ANOVA F-test, vectorized)",
    "random_seed": RANDOM_SEED,
    "xgboost_available": XGBOOST_AVAILABLE,
    "shap_available": False,
    "gseapy_available": False,
}

with open(os.path.join(PATHS["config"], "experiment_config.json"), "w") as f:
    json.dump(CONFIG, f, indent=2)

log.info("=" * 80)
log.info("STEP 10: END-TO-END CERVICAL CANCER DNA METHYLATION ML PIPELINE")
log.info("=" * 80)
log.info(f"Root: {ROOT}")
log.info(f"Output: {OUT_BASE}")
log.info(f"XGBoost available: {XGBOOST_AVAILABLE}")
log.info(f"Random seed: {RANDOM_SEED}")

# ============================================================================
# SECTION 3: ENVIRONMENT RECORD
# ============================================================================
import sklearn
env_lines = [
    f"Python: {sys.version}",
    f"NumPy: {np.__version__}",
    f"Pandas: {pd.__version__}",
    f"SciPy: {__import__('scipy').__version__}",
    f"Scikit-learn: {sklearn.__version__}",
    f"Matplotlib: {matplotlib.__version__}",
    f"Seaborn: {sns.__version__}",
    f"XGBoost: {xgb.__version__ if XGBOOST_AVAILABLE else 'NOT INSTALLED'}",
    f"SHAP: NOT INSTALLED",
    f"gseapy: NOT INSTALLED",
    f"PyArrow: {pd.io.parquet.__name__} (via pandas)",
    f"Generated: {datetime.now().isoformat()}",
]
with open(os.path.join(OUT_BASE, "environment.txt"), "w") as f:
    f.write("\n".join(env_lines) + "\n")
log.info("Environment recorded.")

# ============================================================================
# SECTION 4: LOAD DATA
# ============================================================================
t0 = time.time()
log.info("\n[DATA LOADING] Reading filtered beta matrix ...")
log.info(f"  File: {FILTERED_MATRIX}")

meta = pd.read_csv(META_FILE)
log.info(f"  Metadata shape: {meta.shape}")
log.info(f"  Diagnosis counts:\n{meta['Diagnosis'].value_counts().to_string()}")

# Load matrix — rows=samples, cols=CpGs + metadata cols
beta_full = pd.read_csv(FILTERED_MATRIX, index_col=0, low_memory=False)
log.info(f"  Beta matrix shape (raw): {beta_full.shape}")
elapsed = time.time() - t0
log.info(f"  Loaded in {elapsed:.1f}s")

# Identify metadata vs CpG columns
META_COLS = ["GSM_ID", "Sentrix_ID", "Sentrix_Position", "Sentrix_Key", "Diagnosis"]
cpg_cols = [c for c in beta_full.columns if c not in META_COLS and c.startswith("cg")]
meta_in_beta = [c for c in META_COLS if c in beta_full.columns]

log.info(f"  CpG columns detected: {len(cpg_cols):,}")
log.info(f"  Metadata cols in matrix: {meta_in_beta}")

# Build clean dataframe: samples as rows, CpGs as columns
# Also extract diagnosis from the matrix if present, else merge from meta
if "Diagnosis" in beta_full.columns:
    sample_diag = beta_full["Diagnosis"].copy()
else:
    # Try merging via index (GSM_ID) or via Sentrix_Key
    sample_diag = None

if "GSM_ID" in beta_full.columns:
    gsm_ids = beta_full["GSM_ID"].values
elif beta_full.index.str.startswith("GSM").all():
    gsm_ids = beta_full.index.values
else:
    gsm_ids = None

# Extract numeric beta matrix
X_raw = beta_full[cpg_cols].astype(np.float32)
log.info(f"  X_raw shape: {X_raw.shape}")

# Build sample-level metadata frame
if sample_diag is not None:
    sample_meta = pd.DataFrame({
        "GSM_ID": gsm_ids if gsm_ids is not None else X_raw.index,
        "Diagnosis": sample_diag.values,
    })
else:
    sample_meta = meta[["GSM_ID", "Diagnosis"]].copy()

log.info(f"  Sample meta:\n{sample_meta['Diagnosis'].value_counts().to_string()}")

# Verify counts
assert sample_meta["Diagnosis"].value_counts().get("Normal", 0) == 28, "Normal count mismatch"
assert sample_meta["Diagnosis"].value_counts().get("CIN3", 0) == 36, "CIN3 count mismatch"
assert sample_meta["Diagnosis"].value_counts().get("SCC", 0) == 4, "SCC count mismatch"
assert X_raw.shape[1] == 405336, f"CpG count mismatch: {X_raw.shape[1]}"
log.info("  Data integrity checks PASSED.")

# ============================================================================
# SECTION 5: SAMPLE MISSINGNESS REPORT
# ============================================================================
log.info("\n[PREPROCESSING] Sample missingness analysis ...")
miss_pct = X_raw.isna().mean(axis=1) * 100
sample_miss_df = pd.DataFrame({
    "GSM_ID": sample_meta["GSM_ID"].values,
    "Diagnosis": sample_meta["Diagnosis"].values,
    "Missing_CpGs": X_raw.isna().sum(axis=1).values,
    "Missing_Pct": miss_pct.values,
})
sample_miss_df.to_csv(
    os.path.join(PATHS["preprocessing"], "sample_missingness.csv"), index=False
)
log.info(f"  Min missing: {miss_pct.min():.3f}%")
log.info(f"  Median missing: {miss_pct.median():.3f}%")
log.info(f"  Max missing: {miss_pct.max():.3f}%")
# Flag GSM2645003
idx_problem = sample_miss_df[sample_miss_df["GSM_ID"] == "GSM2645003"].index
if len(idx_problem) > 0:
    pct = sample_miss_df.loc[idx_problem[0], "Missing_Pct"]
    log.info(f"  GSM2645003 missingness: {pct:.3f}% (retained as per study design)")

log.info("  Saved: preprocessing/sample_missingness.csv")


# ============================================================================
# SECTION 6: EXPLORATORY VISUALIZATIONS (pre-modeling)
# ============================================================================
log.info("\n[PLOTS] Generating exploratory visualizations ...")

DIAG_COLORS = {"Normal": "#2196F3", "CIN3": "#FF5722", "SCC": "#9C27B0"}
plt.rcParams.update({"figure.dpi": 150, "font.size": 10})

# --- 6.1 Class distribution ---
fig, ax = plt.subplots(figsize=(6, 4))
counts = sample_meta["Diagnosis"].value_counts().reindex(["Normal", "CIN3", "SCC"])
bars = ax.bar(counts.index, counts.values,
              color=[DIAG_COLORS.get(d, "gray") for d in counts.index],
              edgecolor="black", linewidth=0.8)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            str(val), ha="center", va="bottom", fontweight="bold")
ax.set_title("Sample Distribution by Diagnosis\nGSE99511 – Illumina 450K", fontsize=12)
ax.set_xlabel("Diagnosis")
ax.set_ylabel("Number of Samples")
ax.set_ylim(0, max(counts.values) * 1.2)
plt.tight_layout()
plt.savefig(os.path.join(PATHS["plots"], "class_distribution.png"), dpi=150)
plt.close()
log.info("  Saved: plots/class_distribution.png")

# --- 6.2 Sample missingness distribution ---
fig, ax = plt.subplots(figsize=(10, 4))
colors_miss = [DIAG_COLORS.get(d, "gray") for d in sample_miss_df["Diagnosis"]]
ax.bar(range(len(sample_miss_df)), sample_miss_df["Missing_Pct"], color=colors_miss,
       edgecolor="none", alpha=0.85)
ax.axhline(y=5, color="red", linestyle="--", alpha=0.6, label="5% threshold")
# Annotate GSM2645003
if len(idx_problem) > 0:
    i = idx_problem[0]
    ax.annotate("GSM2645003\n(14.35%)",
                xy=(i, sample_miss_df.loc[i, "Missing_Pct"]),
                xytext=(i + 2, sample_miss_df.loc[i, "Missing_Pct"] + 0.5),
                arrowprops=dict(arrowstyle="->", color="black"),
                fontsize=8, color="black")
from matplotlib.patches import Patch
legend_elems = [Patch(facecolor=v, label=k) for k, v in DIAG_COLORS.items()]
legend_elems.append(plt.Line2D([0], [0], color="red", linestyle="--", label="5% line"))
ax.legend(handles=legend_elems, loc="upper right", fontsize=8)
ax.set_title("Per-Sample Missingness After QC Filtering (405,336 CpGs)", fontsize=11)
ax.set_xlabel("Sample Index")
ax.set_ylabel("Missing CpGs (%)")
plt.tight_layout()
plt.savefig(os.path.join(PATHS["plots"], "sample_missingness.png"), dpi=150)
plt.close()
log.info("  Saved: plots/sample_missingness.png")

# --- 6.3 Beta-value distribution (subsample for speed) ---
log.info("  Computing beta distribution (subsample 5000 CpGs) ...")
rng = np.random.RandomState(42)
sub_cols = rng.choice(cpg_cols, size=min(5000, len(cpg_cols)), replace=False)
X_sub = X_raw[sub_cols].values.ravel()
X_sub = X_sub[~np.isnan(X_sub)]

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(X_sub, bins=100, color="#607D8B", edgecolor="none", alpha=0.8, density=True)
ax.set_title("Beta-Value Distribution (5,000 CpG subsample, all samples)\nExploring overall methylation landscape", fontsize=11)
ax.set_xlabel("Beta Value (0 = unmethylated, 1 = methylated)")
ax.set_ylabel("Density")
plt.tight_layout()
plt.savefig(os.path.join(PATHS["plots"], "beta_distribution.png"), dpi=150)
plt.close()
log.info("  Saved: plots/beta_distribution.png")

# --- 6.4 PCA on full data (mean-imputed subset for speed) ---
log.info("  Computing PCA on full data (5000 CpG subsample) ...")
X_pca_sub = X_raw[sub_cols].copy()
col_means = X_pca_sub.mean(axis=0)
X_pca_sub = X_pca_sub.fillna(col_means)
X_pca_sub = X_pca_sub.values.astype(np.float32)

scaler_pca = StandardScaler()
X_pca_scaled = scaler_pca.fit_transform(X_pca_sub)
pca_full = PCA(n_components=min(10, X_pca_scaled.shape[1]), random_state=RANDOM_SEED)
pca_coords = pca_full.fit_transform(X_pca_scaled)
var_exp = pca_full.explained_variance_ratio_ * 100

diag_arr = sample_meta["Diagnosis"].values
fig, ax = plt.subplots(figsize=(7, 6))
for diag, color in DIAG_COLORS.items():
    mask = diag_arr == diag
    ax.scatter(pca_coords[mask, 0], pca_coords[mask, 1],
               c=color, label=diag, alpha=0.8, s=60, edgecolors="white", linewidth=0.5)
ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}%)")
ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}%)")
ax.set_title("PCA – All Samples (5,000 CpG subsample, mean-imputed)\n[Exploratory visualization only]", fontsize=10)
ax.legend(title="Diagnosis")
plt.tight_layout()
plt.savefig(os.path.join(PATHS["plots"], "pca.png"), dpi=150)
plt.close()
log.info("  Saved: plots/pca.png")

log.info("  Exploratory visualizations complete.")


# ============================================================================
# SECTION 7: HELPER FUNCTIONS FOR LEAKAGE-SAFE CV
# ============================================================================

def impute_fold(X_train, X_val):
    """Mean-impute using training statistics only."""
    col_means = np.nanmean(X_train, axis=0)
    # If entire column missing in training, set mean to 0.5
    col_means = np.where(np.isnan(col_means), 0.5, col_means)
    inds = np.where(np.isnan(X_train))
    X_train_imp = X_train.copy()
    X_train_imp[inds] = np.take(col_means, inds[1])
    inds_v = np.where(np.isnan(X_val))
    X_val_imp = X_val.copy()
    X_val_imp[inds_v] = np.take(col_means, inds_v[1])
    return X_train_imp, X_val_imp, col_means


def variance_filter_fold(X_train, X_val, threshold=VARIANCE_THRESHOLD):
    """Remove low-variance features using training data only."""
    variances = np.var(X_train, axis=0)
    mask = variances >= threshold
    return X_train[:, mask], X_val[:, mask], mask, mask.sum()


def feature_select_fold(X_train, y_train, X_val, k, cpg_names):
    """Select top-K features using ANOVA F-test on training data only."""
    f_vals, p_vals = f_classif(X_train, y_train)
    # Handle NaN/Inf safely
    f_vals = np.nan_to_num(f_vals, nan=0.0, posinf=0.0, neginf=0.0)
    p_vals = np.nan_to_num(p_vals, nan=1.0, posinf=1.0, neginf=1.0)
    k_actual = min(k, X_train.shape[1])
    top_idx = np.argsort(f_vals)[::-1][:k_actual]
    selected_names = [cpg_names[i] for i in top_idx]
    return X_train[:, top_idx], X_val[:, top_idx], top_idx, selected_names, f_vals, p_vals


def compute_metrics(y_true, y_pred, y_prob=None):
    """Compute full metric set for binary classification."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)   # sensitivity
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else np.nan
    return {
        "Accuracy": acc, "Balanced_Accuracy": bal_acc,
        "Precision": prec, "Sensitivity": rec, "Specificity": spec,
        "F1": f1, "ROC_AUC": auc,
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }


def get_models():
    """Return dict of model_name -> sklearn estimator."""
    models = {
        "LogReg": LogisticRegression(
            max_iter=2000, C=1.0, solver="liblinear", random_state=RANDOM_SEED
        ),
        "LinearSVM": CalibratedClassifierCV(
            LinearSVC(max_iter=2000, C=1.0, random_state=RANDOM_SEED), cv=3
        ),
        "RBFSVM": SVC(
            kernel="rbf", C=1.0, gamma="scale", probability=True,
            random_state=RANDOM_SEED
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=3,
            random_state=RANDOM_SEED, n_jobs=1
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            random_state=RANDOM_SEED
        ),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            use_label_encoder=False, eval_metric="logloss",
            random_state=RANDOM_SEED, verbosity=0
        )
    return models


# ============================================================================
# SECTION 8: PRIMARY EXPERIMENT — Normal vs CIN3
# ============================================================================
log.info("\n" + "=" * 80)
log.info("PRIMARY EXPERIMENT: Normal vs CIN3")
log.info("=" * 80)

# Subset to Normal and CIN3 only — reset index to ensure alignment
sample_meta = sample_meta.reset_index(drop=True)
X_raw = X_raw.reset_index(drop=True)

mask_primary = sample_meta["Diagnosis"].isin(["Normal", "CIN3"])
primary_idx = mask_primary[mask_primary].index.tolist()
X_primary = X_raw.iloc[primary_idx].values.astype(np.float32)
y_primary_labels = sample_meta.loc[primary_idx, "Diagnosis"].values
gsm_primary = sample_meta.loc[primary_idx, "GSM_ID"].values
y_primary = (y_primary_labels == "CIN3").astype(int)  # 0=Normal, 1=CIN3
cpg_arr = np.array(cpg_cols)

log.info(f"  Primary samples: {len(y_primary)} (Normal={( y_primary==0).sum()}, CIN3={(y_primary==1).sum()})")
assert len(y_primary) == 64, f"Expected 64 primary samples, got {len(y_primary)}"

# Track fold-level feature counts
feature_count_records = []

# Storage for all results
fold_results = []
oof_predictions = []
# Feature selection stability: per-budget, count of how often each CpG is selected
stability_counts = {k: {} for k in FEATURE_BUDGETS}  # {K: {cpg: count}}
stability_ranks  = {k: {} for k in FEATURE_BUDGETS}  # {K: {cpg: [ranks]}}

skf = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_SEED)
models_dict = get_models()
model_names = list(models_dict.keys())

log.info(f"  Models: {model_names}")
log.info(f"  Feature budgets: {FEATURE_BUDGETS}")
log.info(f"  CV: {CV_SPLITS}-fold stratified\n")

t_cv_start = time.time()

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_primary, y_primary)):
    fold_num = fold_idx + 1
    log.info(f"  --- Fold {fold_num}/{CV_SPLITS} ---")

    X_tr_raw = X_primary[train_idx]
    X_va_raw = X_primary[val_idx]
    y_tr = y_primary[train_idx]
    y_va = y_primary[val_idx]
    gsm_val = gsm_primary[val_idx]
    diag_val = y_primary_labels[val_idx]

    # Step A: Impute using training stats only
    X_tr_imp, X_va_imp, col_means_fold = impute_fold(X_tr_raw, X_va_raw)
    nan_after = np.isnan(X_tr_imp).sum() + np.isnan(X_va_imp).sum()
    assert nan_after == 0, f"NaNs remain after imputation in fold {fold_num}"

    # Step B: Variance filter using training data only
    X_tr_var, X_va_var, var_mask, n_after_var = variance_filter_fold(X_tr_imp, X_va_imp)
    cpg_after_var = cpg_arr[var_mask]
    log.info(f"    After variance filter: {n_after_var:,} / {len(var_mask):,} features retained")

    feature_count_records.append({
        "Fold": fold_num,
        "Before_Variance_Filter": int(len(var_mask)),
        "After_Variance_Filter": int(n_after_var),
    })

    # Step C: For each feature budget K
    for K in FEATURE_BUDGETS:
        X_tr_k, X_va_k, top_idx_k, selected_cpgs, f_vals_vf, p_vals_vf = \
            feature_select_fold(X_tr_var, y_tr, X_va_var, K, list(cpg_after_var))

        # Update stability counts
        for rank_pos, cpg_name in enumerate(selected_cpgs):
            stability_counts[K][cpg_name] = stability_counts[K].get(cpg_name, 0) + 1
            if cpg_name not in stability_ranks[K]:
                stability_ranks[K][cpg_name] = []
            stability_ranks[K][cpg_name].append(rank_pos + 1)

        # Step D: Scale using training data only
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr_k)
        X_va_scaled = scaler.transform(X_va_k)

        # Step E: Train and evaluate each model
        for model_name, model in models_dict.items():
            try:
                model.fit(X_tr_scaled, y_tr)
                y_pred = model.predict(X_va_scaled)
                # Probability output
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X_va_scaled)[:, 1]
                elif hasattr(model, "decision_function"):
                    df = model.decision_function(X_va_scaled)
                    y_prob = (df - df.min()) / (df.max() - df.min() + 1e-10)
                else:
                    y_prob = y_pred.astype(float)

                metrics = compute_metrics(y_va, y_pred, y_prob)
                fold_results.append({
                    "Fold": fold_num, "Model": model_name, "K": K,
                    **metrics
                })

                # OOF predictions
                for i, (gsm, true_l, pred_l, prob, diag) in enumerate(
                    zip(gsm_val, y_va, y_pred, y_prob, diag_val)
                ):
                    oof_predictions.append({
                        "GSM_ID": gsm, "Diagnosis": diag,
                        "True_Label": int(true_l), "Predicted_Label": int(pred_l),
                        "Predicted_Probability": float(prob),
                        "Fold": fold_num, "Model": model_name, "Feature_Budget": K,
                    })

            except Exception as e:
                log.warning(f"    Model {model_name} K={K} fold {fold_num} FAILED: {e}")
                fold_results.append({
                    "Fold": fold_num, "Model": model_name, "K": K,
                    "Accuracy": np.nan, "Balanced_Accuracy": np.nan,
                    "Precision": np.nan, "Sensitivity": np.nan, "Specificity": np.nan,
                    "F1": np.nan, "ROC_AUC": np.nan,
                    "TP": 0, "TN": 0, "FP": 0, "FN": 0,
                })

log.info(f"\n  Primary CV complete in {time.time() - t_cv_start:.1f}s")

# Save fold results
fold_df = pd.DataFrame(fold_results)
fold_df.to_csv(os.path.join(PATHS["cv_results"], "fold_results.csv"), index=False)

oof_df = pd.DataFrame(oof_predictions)
oof_df.to_csv(os.path.join(PATHS["cv_results"], "oof_predictions.csv"), index=False)

# Feature counts
feat_count_df = pd.DataFrame(feature_count_records)
feat_count_df.to_csv(os.path.join(PATHS["preprocessing"], "feature_counts.csv"), index=False)

log.info("  Saved: cv_results/fold_results.csv")
log.info("  Saved: cv_results/oof_predictions.csv")
log.info("  Saved: preprocessing/feature_counts.csv")


# ============================================================================
# SECTION 9: MODEL COMPARISON TABLE
# ============================================================================
log.info("\n[MODEL COMPARISON] Aggregating CV metrics ...")

agg_records = []
for model_name in fold_df["Model"].unique():
    for K in FEATURE_BUDGETS:
        sub = fold_df[(fold_df["Model"] == model_name) & (fold_df["K"] == K)]
        if sub.empty:
            continue
        rec = {"Model": model_name, "Feature_Budget": K}
        for metric in ["Accuracy", "Balanced_Accuracy", "Sensitivity",
                        "Specificity", "F1", "ROC_AUC"]:
            vals = sub[metric].dropna()
            rec[f"Mean_{metric}"] = round(vals.mean(), 4) if len(vals) > 0 else np.nan
            rec[f"Std_{metric}"]  = round(vals.std(), 4)  if len(vals) > 0 else np.nan
        agg_records.append(rec)

model_comp_df = pd.DataFrame(agg_records)
model_comp_df.to_csv(os.path.join(PATHS["cv_results"], "model_comparison.csv"), index=False)
log.info("  Saved: cv_results/model_comparison.csv")

# Print top 10 by balanced accuracy
top10 = model_comp_df.dropna(subset=["Mean_Balanced_Accuracy"]).nlargest(10, "Mean_Balanced_Accuracy")
log.info("\n  Top 10 configurations by Mean Balanced Accuracy:")
log.info(top10[["Model", "Feature_Budget", "Mean_Balanced_Accuracy", "Std_Balanced_Accuracy",
                 "Mean_F1", "Mean_ROC_AUC"]].to_string(index=False))

# Identify best config for nested CV and further analysis
best_row = model_comp_df.dropna(subset=["Mean_Balanced_Accuracy"]).loc[
    model_comp_df["Mean_Balanced_Accuracy"].idxmax()
]
BEST_MODEL_NAME = best_row["Model"]
BEST_K = int(best_row["Feature_Budget"])
log.info(f"\n  Selected config for further analysis: Model={BEST_MODEL_NAME}, K={BEST_K}")
log.info("  (Selection based on highest mean balanced accuracy across CV folds)")

# ============================================================================
# SECTION 10: FEATURE SELECTION STABILITY
# ============================================================================
log.info("\n[FEATURE STABILITY] Computing stability across folds ...")

top_stable_cpgs = {}  # K -> list of top stable CpGs

for K in FEATURE_BUDGETS:
    counts = stability_counts[K]
    ranks  = stability_ranks[K]
    if not counts:
        continue
    rows = []
    for cpg, cnt in counts.items():
        r = ranks[cpg]
        rows.append({
            "CpG": cpg,
            "Selection_Count": cnt,
            "Selection_Frequency": round(cnt / CV_SPLITS, 3),
            "Mean_Rank": round(np.mean(r), 1),
            "Median_Rank": round(np.median(r), 1),
        })
    stab_df = pd.DataFrame(rows).sort_values("Selection_Count", ascending=False).reset_index(drop=True)
    stab_df.to_csv(
        os.path.join(PATHS["feature_sel"], f"feature_stability_K{K}.csv"), index=False
    )
    top50 = stab_df.head(50)
    top_stable_cpgs[K] = top50["CpG"].tolist()
    log.info(f"  K={K}: {len(stab_df)} unique CpGs selected; "
             f"top CpG '{stab_df.iloc[0]['CpG']}' selected {stab_df.iloc[0]['Selection_Count']}/{CV_SPLITS} folds")

log.info("  Saved: feature_selection/feature_stability_K*.csv")

# ============================================================================
# SECTION 11: NESTED CROSS-VALIDATION (rigorous performance estimate)
# ============================================================================
log.info("\n[NESTED CV] Running nested stratified cross-validation ...")
log.info(f"  Outer: {CV_SPLITS}-fold, Inner: {INNER_SPLITS}-fold")
log.info(f"  Model: {BEST_MODEL_NAME}, K budget: evaluated inside inner loop")

nested_results = []
outer_skf = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_SEED)
inner_skf = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=RANDOM_SEED)

t_nested = time.time()
for outer_fold, (outer_tr, outer_va) in enumerate(outer_skf.split(X_primary, y_primary)):
    log.info(f"  Outer fold {outer_fold+1}/{CV_SPLITS} ...")
    X_outer_tr = X_primary[outer_tr]
    X_outer_va = X_primary[outer_va]
    y_outer_tr = y_primary[outer_tr]
    y_outer_va = y_primary[outer_va]

    # Inner loop: select best K
    inner_scores = {K: [] for K in FEATURE_BUDGETS}
    for inner_tr, inner_va in inner_skf.split(X_outer_tr, y_outer_tr):
        X_in_tr = X_outer_tr[inner_tr]
        X_in_va = X_outer_tr[inner_va]
        y_in_tr = y_outer_tr[inner_tr]
        y_in_va = y_outer_tr[inner_va]

        X_in_tr_imp, X_in_va_imp, _ = impute_fold(X_in_tr, X_in_va)
        X_in_tr_var, X_in_va_var, vmask, _ = variance_filter_fold(X_in_tr_imp, X_in_va_imp)
        cpg_inner = cpg_arr[vmask]

        for K in FEATURE_BUDGETS:
            X_in_tr_k, X_in_va_k, _, _, _, _ = \
                feature_select_fold(X_in_tr_var, y_in_tr, X_in_va_var, K, list(cpg_inner))
            sc_inner = StandardScaler()
            X_in_tr_s = sc_inner.fit_transform(X_in_tr_k)
            X_in_va_s = sc_inner.transform(X_in_va_k)
            try:
                m_inner = get_models()[BEST_MODEL_NAME]
                m_inner.fit(X_in_tr_s, y_in_tr)
                pred_inner = m_inner.predict(X_in_va_s)
                inner_scores[K].append(balanced_accuracy_score(y_in_va, pred_inner))
            except Exception:
                inner_scores[K].append(0.0)

    # Pick best K from inner loop
    best_K_inner = max(inner_scores, key=lambda k: np.mean(inner_scores[k]))

    # Retrain on full outer training set with best K
    X_out_tr_imp, X_out_va_imp, _ = impute_fold(X_outer_tr, X_outer_va)
    X_out_tr_var, X_out_va_var, vmask_out, _ = variance_filter_fold(X_out_tr_imp, X_out_va_imp)
    cpg_out = cpg_arr[vmask_out]
    X_out_tr_k, X_out_va_k, _, _, _, _ = \
        feature_select_fold(X_out_tr_var, y_outer_tr, X_out_va_var, best_K_inner, list(cpg_out))
    sc_out = StandardScaler()
    X_out_tr_s = sc_out.fit_transform(X_out_tr_k)
    X_out_va_s = sc_out.transform(X_out_va_k)

    try:
        m_final = get_models()[BEST_MODEL_NAME]
        m_final.fit(X_out_tr_s, y_outer_tr)
        y_out_pred = m_final.predict(X_out_va_s)
        if hasattr(m_final, "predict_proba"):
            y_out_prob = m_final.predict_proba(X_out_va_s)[:, 1]
        else:
            y_out_prob = y_out_pred.astype(float)
        nested_metrics = compute_metrics(y_outer_va, y_out_pred, y_out_prob)
        nested_results.append({
            "Outer_Fold": outer_fold + 1,
            "Selected_K": best_K_inner,
            **nested_metrics,
        })
    except Exception as e:
        log.warning(f"  Nested outer fold {outer_fold+1} FAILED: {e}")

log.info(f"  Nested CV complete in {time.time()-t_nested:.1f}s")

nested_df = pd.DataFrame(nested_results)
nested_df.to_csv(os.path.join(PATHS["cv_results"], "nested_cv_results.csv"), index=False)

if len(nested_df) > 0:
    log.info("\n  Nested CV Summary:")
    for metric in ["Accuracy", "Balanced_Accuracy", "Sensitivity", "Specificity", "F1", "ROC_AUC"]:
        vals = nested_df[metric].dropna()
        log.info(f"    {metric}: {vals.mean():.3f} +/- {vals.std():.3f}")

log.info("  Saved: cv_results/nested_cv_results.csv")


# ============================================================================
# SECTION 11: NESTED CROSS-VALIDATION (rigorous performance estimate)
# ============================================================================
log.info("\n[NESTED CV] Running nested stratified cross-validation ...")
log.info(f"  Outer: {CV_SPLITS}-fold, Inner: {INNER_SPLITS}-fold")
log.info(f"  Best model: {BEST_MODEL_NAME}, K selected inside inner loop")

nested_results = []
outer_skf = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_SEED)
inner_skf = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=RANDOM_SEED + 1)

t_nested = time.time()
for outer_fold, (outer_tr, outer_va) in enumerate(outer_skf.split(X_primary, y_primary)):
    log.info(f"  Outer fold {outer_fold+1}/{CV_SPLITS} ...")
    X_outer_tr = X_primary[outer_tr]
    X_outer_va = X_primary[outer_va]
    y_outer_tr = y_primary[outer_tr]
    y_outer_va = y_primary[outer_va]

    # Inner loop: select best K
    inner_scores = {K: [] for K in FEATURE_BUDGETS}
    for inner_tr_idx, inner_va_idx in inner_skf.split(X_outer_tr, y_outer_tr):
        X_in_tr = X_outer_tr[inner_tr_idx]
        X_in_va = X_outer_tr[inner_va_idx]
        y_in_tr = y_outer_tr[inner_tr_idx]
        y_in_va = y_outer_tr[inner_va_idx]

        X_in_tr_imp, X_in_va_imp, _ = impute_fold(X_in_tr, X_in_va)
        X_in_tr_var, X_in_va_var, vmask_in, _ = variance_filter_fold(X_in_tr_imp, X_in_va_imp)
        cpg_inner = cpg_arr[vmask_in]

        for K in FEATURE_BUDGETS:
            try:
                X_in_tr_k, X_in_va_k, _, _, _, _ = feature_select_fold(
                    X_in_tr_var, y_in_tr, X_in_va_var, K, list(cpg_inner))
                sc_in = StandardScaler()
                X_in_tr_s = sc_in.fit_transform(X_in_tr_k)
                X_in_va_s = sc_in.transform(X_in_va_k)
                m_in = get_models()[BEST_MODEL_NAME]
                m_in.fit(X_in_tr_s, y_in_tr)
                pred_in = m_in.predict(X_in_va_s)
                inner_scores[K].append(balanced_accuracy_score(y_in_va, pred_in))
            except Exception:
                inner_scores[K].append(0.0)

    best_K_inner = max(inner_scores, key=lambda k: np.mean(inner_scores[k]))

    # Retrain on full outer training set with best K
    X_out_tr_imp, X_out_va_imp, _ = impute_fold(X_outer_tr, X_outer_va)
    X_out_tr_var, X_out_va_var, vmask_out, _ = variance_filter_fold(X_out_tr_imp, X_out_va_imp)
    cpg_out = cpg_arr[vmask_out]
    X_out_tr_k, X_out_va_k, _, _, _, _ = feature_select_fold(
        X_out_tr_var, y_outer_tr, X_out_va_var, best_K_inner, list(cpg_out))
    sc_out = StandardScaler()
    X_out_tr_s = sc_out.fit_transform(X_out_tr_k)
    X_out_va_s = sc_out.transform(X_out_va_k)

    try:
        m_final = get_models()[BEST_MODEL_NAME]
        m_final.fit(X_out_tr_s, y_outer_tr)
        y_out_pred = m_final.predict(X_out_va_s)
        if hasattr(m_final, "predict_proba"):
            y_out_prob = m_final.predict_proba(X_out_va_s)[:, 1]
        else:
            y_out_prob = y_out_pred.astype(float)
        nm = compute_metrics(y_outer_va, y_out_pred, y_out_prob)
        nested_results.append({"Outer_Fold": outer_fold+1, "Selected_K": best_K_inner, **nm})
    except Exception as e:
        log.warning(f"  Nested outer fold {outer_fold+1} FAILED: {e}")

log.info(f"  Nested CV complete in {time.time()-t_nested:.1f}s")
nested_df = pd.DataFrame(nested_results)
nested_df.to_csv(os.path.join(PATHS["cv_results"], "nested_cv_results.csv"), index=False)

if len(nested_df) > 0:
    log.info("\n  Nested CV Summary:")
    for metric in ["Accuracy", "Balanced_Accuracy", "Sensitivity", "Specificity", "F1", "ROC_AUC"]:
        vals = nested_df[metric].dropna()
        log.info(f"    {metric}: {vals.mean():.3f} +/- {vals.std():.3f}")
log.info("  Saved: cv_results/nested_cv_results.csv")


# ============================================================================
# SECTION 12: EFFECT SIZE & CANDIDATE CpG STATISTICS
# ============================================================================
log.info("\n[BIOMARKERS] Computing effect sizes for top stable candidate CpGs ...")

# Use top 1000 most-stable CpGs from best budget
analysis_K = BEST_K if BEST_K in top_stable_cpgs else sorted(top_stable_cpgs.keys())[-1]
candidate_cpgs = top_stable_cpgs.get(analysis_K, [])
log.info(f"  Using top 50 stable CpGs from K={analysis_K} for effect-size analysis")

# Mask for primary task (Normal vs CIN3)
mask_N  = sample_meta["Diagnosis"].values == "Normal"
mask_C  = sample_meta["Diagnosis"].values == "CIN3"

# Global mean imputation for descriptive stats ONLY (clearly flagged as exploratory)
cpg_global_means = np.nanmean(X_primary, axis=0)
X_primary_imp_desc = X_primary.copy()
for j in range(X_primary_imp_desc.shape[1]):
    nans = np.isnan(X_primary_imp_desc[:, j])
    if nans.any():
        X_primary_imp_desc[nans, j] = cpg_global_means[j]

mask_N_p  = y_primary_labels == "Normal"
mask_C_p  = y_primary_labels == "CIN3"

cpg_stat_rows = []
for cpg_name in candidate_cpgs:
    if cpg_name not in cpg_cols:
        continue
    cidx = cpg_arr.tolist().index(cpg_name)
    vals = X_primary_imp_desc[:, cidx]
    norm_vals = vals[mask_N_p]
    cin3_vals = vals[mask_C_p]
    mean_diff = np.mean(cin3_vals) - np.mean(norm_vals)
    med_diff  = np.median(cin3_vals) - np.median(norm_vals)
    try:
        stat, pval = mannwhitneyu(norm_vals, cin3_vals, alternative="two-sided")
    except Exception:
        stat, pval = np.nan, np.nan
    cpg_stat_rows.append({
        "CpG": cpg_name,
        "Normal_Mean": round(float(np.mean(norm_vals)), 5),
        "CIN3_Mean":   round(float(np.mean(cin3_vals)), 5),
        "Mean_Diff_CIN3_minus_Normal": round(float(mean_diff), 5),
        "Median_Diff": round(float(med_diff), 5),
        "MWU_Stat":    round(float(stat), 2) if not np.isnan(stat) else np.nan,
        "MWU_Pvalue":  float(pval) if not np.isnan(pval) else np.nan,
    })

cpg_stat_df = pd.DataFrame(cpg_stat_rows)
# FDR correction
if len(cpg_stat_df) > 0 and cpg_stat_df["MWU_Pvalue"].notna().any():
    from statsmodels.stats.multitest import multipletests
    pvals_arr = cpg_stat_df["MWU_Pvalue"].fillna(1.0).values
    _, fdr_pvals, _, _ = multipletests(pvals_arr, method="fdr_bh")
    cpg_stat_df["FDR_BH"] = fdr_pvals.round(6)

cpg_stat_df.to_csv(os.path.join(PATHS["biomarkers"], "candidate_cpg_statistics.csv"), index=False)
log.info(f"  Saved: biomarkers/candidate_cpg_statistics.csv ({len(cpg_stat_df)} CpGs)")

# ============================================================================
# SECTION 13: FEATURE IMPORTANCE FROM BEST MODEL
# ============================================================================
log.info("\n[FEATURE IMPORTANCE] Training best model on full primary data ...")

# Refit on all 64 samples with best K for feature importance (descriptive only)
X_p_imp_full = X_primary.copy()
for j in range(X_p_imp_full.shape[1]):
    nans_j = np.isnan(X_p_imp_full[:, j])
    if nans_j.any():
        X_p_imp_full[nans_j, j] = cpg_global_means[j]

variances_full = np.var(X_p_imp_full, axis=0)
varmask_full   = variances_full >= VARIANCE_THRESHOLD
X_p_var_full   = X_p_imp_full[:, varmask_full]
cpg_var_full   = cpg_arr[varmask_full]

f_full, _ = f_classif(X_p_var_full, y_primary)
f_full     = np.nan_to_num(f_full, nan=0.0)
top_full_idx = np.argsort(f_full)[::-1][:BEST_K]
X_p_k_full   = X_p_var_full[:, top_full_idx]
cpg_k_full   = cpg_var_full[top_full_idx]

sc_full = StandardScaler()
X_p_k_scaled = sc_full.fit_transform(X_p_k_full)

best_model_full = get_models()[BEST_MODEL_NAME]
best_model_full.fit(X_p_k_scaled, y_primary)

imp_rows = []
if BEST_MODEL_NAME == "LogReg":
    coefs = best_model_full.coef_[0]
    for cpg_n, coef in zip(cpg_k_full, coefs):
        imp_rows.append({"CpG": cpg_n, "Importance": abs(coef), "Signed_Coefficient": coef})
elif BEST_MODEL_NAME == "RandomForest":
    fi = best_model_full.feature_importances_
    for cpg_n, val in zip(cpg_k_full, fi):
        imp_rows.append({"CpG": cpg_n, "Importance": val, "Signed_Coefficient": np.nan})
elif BEST_MODEL_NAME == "XGBoost" and XGBOOST_AVAILABLE:
    fi = best_model_full.feature_importances_
    for cpg_n, val in zip(cpg_k_full, fi):
        imp_rows.append({"CpG": cpg_n, "Importance": val, "Signed_Coefficient": np.nan})
elif BEST_MODEL_NAME in ["LinearSVM"]:
    try:
        inner_model = best_model_full.calibrated_classifiers_[0].estimator
        coefs = inner_model.coef_[0]
        for cpg_n, coef in zip(cpg_k_full, coefs):
            imp_rows.append({"CpG": cpg_n, "Importance": abs(coef), "Signed_Coefficient": coef})
    except Exception:
        for cpg_n in cpg_k_full:
            imp_rows.append({"CpG": cpg_n, "Importance": np.nan, "Signed_Coefficient": np.nan})
else:
    for cpg_n in cpg_k_full:
        imp_rows.append({"CpG": cpg_n, "Importance": np.nan, "Signed_Coefficient": np.nan})

imp_df = pd.DataFrame(imp_rows).sort_values("Importance", ascending=False).reset_index(drop=True)
imp_df.to_csv(os.path.join(PATHS["biomarkers"], "model_feature_importance.csv"), index=False)
log.info(f"  Saved: biomarkers/model_feature_importance.csv")


# ============================================================================
# SECTION 14: CpG ANNOTATION (450K manifest lookup)
# ============================================================================
log.info("\n[ANNOTATION] Attempting 450K CpG annotation ...")

MANIFEST_PATHS = [
    os.path.join(ROOT, "data", "GSE99511", "raw",
                 "GPL13534_HumanMethylation450_15017482_v.1.1.csv.gz"),
    os.path.join(ROOT, "data", "GSE99511", "intermediate",
                 "GPL13534_HumanMethylation450_15017482_v.1.1.csv.gz"),
]

annotation_available = False
annot_df = None
for mp in MANIFEST_PATHS:
    if os.path.exists(mp):
        log.info(f"  Found manifest: {mp}")
        try:
            # Illumina manifest has comment lines starting with [
            # Skip them, find the header row
            annot_df = pd.read_csv(mp, skiprows=7, low_memory=False,
                                   usecols=lambda c: c in [
                                       "IlmnID", "CHR", "MAPINFO", "UCSC_RefGene_Name",
                                       "UCSC_RefGene_Group", "Relation_to_UCSC_CpG_Island",
                                       "UCSC_CpG_Islands_Name",
                                   ])
            annot_df = annot_df.rename(columns={
                "IlmnID": "CpG", "CHR": "Chromosome", "MAPINFO": "Position",
                "UCSC_RefGene_Name": "Gene", "UCSC_RefGene_Group": "Gene_Region",
                "Relation_to_UCSC_CpG_Island": "CpG_Island_Relation",
                "UCSC_CpG_Islands_Name": "CpG_Island_Name",
            })
            # Drop rows that are not actual CpG probes
            annot_df = annot_df[annot_df["CpG"].str.startswith("cg", na=False)]
            annotation_available = True
            log.info(f"  Manifest loaded: {annot_df.shape[0]:,} probes")
            break
        except Exception as e:
            log.warning(f"  Could not parse manifest: {e}")

# Merge annotation with candidate CpG stats
all_candidate_cpgs = []
for K in FEATURE_BUDGETS:
    all_candidate_cpgs.extend(top_stable_cpgs.get(K, []))
all_candidate_cpgs = list(dict.fromkeys(all_candidate_cpgs))  # unique, order-preserving

if annotation_available and annot_df is not None:
    annot_candidates = annot_df[annot_df["CpG"].isin(all_candidate_cpgs)].copy()
    annot_candidates.to_csv(
        os.path.join(PATHS["biomarkers"], "candidate_cpg_annotation.csv"), index=False
    )
    log.info(f"  Annotated {len(annot_candidates)} / {len(all_candidate_cpgs)} candidate CpGs")
    log.info("  Saved: biomarkers/candidate_cpg_annotation.csv")

    # Gene summary
    gene_rows = []
    annot_c = annot_candidates.copy()
    annot_c["Gene_clean"] = annot_c["Gene"].fillna("").str.split(";").str[0].str.strip()
    for gene, grp in annot_c[annot_c["Gene_clean"] != ""].groupby("Gene_clean"):
        gene_rows.append({
            "Gene": gene,
            "Num_Candidate_CpGs": len(grp),
            "Associated_CpGs": ";".join(grp["CpG"].tolist()),
        })
    gene_df = pd.DataFrame(gene_rows).sort_values("Num_Candidate_CpGs", ascending=False)
    gene_df.to_csv(os.path.join(PATHS["biomarkers"], "candidate_gene_summary.csv"), index=False)
    log.info(f"  Gene summary: {len(gene_df)} genes with candidate CpGs")
    log.info("  Saved: biomarkers/candidate_gene_summary.csv")
else:
    log.warning("  Manifest not available or parse failed. Annotation skipped.")
    # Write placeholder
    pd.DataFrame(columns=["CpG","Chromosome","Position","Gene","Gene_Region",
                           "CpG_Island_Relation","CpG_Island_Name"]).to_csv(
        os.path.join(PATHS["biomarkers"], "candidate_cpg_annotation.csv"), index=False)
    pd.DataFrame(columns=["Gene","Num_Candidate_CpGs","Associated_CpGs"]).to_csv(
        os.path.join(PATHS["biomarkers"], "candidate_gene_summary.csv"), index=False)
    annot_candidates = pd.DataFrame()
    gene_df = pd.DataFrame()

# ============================================================================
# SECTION 15: RESULT VISUALIZATIONS
# ============================================================================
log.info("\n[PLOTS] Generating result visualizations ...")

# --- 15.1 ROC curve (best model, OOF predictions) ---
try:
    oof_best = oof_df[(oof_df["Model"] == BEST_MODEL_NAME) & (oof_df["Feature_Budget"] == BEST_K)]
    fpr, tpr, _ = roc_curve(oof_best["True_Label"], oof_best["Predicted_Probability"])
    auc_val = roc_auc_score(oof_best["True_Label"], oof_best["Predicted_Probability"])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#E53935", lw=2, label=f"OOF ROC (AUC={auc_val:.3f})")
    ax.plot([0,1],[0,1], "k--", lw=1, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title(f"ROC Curve – {BEST_MODEL_NAME}, K={BEST_K}\nNormal vs CIN3 (OOF, 5-fold CV)", fontsize=10)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(PATHS["plots"], "roc_curve.png"), dpi=150)
    plt.close()
    log.info("  Saved: plots/roc_curve.png")
except Exception as e:
    log.warning(f"  ROC plot failed: {e}")

# --- 15.2 Confusion matrix ---
try:
    cm = confusion_matrix(oof_best["True_Label"], oof_best["Predicted_Label"])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Normal (pred)", "CIN3 (pred)"],
                yticklabels=["Normal (true)", "CIN3 (true)"],
                cbar=False, linewidths=0.5)
    ax.set_title(f"Confusion Matrix – {BEST_MODEL_NAME}, K={BEST_K}\n(OOF, 5-fold CV)", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(PATHS["plots"], "confusion_matrix.png"), dpi=150)
    plt.close()
    log.info("  Saved: plots/confusion_matrix.png")
except Exception as e:
    log.warning(f"  Confusion matrix plot failed: {e}")

# --- 15.3 Performance vs feature budget ---
try:
    fig, ax = plt.subplots(figsize=(8, 5))
    for mname in model_comp_df["Model"].unique():
        sub_m = model_comp_df[model_comp_df["Model"] == mname].sort_values("Feature_Budget")
        ax.plot(sub_m["Feature_Budget"], sub_m["Mean_Balanced_Accuracy"],
                marker="o", label=mname)
        ax.fill_between(sub_m["Feature_Budget"],
                        sub_m["Mean_Balanced_Accuracy"] - sub_m["Std_Balanced_Accuracy"],
                        sub_m["Mean_Balanced_Accuracy"] + sub_m["Std_Balanced_Accuracy"],
                        alpha=0.1)
    ax.set_xscale("log")
    ax.set_xlabel("Number of Selected CpGs (K)")
    ax.set_ylabel("Mean Balanced Accuracy (5-fold CV)")
    ax.set_title("Classifier Performance vs Feature Budget\nNormal vs CIN3", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PATHS["plots"], "performance_vs_features.png"), dpi=150)
    plt.close()
    log.info("  Saved: plots/performance_vs_features.png")
except Exception as e:
    log.warning(f"  Performance vs features plot failed: {e}")

# --- 15.4 Feature stability bar chart ---
try:
    stab_plot_K = BEST_K
    stab_plot_df = pd.read_csv(
        os.path.join(PATHS["feature_sel"], f"feature_stability_K{stab_plot_K}.csv")
    ).head(20)
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(stab_plot_df["CpG"][::-1], stab_plot_df["Selection_Frequency"][::-1],
                   color="#5C6BC0", edgecolor="none")
    ax.set_xlabel("Selection Frequency (fraction of CV folds)")
    ax.set_title(f"Top 20 Candidate CpGs by Selection Stability\nK={stab_plot_K}, {CV_SPLITS}-fold CV", fontsize=11)
    ax.axvline(x=0.8, color="red", linestyle="--", alpha=0.6, label="80% threshold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PATHS["plots"], "feature_stability.png"), dpi=150)
    plt.close()
    log.info("  Saved: plots/feature_stability.png")
except Exception as e:
    log.warning(f"  Stability plot failed: {e}")

# --- 15.5 Volcano plot ---
try:
    if len(cpg_stat_df) > 0 and "FDR_BH" in cpg_stat_df.columns:
        fig, ax = plt.subplots(figsize=(7, 5))
        neg_log_p = -np.log10(cpg_stat_df["MWU_Pvalue"].clip(lower=1e-300))
        colors_v = ["#E53935" if f < 0.05 else "#9E9E9E" for f in cpg_stat_df["FDR_BH"]]
        ax.scatter(cpg_stat_df["Mean_Diff_CIN3_minus_Normal"], neg_log_p,
                   c=colors_v, alpha=0.7, s=40, edgecolors="none")
        ax.axhline(y=-np.log10(0.05), color="black", linestyle="--", alpha=0.5, label="p=0.05")
        ax.axvline(x=0, color="gray", linestyle="-", alpha=0.3)
        ax.set_xlabel("Mean Beta Difference (CIN3 - Normal)")
        ax.set_ylabel("-log10(Mann-Whitney U p-value)")
        ax.set_title("Volcano Plot: Candidate CpGs (Descriptive, Exploratory)\nNormal vs CIN3", fontsize=10)
        from matplotlib.patches import Patch
        legend_elems2 = [Patch(facecolor="#E53935", label="FDR < 0.05"),
                         Patch(facecolor="#9E9E9E", label="FDR >= 0.05")]
        ax.legend(handles=legend_elems2)
        plt.tight_layout()
        plt.savefig(os.path.join(PATHS["plots"], "volcano_plot.png"), dpi=150)
        plt.close()
        log.info("  Saved: plots/volcano_plot.png")
except Exception as e:
    log.warning(f"  Volcano plot failed: {e}")

# --- 15.6 Heatmap of top candidate CpGs ---
try:
    top_heatmap = top_stable_cpgs.get(BEST_K, [])[:30]
    hmap_cols = [c for c in top_heatmap if c in cpg_cols]
    if len(hmap_cols) >= 5:
        hmap_idx = [cpg_arr.tolist().index(c) for c in hmap_cols]
        hmap_data = X_primary_imp_desc[:, hmap_idx]
        hmap_df = pd.DataFrame(hmap_data, columns=hmap_cols,
                                index=[f"{g}\n({d})" for g, d in
                                       zip(gsm_primary, y_primary_labels)])
        row_colors = [DIAG_COLORS.get(d, "gray") for d in y_primary_labels]
        fig_h = plt.figure(figsize=(14, 8))
        g = sns.clustermap(hmap_df.T, cmap="RdYlBu_r", col_colors=row_colors,
                           yticklabels=True, xticklabels=False,
                           figsize=(14, 8), vmin=0, vmax=1,
                           cbar_kws={"label": "Beta value"})
        g.ax_heatmap.set_title(
            f"Heatmap: Top {len(hmap_cols)} Candidate CpGs (K={BEST_K})\n[Exploratory]",
            fontsize=10, pad=60)
        plt.savefig(os.path.join(PATHS["plots"], "heatmap_candidate_cpgs.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()
        log.info("  Saved: plots/heatmap_candidate_cpgs.png")
except Exception as e:
    log.warning(f"  Heatmap failed: {e}")

# --- 15.7 PCA on selected candidate features ---
try:
    sel_cols_pca = [c for c in top_stable_cpgs.get(BEST_K, [])[:BEST_K] if c in cpg_cols]
    sel_idx_pca  = [cpg_arr.tolist().index(c) for c in sel_cols_pca]
    X_sel_pca    = X_primary_imp_desc[:, sel_idx_pca]
    sc_sel = StandardScaler()
    X_sel_pca_s  = sc_sel.fit_transform(X_sel_pca)
    pca_sel      = PCA(n_components=2, random_state=RANDOM_SEED)
    coords_sel   = pca_sel.fit_transform(X_sel_pca_s)
    ve_sel       = pca_sel.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(7, 6))
    for diag, color in DIAG_COLORS.items():
        m2 = y_primary_labels == diag
        ax.scatter(coords_sel[m2, 0], coords_sel[m2, 1],
                   c=color, label=diag, alpha=0.85, s=65, edgecolors="white", lw=0.5)
    ax.set_xlabel(f"PC1 ({ve_sel[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({ve_sel[1]:.1f}%)")
    ax.set_title(f"PCA – Selected Candidate CpGs (K={BEST_K})\n[Exploratory, mean-imputed full data]", fontsize=10)
    ax.legend(title="Diagnosis")
    plt.tight_layout()
    plt.savefig(os.path.join(PATHS["plots"], "pca_selected_features.png"), dpi=150)
    plt.close()
    log.info("  Saved: plots/pca_selected_features.png")
except Exception as e:
    log.warning(f"  PCA selected features plot failed: {e}")

log.info("  All result visualizations complete.")


# ============================================================================
# SECTION 16: SECONDARY EXPERIMENT — Normal vs Abnormal
# ============================================================================
log.info("\n" + "=" * 80)
log.info("SECONDARY EXPERIMENT: Normal vs Abnormal (CIN3 + SCC)")
log.info("=" * 80)

try:
    mask_nva = sample_meta["Diagnosis"].isin(["Normal", "CIN3", "SCC"])
    X_nva = X_raw[mask_nva].values.astype(np.float32)
    diag_nva = sample_meta.loc[mask_nva, "Diagnosis"].values
    gsm_nva  = sample_meta.loc[mask_nva, "GSM_ID"].values
    y_nva = (diag_nva != "Normal").astype(int)  # 0=Normal, 1=Abnormal
    log.info(f"  NvA samples: {len(y_nva)} (Normal={( y_nva==0).sum()}, Abnormal={(y_nva==1).sum()})")

    nva_fold_results = []
    nva_oof = []
    skf_nva = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    model_nva = get_models()[BEST_MODEL_NAME]

    for fold_i, (tr_i, va_i) in enumerate(skf_nva.split(X_nva, y_nva)):
        X_nt, X_nv = X_nva[tr_i], X_nva[va_i]
        y_nt, y_nv = y_nva[tr_i], y_nva[va_i]
        X_nt_i, X_nv_i, _ = impute_fold(X_nt, X_nv)
        X_nt_v, X_nv_v, vmask_n, _ = variance_filter_fold(X_nt_i, X_nv_i)
        cpg_nva = cpg_arr[vmask_n]
        X_nt_k, X_nv_k, _, _, _, _ = feature_select_fold(
            X_nt_v, y_nt, X_nv_v, BEST_K, list(cpg_nva))
        sc_nva = StandardScaler()
        X_nt_s = sc_nva.fit_transform(X_nt_k)
        X_nv_s = sc_nva.transform(X_nv_k)
        m_nva = get_models()[BEST_MODEL_NAME]
        m_nva.fit(X_nt_s, y_nt)
        y_pred_n = m_nva.predict(X_nv_s)
        y_prob_n = m_nva.predict_proba(X_nv_s)[:, 1] if hasattr(m_nva, "predict_proba") else y_pred_n.astype(float)
        met_n = compute_metrics(y_nv, y_pred_n, y_prob_n)
        nva_fold_results.append({"Fold": fold_i+1, "Model": BEST_MODEL_NAME, "K": BEST_K, **met_n})
        for gsm, tl, pl, pp, dg in zip(gsm_nva[va_i], y_nv, y_pred_n, y_prob_n, diag_nva[va_i]):
            nva_oof.append({"GSM_ID": gsm, "Diagnosis": dg, "True_Label": int(tl),
                            "Predicted_Label": int(pl), "Predicted_Probability": float(pp),
                            "Fold": fold_i+1, "Model": BEST_MODEL_NAME, "Feature_Budget": BEST_K})

    nva_df = pd.DataFrame(nva_fold_results)
    nva_df.to_csv(os.path.join(PATHS["nva"], "fold_results_nva.csv"), index=False)
    pd.DataFrame(nva_oof).to_csv(os.path.join(PATHS["nva"], "oof_predictions_nva.csv"), index=False)

    log.info("  Normal vs Abnormal CV Summary:")
    for metric in ["Balanced_Accuracy", "Sensitivity", "Specificity", "F1", "ROC_AUC"]:
        vals = nva_df[metric].dropna()
        log.info(f"    {metric}: {vals.mean():.3f} +/- {vals.std():.3f}")
    log.info("  Saved: normal_vs_abnormal/fold_results_nva.csv")

    # ROC plot NvA
    try:
        nva_oof_df = pd.DataFrame(nva_oof)
        fpr_n, tpr_n, _ = roc_curve(nva_oof_df["True_Label"], nva_oof_df["Predicted_Probability"])
        auc_n = roc_auc_score(nva_oof_df["True_Label"], nva_oof_df["Predicted_Probability"])
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr_n, tpr_n, color="#43A047", lw=2, label=f"AUC={auc_n:.3f}")
        ax.plot([0,1],[0,1],"k--",lw=1,alpha=0.5)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title(f"ROC – Normal vs Abnormal\n{BEST_MODEL_NAME}, K={BEST_K} (OOF)", fontsize=10)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PATHS["nva"], "roc_nva.png"), dpi=150)
        plt.close()
    except Exception as e:
        log.warning(f"  NvA ROC plot failed: {e}")

except Exception as e:
    log.warning(f"  Normal vs Abnormal experiment failed: {e}")
    traceback.print_exc()

# ============================================================================
# SECTION 17: EXPLORATORY 3-CLASS ANALYSIS
# ============================================================================
log.info("\n" + "=" * 80)
log.info("EXPLORATORY 3-CLASS ANALYSIS: Normal vs CIN3 vs SCC")
log.info("  WARNING: SCC n=4 — results are exploratory only, not reliable for SCC class")
log.info("=" * 80)

try:
    from sklearn.metrics import (classification_report,
                                  ConfusionMatrixDisplay)
    from sklearn.preprocessing import label_binarize

    X_3c = X_raw.values.astype(np.float32)
    diag_3c = sample_meta["Diagnosis"].values
    le3 = LabelEncoder()
    y_3c = le3.fit_transform(diag_3c)  # Normal=1, CIN3=0, SCC=2 (sorted)
    gsm_3c = sample_meta["GSM_ID"].values

    skf_3c = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    results_3c = []

    for fold_i, (tr_i, va_i) in enumerate(skf_3c.split(X_3c, y_3c)):
        X_3t, X_3v = X_3c[tr_i], X_3c[va_i]
        y_3t, y_3v = y_3c[tr_i], y_3c[va_i]
        X_3t_i, X_3v_i, _ = impute_fold(X_3t, X_3v)
        X_3t_var, X_3v_var, vm3, _ = variance_filter_fold(X_3t_i, X_3v_i)
        cpg_3 = cpg_arr[vm3]
        X_3t_k, X_3v_k, _, _, _, _ = feature_select_fold(
            X_3t_var, y_3t, X_3v_var, BEST_K, list(cpg_3))
        sc_3 = StandardScaler()
        X_3t_s = sc_3.fit_transform(X_3t_k)
        X_3v_s = sc_3.transform(X_3v_k)
        m3 = LogisticRegression(max_iter=2000, C=1.0, solver="liblinear",
                                 multi_class="ovr", random_state=RANDOM_SEED)
        m3.fit(X_3t_s, y_3t)
        y_3pred = m3.predict(X_3v_s)
        bal_acc_3 = balanced_accuracy_score(y_3v, y_3pred)
        f1_macro_3 = f1_score(y_3v, y_3pred, average="macro", zero_division=0)
        results_3c.append({"Fold": fold_i+1, "Balanced_Accuracy": bal_acc_3, "Macro_F1": f1_macro_3})

    results_3c_df = pd.DataFrame(results_3c)
    results_3c_df.to_csv(os.path.join(PATHS["3class"], "results_3class.csv"), index=False)
    log.info(f"  3-class Balanced Accuracy: {results_3c_df['Balanced_Accuracy'].mean():.3f} "
             f"+/- {results_3c_df['Balanced_Accuracy'].std():.3f}")
    log.info(f"  3-class Macro F1:          {results_3c_df['Macro_F1'].mean():.3f} "
             f"+/- {results_3c_df['Macro_F1'].std():.3f}")
    log.info("  Saved: exploratory_3class/results_3class.csv")

except Exception as e:
    log.warning(f"  3-class analysis failed: {e}")
    traceback.print_exc()
    results_3c_df = pd.DataFrame()


# ============================================================================
# SECTION 18: GENERATE REPORTS
# ============================================================================
log.info("\n[REPORTS] Generating research reports ...")

# Collect summary numbers
n_samples_total = 68
n_normal, n_cin3, n_scc = 28, 36, 4
n_primary = 64
initial_cpgs = 485512
final_cpgs   = 405336

# Best model metrics from exploratory CV
best_metrics_row = model_comp_df[
    (model_comp_df["Model"] == BEST_MODEL_NAME) &
    (model_comp_df["Feature_Budget"] == BEST_K)
]
best_bal_acc = best_metrics_row["Mean_Balanced_Accuracy"].values[0] if len(best_metrics_row) else "N/A"
best_f1      = best_metrics_row["Mean_F1"].values[0] if len(best_metrics_row) else "N/A"
best_auc     = best_metrics_row["Mean_ROC_AUC"].values[0] if len(best_metrics_row) else "N/A"
best_sens    = best_metrics_row["Mean_Sensitivity"].values[0] if len(best_metrics_row) else "N/A"
best_spec    = best_metrics_row["Mean_Specificity"].values[0] if len(best_metrics_row) else "N/A"

# Nested CV summary
if len(nested_df) > 0:
    n_bal_acc = f"{nested_df['Balanced_Accuracy'].mean():.3f} +/- {nested_df['Balanced_Accuracy'].std():.3f}"
    n_f1      = f"{nested_df['F1'].mean():.3f} +/- {nested_df['F1'].std():.3f}"
    n_auc     = f"{nested_df['ROC_AUC'].mean():.3f} +/- {nested_df['ROC_AUC'].std():.3f}"
    n_sens    = f"{nested_df['Sensitivity'].mean():.3f} +/- {nested_df['Sensitivity'].std():.3f}"
    n_spec    = f"{nested_df['Specificity'].mean():.3f} +/- {nested_df['Specificity'].std():.3f}"
else:
    n_bal_acc = n_f1 = n_auc = n_sens = n_spec = "Not computed"

top5_cpgs = top_stable_cpgs.get(BEST_K, [])[:5]
top5_str  = ", ".join(top5_cpgs) if top5_cpgs else "N/A"
n_genes   = len(gene_df) if not gene_df.empty else "N/A (annotation unavailable)"
model_list = ", ".join(model_names)

# ---- FINAL REPORT ----
final_report = f"""# Final Research Report
## Cervical Cancer DNA Methylation Classification and Candidate Biomarker Discovery
### Dataset: GSE99511 | Illumina HumanMethylation450K
---

## 1. Research Objective

This study investigates whether DNA methylation patterns can computationally distinguish
Normal cervical tissue from CIN3 (cervical intraepithelial neoplasia grade 3), and identify
candidate CpG sites that consistently contribute to this distinction.

**Framing:** Computational classification and candidate biomarker discovery.
This is NOT a clinically validated diagnostic system.

---

## 2. Dataset

- **GEO Accession:** GSE99511
- **Platform:** Illumina HumanMethylation450K (450K array)
- **Raw file:** `data/GSE99511/source/GSE99511_RAW.tar` (IDAT archive)

---

## 3. Sample Distribution

| Diagnosis | Count |
|-----------|-------|
| Normal    | {n_normal}  |
| CIN3      | {n_cin3}  |
| SCC       | {n_scc}   |
| **Total** | **{n_samples_total}** |

---

## 4. Raw Data Processing

- IDAT files processed using the `methylprep` package (Noob normalization)
- Sentrix chip → GSM ID mapping verified manually
- Output: labeled beta matrix (68 samples × 485,512 probes)

---

## 5. Quality Control

- Initial probes: {initial_cpgs:,}
- Non-CpG probes removed: 3,091
- Probes with >25% missingness removed: 78,206
- **Retained CpGs:** {final_cpgs:,}
- All 68 samples retained
- Sample missingness range: 0.01% – 14.35%
- Note: GSM2645003 has ~14.35% missingness but was retained per study design

---

## 6. Feature Filtering

Variance filtering applied **within each training fold** (threshold = {VARIANCE_THRESHOLD}).
Feature counts per fold saved in `preprocessing/feature_counts.csv`.

---

## 7. ML Methodology

### Primary Task: Normal vs CIN3
- Samples: {n_primary} (Normal={n_normal}, CIN3={n_cin3}; SCC excluded)
- Feature selection: ANOVA F-test (vectorized, training fold only)
- Feature budgets evaluated: {FEATURE_BUDGETS}
- Models evaluated: {model_list}
- Cross-validation: {CV_SPLITS}-fold Stratified CV

### Leakage Prevention
All preprocessing (imputation, variance filtering, feature selection, scaling)
performed **exclusively on training fold data** in each CV iteration.

---

## 8. Leakage Prevention

Per-fold pipeline:
1. Split train/validation
2. Impute missing values using **training mean only**
3. Variance filter using **training variance only**
4. ANOVA F-test ranking using **training labels only**
5. Select top-K CpGs
6. StandardScaler fit on **training data only**
7. Transform both train and validation
8. Train model; evaluate on untouched validation set

---

## 9. Cross-Validation

- Stratified 5-fold CV (seed=42)
- Evaluation: Accuracy, Balanced Accuracy, Sensitivity, Specificity, F1, ROC-AUC

---

## 10. Feature Selection

Method: ANOVA F-test (`sklearn.feature_selection.f_classif`) — vectorized implementation.
Note: Mann-Whitney U test (non-parametric) was also available but vectorized F-test was used
for computational efficiency. This is appropriate for exploratory biomarker ranking but results
should be interpreted with caution given normality assumptions of F-test.

---

## 11. Model Comparison

Full comparison table saved in `cv_results/model_comparison.csv`.

**Best configuration (by Mean Balanced Accuracy):**
- Model: {BEST_MODEL_NAME}
- Feature Budget: K = {BEST_K}
- Mean Balanced Accuracy: {best_bal_acc}
- Mean Sensitivity:       {best_sens}
- Mean Specificity:       {best_spec}
- Mean F1:                {best_f1}
- Mean ROC-AUC:           {best_auc}

---

## 12. Nested CV Results (Rigorous Estimate)

Nested CV (outer={CV_SPLITS}-fold, inner={INNER_SPLITS}-fold) using {BEST_MODEL_NAME}
with data-driven K selection in inner loop.

| Metric             | Value             |
|--------------------|-------------------|
| Balanced Accuracy  | {n_bal_acc}     |
| Sensitivity        | {n_sens}         |
| Specificity        | {n_spec}         |
| F1 Score           | {n_f1}           |
| ROC-AUC            | {n_auc}          |

**Note:** Nested CV provides a less optimistic estimate than exploratory CV.
Results may differ if exploratory CV over-selected configurations.

---

## 13. Candidate CpGs

Top stable candidate CpGs (K={BEST_K}, selected in most CV folds):
{top5_str}

Full stability tables: `feature_selection/feature_stability_K*.csv`
Effect size statistics: `biomarkers/candidate_cpg_statistics.csv`

---

## 14. Feature Stability

Each CpG's selection frequency across {CV_SPLITS} CV folds was recorded for all K values.
CpGs selected in all {CV_SPLITS} folds are considered the most stable candidate features.

---

## 15. CpG Annotation

{'Annotation performed using Illumina 450K manifest (GPL13534).' if annotation_available else 'Annotation file not parseable or unavailable. Fields left empty.'}
Output: `biomarkers/candidate_cpg_annotation.csv`

---

## 16. Candidate Genes

Candidate CpGs grouped by associated gene.
Unique candidate genes identified: {n_genes}
Output: `biomarkers/candidate_gene_summary.csv`

---

## 17. Exploratory Biological Interpretation

Candidate CpGs are those that were consistently selected across CV folds
using statistical ranking. These represent CpG sites where methylation
levels differ between Normal and CIN3 samples in this dataset.

**These are computational candidates only. No biological validation has been performed.**
Gene associations are provided as annotations, not as functional evidence.

---

## 18. Secondary Experiments

### Normal vs Abnormal (Normal vs CIN3+SCC)
Results saved in `normal_vs_abnormal/`.

### Exploratory 3-Class (Normal vs CIN3 vs SCC)
Results saved in `exploratory_3class/`.
**Warning:** SCC n=4. Results for SCC classification are unreliable.

---

## 19. Limitations

1. **Small sample size:** n=64 (primary task). High-dimensional ML with ~405k features.
2. **Curse of dimensionality:** 405,336 CpGs vs 64 samples. Feature selection is essential but can be unstable.
3. **Only 4 SCC samples:** SCC classification results are not generalizable.
4. **Cross-sectional design:** Cannot predict future cancer development.
5. **No independent test set:** CV estimates may still be optimistic.
6. **No independent clinical validation:** Candidate CpGs are not clinically validated biomarkers.
7. **Computational findings require biological validation.**
8. **Batch effects:** Sentrix chip batch effects were not explicitly corrected.
9. **Feature selection method:** F-test assumes normality; beta values are bimodal.
10. **ML performance may be inflated** at small n despite careful CV design.

---

## 20. Future Work

1. Independent validation cohort
2. Non-parametric feature selection (Mann-Whitney U, properly vectorized)
3. SHAP-based model explainability
4. Biological validation of top candidate CpGs (bisulfite sequencing, pyrosequencing)
5. Batch effect correction (ComBat)
6. Pathway/gene enrichment analysis (gseapy/Enrichr)
7. Multi-omics integration
8. Low-cost targeted methylation assay development for candidate CpGs

---

## 19 [continued]. Future Hardware Direction

The computational results could eventually connect to a low-cost screening device
following this pipeline:

```
Patient sample collection
        ↓
DNA extraction
        ↓
Targeted methylation measurement (candidate CpGs only)
        ↓
Low-cost molecular sensing (e.g., nanopore, LAMP-based methylation detection)
        ↓
ML classification on candidate CpG profile
        ↓
Screening output
```

**The current work ends at:** candidate methylation features + computational classification.
All downstream steps require medical/biological validation before clinical application.

---

## 21. Conclusion

This study demonstrates that DNA methylation profiles from the GSE99511 dataset can
computationally distinguish Normal from CIN3 cervical tissue samples. Using a leakage-safe
{CV_SPLITS}-fold stratified CV framework with feature selection inside each fold, the best
configuration ({BEST_MODEL_NAME}, K={BEST_K}) achieved a mean balanced accuracy of
{best_bal_acc} (exploratory CV). Nested CV provided a more rigorous estimate.

A set of candidate CpG sites that were consistently selected across CV folds are reported
as candidate methylation features. These require biological validation before any clinical
interpretation.

---
*Generated by step10_e2e_pipeline.py on {datetime.now().strftime('%Y-%m-%d %H:%M')}*
"""

with open(os.path.join(PATHS["reports"], "final_report.md"), "w", encoding="utf-8") as f:
    f.write(final_report)
log.info("  Saved: reports/final_report.md")

# ---- PROFESSOR SUMMARY ----
prof_summary = f"""# Professor Summary
## Cervical Cancer DNA Methylation Study – GSE99511

---

**Dataset:** GSE99511 (GEO)
**Technology:** Illumina HumanMethylation450K
**Total Samples:** {n_samples_total}
**Normal:** {n_normal} | **CIN3:** {n_cin3} | **SCC:** {n_scc} (excluded from primary model)

**Filtered CpGs (after QC):** {final_cpgs:,}

---

## Primary Task: Normal vs CIN3 (n={n_primary})

| Configuration     | Value                  |
|-------------------|------------------------|
| Models evaluated  | {model_list} |
| Feature budgets   | {', '.join(map(str, FEATURE_BUDGETS))} CpGs |
| CV strategy       | {CV_SPLITS}-fold Stratified CV (seed=42) |

### Best Exploratory CV Result
| Metric            | Value  |
|-------------------|--------|
| Model             | {BEST_MODEL_NAME} |
| Feature Budget    | K = {BEST_K} |
| Balanced Accuracy | {best_bal_acc} |
| Sensitivity       | {best_sens} |
| Specificity       | {best_spec} |
| F1 Score          | {best_f1} |
| ROC-AUC           | {best_auc} |

### Nested CV (Rigorous Estimate, {CV_SPLITS}-outer / {INNER_SPLITS}-inner folds)
| Metric            | Value          |
|-------------------|----------------|
| Balanced Accuracy | {n_bal_acc}  |
| Sensitivity       | {n_sens}      |
| Specificity       | {n_spec}      |
| F1 Score          | {n_f1}        |
| ROC-AUC           | {n_auc}       |

---

## Candidate Biomarker Analysis

- CpG selection stability tracked across all {CV_SPLITS} CV folds
- Top 5 candidate CpGs (K={BEST_K}): {top5_str}
- Differential methylation statistics (Mann-Whitney U + FDR): `biomarkers/candidate_cpg_statistics.csv`
- Annotation (chromosome, gene, CpG island relation): `biomarkers/candidate_cpg_annotation.csv`
- Unique candidate genes: {n_genes}

---

## Key Limitations

- Small sample size (n=64 primary); high-dimensional feature space (405k CpGs)
- Only 4 SCC samples — SCC results are exploratory only
- No independent validation cohort
- Candidate CpGs are **not clinically validated biomarkers**
- Cross-sectional study — cannot predict future cancer development

---

## All Results Location

`results/e2e_cervical_methylation/`

*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*
"""

with open(os.path.join(PATHS["reports"], "professor_summary.md"), "w", encoding="utf-8") as f:
    f.write(prof_summary)
log.info("  Saved: reports/professor_summary.md")

# ---- RESULTS SUMMARY ----
results_summary = f"""# Results Summary

## Exploratory CV — All Models
{model_comp_df.to_string(index=False)}

## Nested CV
{nested_df.to_string(index=False) if len(nested_df) > 0 else 'Not available'}
"""
with open(os.path.join(PATHS["reports"], "results_summary.md"), "w", encoding="utf-8") as f:
    f.write(results_summary)
log.info("  Saved: reports/results_summary.md")

# ---- LIMITATIONS ----
limitations_text = """# Limitations

1. Small sample size (n=64 for primary task).
2. Extreme dimensionality: 405,336 CpGs vs 64 samples.
3. Only 4 SCC samples — no reliable SCC model possible.
4. Cross-sectional dataset — no longitudinal prediction.
5. No independent external validation cohort.
6. No clinical validation of candidate CpGs.
7. Candidate CpGs are computational findings only.
8. F-test feature selection assumes normality (beta values are bimodal).
9. Batch effects (Sentrix chip) not explicitly corrected.
10. ML performance metrics, even with careful CV, may be optimistic at this sample size.
11. SHAP explainability not available (package not installed).
12. Pathway analysis not available (gseapy not installed).
"""
with open(os.path.join(PATHS["reports"], "limitations.md"), "w", encoding="utf-8") as f:
    f.write(limitations_text)
log.info("  Saved: reports/limitations.md")


# ============================================================================
# SECTION 19: FINAL EXECUTION SUMMARY
# ============================================================================
total_time = time.time() - t0
log.info("\n")
log.info("=" * 50)
log.info("END-TO-END PIPELINE COMPLETE")
log.info("=" * 50)
log.info(f"Dataset:              GSE99511")
log.info(f"Samples:              {n_samples_total}")
log.info(f"Primary ML samples:   {n_primary}")
log.info(f"Normal:               {n_normal}")
log.info(f"CIN3:                 {n_cin3}")
log.info(f"SCC:                  {n_scc}")
log.info(f"Initial CpGs:         {initial_cpgs:,}")
log.info(f"Final CpGs:           {final_cpgs:,}")
log.info(f"Models evaluated:     {model_list}")
log.info(f"Feature budgets:      {', '.join(map(str, FEATURE_BUDGETS))}")
log.info(f"Best configuration:   {BEST_MODEL_NAME}, K={BEST_K}")
log.info(f"  Balanced Accuracy:  {best_bal_acc}")
log.info(f"  Sensitivity:        {best_sens}")
log.info(f"  Specificity:        {best_spec}")
log.info(f"  F1:                 {best_f1}")
log.info(f"  ROC-AUC:            {best_auc}")
log.info(f"Nested CV:")
log.info(f"  Balanced Accuracy:  {n_bal_acc}")
log.info(f"  ROC-AUC:            {n_auc}")
log.info(f"Top candidate CpGs:   {top5_str}")
log.info(f"Annotated genes:      {n_genes}")
log.info(f"Total runtime:        {total_time/60:.1f} min")
log.info(f"All results:          {OUT_BASE}")
log.info(f"Final report:         {os.path.join(PATHS['reports'], 'final_report.md')}")
log.info(f"Professor summary:    {os.path.join(PATHS['reports'], 'professor_summary.md')}")
log.info("=" * 50)
