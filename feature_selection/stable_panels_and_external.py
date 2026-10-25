"""
stable_panels_and_external.py
-----------------------------
Implements Section 4:
1. Run stability selection (100 subsamples/bootstraps) on GSE99511.
2. Select stable features (selection frequency >= 60% or top ranked).
3. Compute Kuncheva stability index and Jaccard index.
4. Build final 5, 10, 20, 50 marker panels.
5. Harmonize with GSE46306 common probe intersection.
6. Train on GSE99511, test once on GSE46306 with zero tuning on it.
7. Output external validation table and cross-dataset shift analysis.
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from data_loader import load_primary_data, impute_fold, variance_prefilter
from validate_gse46306 import load_external
from evaluator import compute_metrics, _get_classifier_proba
from stability import kuncheva_index, jaccard_pairwise

warnings.filterwarnings("ignore")
CONFIG_PATH = os.path.join(HERE, "config.json")
OUT_DIR = os.path.join(ROOT, "results", "feature_selection_benchmark")


def run_stability_selection_100(X: np.ndarray, y: np.ndarray, cpg_names: list,
                               n_bootstraps: int = 100, subsample_fraction: float = 0.70,
                               seed: int = 42, verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """
    Run 100 stratified subsamples to compute feature selection frequency and stability indices.
    """
    if verbose:
        print(f"[StabilitySelection] Running {n_bootstraps} subsamples (fraction={subsample_fraction})...")
        
    rng = np.random.RandomState(seed)
    n_samples = len(y)
    n_sub = int(n_samples * subsample_fraction)
    
    # In-fold pre-filter to top 5000 by variance on full X for computational feasibility
    var_full = np.nanvar(X, axis=0)
    top_var_idx = np.argsort(var_full)[::-1][:min(5000, X.shape[1])]
    X_sub = X[:, top_var_idx]
    cpg_sub = [cpg_names[i] for i in top_var_idx]
    
    # M-value transform
    eps = 1e-5
    X_sub = np.clip(X_sub, eps, 1.0 - eps)
    X_sub = np.log2(X_sub / (1.0 - X_sub))
    
    idx_class0 = np.where(y == 0)[0]
    idx_class1 = np.where(y == 1)[0]
    
    n_c0 = int(len(idx_class0) * subsample_fraction)
    n_c1 = int(len(idx_class1) * subsample_fraction)
    
    selection_counts = {c: 0 for c in cpg_sub}
    selected_lists_by_k = {k: [] for k in [5, 10, 20, 50]}
    
    for b in range(n_bootstraps):
        sub_c0 = rng.choice(idx_class0, size=n_c0, replace=False)
        sub_c1 = rng.choice(idx_class1, size=n_c1, replace=False)
        sub_idx = np.concatenate([sub_c0, sub_c1])
        
        X_b = X_sub[sub_idx]
        y_b = y[sub_idx]
        
        # Rank by Mann-Whitney U test p-values
        p_vals = []
        for col_i in range(X_b.shape[1]):
            try:
                _, p = stats.mannwhitneyu(X_b[y_b == 0, col_i], X_b[y_b == 1, col_i])
            except Exception:
                p = 1.0
            p_vals.append(p)
            
        ranked_idx = np.argsort(p_vals)
        for k in [5, 10, 20, 50]:
            chosen_cpgs = [cpg_sub[i] for i in ranked_idx[:k]]
            selected_lists_by_k[k].append(chosen_cpgs)
            
        # Top 50 count for general stability
        for i in ranked_idx[:50]:
            selection_counts[cpg_sub[i]] += 1

    freq_df = pd.DataFrame([
        {"CpG": c, "Selection_Frequency": count / n_bootstraps}
        for c, count in selection_counts.items()
    ]).sort_values("Selection_Frequency", ascending=False).reset_index(drop=True)
    
    # Compute Kuncheva and Jaccard across bootstrap runs
    stability_metrics = {}
    for k in [5, 10, 20, 50]:
        k_lists = selected_lists_by_k[k]
        mean_j = jaccard_pairwise(k_lists)
        mean_k = kuncheva_index(k_lists, k=k, p=len(cpg_sub))
        stability_metrics[k] = {
            "mean_jaccard": float(mean_j),
            "mean_kuncheva": float(mean_k)
        }
        if verbose:
            print(f"  k={k:2d}: Jaccard={mean_j:.3f}, Kuncheva={mean_k:.3f}")

    return freq_df, stability_metrics



def run_stable_panels_and_external_validation(verbose: bool = True):
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    # 1. Load both cohorts
    train = load_primary_data(config, verbose=verbose)
    X_train_full = train["X"]
    y_train = train["y"]
    train_cpgs = train["cpg_names"]
    
    external_df, y_external, ext_meta = load_external(config)
    
    # Common probes intersection
    common_cpgs = sorted(set(train_cpgs).intersection(set(external_df.columns)))
    if verbose:
        print(f"\n[CommonProbes] Intersection of GSE99511 and GSE46306: {len(common_cpgs):,} CpGs.")
        
    train_cpg_map = {name: i for i, name in enumerate(train_cpgs)}
    train_indices = np.array([train_cpg_map[c] for c in common_cpgs], dtype=int)
    
    X_train = X_train_full[:, train_indices]
    X_external = external_df.loc[:, common_cpgs].to_numpy(dtype=np.float32)
    
    # Preprocessing consistency check
    print(f"[PreprocessingConsistency] GSE99511 beta range: [{np.nanmin(X_train):.3f}, {np.nanmax(X_train):.3f}], mean: {np.nanmean(X_train):.3f}")
    print(f"[PreprocessingConsistency] GSE46306 beta range: [{np.nanmin(X_external):.3f}, {np.nanmax(X_external):.3f}], mean: {np.nanmean(X_external):.3f}")

    # 2. Run 100-bootstrap stability selection on GSE99511
    freq_df, stab_dict = run_stability_selection_100(X_train, y_train, common_cpgs, n_bootstraps=100, seed=42, verbose=verbose)
    
    # Save stability frequency
    freq_csv = os.path.join(OUT_DIR, "panels", "stability_selection_frequencies.csv")
    os.makedirs(os.path.dirname(freq_csv), exist_ok=True)
    freq_df.to_csv(freq_csv, index=False)
    
    # 3. Form stable consensus panels
    # Filter features with frequency >= 60%, fallback to top k
    stable_features_60 = freq_df[freq_df["Selection_Frequency"] >= 0.60]["CpG"].tolist()
    if verbose:
        print(f"[StablePanels] {len(stable_features_60)} CpGs have selection frequency >= 60%.")

    panel_cpgs_dict = {}
    for k in [5, 10, 20, 50]:
        chosen = freq_df.iloc[:k]["CpG"].tolist()
        panel_cpgs_dict[k] = chosen

    # Save stable panels
    panel_rows = []
    for k, c_list in panel_cpgs_dict.items():
        for rank, c in enumerate(c_list, start=1):
            panel_rows.append({"k": k, "rank": rank, "CpG": c, "Selection_Frequency": freq_df[freq_df["CpG"] == c]["Selection_Frequency"].values[0]})
    panels_csv = os.path.join(OUT_DIR, "panels", "stable_consensus_panels.csv")
    pd.DataFrame(panel_rows).to_csv(panels_csv, index=False)
    if verbose:
        print(f"Saved stable panels to {panels_csv}")

    # 4. Train on GSE99511 and Test Once on GSE46306
    # Convert to M-values
    eps = 1e-5
    X_tr_m = np.log2(np.clip(X_train, eps, 1.0 - eps) / (1.0 - np.clip(X_train, eps, 1.0 - eps)))
    X_te_m = np.log2(np.clip(X_external, eps, 1.0 - eps) / (1.0 - np.clip(X_external, eps, 1.0 - eps)))
    
    common_idx_map = {name: i for i, name in enumerate(common_cpgs)}
    
    results = []
    shift_rows = []
    
    for k in [5, 10, 20, 50]:
        cpg_k = panel_cpgs_dict[k]
        k_indices = np.array([common_idx_map[c] for c in cpg_k], dtype=int)
        
        X_tr_k = X_tr_m[:, k_indices]
        X_te_k = X_te_m[:, k_indices]
        
        # Scaling fit on GSE99511 only
        scaler = StandardScaler().fit(X_tr_k)
        X_tr_sc = scaler.transform(X_tr_k)
        X_te_sc = scaler.transform(X_te_k)
        
        # Check cross-dataset shift on this panel
        for col_idx, cpg in enumerate(cpg_k):
            val_tr = X_tr_k[:, col_idx]
            val_te = X_te_k[:, col_idx]
            ks_stat, ks_pval = stats.ks_2samp(val_tr, val_te)
            shift_rows.append({
                "k": k,
                "CpG": cpg,
                "GSE99511_Mean_M": round(float(val_tr.mean()), 3),
                "GSE46306_Mean_M": round(float(val_te.mean()), 3),
                "Mean_M_Difference": round(float(val_te.mean() - val_tr.mean()), 3),
                "KS_Stat": round(float(ks_stat), 4),
                "KS_Pval": round(float(ks_pval), 6),
                "Significant_Shift_p05": bool(ks_pval < 0.05)
            })
            
        for clf_name in ["logreg", "linear_svm"]:
            if clf_name == "logreg":
                clf = LogisticRegression(C=1.0, solver="liblinear", random_state=42)
            else:
                clf = CalibratedClassifierCV(LinearSVC(C=1.0, random_state=42), cv=2)
                
            clf.fit(X_tr_sc, y_train)
            y_pred = clf.predict(X_te_sc)
            y_prob = _get_classifier_proba(clf, X_te_sc)
            
            m = compute_metrics(y_external, y_pred, y_prob)
            results.append({
                "Cohort_Train": "GSE99511",
                "Cohort_Test": "GSE46306",
                "k": k,
                "Classifier": clf_name,
                "Jaccard_Stability": round(stab_dict[k]["mean_jaccard"], 4),
                "Kuncheva_Index": round(stab_dict[k]["mean_kuncheva"], 4),
                **m
            })
            
    res_df = pd.DataFrame(results)
    res_csv = os.path.join(OUT_DIR, "stable_external_validation.csv")
    res_df.to_csv(res_csv, index=False)

    shift_df = pd.DataFrame(shift_rows).drop_duplicates(subset=["CpG"])
    shift_csv = os.path.join(OUT_DIR, "cross_dataset_shift.csv")
    shift_df.to_csv(shift_csv, index=False)

    sig_shift_pct = (shift_df["Significant_Shift_p05"].sum() / len(shift_df) * 100) if len(shift_df) > 0 else 0.0

    print("\n" + "="*70)
    print("EXTERNAL VALIDATION RESULTS (Train: GSE99511 -> Test: GSE46306)")
    print("="*70)
    print(res_df[["k", "Classifier", "Balanced_Accuracy", "ROC_AUC", "Sensitivity", "Specificity", "F1", "Accuracy"]].to_string(index=False))
    print(f"\nCross-Dataset Distribution Shift: {shift_df['Significant_Shift_p05'].sum()}/{len(shift_df)} ({sig_shift_pct:.1f}%) panel CpGs show significant distribution shift between cohorts (KS test p < 0.05).")
    print(f"Saved external validation to {res_csv}")
    print(f"Saved shift analysis to {shift_csv}")
    
    return res_df, shift_df


if __name__ == "__main__":
    run_stable_panels_and_external_validation(verbose=True)
