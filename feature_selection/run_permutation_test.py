"""
run_permutation_test.py
-----------------------
Runs a 100-permutation test to establish empirical null distributions
for Balanced Accuracy and ROC-AUC, testing whether the observed ~60-65%
balanced accuracy statistically beats random chance.
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from data_loader import load_primary_data, impute_fold, variance_prefilter
from evaluator import compute_metrics, _get_classifier_proba

warnings.filterwarnings("ignore")
CONFIG_PATH = os.path.join(HERE, "config.json")
OUT_DIR = os.path.join(ROOT, "results", "feature_selection_benchmark")


def run_permutation_benchmark(n_permutations: int = 100, k: int = 50, verbose: bool = True):
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    data = load_primary_data(config, verbose=verbose)
    X = data["X"]
    y = data["y"]
    cpg_names = data["cpg_names"]

    # Observed score (true labels) with M-values, 5-fold CV, MW filter, Linear SVM
    base_seed = 42
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=base_seed)

    def _eval_pipeline(X_curr, y_curr, seed):
        fold_scores_bal = []
        fold_scores_auc = []
        for fold_i, (tr, va) in enumerate(skf.split(X_curr, y_curr)):
            X_tr, X_va = X_curr[tr], X_curr[va]
            y_tr, y_va = y_curr[tr], y_curr[va]

            X_tr, X_va, _ = impute_fold(X_tr, X_va)

            # M-values
            eps = 1e-5
            X_tr = np.clip(X_tr, eps, 1.0 - eps)
            X_va = np.clip(X_va, eps, 1.0 - eps)
            X_tr = np.log2(X_tr / (1.0 - X_tr))
            X_va = np.log2(X_va / (1.0 - X_va))

            # Pre-filter top 2000
            var_tr = np.var(X_tr, axis=0)
            top_var_idx = np.argsort(var_tr)[::-1][:min(2000, X_tr.shape[1])]
            X_tr_pf = X_tr[:, top_var_idx]
            X_va_pf = X_va[:, top_var_idx]

            # In-fold MW ranking
            p_vals = []
            for col in range(X_tr_pf.shape[1]):
                try:
                    _, p = stats.mannwhitneyu(X_tr_pf[y_tr == 0, col], X_tr_pf[y_tr == 1, col])
                except Exception:
                    p = 1.0
                p_vals.append(p)
            top_k = np.argsort(p_vals)[:k]

            scaler = StandardScaler()
            X_tr_k = scaler.fit_transform(X_tr_pf[:, top_k])
            X_va_k = scaler.transform(X_va_pf[:, top_k])

            clf = CalibratedClassifierCV(LinearSVC(C=1.0, random_state=seed + fold_i), cv=2)
            clf.fit(X_tr_k, y_tr)
            y_pred = clf.predict(X_va_k)
            y_prob = _get_classifier_proba(clf, X_va_k)

            m = compute_metrics(y_va, y_pred, y_prob)
            fold_scores_bal.append(m["Balanced_Accuracy"])
            fold_scores_auc.append(m["ROC_AUC"])

        return float(np.mean(fold_scores_bal)), float(np.mean(fold_scores_auc))

    if verbose:
        print("\nComputing true observed score...")
    obs_bal, obs_auc = _eval_pipeline(X, y, seed=base_seed)
    if verbose:
        print(f"Observed Balanced Accuracy: {obs_bal:.4f}, ROC-AUC: {obs_auc:.4f}")

    if verbose:
        print(f"\nRunning {n_permutations} label permutations...")
    
    rng = np.random.RandomState(base_seed)
    null_bal = []
    null_auc = []
    t0 = time.time()

    for p_i in range(n_permutations):
        y_shuffled = rng.permutation(y)
        bal_p, auc_p = _eval_pipeline(X, y_shuffled, seed=base_seed + p_i + 1)
        null_bal.append(bal_p)
        null_auc.append(auc_p)
        if verbose and (p_i + 1) % 20 == 0:
            el = time.time() - t0
            print(f"  Permutation {p_i+1}/{n_permutations} done ({el:.1f}s elapsed)")

    # Empirical p-values
    p_val_bal = (np.sum(np.array(null_bal) >= obs_bal) + 1) / (n_permutations + 1)
    p_val_auc = (np.sum(np.array(null_auc) >= obs_auc) + 1) / (n_permutations + 1)

    results = {
        "n_permutations": n_permutations,
        "k": k,
        "observed_balanced_accuracy": round(obs_bal, 4),
        "observed_roc_auc": round(obs_auc, 4),
        "null_balanced_accuracy_mean": round(float(np.mean(null_bal)), 4),
        "null_balanced_accuracy_std": round(float(np.std(null_bal)), 4),
        "null_balanced_accuracy_95th_percentile": round(float(np.percentile(null_bal, 95)), 4),
        "null_roc_auc_mean": round(float(np.mean(null_auc)), 4),
        "null_roc_auc_std": round(float(np.std(null_auc)), 4),
        "null_roc_auc_95th_percentile": round(float(np.percentile(null_auc, 95)), 4),
        "empirical_p_value_balanced_accuracy": round(float(p_val_bal), 4),
        "empirical_p_value_roc_auc": round(float(p_val_auc), 4),
        "statistically_significant_above_chance_p05": bool(p_val_bal < 0.05),
    }

    out_file = os.path.join(OUT_DIR, "permutation_test_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    df_null = pd.DataFrame({"permutation": range(1, n_permutations + 1), "null_balanced_accuracy": null_bal, "null_roc_auc": null_auc})
    df_null.to_csv(os.path.join(OUT_DIR, "permutation_null_distribution.csv"), index=False)

    if verbose:
        print("\n" + "="*70)
        print("PERMUTATION TEST RESULTS")
        print("="*70)
        print(f"Observed Balanced Accuracy: {obs_bal:.4f} vs Null: {np.mean(null_bal):.4f} ± {np.std(null_bal):.4f} (p = {p_val_bal:.4f})")
        print(f"Observed ROC-AUC:          {obs_auc:.4f} vs Null: {np.mean(null_auc):.4f} ± {np.std(null_auc):.4f} (p = {p_val_auc:.4f})")
        print(f"Statistical significance (p < 0.05): {results['statistically_significant_above_chance_p05']}")
        print(f"Saved results to {out_file}")

    return results


if __name__ == "__main__":
    run_permutation_benchmark(n_permutations=100, k=50, verbose=True)
