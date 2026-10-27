"""
feature_selection/rfe_stability.py
----------------------------------
Recursive Feature Elimination (RFE) combined with Bootstrap Stability Selection.
Selects robust, reproducible 5, 10, 20, and 50-CpG panels using training-fold-only data.
Computes:
1. Probe-level selection frequencies across B=100 bootstrap resamplings
2. Kuncheva and Nogueira stability indices
3. Stable consensus panels of sizes k in [5, 10, 20, 50]
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple


def compute_kuncheva_index(selected_subsets: List[set], n_total_features: int) -> float:
    """
    Computes Kuncheva's stability index for a collection of selected feature sets.
    KI ranges from -1 to 1, where 1 means identical subsets across all bootstraps,
    0 means random overlap, and < 0 means less overlap than expected by chance.
    """
    m = len(selected_subsets)
    if m < 2:
        return 1.0
    
    k_sizes = [len(s) for s in selected_subsets]
    mean_k = np.mean(k_sizes)
    if mean_k == 0 or mean_k == n_total_features:
        return 1.0

    sum_ki = 0.0
    pair_count = 0
    for i in range(m):
        for j in range(i + 1, m):
            s_i, s_j = selected_subsets[i], selected_subsets[j]
            r = len(s_i.intersection(s_j))
            k_i, k_j = len(s_i), len(s_j)
            denom = min(k_i, k_j) - (k_i * k_j) / n_total_features
            if abs(denom) > 1e-12:
                ki = (r - (k_i * k_j) / n_total_features) / denom
                sum_ki += ki
            else:
                sum_ki += 1.0
            pair_count += 1
            
    return sum_ki / pair_count if pair_count > 0 else 0.0


def compute_nogueira_index(selection_matrix: np.ndarray) -> float:
    """
    Computes Nogueira et al. (JMLR 2018) stability index.
    selection_matrix: shape (n_bootstraps, n_features), binary indicator.
    """
    M, d = selection_matrix.shape
    if M < 2:
        return 1.0
    
    p_hat = np.mean(selection_matrix, axis=0)  # selection frequencies
    k_bar = np.sum(p_hat)
    if k_bar == 0 or k_bar == d:
        return 1.0
    
    sample_var = np.sum(p_hat * (1.0 - p_hat)) * (M / (M - 1.0))
    expected_var = (k_bar / d) * (1.0 - k_bar / d) * d
    if expected_var <= 1e-12:
        return 1.0
    
    phi = 1.0 - sample_var / expected_var
    return float(np.clip(phi, -1.0, 1.0))


def rfe_bootstrap_stability_selection(
    X: np.ndarray,
    y: np.ndarray,
    cpg_names: list,
    target_k_list: list = [5, 10, 20, 50],
    n_bootstraps: int = 100,
    estimator_type: str = "elastic_net",
    prefilter_top_m: int = 500,
    random_seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """
    Perform RFE with bootstrap stability selection.
    
    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Training feature matrix.
    y : np.ndarray, shape (n_samples,)
        Binary target labels.
    cpg_names : list of str
        Feature names.
    target_k_list : list of int
        Panel sizes to extract consensus features for.
    n_bootstraps : int
        Number of bootstrap iterations.
    estimator_type : str
        Base estimator: 'elastic_net', 'svm', or 'logreg'.
    prefilter_top_m : int
        Fast in-fold prefilter to candidate probes before RFE.
    random_seed : int
        RNG seed.

    Returns
    -------
    dict with:
        'selection_frequencies': pd.Series mapping CpG name -> frequency
        'consensus_panels': dict mapping k -> list of top k CpGs
        'stability_indices': dict mapping k -> {'kuncheva': float, 'nogueira': float}
        'bootstrap_subsets': dict mapping k -> list of selected probe sets
    """
    n_samples, n_features = X.shape
    rng = np.random.RandomState(random_seed)
    
    # In-fold prefiltering to top M variable features if feature count is large
    if n_features > prefilter_top_m:
        vars_ = np.var(X, axis=0)
        top_idx = np.argsort(vars_)[-prefilter_top_m:]
        X_sub = X[:, top_idx]
        sub_cpgs = [cpg_names[i] for i in top_idx]
    else:
        X_sub = X
        sub_cpgs = list(cpg_names)
        
    n_sub = X_sub.shape[1]
    max_k = max(target_k_list)
    
    # Store selection across bootstraps for each k
    subsets_by_k = {k: [] for k in target_k_list}
    counts_by_probe = {c: 0 for c in sub_cpgs}
    
    for b in range(n_bootstraps):
        # Stratified bootstrap sampling
        idx_0 = np.where(y == 0)[0]
        idx_1 = np.where(y == 1)[0]
        b_idx_0 = rng.choice(idx_0, size=len(idx_0), replace=True)
        b_idx_1 = rng.choice(idx_1, size=len(idx_1), replace=True)
        boot_idx = np.concatenate([b_idx_0, b_idx_1])
        
        X_b, y_b = X_sub[boot_idx], y[boot_idx]
        
        # Scale in-fold
        scaler = StandardScaler()
        X_b_scaled = scaler.fit_transform(X_b)
        
        if estimator_type == "svm":
            base_estimator = LinearSVC(C=1.0, max_iter=2000, random_state=random_seed + b)
        elif estimator_type == "elastic_net":
            base_estimator = LogisticRegression(
                penalty="elasticnet", solver="saga", l1_ratio=0.5, C=0.5,
                max_iter=500, random_state=random_seed + b
            )
        else:
            base_estimator = LogisticRegression(
                penalty="l2", solver="liblinear", C=1.0, max_iter=1000,
                random_state=random_seed + b
            )
            
        # Run RFE down to min(target_k_list)
        # We step down progressively to retrieve rankings
        rfe = RFE(estimator=base_estimator, n_features_to_select=min(target_k_list), step=0.2)
        try:
            rfe.fit(X_b_scaled, y_b)
            rankings = rfe.ranking_  # 1 means top selected
            ranked_order = np.argsort(rankings)
        except Exception:
            # Fallback to feature importance or variances if solver did not converge
            coef = np.abs(np.mean(X_b_scaled[y_b==1], axis=0) - np.mean(X_b_scaled[y_b==0], axis=0))
            ranked_order = np.argsort(coef)[::-1]

        for k in target_k_list:
            top_k_indices = ranked_order[:k]
            top_k_probes = set(sub_cpgs[i] for i in top_k_indices)
            subsets_by_k[k].append(top_k_probes)
            if k == max_k:
                for p in top_k_probes:
                    counts_by_probe[p] += 1
                    
        if verbose and ((b + 1) % 25 == 0 or (b + 1) == n_bootstraps):
            print(f"  Completed bootstrap {b+1}/{n_bootstraps}...")

    # Selection frequencies
    freq_series = pd.Series({c: counts_by_probe[c] / n_bootstraps for c in sub_cpgs}).sort_values(ascending=False)
    
    consensus_panels = {}
    stability_indices = {}
    
    for k in target_k_list:
        # Top k probes with highest selection frequency across bootstraps
        # Subsets for this k
        sub_list = subsets_by_k[k]
        
        # Calculate selection frequencies specifically for this panel size k
        k_counts = {c: 0 for c in sub_cpgs}
        for s in sub_list:
            for p in s:
                k_counts[p] += 1
        k_freq = pd.Series(k_counts).sort_values(ascending=False)
        top_k = list(k_freq.index[:k])
        consensus_panels[k] = top_k
        
        # Compute stability metrics
        ki = compute_kuncheva_index(sub_list, n_total_features=n_sub)
        
        # Binary selection matrix for Nogueira index
        mat = np.zeros((n_bootstraps, n_sub), dtype=int)
        cpg_to_idx = {c: i for i, c in enumerate(sub_cpgs)}
        for b_idx, s in enumerate(sub_list):
            for p in s:
                mat[b_idx, cpg_to_idx[p]] = 1
        nogueira = compute_nogueira_index(mat)
        
        stability_indices[k] = {
            "kuncheva_index": round(ki, 4),
            "nogueira_index": round(nogueira, 4),
            "mean_pairwise_jaccard": round(
                np.mean([len(s1.intersection(s2)) / len(s1.union(s2))
                         for i, s1 in enumerate(sub_list) for s2 in sub_list[i+1:]]), 4
            ) if len(sub_list) > 1 else 1.0
        }
        
    return {
        "selection_frequencies": freq_series,
        "consensus_panels": consensus_panels,
        "stability_indices": stability_indices,
        "bootstrap_subsets": subsets_by_k,
    }
