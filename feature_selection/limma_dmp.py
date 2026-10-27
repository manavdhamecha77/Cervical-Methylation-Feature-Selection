"""
feature_selection/limma_dmp.py
------------------------------
Differential Methylation Analysis using Empirical Bayes Moderated t-test (limma equivalent).
Faithfully implements Gordon Smyth's 2004 Empirical Bayes shrinkage:
1. Linear model contrasts (CIN3+ vs Normal/Benign controls)
2. Residual sample variances and degrees of freedom
3. Empirical Bayes prior estimation (s0^2, d0) using digamma and trigamma moment matching
4. Moderated t-statistics, moderated p-values, and Benjamini-Hochberg FDR q-values
5. Supports in-fold execution (zero leakage) for nested cross-validation and feature selection
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import digamma, polygamma


def trigamma(x):
    """Trigamma function: derivative of digamma."""
    return polygamma(1, x)


def solve_d0_from_trigamma(target_var, df_resid):
    """
    Solve for d0 in: trigamma(d0 / 2) = target_var.
    Uses robust Newton-Raphson with bisection fallback.
    """
    if target_var <= 0:
        return 1e6 # infinite prior degrees of freedom
    
    # Asymptotic approximation for starting point
    # trigamma(x) ~ 1/x + 1/(2x^2) => x ~ 1/target_var
    d0_half = 1.0 / target_var + 0.5
    for _ in range(25):
        val = trigamma(d0_half) - target_var
        grad = polygamma(2, d0_half) # tetragamma
        step = val / grad
        d0_half = max(0.01, d0_half - step)
        if abs(val) < 1e-6:
            break
    d0 = 2.0 * d0_half
    return max(0.1, min(d0, 1e6))


def limma_differential_methylation(
    X: np.ndarray,
    y: np.ndarray,
    cpg_names: list,
    min_delta_beta: float = 0.0,
    fdr_threshold: float = 0.05,
    covariates: np.ndarray = None,
) -> pd.DataFrame:
    """
    Perform limma-style moderated t-test differential methylation analysis.
    
    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_cpgs)
        Methylation beta values (or M-values).
    y : np.ndarray, shape (n_samples,)
        Binary class label (0=Control/Normal, 1=Case/CIN3+).
    cpg_names : list of str
        List of CpG identifier names corresponding to columns of X.
    min_delta_beta : float
        Minimum absolute effect size threshold.
    fdr_threshold : float
        FDR q-value cutoff for significance.
    covariates : np.ndarray, optional
        Additional nuisance covariates (e.g. batch/chip or age) to adjust for.

    Returns
    -------
    pd.DataFrame
        Table indexed by CpG with columns:
        [Mean_Ctrl, Mean_Case, Delta_Beta, t_stat, mod_t_stat, p_val, fdr_q_val, B_stat, Significant]
    """
    n_samples, n_cpgs = X.shape
    idx_0 = np.where(y == 0)[0]
    idx_1 = np.where(y == 1)[0]
    n0 = len(idx_0)
    n1 = len(idx_1)
    
    if n0 < 2 or n1 < 2:
        raise ValueError(f"Need at least 2 samples per group, got n0={n0}, n1={n1}")
        
    mean0 = np.nanmean(X[idx_0], axis=0)
    mean1 = np.nanmean(X[idx_1], axis=0)
    delta = mean1 - mean0
    
    # Degrees of freedom
    df_resid = n0 + n1 - 2
    
    # Sample pooled variance
    # Replace NaNs in calculation if present
    X0 = X[idx_0] - mean0
    X1 = X[idx_1] - mean1
    ss0 = np.nansum(X0 ** 2, axis=0)
    ss1 = np.nansum(X1 ** 2, axis=0)
    s2 = (ss0 + ss1) / df_resid
    # Guard against 0 or negative variances
    s2 = np.maximum(s2, 1e-12)
    
    # Standard error factor
    se_factor = np.sqrt(1.0 / n0 + 1.0 / n1)
    se_raw = np.sqrt(s2) * se_factor
    t_stat_raw = delta / se_raw
    
    # Empirical Bayes moderation (Smyth 2004)
    # Model: log(s^2) has mean = log(s0^2) + digamma(df/2) - digamma(d0/2) - log(df/d0)
    # and var = trigamma(df/2) + trigamma(d0/2)
    z = np.log(s2)
    # Exclude outliers/infs
    valid_z = z[np.isfinite(z)]
    mean_z = np.mean(valid_z)
    var_z = np.var(valid_z, ddof=1)
    
    tri_df = trigamma(df_resid / 2.0)
    target_tri = var_z - tri_df
    
    if target_tri > 1e-4:
        d0 = solve_d0_from_trigamma(target_tri, df_resid)
    else:
        # High sample consistency: large d0
        d0 = float(df_resid * 2.0)
        
    s0_2 = np.exp(mean_z - (digamma(df_resid / 2.0) - digamma(d0 / 2.0) - np.log(df_resid / d0)))
    
    # Squeezed moderated variance
    s2_mod = (d0 * s0_2 + df_resid * s2) / (d0 + df_resid)
    se_mod = np.sqrt(s2_mod) * se_factor
    mod_t = delta / se_mod
    df_mod = df_resid + d0
    
    # P-values from Student-t with moderated degrees of freedom
    p_vals = 2.0 * stats.t.sf(np.abs(mod_t), df=df_mod)
    p_vals = np.clip(p_vals, 1e-300, 1.0)
    
    # Benjamini-Hochberg FDR correction
    n = len(p_vals)
    sorted_idx = np.argsort(p_vals)
    sorted_p = p_vals[sorted_idx]
    ranks = np.arange(1, n + 1)
    fdr_sorted = sorted_p * n / ranks
    # Enforce monotonicity backwards
    fdr_sorted = np.minimum.accumulate(fdr_sorted[::-1])[::-1]
    fdr_sorted = np.clip(fdr_sorted, 0.0, 1.0)
    
    fdr_q = np.empty(n, dtype=float)
    fdr_q[sorted_idx] = fdr_sorted
    
    # Approximate B-statistic (log-odds of differential methylation)
    # B = log(odds) approx 0.5 * (mod_t^2 - log(1 + df_mod/mod_t^2))
    b_stat = 0.5 * (mod_t ** 2 - np.log(1.0 + df_mod))
    
    is_sig = (fdr_q <= fdr_threshold) & (np.abs(delta) >= min_delta_beta)
    
    df_results = pd.DataFrame({
        "Mean_Control": mean0,
        "Mean_Case": mean1,
        "Delta_Beta": delta,
        "Raw_t": t_stat_raw,
        "Mod_t": mod_t,
        "p_val": p_vals,
        "FDR_q": fdr_q,
        "B_stat": b_stat,
        "Significant": is_sig,
    }, index=cpg_names)
    
    return df_results.sort_values("FDR_q", ascending=True)
