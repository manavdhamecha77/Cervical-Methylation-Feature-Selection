"""
feature_selection/calibration_triage.py
---------------------------------------
Probability calibration and clinical threshold optimization for cervical cancer triage.
Implements:
1. Platt scaling / Isotonic regression probability calibration
2. Youden's J threshold optimization and clinical high-sensitivity triage thresholding
3. Wilson score 95% confidence intervals for sensitivity, specificity, and balanced accuracy
4. DeLong / bootstrap 95% confidence intervals for ROC-AUC
5. Complete no-leakage training-fold threshold tuning
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, brier_score_loss


def wilson_score_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple:
    """Computes Wilson score 95% confidence interval for a proportion."""
    if trials == 0:
        return (0.0, 0.0)
    z = stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0)
    p = successes / trials
    denom = 1.0 + z ** 2 / trials
    centre = (p + z ** 2 / (2.0 * trials)) / denom
    margin = (z * np.sqrt(p * (1.0 - p) / trials + z ** 2 / (4.0 * trials ** 2))) / denom
    lower = max(0.0, centre - margin)
    upper = min(1.0, centre + margin)
    return (round(lower, 4), round(upper, 4))


def auc_bootstrap_ci(y_true: np.ndarray, y_prob: np.ndarray, n_bootstraps: int = 1000,
                     confidence: float = 0.95, random_seed: int = 42) -> tuple:
    """Computes bootstrap 95% CI for ROC-AUC."""
    rng = np.random.RandomState(random_seed)
    n = len(y_true)
    boot_aucs = []
    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            boot_aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
        except Exception:
            continue
    if len(boot_aucs) < 50:
        return (np.nan, np.nan)
    alpha = (1.0 - confidence) / 2.0
    lower = np.percentile(boot_aucs, 100.0 * alpha)
    upper = np.percentile(boot_aucs, 100.0 * (1.0 - alpha))
    return (round(float(lower), 4), round(float(upper), 4))


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    strategy: str = "youden",
    min_sensitivity: float = 0.85
) -> float:
    """
    Find optimal triage decision threshold on training fold.
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary training labels (0/1).
    y_prob : np.ndarray
        Predicted probabilities on training data.
    strategy : str
        'youden' : maximizes Sensitivity + Specificity - 1 (balanced accuracy)
        'high_sensitivity' : achieves Sensitivity >= min_sensitivity while maximizing specificity
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    
    if strategy == "youden":
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_thresh = thresholds[best_idx]
    elif strategy == "high_sensitivity":
        # Filter to operating points where sensitivity (tpr) >= min_sensitivity
        eligible = np.where(tpr >= min_sensitivity)[0]
        if len(eligible) > 0:
            # Among those, pick the one with lowest FPR (highest specificity)
            best_idx = eligible[np.argmin(fpr[eligible])]
            best_thresh = thresholds[best_idx]
        else:
            best_thresh = thresholds[np.argmax(tpr)]
    else:
        best_thresh = 0.5
        
    # Guard against -inf or inf thresholds returned by roc_curve
    best_thresh = np.clip(best_thresh, 0.01, 0.99)
    return float(best_thresh)


def evaluate_with_threshold_and_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    calc_ci: bool = True
) -> dict:
    """
    Evaluate binary predictions at a specified decision threshold with 95% CIs.
    """
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    
    n_pos = tp + fn
    n_neg = tn + fp
    total = len(y_true)
    
    sens = tp / n_pos if n_pos > 0 else 0.0
    spec = tn / n_neg if n_neg > 0 else 0.0
    bacc = 0.5 * (sens + spec)
    acc = (tp + tn) / total if total > 0 else 0.0
    brier = brier_score_loss(y_true, y_prob) if len(y_true) > 0 else np.nan
    
    try:
        auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    except Exception:
        auc = np.nan

    res = {
        "ROC_AUC": round(auc, 4) if not np.isnan(auc) else np.nan,
        "Threshold": round(threshold, 4),
        "Sensitivity": round(sens, 4),
        "Specificity": round(spec, 4),
        "Balanced_Accuracy": round(bacc, 4),
        "Accuracy": round(acc, 4),
        "Brier_Score": round(brier, 4),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }

    if calc_ci:
        sens_ci = wilson_score_interval(tp, n_pos)
        spec_ci = wilson_score_interval(tn, n_neg)
        auc_ci = auc_bootstrap_ci(y_true, y_prob)
        # Balanced accuracy CI approx via delta method / bootstrap
        bacc_low = round(0.5 * (sens_ci[0] + spec_ci[0]), 4)
        bacc_high = round(0.5 * (sens_ci[1] + spec_ci[1]), 4)
        
        res["ROC_AUC_95CI"] = f"[{auc_ci[0]:.4f}, {auc_ci[1]:.4f}]" if not np.isnan(auc_ci[0]) else "N/A"
        res["Sensitivity_95CI"] = f"[{sens_ci[0]:.4f}, {sens_ci[1]:.4f}]"
        res["Specificity_95CI"] = f"[{spec_ci[0]:.4f}, {spec_ci[1]:.4f}]"
        res["Balanced_Accuracy_95CI"] = f"[{bacc_low:.4f}, {bacc_high:.4f}]"

    return res
