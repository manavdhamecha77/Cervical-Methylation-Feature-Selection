"""
feature_selection/run_nested_cv_benchmark.py
--------------------------------------------
Master benchmarking module implementing Steps 3-6 of the plan:
1. Restricts analysis to intended population: hrHPV-positive women, CIN3+ outcome
2. Evaluates Elastic Net, Linear SVM, XGBoost, and LightGBM using nested cross-validation
3. Selects stable 5/10/20/50-CpG panels using training-fold-only selection (Limma DMP and RFE stability)
4. Applies in-fold ComBat batch adjustment (strictly zero leakage)
5. Applies calibration and optimal triage thresholding
6. Evaluates on completely untouched external cohort(s)
7. Reports ROC-AUC, Sensitivity, Specificity, Balanced Accuracy, and 95% CIs
"""

import os
import sys
import json
import warnings
from typing import Tuple, List, Dict
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "feature_selection"))

from limma_dmp import limma_differential_methylation
from rfe_stability import rfe_bootstrap_stability_selection
from combat_correction import ComBatBatchCorrector
from calibration_triage import (
    find_optimal_threshold, evaluate_with_threshold_and_ci,
    wilson_score_interval, auc_bootstrap_ci
)
from multi_cohort_loader import get_harmonized_cohorts
from annotator import load_manifest

warnings.filterwarnings("ignore")
OUT_DIR = os.path.join(ROOT, "results", "feature_selection_benchmark")
PANELS_DIR = os.path.join(OUT_DIR, "panels")
os.makedirs(PANELS_DIR, exist_ok=True)


def get_base_classifier(model_name: str, seed: int = 42):
    """Factory for the 4 compared models."""
    if model_name == "elastic_net":
        return LogisticRegression(
            penalty="elasticnet", solver="saga", l1_ratio=0.5, C=0.2,
            max_iter=1000, random_state=seed
        )
    elif model_name == "linear_svm":
        base_svm = LinearSVC(C=0.1, max_iter=2000, random_state=seed)
        return CalibratedClassifierCV(base_svm, cv=3)
    elif model_name == "xgboost":
        return xgb.XGBClassifier(
            n_estimators=60, max_depth=2, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            random_state=seed, verbosity=0
        )
    elif model_name == "lightgbm":
        return lgb.LGBMClassifier(
            n_estimators=60, max_depth=2, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=5,
            random_state=seed, verbose=-1
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def run_nested_cv_experiment(
    X: np.ndarray,
    y: np.ndarray,
    cohort: np.ndarray,
    cpg_names: list,
    panel_sizes: list = [5, 10, 20, 50],
    n_splits: int = 5,
    random_seed: int = 42,
    verbose: bool = True,
) -> Tuple:
    """
    Nested cross-validation over the development cohort:
    - In-fold Limma DMP selection & RFE stability selection
    - In-fold ComBat batch adjustment across cohorts
    - Comparison of Elastic Net, Linear SVM, XGBoost, and LightGBM
    - Optimal thresholding on training fold (Youden's J)
    """
    models = ["elastic_net", "linear_svm", "xgboost", "lightgbm"]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    records = []
    fold_selected_panels = {k: [] for k in panel_sizes}

    if verbose:
        print(f"\n[Nested CV] Running {n_splits}-fold CV over {len(y)} hrHPV+ samples...")
        print(f"  Cases (CIN3+): {sum(y==1)}, Controls: {sum(y==0)}")
        print(f"  Comparing models: {models}")
        print(f"  Panel sizes: {panel_sizes}")

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        if verbose:
            print(f"\n--- FOLD {fold_idx + 1}/{n_splits} ---")
        X_tr, y_tr, c_tr = X[tr_idx], y[tr_idx], cohort[tr_idx]
        X_va, y_va, c_va = X[val_idx], y[val_idx], cohort[val_idx]

        # 1. In-fold batch adjustment using ComBat (Zero leakage!)
        unique_tr_batches = np.unique(c_tr)
        if len(unique_tr_batches) > 1:
            combat = ComBatBatchCorrector(eb=True)
            X_tr_adj = combat.fit_transform(X_tr, batch=c_tr, y=y_tr)
            X_va_adj = combat.transform(X_va, batch=c_va)
        else:
            X_tr_adj = X_tr.copy()
            X_va_adj = X_va.copy()

        # 2. In-fold Limma Differential Methylation Analysis
        dmp_df = limma_differential_methylation(
            X=X_tr_adj, y=y_tr, cpg_names=cpg_names, min_delta_beta=0.0
        )
        ranked_cpgs = list(dmp_df.index)

        # 3. In-fold RFE Stability Selection to refine robust panels
        rfe_res = rfe_bootstrap_stability_selection(
            X=X_tr_adj, y=y_tr, cpg_names=cpg_names,
            target_k_list=panel_sizes, n_bootstraps=30,  # 30 per fold for inner speed
            prefilter_top_m=300, random_seed=random_seed + fold_idx, verbose=False
        )

        for k in panel_sizes:
            # We compare Limma top-k and RFE consensus top-k
            k_cpgs_limma = ranked_cpgs[:k]
            k_cpgs_rfe = rfe_res["consensus_panels"][k]
            fold_selected_panels[k].append(k_cpgs_rfe)

            for sel_name, chosen_cpgs in [("Limma_eBayes", k_cpgs_limma), ("RFE_Stability", k_cpgs_rfe)]:
                cpg_to_idx = {name: i for i, name in enumerate(cpg_names)}
                feat_indices = [cpg_to_idx[p] for p in chosen_cpgs]

                X_tr_k = X_tr_adj[:, feat_indices]
                X_va_k = X_va_adj[:, feat_indices]

                # Standardize in-fold
                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr_k)
                X_va_s = scaler.transform(X_va_k)

                for m_name in models:
                    clf = get_base_classifier(m_name, seed=random_seed + fold_idx)
                    clf.fit(X_tr_s, y_tr)

                    # In-fold predicted probabilities for training threshold tuning
                    if hasattr(clf, "predict_proba"):
                        p_tr = clf.predict_proba(X_tr_s)[:, 1]
                        p_va = clf.predict_proba(X_va_s)[:, 1]
                    else:
                        df_tr = clf.decision_function(X_tr_s)
                        df_va = clf.decision_function(X_va_s)
                        p_tr = 1.0 / (1.0 + np.exp(-df_tr))
                        p_va = 1.0 / (1.0 + np.exp(-df_va))

                    # Find optimal triage threshold on training fold ONLY
                    opt_thresh = find_optimal_threshold(y_tr, p_tr, strategy="youden")
                    
                    # Evaluate on held-out validation fold
                    metrics = evaluate_with_threshold_and_ci(
                        y_va, p_va, threshold=opt_thresh, calc_ci=False
                    )

                    rec = {
                        "Fold": fold_idx + 1,
                        "Selector": sel_name,
                        "Panel_Size_k": k,
                        "Model": m_name,
                        "ROC_AUC": metrics["ROC_AUC"],
                        "Threshold": metrics["Threshold"],
                        "Sensitivity": metrics["Sensitivity"],
                        "Specificity": metrics["Specificity"],
                        "Balanced_Accuracy": metrics["Balanced_Accuracy"],
                        "Accuracy": metrics["Accuracy"],
                    }
                    records.append(rec)

    df_results = pd.DataFrame(records)

    # Summary table with mean +/- std and 95% CIs
    summary_rows = []
    grouped = df_results.groupby(["Selector", "Panel_Size_k", "Model"])
    for (sel, k, model), grp in grouped:
        auc_mean = grp["ROC_AUC"].mean()
        auc_std = grp["ROC_AUC"].std()
        sens_mean = grp["Sensitivity"].mean()
        sens_std = grp["Sensitivity"].std()
        spec_mean = grp["Specificity"].mean()
        spec_std = grp["Specificity"].std()
        bacc_mean = grp["Balanced_Accuracy"].mean()
        bacc_std = grp["Balanced_Accuracy"].std()
        th_mean = grp["Threshold"].mean()

        # 95% CI on mean using normal approximation for CV folds
        ci_factor = 1.96 / np.sqrt(len(grp))
        auc_ci = (max(0.0, auc_mean - ci_factor * auc_std), min(1.0, auc_mean + ci_factor * auc_std))
        bacc_ci = (max(0.0, bacc_mean - ci_factor * bacc_std), min(1.0, bacc_mean + ci_factor * bacc_std))

        summary_rows.append({
            "Selector": sel,
            "Panel_Size_k": k,
            "Model": model,
            "ROC_AUC_Mean": round(auc_mean, 4),
            "ROC_AUC_Std": round(auc_std, 4),
            "ROC_AUC_95CI": f"[{auc_ci[0]:.4f}, {auc_ci[1]:.4f}]",
            "Optimal_Threshold_Mean": round(th_mean, 4),
            "Sensitivity_Mean": round(sens_mean, 4),
            "Sensitivity_Std": round(sens_std, 4),
            "Specificity_Mean": round(spec_mean, 4),
            "Specificity_Std": round(spec_std, 4),
            "Balanced_Accuracy_Mean": round(bacc_mean, 4),
            "Balanced_Accuracy_Std": round(bacc_std, 4),
            "Balanced_Accuracy_95CI": f"[{bacc_ci[0]:.4f}, {bacc_ci[1]:.4f}]",
        })

    df_summary = pd.DataFrame(summary_rows).sort_values("ROC_AUC_Mean", ascending=False)
    return df_results, df_summary, fold_selected_panels


def evaluate_untouched_external_cohort(
    dev_cohort: dict,
    ext_cohort: dict,
    consensus_panels: dict,
    models: list = ["elastic_net", "linear_svm", "xgboost", "lightgbm"],
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Evaluates models trained on the development cohort on completely untouched external data.
    Strict single-shot evaluation:
    - Zero tuning on external data
    - Decision threshold tuned on development cohort
    - Reports ROC-AUC, Sensitivity, Specificity, Balanced Accuracy with 95% CIs
    """
    if verbose:
        print("\n" + "=" * 75)
        print("EVALUATING ON COMPLETELY UNTOUCHED EXTERNAL COHORT")
        print("=" * 75)
        print(f"  Dev cohort: {len(dev_cohort['y'])} samples (Cases: {sum(dev_cohort['y']==1)}, Controls: {sum(dev_cohort['y']==0)})")
        print(f"  External cohort: {len(ext_cohort['y'])} samples (Cases: {sum(ext_cohort['y']==1)}, Controls: {sum(ext_cohort['y']==0)})")

    cpg_to_idx_dev = {name: i for i, name in enumerate(dev_cohort["cpg_names"])}
    cpg_to_idx_ext = {name: i for i, name in enumerate(ext_cohort["cpg_names"])}

    records = []

    for k, cpg_list in consensus_panels.items():
        dev_indices = [cpg_to_idx_dev[p] for p in cpg_list if p in cpg_to_idx_dev]
        ext_indices = [cpg_to_idx_ext[p] for p in cpg_list if p in cpg_to_idx_ext]

        if len(dev_indices) != len(cpg_list) or len(ext_indices) != len(cpg_list):
            continue

        X_dev_k = dev_cohort["X"][:, dev_indices]
        y_dev = dev_cohort["y"]
        X_ext_k = ext_cohort["X"][:, ext_indices]
        y_ext = ext_cohort["y"]

        scaler = StandardScaler()
        X_dev_s = scaler.fit_transform(X_dev_k)
        X_ext_s = scaler.transform(X_ext_k)

        for m_name in models:
            clf = get_base_classifier(m_name, seed=42)
            clf.fit(X_dev_s, y_dev)

            # In-fold predicted probabilities on dev set for threshold tuning
            if hasattr(clf, "predict_proba"):
                p_dev = clf.predict_proba(X_dev_s)[:, 1]
                p_ext = clf.predict_proba(X_ext_s)[:, 1]
            else:
                df_dev = clf.decision_function(X_dev_s)
                df_ext = clf.decision_function(X_ext_s)
                p_dev = 1.0 / (1.0 + np.exp(-df_dev))
                p_ext = 1.0 / (1.0 + np.exp(-df_ext))

            opt_th = find_optimal_threshold(y_dev, p_dev, strategy="youden")
            metrics = evaluate_with_threshold_and_ci(y_ext, p_ext, threshold=opt_th, calc_ci=True)

            rec = {
                "External_Cohort": ext_cohort.get("cohort", ["External"])[0] if "cohort" in ext_cohort else "External",
                "Panel_Size_k": k,
                "Model": m_name,
                "ROC_AUC": metrics["ROC_AUC"],
                "ROC_AUC_95CI": metrics["ROC_AUC_95CI"],
                "Optimal_Threshold": metrics["Threshold"],
                "Sensitivity": metrics["Sensitivity"],
                "Sensitivity_95CI": metrics["Sensitivity_95CI"],
                "Specificity": metrics["Specificity"],
                "Specificity_95CI": metrics["Specificity_95CI"],
                "Balanced_Accuracy": metrics["Balanced_Accuracy"],
                "Balanced_Accuracy_95CI": metrics["Balanced_Accuracy_95CI"],
                "Accuracy": metrics["Accuracy"],
                "TP": metrics["TP"], "TN": metrics["TN"],
                "FP": metrics["FP"], "FN": metrics["FN"],
            }
            records.append(rec)

    df_ext = pd.DataFrame(records).sort_values("ROC_AUC", ascending=False)
    if verbose:
        print("\n--- EXTERNAL VALIDATION RESULTS ---")
        print(df_ext[["External_Cohort", "Panel_Size_k", "Model", "ROC_AUC", "ROC_AUC_95CI", "Sensitivity", "Specificity", "Balanced_Accuracy"]])

    out_csv = os.path.join(OUT_DIR, "untouched_external_validation_ritweek2.csv")
    df_ext.to_csv(out_csv, index=False)
    print(f"\n[External] Saved external validation results to {out_csv}")
    return df_ext


def run_sensor_stress_on_consensus(
    dev_cohort: dict,
    consensus_cpgs: list,
    model_name: str = "elastic_net",
    tau_kinetics: float = 5.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Stress-tests the final consensus biomarker panel under measured assay noise and kinetics:
    1. Cellular dilution: 10%, 25%, 50%, 100% CIN3 fraction
    2. Langmuir binding kinetics: 5, 10, 15, 30, 45, 60 minutes
    3. Assay measurement noise: sigma in [0.01, 0.03, 0.05, 0.10]
    """
    if verbose:
        print("\n" + "=" * 75)
        print("TESTING CONSENSUS PANEL UNDER MEASURED ASSAY NOISE & DETECTION LIMITS")
        print("=" * 75)

    cpg_to_idx = {name: i for i, name in enumerate(dev_cohort["cpg_names"])}
    feat_idx = [cpg_to_idx[p] for p in consensus_cpgs if p in cpg_to_idx]

    X = dev_cohort["X"][:, feat_idx]
    y = dev_cohort["y"]

    # Baseline normal background profile
    normal_bg = np.median(X[y == 0], axis=0)

    # Train reference model
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    clf = get_base_classifier(model_name, seed=42)
    clf.fit(X_s, y)
    p_ref = clf.predict_proba(X_s)[:, 1] if hasattr(clf, "predict_proba") else 1.0 / (1.0 + np.exp(-clf.decision_function(X_s)))
    th_ref = find_optimal_threshold(y, p_ref, strategy="youden")

    dilution_fractions = [0.10, 0.25, 0.50, 0.75, 1.00]
    times_min = [5.0, 10.0, 15.0, 30.0, 45.0, 60.0]
    noise_sigmas = [0.01, 0.03, 0.05, 0.10]

    records = []
    rng = np.random.RandomState(42)

    for dil in dilution_fractions:
        for t in times_min:
            theta_bind = 1.0 - np.exp(-t / tau_kinetics)
            for sig in noise_sigmas:
                # Apply dilution to CIN3 cases
                X_perturbed = X.copy()
                cin3_mask = (y == 1)
                X_perturbed[cin3_mask] = dil * X[cin3_mask] + (1.0 - dil) * normal_bg

                # Apply Langmuir signal reduction
                X_perturbed = X_perturbed * theta_bind

                # Add assay noise
                noise = rng.normal(0.0, sig, size=X_perturbed.shape)
                X_perturbed = np.clip(X_perturbed + noise, 0.0, 1.0)

                # Evaluate with reference model
                X_pert_s = scaler.transform(X_perturbed)
                p_pert = clf.predict_proba(X_pert_s)[:, 1] if hasattr(clf, "predict_proba") else 1.0 / (1.0 + np.exp(-clf.decision_function(X_pert_s)))
                m = evaluate_with_threshold_and_ci(y, p_pert, threshold=th_ref, calc_ci=False)

                records.append({
                    "Dilution_Fraction": dil,
                    "Incubation_Time_Min": t,
                    "Binding_Fraction_Theta": round(theta_bind, 3),
                    "Assay_Noise_Sigma": sig,
                    "ROC_AUC": m["ROC_AUC"],
                    "Sensitivity": m["Sensitivity"],
                    "Specificity": m["Specificity"],
                    "Balanced_Accuracy": m["Balanced_Accuracy"],
                })

    df_stress = pd.DataFrame(records)
    out_csv = os.path.join(OUT_DIR, "sensor_combined_stress_ritweek2.csv")
    df_stress.to_csv(out_csv, index=False)
    if verbose:
        print(f"[Sensor] Saved sensor stress grid to {out_csv}")
        # Preview at 30 min incubation, 0.03 noise
        pview = df_stress[(df_stress["Incubation_Time_Min"] == 30.0) & (df_stress["Assay_Noise_Sigma"] == 0.03)]
        print("\nSensor performance across Pap cellular dilutions (t=30min, sigma=0.03):")
        print(pview[["Dilution_Fraction", "ROC_AUC", "Sensitivity", "Specificity", "Balanced_Accuracy"]])

    return df_stress
