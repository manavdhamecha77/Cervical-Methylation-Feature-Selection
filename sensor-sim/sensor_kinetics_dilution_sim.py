"""
sensor_kinetics_dilution_sim.py
-------------------------------
Implements Section 5:
1. Diffusion-limited transport + Langmuir binding kinetics:
   Signal vs incubation time t, SNR degradation at short incubation.
2. Pap-style cellular dilution experiment:
   CIN3 profile diluted with Normal background at 10%, 25%, 50%, 100% CIN3 fraction.
3. Combined stress test:
   Dilution fraction x Sensor Noise x Incubation Time.
4. Outputs:
   - accuracy vs incubation time (csv & plot)
   - accuracy vs dilution (csv & plot)
   - combined heatmap table and visual
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FS_DIR = os.path.join(ROOT, "feature_selection")
sys.path.insert(0, FS_DIR)

from data_loader import load_primary_data, impute_fold, variance_prefilter
from evaluator import compute_metrics, _get_classifier_proba

warnings.filterwarnings("ignore")
CONFIG_PATH = os.path.join(FS_DIR, "config.json")
OUT_DIR = os.path.join(ROOT, "results", "feature_selection_benchmark")
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def langmuir_binding_fraction(t_minutes: float, tau: float = 5.0) -> float:
    """
    Fractional surface coverage theta(t) of biosensor under Langmuir binding / diffusion limit:
    theta(t) = 1.0 - exp(-t / tau).
    """
    if t_minutes <= 0:
        return 0.01
    return float(1.0 - np.exp(-t_minutes / tau))


def run_sensor_experiments(verbose: bool = True):
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    data = load_primary_data(config, verbose=verbose)
    X = data["X"]
    y = data["y"]
    cpg_names = data["cpg_names"]

    # Pre-filter top 2000 features by variance on full X for speed and stability
    X_imp, _, _ = impute_fold(X, X)
    var_all = np.nanvar(X_imp, axis=0)
    top_2000_idx = np.argsort(var_all)[::-1][:2000]
    X_sub = X_imp[:, top_2000_idx]
    cpg_sub = [cpg_names[i] for i in top_2000_idx]

    # Convert to M-values
    eps = 1e-5
    X_m = np.log2(np.clip(X_sub, eps, 1.0 - eps) / (1.0 - np.clip(X_sub, eps, 1.0 - eps)))

    # Select top 50 markers using MW filter on full cohort
    p_vals = []
    for col in range(X_m.shape[1]):
        try:
            _, p = stats.mannwhitneyu(X_m[y == 0, col], X_m[y == 1, col])
        except Exception:
            p = 1.0
        p_vals.append(p)
    top_50_idx = np.argsort(p_vals)[:50]
    X_50 = X_m[:, top_50_idx]
    
    # Baseline normal reference profile (for dilution)
    normal_profile = np.mean(X_50[y == 0], axis=0)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # -------------------------------------------------------------
    # 1. Diffusion / Kinetics: Signal vs Incubation Time
    # -------------------------------------------------------------
    if verbose:
        print("\n" + "="*70)
        print("SIMULATION 1: Diffusion-Limited Transport & Langmuir Kinetics")
        print("="*70)

    incubation_times = [1, 2, 3, 5, 7.5, 10, 15, 20, 30]
    kinetics_rows = []

    for t in incubation_times:
        theta = langmuir_binding_fraction(t, tau=5.0)
        # Transient noise decreases as signal reaches saturation
        noise_sd = 0.02 + 0.04 * (1.0 - theta)
        
        fold_bals = []
        fold_aucs = []
        fold_sens = []
        fold_spec = []

        for tr, va in skf.split(X_50, y):
            X_tr, X_va = X_50[tr], X_50[va]
            y_tr, y_va = y[tr], y[va]

            scaler = StandardScaler().fit(X_tr)
            X_tr_sc = scaler.transform(X_tr)

            # Apply kinetics attenuation and noise to held-out test fold only
            rng = np.random.RandomState(42 + int(t * 10))
            X_va_kin = X_va * theta + rng.normal(0, noise_sd, size=X_va.shape)
            X_va_sc = scaler.transform(X_va_kin)

            clf = LogisticRegression(C=1.0, solver="liblinear", random_state=42)
            clf.fit(X_tr_sc, y_tr)
            y_pred = clf.predict(X_va_sc)
            y_prob = _get_classifier_proba(clf, X_va_sc)

            m = compute_metrics(y_va, y_pred, y_prob)
            fold_bals.append(m["Balanced_Accuracy"])
            fold_aucs.append(m["ROC_AUC"])
            fold_sens.append(m["Sensitivity"])
            fold_spec.append(m["Specificity"])

        kinetics_rows.append({
            "Incubation_Time_Min": t,
            "Binding_Saturation_Theta": round(theta, 4),
            "Effective_Noise_SD": round(noise_sd, 4),
            "Balanced_Accuracy_Mean": round(float(np.mean(fold_bals)), 4),
            "Balanced_Accuracy_Std": round(float(np.std(fold_bals)), 4),
            "ROC_AUC_Mean": round(float(np.mean(fold_aucs)), 4),
            "Sensitivity_Mean": round(float(np.mean(fold_sens)), 4),
            "Specificity_Mean": round(float(np.mean(fold_spec)), 4),
        })

    df_kin = pd.DataFrame(kinetics_rows)
    kin_csv = os.path.join(OUT_DIR, "sensor_incubation_kinetics.csv")
    df_kin.to_csv(kin_csv, index=False)
    if verbose:
        print(df_kin[["Incubation_Time_Min", "Binding_Saturation_Theta", "Balanced_Accuracy_Mean", "ROC_AUC_Mean"]].to_string(index=False))

    # -------------------------------------------------------------
    # 2. Pap-Style Cellular Dilution Experiment
    # -------------------------------------------------------------
    if verbose:
        print("\n" + "="*70)
        print("SIMULATION 2: Pap-Style Precancer Cellular Dilution (10% to 100%)")
        print("="*70)

    dilution_fractions = [0.10, 0.25, 0.50, 0.75, 1.00]
    dilution_rows = []

    for alpha in dilution_fractions:
        fold_bals = []
        fold_aucs = []
        fold_sens = []
        fold_spec = []

        for tr, va in skf.split(X_50, y):
            X_tr, X_va = X_50[tr], X_50[va]
            y_tr, y_va = y[tr], y[va]

            scaler = StandardScaler().fit(X_tr)
            X_tr_sc = scaler.transform(X_tr)

            # Dilute only CIN3 (y=1) test samples with normal background
            X_va_dil = X_va.copy()
            cin3_test_idx = np.where(y_va == 1)[0]
            X_va_dil[cin3_test_idx] = alpha * X_va[cin3_test_idx] + (1.0 - alpha) * normal_profile

            X_va_sc = scaler.transform(X_va_dil)

            clf = LogisticRegression(C=1.0, solver="liblinear", random_state=42)
            clf.fit(X_tr_sc, y_tr)
            y_pred = clf.predict(X_va_sc)
            y_prob = _get_classifier_proba(clf, X_va_sc)

            m = compute_metrics(y_va, y_pred, y_prob)
            fold_bals.append(m["Balanced_Accuracy"])
            fold_aucs.append(m["ROC_AUC"])
            fold_sens.append(m["Sensitivity"])
            fold_spec.append(m["Specificity"])

        dilution_rows.append({
            "CIN3_Dilution_Fraction": alpha,
            "Balanced_Accuracy_Mean": round(float(np.mean(fold_bals)), 4),
            "Balanced_Accuracy_Std": round(float(np.std(fold_bals)), 4),
            "ROC_AUC_Mean": round(float(np.mean(fold_aucs)), 4),
            "Sensitivity_Mean": round(float(np.mean(fold_sens)), 4),
            "Specificity_Mean": round(float(np.mean(fold_spec)), 4),
        })

    df_dil = pd.DataFrame(dilution_rows)
    dil_csv = os.path.join(OUT_DIR, "sensor_dilution_experiment.csv")
    df_dil.to_csv(dil_csv, index=False)
    if verbose:
        print(df_dil.to_string(index=False))

    # -------------------------------------------------------------
    # 3. Combined Stress Test (Heatmap: Dilution x Noise x Time)
    # -------------------------------------------------------------
    if verbose:
        print("\n" + "="*70)
        print("SIMULATION 3: Combined Stress Factorial Matrix")
        print("="*70)

    noise_levels = [0.00, 0.02, 0.05, 0.10]
    time_levels = [2.0, 5.0, 15.0]
    dil_levels = [0.10, 0.25, 0.50, 1.00]

    grid_rows = []
    heatmap_matrix = np.zeros((len(dil_levels), len(noise_levels)))

    for i, dil in enumerate(dil_levels):
        for j, noise in enumerate(noise_levels):
            # Evaluate at standard 10 min incubation
            theta_10 = langmuir_binding_fraction(10.0, tau=5.0)
            
            fold_bals = []
            for tr, va in skf.split(X_50, y):
                X_tr, X_va = X_50[tr], X_50[va]
                y_tr, y_va = y[tr], y[va]

                scaler = StandardScaler().fit(X_tr)
                X_tr_sc = scaler.transform(X_tr)

                X_va_stress = X_va.copy()
                cin3_idx = np.where(y_va == 1)[0]
                X_va_stress[cin3_idx] = dil * X_va[cin3_idx] + (1.0 - dil) * normal_profile
                
                # Kinetics scaling + noise
                rng = np.random.RandomState(int(dil * 100 + noise * 1000))
                X_va_stress = X_va_stress * theta_10 + rng.normal(0, noise, size=X_va_stress.shape)
                X_va_sc = scaler.transform(X_va_stress)

                clf = LogisticRegression(C=1.0, solver="liblinear", random_state=42)
                clf.fit(X_tr_sc, y_tr)
                y_pred = clf.predict(X_va_sc)
                y_prob = _get_classifier_proba(clf, X_va_sc)

                m = compute_metrics(y_va, y_pred, y_prob)
                fold_bals.append(m["Balanced_Accuracy"])

            mean_bal = float(np.mean(fold_bals))
            heatmap_matrix[i, j] = mean_bal
            grid_rows.append({
                "CIN3_Fraction": dil,
                "Noise_SD": noise,
                "Incubation_Time_Min": 10.0,
                "Mean_Balanced_Accuracy": round(mean_bal, 4)
            })

    df_grid = pd.DataFrame(grid_rows)
    grid_csv = os.path.join(OUT_DIR, "sensor_combined_stress_grid.csv")
    df_grid.to_csv(grid_csv, index=False)

    # -------------------------------------------------------------
    # 4. Generate Visual Plots
    # -------------------------------------------------------------
    # Plot 1: Accuracy vs Incubation Time
    plt.figure(figsize=(7, 4.5))
    plt.plot(df_kin["Incubation_Time_Min"], df_kin["Balanced_Accuracy_Mean"], "o-", color="#1f77b4", label="Balanced Accuracy")
    plt.plot(df_kin["Incubation_Time_Min"], df_kin["ROC_AUC_Mean"], "s--", color="#ff7f0e", label="ROC-AUC")
    plt.axvline(x=5.0, color="gray", linestyle=":", label="Time Constant tau (5 min)")
    plt.xlabel("Incubation Time (minutes)", fontsize=11)
    plt.ylabel("Cross-Validation Score", fontsize=11)
    plt.title("Sensor Simulation: Model Performance vs Incubation Time", fontsize=12, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True)
    plt.tight_layout()
    plot_kin_path = os.path.join(PLOTS_DIR, "sensor_incubation_time.png")
    plt.savefig(plot_kin_path, dpi=150)
    plt.close()

    # Plot 2: Accuracy vs Dilution
    plt.figure(figsize=(7, 4.5))
    plt.plot(df_dil["CIN3_Dilution_Fraction"] * 100, df_dil["Balanced_Accuracy_Mean"], "o-", color="#2ca02c", label="Balanced Accuracy")
    plt.plot(df_dil["CIN3_Dilution_Fraction"] * 100, df_dil["Sensitivity_Mean"], "^--", color="#d62728", label="Sensitivity (Recall)")
    plt.plot(df_dil["CIN3_Dilution_Fraction"] * 100, df_dil["Specificity_Mean"], "v-.", color="#9467bd", label="Specificity")
    plt.xlabel("CIN3 Precancer Cell Purity (%)", fontsize=11)
    plt.ylabel("Score", fontsize=11)
    plt.title("Pap-Style Cellular Dilution: Detection vs Precancer Fraction", fontsize=12, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True)
    plt.tight_layout()
    plot_dil_path = os.path.join(PLOTS_DIR, "sensor_pap_dilution.png")
    plt.savefig(plot_dil_path, dpi=150)
    plt.close()

    # Plot 3: Combined Heatmap
    plt.figure(figsize=(7, 5))
    heatmap_df = pd.DataFrame(
        heatmap_matrix,
        index=[f"{int(d*100)}% CIN3" for d in dil_levels],
        columns=[f"Noise SD={n:.2f}" for n in noise_levels]
    )
    sns.heatmap(heatmap_df, annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={"label": "Balanced Accuracy"})
    plt.title("Combined Stress Heatmap: Cellular Dilution vs Sensor Noise", fontsize=12, fontweight="bold")
    plt.ylabel("Cellular Dilution (Purity)", fontsize=11)
    plt.xlabel("Assumed Measurement Noise", fontsize=11)
    plt.tight_layout()
    plot_hm_path = os.path.join(PLOTS_DIR, "sensor_combined_stress_heatmap.png")
    plt.savefig(plot_hm_path, dpi=150)
    plt.close()

    if verbose:
        print(f"\nGenerated plots:\n  {plot_kin_path}\n  {plot_dil_path}\n  {plot_hm_path}")

    return df_kin, df_dil, df_grid


if __name__ == "__main__":
    run_sensor_experiments(verbose=True)
