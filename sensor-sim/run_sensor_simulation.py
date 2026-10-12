"""Scenario-based robustness simulation for selected CpG panels.

This does not model a real device. It perturbs held-out array beta values using
explicit assumptions to explore how measurement error could affect classifiers.
"""
from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FS_DIR = os.path.join(ROOT, "feature_selection")
sys.path.insert(0, FS_DIR)
from data_loader import load_primary_data, impute_fold, variance_prefilter
from fs_selectors import get_selector
from validate_gse46306 import load_external

warnings.filterwarnings("ignore")
N_SIMULATIONS = 50
SEED = 271828
PREFILTER_N = 1000
SCENARIOS = [
    {"name": "clean", "kind": "clean", "level": 0},
    {"name": "noise_sd_0.01", "kind": "noise", "level": 0.01},
    {"name": "noise_sd_0.03", "kind": "noise", "level": 0.03},
    {"name": "noise_sd_0.05", "kind": "noise", "level": 0.05},
    {"name": "offset_pm_0.02", "kind": "offset", "level": 0.02},
    {"name": "offset_pm_0.05", "kind": "offset", "level": 0.05},
    {"name": "gain_0.9_1.1", "kind": "gain", "level": 0.10},
    {"name": "gain_0.8_1.2", "kind": "gain", "level": 0.20},
    {"name": "drift_pm_0.02", "kind": "drift", "level": 0.02},
    {"name": "drift_pm_0.05", "kind": "drift", "level": 0.05},
    {"name": "missing_05pct", "kind": "missing", "level": 0.05},
    {"name": "missing_10pct", "kind": "missing", "level": 0.10},
    {"name": "missing_20pct", "kind": "missing", "level": 0.20},
    {"name": "combined_moderate", "kind": "combined", "level": 0.03},
    {"name": "combined_severe", "kind": "combined", "level": 0.05},
]


def cohorts(config):
    primary = load_primary_data(config, verbose=True)
    ext_x, ext_y, ext_meta = load_external(config)
    external = {
        "X": ext_x.to_numpy(dtype=np.float32), "y": ext_y,
        "cpg_names": list(ext_x.columns),
        "sample_meta": ext_meta, "n_samples": len(ext_y),
        "n_cpgs": ext_x.shape[1],
    }
    return [("GSE99511", primary), ("GSE46306", external)]


def perturb(x, kind, level, train_medians, rng):
    """Perturb one fold's held-out panel values; always clip beta to [0, 1]."""
    z = np.asarray(x, dtype=np.float64).copy()
    n, p = z.shape
    if kind in ("noise", "combined"):
        sd = level if kind == "noise" else level
        z += rng.normal(0, sd, size=z.shape)
    if kind == "offset":
        z += rng.uniform(-level, level, size=(n, 1))
    elif kind == "gain":
        gain = rng.uniform(1-level, 1+level, size=(n, 1))
        z = 0.5 + gain * (z - 0.5)
    elif kind == "drift":
        # Randomize acquisition order independently of labels; the endpoints
        # span the assumed drift range. This is not observed time drift.
        drift = rng.permutation(np.linspace(-level, level, n, dtype=float))
        z += drift[:, None]
    elif kind == "missing":
        mask = rng.random(z.shape) < level
        z[mask] = np.broadcast_to(train_medians, z.shape)[mask]
    elif kind == "combined":
        # Combined moderate/severe levels: noise SD=level, offset +/- level,
        # gain +/- 10% (moderate) or +/- 20% (severe), drift +/- level,
        # and 10% / 20% missing values.
        z += rng.uniform(-level, level, size=(n, 1))
        gain_range = 0.10 if level <= 0.03 else 0.20
        z = 0.5 + rng.uniform(1-gain_range, 1+gain_range, size=(n, 1)) * (z - 0.5)
        drift = rng.permutation(np.linspace(-level, level, n, dtype=float))
        z += drift[:, None]
        missing = 0.10 if level <= 0.03 else 0.20
        mask = rng.random(z.shape) < missing
        z[mask] = np.broadcast_to(train_medians, z.shape)[mask]
    return np.clip(z, 0, 1)


def metrics(y, pred, score):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "Accuracy": accuracy_score(y, pred),
        "Balanced_Accuracy": balanced_accuracy_score(y, pred),
        "ROC_AUC": roc_auc_score(y, score) if len(np.unique(y)) == 2 else np.nan,
        "Sensitivity": recall_score(y, pred, zero_division=0),
        "Specificity": tn / (tn + fp) if tn + fp else 0.0,
        "Precision": precision_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }


def fit_model(name, x, y, seed):
    if name == "logreg":
        model = LogisticRegression(C=1.0, solver="liblinear", max_iter=3000,
                                   random_state=seed)
    else:
        model = CalibratedClassifierCV(
            LinearSVC(C=1.0, max_iter=3000, random_state=seed), cv=2
        )
    model.fit(x, y)
    return model


def scores(model, x):
    pred = model.predict(x)
    if hasattr(model, "predict_proba"):
        score = model.predict_proba(x)[:, 1]
    else:
        score = model.decision_function(x)
    return pred, score


def choose_models(dataset):
    path = os.path.join(ROOT, "results", "feature_selection_benchmark",
                        "all_datasets", dataset, "summary_metrics.csv")
    summary = pd.read_csv(path)
    rows = []
    for k, part in summary.groupby("k"):
        best = part.sort_values(
            ["mean_Balanced_Accuracy", "mean_ROC_AUC"], ascending=False
        ).iloc[0]
        rows.append({
            "dataset": dataset, "k": int(k), "selector": best.selector,
            "classifier": best.classifier,
            "benchmark_balanced_accuracy": best.mean_Balanced_Accuracy,
            "benchmark_roc_auc": best.mean_ROC_AUC,
            "selection_note": "Highest reported within-cohort CV balanced accuracy at this k; exploratory winner selection.",
        })
    return rows


def run_model(dataset, data, choice, config):
    X, y = data["X"], np.asarray(data["y"], dtype=int)
    cpg_names = data["cpg_names"]
    fold_rows = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (tr, va) in enumerate(cv.split(X, y)):
        xtr_imp, xva_imp, _ = impute_fold(X[tr], X[va])
        xtr_pf, xva_pf, names, _ = variance_prefilter(
            xtr_imp, xva_imp, cpg_names,
            threshold=config["prefilter"]["variance_threshold"], top_n=PREFILTER_N,
        )
        selector = get_selector(
            choice["selector"], int(choice["k"]), config,
            random_state=(42 + fold) + (
                sum((i + 1) * ord(ch) for i, ch in enumerate(choice["selector"]))
                % 1000
            ),
        )
        selector.fit(xtr_pf, y[tr])
        idx = selector.selected_indices_
        xtr = xtr_pf[:, idx]
        xva = xva_pf[:, idx]
        scaler = StandardScaler()
        xtr = scaler.fit_transform(xtr)
        xva = scaler.transform(xva)
        model = fit_model(choice["classifier"], xtr, y[tr], 42 + fold)

        # Training medians are used only for simulated missing readings.
        medians = np.nanmedian(xtr_pf[:, idx], axis=0)
        # Store fold-level predictions by simulation and scenario.
        for scenario in SCENARIOS:
            repeats = 1 if scenario["kind"] == "clean" else N_SIMULATIONS
            for rep in range(repeats):
                rng = np.random.default_rng(SEED + fold * 100000 + rep * 1000
                                            + SCENARIOS.index(scenario))
                x_sim = perturb(xva_pf[:, idx], scenario["kind"],
                                scenario["level"], medians, rng)
                # For dropout, medians above are from prefiltered beta scale.
                # Apply the already-fitted training scaler and classifier.
                x_sim = scaler.transform(x_sim)
                pred, score = scores(model, x_sim)
                fold_rows.append({
                    "dataset": dataset, "k": int(choice["k"]),
                    "selector": choice["selector"],
                    "classifier": choice["classifier"],
                    "scenario": scenario["name"], "simulation": rep,
                    "fold": fold, "y": y[va].copy(),
                    "pred": pred, "score": score,
                })
    # Pool OOF predictions for each Monte Carlo run. Fold-level ordering differs,
    # so concatenate rows by fold and calculate the metric on all held-out people.
    out = []
    for scenario in SCENARIOS:
        reps = [0] if scenario["kind"] == "clean" else range(N_SIMULATIONS)
        for rep in reps:
            subset = [r for r in fold_rows if r["scenario"] == scenario["name"]
                      and r["simulation"] == rep]
            yy = np.concatenate([r["y"] for r in subset])
            pp = np.concatenate([r["pred"] for r in subset])
            ss = np.concatenate([r["score"] for r in subset])
            out.append({"dataset": dataset, "k": int(choice["k"]),
                        "selector": choice["selector"],
                        "classifier": choice["classifier"],
                        "scenario": scenario["name"], "simulation": rep,
                        **metrics(yy, pp, ss)})
    return out


def main():
    cfg_path = os.path.join(FS_DIR, "config.json")
    with open(cfg_path, encoding="utf-8") as f:
        config = json.load(f)
    config["selectors"]["stability_selection"]["n_subsamples"] = 10
    config["selectors"]["rf_importance"]["n_estimators"] = 100
    config["selectors"]["svm_rfe"]["step_fraction"] = 0.5

    output_dir = os.path.dirname(__file__)
    all_choices, all_runs = [], []
    cohort_map = dict(cohorts(config))
    for dataset in ("GSE99511", "GSE46306"):
        choices = choose_models(dataset)
        all_choices.extend(choices)
        data = cohort_map[dataset]
        print(f"\n=== {dataset}: {data['n_samples']} samples, {data['n_cpgs']:,} CpGs ===")
        for choice in choices:
            print(f"  k={choice['k']}: {choice['selector']} + {choice['classifier']}", flush=True)
            all_runs.extend(run_model(dataset, data, choice, config))

    runs = pd.DataFrame(all_runs)
    runs.to_csv(os.path.join(output_dir, "simulation_runs.csv"), index=False)
    choices_df = pd.DataFrame(all_choices)
    choices_df.to_csv(os.path.join(output_dir, "model_choices.csv"), index=False)
    metrics_cols = ["Accuracy", "Balanced_Accuracy", "ROC_AUC", "Sensitivity",
                    "Specificity", "Precision", "F1", "TP", "TN", "FP", "FN"]
    summary = runs.groupby(
        ["dataset", "k", "selector", "classifier", "scenario"]
    )[metrics_cols].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(str(c) for c in col if c).rstrip("_")
                       if isinstance(col, tuple) else col for col in summary.columns]
    summary = summary.reset_index(drop=True)
    summary.to_csv(os.path.join(output_dir, "simulation_summary.csv"), index=False)
    print("Saved simulation outputs in", output_dir)


if __name__ == "__main__":
    main()
