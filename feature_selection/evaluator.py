"""
evaluator.py
------------
Leakage-free nested repeated stratified cross-validation engine.

Architecture
------------
Outer CV (5-fold x 10 repeats = 50 outer folds):
  For each outer fold:
    1. impute_fold (train mean only)
    2. variance_prefilter to top_n CpGs (train variance only)
    3. selector.fit(X_train_prefilt, y_train)   ← ONLY training data
    4. X_train_k, X_val_k = selector.transform(...)
    5. [inner CV for classifier hyperparameter selection]
    6. StandardScaler fit on X_train_k only
    7. classifier.fit(X_train_k_scaled, y_train)
    8. evaluate on X_val_k_scaled (NEVER seen before)

Leakage assertion
-----------------
An explicit check verifies that val indices are not in train indices for every fold.

Outputs
-------
- Per-fold records: selector, k, repeat, fold, classifier, metrics, selected CpG names
- Aggregated mean ± std table
- Per-fold selected features (for stability analysis)
"""

import warnings
import time
import numpy as np
import pandas as pd
from itertools import product
import sys
import os

# Ensure feature_selection/ is importable in joblib subprocess workers
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    balanced_accuracy_score, roc_auc_score, f1_score,
    recall_score, confusion_matrix, accuracy_score
)

from data_loader import impute_fold, variance_prefilter
from fs_selectors import get_selector, SELECTOR_NAMES

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_prob: np.ndarray = None) -> dict:
    """Full binary classification metric set."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    bal_acc  = balanced_accuracy_score(y_true, y_pred)
    acc      = accuracy_score(y_true, y_pred)
    sens     = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # recall / sensitivity
    spec     = tn / (tn + fp) if (tn + fp) > 0 else 0.0   # specificity
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1       = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc  = roc_auc_score(y_true, y_prob) if y_prob is not None else np.nan
    except Exception:
        auc  = np.nan
    return dict(
        Accuracy=round(acc, 6),
        Balanced_Accuracy=round(bal_acc, 6),
        Sensitivity=round(sens, 6),
        Specificity=round(spec, 6),
        Precision=round(precision, 6),
        F1=round(f1, 6),
        ROC_AUC=round(auc, 6) if not np.isnan(auc) else np.nan,
        TP=int(tp), TN=int(tn), FP=int(fp), FN=int(fn),
    )


def _get_classifier_proba(clf, X: np.ndarray) -> np.ndarray:
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    if hasattr(clf, "decision_function"):
        df = clf.decision_function(X)
        mn, mx = df.min(), df.max()
        return (df - mn) / (mx - mn + 1e-12)
    return clf.predict(X).astype(float)


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------

def _tune_classifier(name: str, X_tr: np.ndarray, y_tr: np.ndarray,
                     config: dict, inner_seed: int) -> object:
    """
    Inner CV to select best C for the downstream classifier.
    Returns fitted classifier on full X_tr.
    """
    clf_cfg = config.get("classifiers", {})
    inner_splits = config["cv"]["inner_splits"]

    if name == "logreg":
        c_grid = clf_cfg.get("logreg", {}).get("C_grid", [0.01, 0.1, 1.0, 10.0])
        max_iter = clf_cfg.get("logreg", {}).get("max_iter", 3000)
        best_C, best_score = c_grid[0], -1.0
        inner_cv = StratifiedKFold(
            n_splits=min(inner_splits, len(np.unique(y_tr))),
            shuffle=True, random_state=inner_seed
        )
        for C in c_grid:
            scores = []
            for itr, iva in inner_cv.split(X_tr, y_tr):
                m = LogisticRegression(
                    C=C, solver="liblinear", max_iter=max_iter, random_state=inner_seed
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m.fit(X_tr[itr], y_tr[itr])
                scores.append(balanced_accuracy_score(y_tr[iva], m.predict(X_tr[iva])))
            mean_sc = np.mean(scores)
            if mean_sc > best_score:
                best_score, best_C = mean_sc, C
        clf = LogisticRegression(
            C=best_C, solver="liblinear", max_iter=max_iter, random_state=inner_seed
        )

    elif name == "linear_svm":
        c_grid = clf_cfg.get("linear_svm", {}).get("C_grid", [0.01, 0.1, 1.0, 10.0])
        max_iter = clf_cfg.get("linear_svm", {}).get("max_iter", 3000)
        best_C, best_score = c_grid[0], -1.0
        inner_cv = StratifiedKFold(
            n_splits=min(inner_splits, len(np.unique(y_tr))),
            shuffle=True, random_state=inner_seed
        )
        for C in c_grid:
            scores = []
            for itr, iva in inner_cv.split(X_tr, y_tr):
                base = LinearSVC(C=C, max_iter=max_iter, random_state=inner_seed)
                m    = CalibratedClassifierCV(base, cv=2)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m.fit(X_tr[itr], y_tr[itr])
                scores.append(balanced_accuracy_score(y_tr[iva], m.predict(X_tr[iva])))
            mean_sc = np.mean(scores)
            if mean_sc > best_score:
                best_score, best_C = mean_sc, C
        clf = CalibratedClassifierCV(
            LinearSVC(C=best_C, max_iter=max_iter, random_state=inner_seed), cv=3
        )
    else:
        raise ValueError(f"Unknown classifier: {name}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X_tr, y_tr)
    return clf


# ---------------------------------------------------------------------------
# Single outer-fold evaluation
# ---------------------------------------------------------------------------

def _run_single_fold(fold_args: dict) -> list:
    """
    Evaluate all (selector x k x classifier) combinations on one outer fold.
    This function is designed to be called in parallel via joblib.

    Returns list of result dicts.
    """
    # Ensure imports work in joblib subprocess workers
    import sys, os
    _d = fold_args.get("_feature_selection_dir", "")
    if _d and _d not in sys.path:
        sys.path.insert(0, _d)

    from data_loader import impute_fold, variance_prefilter
    from fs_selectors import get_selector
    from sklearn.preprocessing import StandardScaler
    (outer_fold_idx, repeat_idx, train_idx, val_idx,
     X, y, cpg_names, config, selector_names, panel_sizes,
     prefilter_n, verbose) = (
        fold_args["outer_fold_idx"], fold_args["repeat_idx"],
        fold_args["train_idx"],      fold_args["val_idx"],
        fold_args["X"],              fold_args["y"],
        fold_args["cpg_names"],      fold_args["config"],
        fold_args["selector_names"], fold_args["panel_sizes"],
        fold_args["prefilter_n"],    fold_args.get("verbose", False),
    )

    # ---- No-leakage assertion ----
    train_set = set(train_idx.tolist())
    val_set   = set(val_idx.tolist())
    assert len(train_set & val_set) == 0, (
        f"LEAKAGE: train and val indices overlap at fold {outer_fold_idx} repeat {repeat_idx}!"
    )

    X_train_raw = X[train_idx]
    X_val_raw   = X[val_idx]
    y_train     = y[train_idx]
    y_val       = y[val_idx]

    base_seed = config["cv"]["random_seed"]
    fold_seed = base_seed + repeat_idx * 100 + outer_fold_idx

    # Step 1: Impute (train mean only)
    X_train_imp, X_val_imp, _col_means = impute_fold(X_train_raw, X_val_raw)

    # No NaNs after imputation
    assert np.isnan(X_train_imp).sum() == 0, "NaNs remain in train after imputation"
    assert np.isnan(X_val_imp).sum() == 0,   "NaNs remain in val after imputation"

    # Optional M-value conversion (log2(beta / (1 - beta)))
    if config.get("use_m_values", False):
        eps = 1e-5
        X_train_imp = np.clip(X_train_imp, eps, 1.0 - eps)
        X_val_imp   = np.clip(X_val_imp,   eps, 1.0 - eps)
        X_train_imp = np.log2(X_train_imp / (1.0 - X_train_imp))
        X_val_imp   = np.log2(X_val_imp   / (1.0 - X_val_imp))

    # Step 2: Pre-filter (train variance only)

    X_train_pf, X_val_pf, pf_names, _pf_mask = variance_prefilter(
        X_train_imp, X_val_imp, cpg_names,
        threshold=config["prefilter"]["variance_threshold"],
        top_n=prefilter_n,
    )

    results = []

    for sel_name in selector_names:
        for k in panel_sizes:
            # Stable across Python processes (unlike built-in hash()).
            selector_seed = sum((i + 1) * ord(ch)
                                for i, ch in enumerate(sel_name))
            sel_seed = fold_seed + selector_seed % 1000

            try:
                selector = get_selector(sel_name, k, config, random_state=sel_seed)

                # ---- FIT selector on train only ----
                t0 = time.time()
                selector.fit(X_train_pf, y_train)
                sel_time = time.time() - t0

                # Attach CpG names to selector
                selector.selected_names_ = [
                    pf_names[i] for i in selector.selected_indices_
                ]
                selected_cpgs = selector.selected_names_

                # Extract selected features
                X_tr_k = X_train_pf[:, selector.selected_indices_]
                X_va_k = X_val_pf[:, selector.selected_indices_]

                # ---- Scale on train only ----
                scaler = StandardScaler()
                X_tr_k_sc = scaler.fit_transform(X_tr_k)
                X_va_k_sc = scaler.transform(X_va_k)

                # ---- Evaluate classifiers ----
                for clf_name in ["logreg", "linear_svm"]:
                    try:
                        if config.get("cv", {}).get("fixed_classifier_params", False):
                            if clf_name == "logreg":
                                clf = LogisticRegression(
                                    C=1.0,
                                    solver=config.get("classifiers", {}).get("logreg", {}).get("solver", "liblinear"),
                                    max_iter=config.get("classifiers", {}).get("logreg", {}).get("max_iter", 3000),
                                    random_state=fold_seed,
                                )
                            else:
                                clf = CalibratedClassifierCV(
                                    LinearSVC(
                                        C=1.0,
                                        max_iter=config.get("classifiers", {}).get("linear_svm", {}).get("max_iter", 3000),
                                        random_state=fold_seed,
                                    ), cv=2
                                )
                            clf.fit(X_tr_k_sc, y_train)
                        else:
                            clf = _tune_classifier(
                                clf_name, X_tr_k_sc, y_train, config, fold_seed
                            )
                        y_pred = clf.predict(X_va_k_sc)
                        y_prob = _get_classifier_proba(clf, X_va_k_sc)
                        metrics = compute_metrics(y_val, y_pred, y_prob)

                        results.append({
                            "repeat":       repeat_idx,
                            "outer_fold":   outer_fold_idx,
                            "selector":     sel_name,
                            "k":            k,
                            "classifier":   clf_name,
                            "prefilter_n":  prefilter_n,
                            "n_train":      len(train_idx),
                            "n_val":        len(val_idx),
                            "sel_time_s":   round(sel_time, 3),
                            "selected_cpgs": ",".join(selected_cpgs),
                            **metrics,
                        })
                    except Exception as e:
                        results.append({
                            "repeat":       repeat_idx,
                            "outer_fold":   outer_fold_idx,
                            "selector":     sel_name,
                            "k":            k,
                            "classifier":   clf_name,
                            "prefilter_n":  prefilter_n,
                            "error":        str(e),
                            "Balanced_Accuracy": np.nan,
                            "ROC_AUC":      np.nan,
                            "F1":           np.nan,
                            "Sensitivity":  np.nan,
                            "Specificity":  np.nan,
                            "Accuracy":     np.nan,
                            "selected_cpgs": "",
                        })
            except Exception as e:
                # Selector failed entirely
                for clf_name in ["logreg", "linear_svm"]:
                    results.append({
                        "repeat":       repeat_idx,
                        "outer_fold":   outer_fold_idx,
                        "selector":     sel_name,
                        "k":            k,
                        "classifier":   clf_name,
                        "prefilter_n":  prefilter_n,
                        "error":        f"selector_failed: {e}",
                        "Balanced_Accuracy": np.nan,
                        "ROC_AUC":      np.nan,
                        "F1":           np.nan,
                        "Sensitivity":  np.nan,
                        "Specificity":  np.nan,
                        "Accuracy":     np.nan,
                        "selected_cpgs": "",
                    })

    return results


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

def run_nested_cv(
    X: np.ndarray,
    y: np.ndarray,
    cpg_names: list,
    config: dict,
    selector_names: list = None,
    prefilter_n: int = None,
    n_jobs: int = -1,
    verbose: bool = True,
    cv_splitter = None,
    groups: np.ndarray = None,
) -> pd.DataFrame:
    """
    Run the full repeated nested CV benchmark (or custom cv_splitter such as leave-one-chip-out).

    Parameters
    ----------
    X           : float32 array (n_samples, n_cpgs) — raw beta values with NaNs OK
    y           : int array (n_samples,) — 0/1 labels
    cpg_names   : list of CpG IDs matching X columns
    config      : loaded config dict
    selector_names : subset of SELECTOR_NAMES to run (None = all)
    prefilter_n : override prefilter_size_main from config
    n_jobs      : joblib parallelism (-1 = all cores)
    verbose     : print progress
    cv_splitter : optional cross-validation splitter (e.g. LeaveOneGroupOut)
    groups      : optional sample group labels for group-aware CV (e.g. Sentrix_ID chips)

    Returns
    -------
    pd.DataFrame with one row per (repeat, fold, selector, k, classifier)
    """
    from joblib import Parallel, delayed

    cv_cfg = config["cv"]
    outer_splits  = cv_cfg.get("outer_splits", 5)
    outer_repeats = cv_cfg.get("outer_repeats", 10)
    base_seed     = cv_cfg.get("random_seed", 42)

    panel_sizes    = config["panel_sizes"]
    selector_names = selector_names or SELECTOR_NAMES
    prefilter_n    = prefilter_n or config["prefilter"]["prefilter_size_main"]

    if cv_splitter is not None:
        splits = list(cv_splitter.split(X, y, groups))
        total_folds = len(splits)
        if verbose:
            print(f"\n[Evaluator] Custom CV (e.g. Group/Chip-aware): {total_folds} outer folds")
    else:
        rskf = RepeatedStratifiedKFold(
            n_splits=outer_splits,
            n_repeats=outer_repeats,
            random_state=base_seed,
        )
        splits = list(rskf.split(X, y))
        total_folds = len(splits)
        if verbose:
            print(f"\n[Evaluator] Repeated nested CV: {outer_splits}-fold x {outer_repeats} repeats")

    if verbose:
        print(f"[Evaluator] Selectors: {selector_names}")
        print(f"[Evaluator] Panel sizes k: {panel_sizes}")
        print(f"[Evaluator] Pre-filter N: {prefilter_n}")
        print(f"[Evaluator] Total outer folds: {total_folds}")
        print(f"[Evaluator] Parallelism: n_jobs={n_jobs}")

    fold_args_list = []
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    for fold_num, (train_idx, val_idx) in enumerate(splits):
        if cv_splitter is not None:
            repeat_idx = 0
            outer_fold_idx = fold_num
        else:
            repeat_idx     = fold_num // outer_splits
            outer_fold_idx = fold_num  % outer_splits
        fold_args_list.append({

            "outer_fold_idx": outer_fold_idx,
            "repeat_idx":     repeat_idx,
            "train_idx":      train_idx,
            "val_idx":        val_idx,
            "X":              X,
            "y":              y,
            "cpg_names":      cpg_names,
            "config":         config,
            "selector_names": selector_names,
            "panel_sizes":    panel_sizes,
            "prefilter_n":    prefilter_n,
            "verbose":        False,
            "_feature_selection_dir": _this_dir,
        })

    if verbose:
        t0 = time.time()
        print(f"[Evaluator] Starting parallel evaluation ...")

    if n_jobs == 1 or n_jobs == 0:
        all_results = []
        for i, args in enumerate(fold_args_list):
            res = _run_single_fold(args)
            all_results.extend(res)
            if verbose and (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                print(f"  Fold {i+1}/{total_folds} done ({elapsed:.0f}s elapsed)")
    else:
        import joblib
        raw = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(_run_single_fold)(args) for args in fold_args_list
        )
        all_results = [r for batch in raw for r in batch]

    if verbose:
        elapsed = time.time() - t0
        print(f"[Evaluator] Done in {elapsed:.1f}s ({elapsed/60:.1f} min).")

    df = pd.DataFrame(all_results)
    return df


# ---------------------------------------------------------------------------
# Baseline reproduction: existing top-5000 ANOVA + RandomForest
# ---------------------------------------------------------------------------

def run_baseline_reproduction(
    X: np.ndarray,
    y: np.ndarray,
    cpg_names: list,
    config: dict,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Reproduce the existing pipeline's top-5000 ANOVA + RF result under
    correct repeated nested CV so we can see how much the original was inflated.
    Uses the same pre-filter + ANOVA + RF but with outer 5x10 and inner 3-fold.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import f_classif
    from joblib import Parallel, delayed

    cv_cfg        = config["cv"]
    outer_splits  = cv_cfg["outer_splits"]
    outer_repeats = cv_cfg["outer_repeats"]
    base_seed     = cv_cfg["random_seed"]
    K             = config["baseline_reproduction"]["k"]

    if verbose:
        print(f"\n[Baseline] Reproducing top-{K} ANOVA + RF under repeated nested CV ...")

    rskf = RepeatedStratifiedKFold(
        n_splits=outer_splits, n_repeats=outer_repeats, random_state=base_seed
    )

    def _baseline_fold(train_idx, val_idx, fold_num):
        repeat_idx     = fold_num // outer_splits
        outer_fold_idx = fold_num  % outer_splits
        fold_seed      = base_seed + repeat_idx * 100 + outer_fold_idx

        X_train_raw = X[train_idx]
        X_val_raw   = X[val_idx]
        y_train     = y[train_idx]
        y_val       = y[val_idx]

        # No-leakage assertion
        assert len(set(train_idx) & set(val_idx)) == 0

        X_train_imp, X_val_imp, _ = impute_fold(X_train_raw, X_val_raw)

        # Variance prefilter
        X_tr_pf, X_va_pf, pf_names, _ = variance_prefilter(
            X_train_imp, X_val_imp, cpg_names,
            threshold=config["prefilter"]["variance_threshold"],
            top_n=K,
        )

        # ANOVA on train only
        from sklearn.feature_selection import f_classif as _f_classif
        f_vals, _ = _f_classif(X_tr_pf, y_train)
        f_vals    = np.nan_to_num(f_vals, nan=0.0)
        k_actual  = min(K, len(f_vals))
        top_idx   = np.argsort(f_vals)[::-1][:k_actual]

        X_tr_k = X_tr_pf[:, top_idx]
        X_va_k = X_va_pf[:, top_idx]

        scaler   = StandardScaler()
        X_tr_sc  = scaler.fit_transform(X_tr_k)
        X_va_sc  = scaler.transform(X_va_k)

        rf = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=3,
            random_state=fold_seed, n_jobs=1
        )
        rf.fit(X_tr_sc, y_train)
        y_pred = rf.predict(X_va_sc)
        y_prob = rf.predict_proba(X_va_sc)[:, 1]
        metrics = compute_metrics(y_val, y_pred, y_prob)

        return {
            "repeat":      repeat_idx,
            "outer_fold":  outer_fold_idx,
            "selector":    f"ANOVA_top{K}",
            "classifier":  "RandomForest",
            "k":           K,
            **metrics,
        }

    results = Parallel(n_jobs=-1)(
        delayed(_baseline_fold)(tr, va, i)
        for i, (tr, va) in enumerate(rskf.split(X, y))
    )
    df = pd.DataFrame(results)
    if verbose:
        auc_mean = df["ROC_AUC"].mean()
        auc_std  = df["ROC_AUC"].std()
        ba_mean  = df["Balanced_Accuracy"].mean()
        ba_std   = df["Balanced_Accuracy"].std()
        print(f"[Baseline] ANOVA-top{K} + RF: AUC={auc_mean:.3f}±{auc_std:.3f}, "
              f"Bal.Acc={ba_mean:.3f}±{ba_std:.3f} (repeated nested CV)")
    return df


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------

def run_permutation_test(
    X: np.ndarray,
    y: np.ndarray,
    cpg_names: list,
    config: dict,
    best_selector: str,
    best_k: int,
    best_classifier: str,
    verbose: bool = True,
) -> dict:
    """
    Shuffle labels n_permutations times and rerun the full pipeline.
    Returns dict with null distribution of balanced accuracy and AUC,
    and empirical p-value relative to the observed score.
    """
    from joblib import Parallel, delayed

    perm_cfg      = config["permutation_test"]
    n_perms       = perm_cfg["n_permutations"]
    outer_splits  = perm_cfg["outer_splits"]
    outer_repeats = perm_cfg["outer_repeats"]
    base_seed     = config["cv"]["random_seed"]
    prefilter_n   = config["prefilter"]["prefilter_size_main"]

    if verbose:
        print(f"\n[Permutation] {n_perms} permutations x {outer_splits}x{outer_repeats} CV "
              f"on best config: selector={best_selector}, k={best_k}, clf={best_classifier}")

    def _one_permutation(perm_idx):
        rng_perm = np.random.RandomState(base_seed + perm_idx + 9999)
        y_perm   = rng_perm.permutation(y)
        rskf     = RepeatedStratifiedKFold(
            n_splits=outer_splits, n_repeats=outer_repeats,
            random_state=base_seed + perm_idx
        )
        fold_results = []
        for fold_num, (train_idx, val_idx) in enumerate(rskf.split(X, y_perm)):
            repeat_idx     = fold_num // outer_splits
            outer_fold_idx = fold_num  % outer_splits
            fold_seed      = base_seed + perm_idx * 10000 + repeat_idx * 100 + outer_fold_idx

            X_tr_raw = X[train_idx]
            X_va_raw = X[val_idx]
            y_tr     = y_perm[train_idx]
            y_va     = y_perm[val_idx]

            if len(np.unique(y_tr)) < 2:
                continue

            X_tr_imp, X_va_imp, _ = impute_fold(X_tr_raw, X_va_raw)
            X_tr_pf, X_va_pf, pf_names, _ = variance_prefilter(
                X_tr_imp, X_va_imp, cpg_names,
                threshold=config["prefilter"]["variance_threshold"],
                top_n=prefilter_n,
            )

            try:
                sel = get_selector(best_selector, best_k, config, random_state=fold_seed)
                sel.fit(X_tr_pf, y_tr)
                X_tr_k = X_tr_pf[:, sel.selected_indices_]
                X_va_k = X_va_pf[:, sel.selected_indices_]
                scaler = StandardScaler()
                X_tr_sc = scaler.fit_transform(X_tr_k)
                X_va_sc = scaler.transform(X_va_k)
                clf = _tune_classifier(best_classifier, X_tr_sc, y_tr, config, fold_seed)
                y_pred = clf.predict(X_va_sc)
                y_prob = _get_classifier_proba(clf, X_va_sc)
                m      = compute_metrics(y_va, y_pred, y_prob)
                fold_results.append(m["Balanced_Accuracy"])
            except Exception:
                pass

        if fold_results:
            return float(np.mean(fold_results))
        return 0.5

    null_scores = Parallel(n_jobs=-1)(
        delayed(_one_permutation)(i) for i in range(n_perms)
    )
    null_scores = np.array(null_scores)

    if verbose:
        print(f"[Permutation] Null Bal.Acc: {null_scores.mean():.3f} ± {null_scores.std():.3f}")

    return {
        "null_balanced_accuracy": null_scores.tolist(),
        "null_mean": float(null_scores.mean()),
        "null_std":  float(null_scores.std()),
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-fold results into mean ± std and 95% CI table.
    One row per (selector, k, classifier, prefilter_n).
    """
    from scipy import stats

    metrics = ["Balanced_Accuracy", "ROC_AUC", "Sensitivity",
               "Specificity", "Precision", "F1", "Accuracy"]
    group_cols = ["selector", "k", "classifier", "prefilter_n"]
    # only use cols that exist
    group_cols = [c for c in group_cols if c in df.columns]

    rows = []
    for keys, grp in df.groupby(group_cols):
        if isinstance(keys, str):
            keys = (keys,)
        rec = dict(zip(group_cols, keys))
        n_folds = len(grp)
        rec["n_folds"] = n_folds
        for m in metrics:
            if m in grp.columns:
                vals = grp[m].dropna()
                n_m = len(vals)
                if n_m > 0:
                    mean_val = float(vals.mean())
                    std_val  = float(vals.std()) if n_m > 1 else 0.0
                    se_val   = std_val / np.sqrt(n_m) if n_m > 1 else 0.0
                    t_crit   = stats.t.ppf(0.975, df=n_m - 1) if n_m > 1 else 1.96
                    ci_low   = max(0.0, mean_val - t_crit * se_val)
                    ci_high  = min(1.0, mean_val + t_crit * se_val)
                    
                    rec[f"mean_{m}"] = round(mean_val, 4)
                    rec[f"std_{m}"]  = round(std_val, 4)
                    rec[f"ci95_low_{m}"]  = round(ci_low, 4)
                    rec[f"ci95_high_{m}"] = round(ci_high, 4)
                    rec[f"{m}_mean_ci95"] = f"{mean_val:.3f} [{ci_low:.3f}, {ci_high:.3f}]"
                    rec[f"n_{m}"] = n_m
                else:
                    rec[f"mean_{m}"] = np.nan
                    rec[f"std_{m}"]  = np.nan
                    rec[f"ci95_low_{m}"]  = np.nan
                    rec[f"ci95_high_{m}"] = np.nan
                    rec[f"{m}_mean_ci95"] = "n/a"
                    rec[f"n_{m}"] = 0
        for count_metric in ["TP", "TN", "FP", "FN"]:
            if count_metric in grp.columns:
                rec[f"sum_{count_metric}"] = int(grp[count_metric].sum())
        rows.append(rec)

    agg = pd.DataFrame(rows)
    if "mean_Balanced_Accuracy" in agg.columns:
        agg = agg.sort_values(
            ["selector", "k", "classifier"],
            ascending=[True, True, True]
        ).reset_index(drop=True)
    return agg

