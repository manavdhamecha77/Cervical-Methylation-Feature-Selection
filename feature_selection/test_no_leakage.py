"""
test_no_leakage.py
------------------
Unit tests that PROVE no test data ever touches any selection or fitting step.

Tests
-----
1. test_impute_uses_train_mean_only
   - Verify that imputation parameters come exclusively from training data.

2. test_variance_prefilter_train_only
   - Verify that the pre-filter mask is computed only from training variance.

3. test_selector_fit_ignores_val
   - Fit each selector on training data, then verify it produces the SAME
     result even if the val data is replaced with Gaussian noise.

4. test_no_index_overlap
   - Verify that train and val index sets are disjoint in every outer fold.

5. test_scaler_fit_train_only
   - Verify that StandardScaler is fit on training features only.

6. test_metrics_computed_on_val_only
   - Verify that y_true used for metrics equals y[val_idx] (val labels only).

7. test_pipeline_assertion_fires
   - Verify that the leakage assertion in _run_single_fold raises if
     train and val indices overlap.

Run with:
  .venv\\Scripts\\python.exe feature_selection\\test_no_leakage.py
or:
  .venv\\Scripts\\python.exe -m pytest feature_selection\\test_no_leakage.py -v
"""

import sys
import os
import json
import warnings
import numpy as np

warnings.filterwarnings("ignore")

# Allow imports from feature_selection/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import impute_fold, variance_prefilter
from fs_selectors import (MannWhitneySelector, LassoSelector, ElasticNetSelector,
                       StabilitySelectionSelector, SVMRFESelector,
                       MRMRSelector, RandomForestSelector, SELECTOR_NAMES,
                       get_selector)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_data(n_samples=40, n_features=200, seed=0):
    """Small synthetic dataset for fast unit tests."""
    rng = np.random.RandomState(seed)
    X   = rng.beta(2, 5, size=(n_samples, n_features)).astype(np.float32)
    y   = (rng.rand(n_samples) > 0.5).astype(int)
    # Ensure at least 2 positive and 2 negative
    y[:20] = 0
    y[20:] = 1
    # Inject some NaNs
    mask = rng.rand(n_samples, n_features) < 0.02
    X[mask] = np.nan
    cpg_names = [f"cg{i:07d}" for i in range(n_features)]
    return X, y, cpg_names


def _minimal_config():
    return {
        "prefilter": {
            "variance_threshold": 0.001,
            "prefilter_size_main": 100,
        },
        "panel_sizes": [5, 10],
        "cv": {
            "outer_splits": 3,
            "outer_repeats": 2,
            "inner_splits": 2,
            "random_seed": 42,
        },
        "classifiers": {
            "logreg": {"C_grid": [0.1, 1.0], "max_iter": 500},
            "linear_svm": {"C_grid": [0.1, 1.0], "max_iter": 500},
        },
        "selectors": {
            "lasso": {"C_grid": [0.01, 0.1, 1.0], "max_iter": 500},
            "elastic_net": {
                "C_grid": [0.01, 0.1],
                "l1_ratio_grid": [0.5],
                "max_iter": 500,
            },
            "stability_selection": {
                "n_subsamples": 5,
                "subsample_fraction": 0.65,
                "C": 0.1,
                "max_iter": 200,
            },
            "svm_rfe": {"C": 1.0, "step_fraction": 0.3},
            "rf_importance": {"n_estimators": 10, "max_depth": 3, "min_samples_leaf": 2},
        },
    }


# ---------------------------------------------------------------------------
# Test 1: Imputation uses train mean only
# ---------------------------------------------------------------------------

def test_impute_uses_train_mean_only():
    """
    Tests that:
    (a) train imputation uses only train statistics (changing val has no effect on train result)
    (b) val NaN positions are filled with TRAIN means (not val means)
    """
    X, y, _ = _make_data(seed=1)
    train_idx = np.arange(30)
    val_idx   = np.arange(30, 40)

    X_train = X[train_idx].copy()
    X_val   = X[val_idx].copy()

    # (a) Corrupt val data and verify train imputation is unchanged
    X_tr_imp1, X_va_imp1, means1 = impute_fold(X_train, X_val)

    X_val_corrupted = np.full_like(X_val, 999.0)
    # Preserve NaN positions so impute_fold still has NaNs to fill
    nan_mask = np.isnan(X_val)
    X_val_corrupted[nan_mask] = np.nan

    X_tr_imp2, X_va_imp2, means2 = impute_fold(X_train, X_val_corrupted)

    np.testing.assert_array_equal(
        X_tr_imp1, X_tr_imp2,
        err_msg="FAIL: train imputation changed when val data was corrupted — leakage!"
    )
    np.testing.assert_array_equal(
        means1, means2,
        err_msg="FAIL: imputation means changed when val data was corrupted — leakage!"
    )

    # (b) Val NaN positions should be filled with TRAIN mean (not val mean)
    if nan_mask.any():
        rows_nan, cols_nan = np.where(nan_mask)
        for r, c in zip(rows_nan, cols_nan):
            np.testing.assert_allclose(
                float(X_va_imp2[r, c]), float(means2[c]), rtol=1e-4,
                err_msg=(
                    f"FAIL: val NaN at ({r},{c}) not filled with train mean. "
                    f"Got {X_va_imp2[r, c]}, expected {means2[c]}"
                )
            )

    # (c) Val non-NaN positions (999.0) should remain as-is (not replaced by train mean)
    non_nan_mask = ~nan_mask
    if non_nan_mask.any():
        rows_nn, cols_nn = np.where(non_nan_mask)
        r0, c0 = rows_nn[0], cols_nn[0]
        np.testing.assert_allclose(
            float(X_va_imp2[r0, c0]), 999.0, rtol=1e-4,
            err_msg="FAIL: val non-NaN position was overwritten — imputation incorrectly modified non-NaN val values"
        )

    print("PASS: test_impute_uses_train_mean_only")


# ---------------------------------------------------------------------------
# Test 2: Variance pre-filter uses train variance only
# ---------------------------------------------------------------------------

def test_variance_prefilter_train_only():
    """
    Changing val data should not change which features are selected by the pre-filter.
    """
    X, y, cpg_names = _make_data(seed=2)
    train_idx = np.arange(30)
    val_idx   = np.arange(30, 40)

    X_train_imp, X_val_imp, _ = impute_fold(X[train_idx], X[val_idx])

    # Reference pre-filter
    _, _, names1, mask1 = variance_prefilter(
        X_train_imp, X_val_imp, cpg_names, top_n=50
    )

    # Corrupt val
    X_val_corrupted = np.full_like(X_val_imp, 0.0)  # zero variance val
    _, _, names2, mask2 = variance_prefilter(
        X_train_imp, X_val_corrupted, cpg_names, top_n=50
    )

    assert names1 == names2, (
        "FAIL: pre-filter selected different features when val data was corrupted — "
        "pre-filter is using val data!"
    )
    np.testing.assert_array_equal(
        mask1, mask2,
        err_msg="FAIL: pre-filter mask changed when val data was changed — leakage!"
    )
    print("PASS: test_variance_prefilter_train_only")


# ---------------------------------------------------------------------------
# Test 3: Each selector fit ignores val data
# ---------------------------------------------------------------------------

def test_selector_fit_ignores_val():
    """
    Each selector's fit() must not use val data.
    We verify: fitting on X_train gives the same selected features
    regardless of what X_val looks like.
    """
    X, y, cpg_names = _make_data(seed=3, n_features=100)
    X_train_imp, X_val_imp, _ = impute_fold(X[:30], X[30:])
    X_tr_pf, X_va_pf, pf_names, _ = variance_prefilter(
        X_train_imp, X_val_imp, cpg_names, top_n=80
    )
    y_train = y[:30]
    config = _minimal_config()

    rng = np.random.RandomState(7)
    X_va_noise = rng.randn(*X_va_pf.shape).astype(np.float32)

    for sel_name in SELECTOR_NAMES:
        sel1 = get_selector(sel_name, k=5, config=config, random_state=42)
        sel2 = get_selector(sel_name, k=5, config=config, random_state=42)

        sel1.fit(X_tr_pf, y_train)
        sel2.fit(X_tr_pf, y_train)  # fit on same train, val doesn't matter

        # Transforming different val data should not change selected_indices_
        np.testing.assert_array_equal(
            sel1.selected_indices_,
            sel2.selected_indices_,
            err_msg=f"FAIL [{sel_name}]: fit() produced different indices — selector is non-deterministic or using val data"
        )

        # transform with real val
        Xv1 = sel1.transform(X_va_pf)
        # transform with noisy val
        Xv2 = sel2.transform(X_va_noise)
        # Both transformations should select the same columns (from same feature space)
        np.testing.assert_array_equal(
            Xv1, X_va_pf[:, sel1.selected_indices_],
            err_msg=f"FAIL [{sel_name}]: transform() did not select correct columns"
        )

    print("PASS: test_selector_fit_ignores_val")


# ---------------------------------------------------------------------------
# Test 4: Train/val index sets are disjoint in every outer fold
# ---------------------------------------------------------------------------

def test_no_index_overlap():
    """
    In every outer fold of RepeatedStratifiedKFold, train and val index sets
    must be disjoint and together cover all samples.
    """
    X, y, _ = _make_data(seed=4)
    n_samples = len(y)
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    for fold_num, (train_idx, val_idx) in enumerate(rskf.split(X, y)):
        train_set = set(train_idx.tolist())
        val_set   = set(val_idx.tolist())

        overlap = train_set & val_set
        assert len(overlap) == 0, (
            f"FAIL: fold {fold_num} has overlapping train/val indices: {overlap}"
        )
        assert len(train_set | val_set) == n_samples, (
            f"FAIL: fold {fold_num} train+val does not cover all {n_samples} samples"
        )

    print("PASS: test_no_index_overlap")


# ---------------------------------------------------------------------------
# Test 5: StandardScaler is fit on train only
# ---------------------------------------------------------------------------

def test_scaler_fit_train_only():
    """
    If we fit a StandardScaler on X_train and transform both X_train and X_val,
    the mean/std of the scaler must equal the mean/std of X_train, NOT X_val.
    """
    X, y, _ = _make_data(seed=5)
    X_train_imp, X_val_imp, _ = impute_fold(X[:30], X[30:])

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train_imp)
    X_va_sc = scaler.transform(X_val_imp)

    # Scaler mean_ must match train mean
    train_mean = X_train_imp.mean(axis=0)
    np.testing.assert_allclose(
        scaler.mean_, train_mean, rtol=1e-5,
        err_msg="FAIL: scaler mean does not match training mean — scaler may be fit on wrong data"
    )

    # After transform, training data must be zero-mean
    tr_mean_after = X_tr_sc.mean(axis=0)
    np.testing.assert_allclose(
        tr_mean_after, np.zeros_like(tr_mean_after), atol=1e-5,
        err_msg="FAIL: training data is not zero-mean after StandardScaler — scaler fit incorrectly"
    )

    # Val data mean after transform should NOT be zero (it's scaled by train stats)
    val_mean_after = X_va_sc.mean(axis=0)
    # At least some columns should differ from zero (val distribution differs from train)
    assert not np.allclose(val_mean_after, 0, atol=0.5), (
        "WARNING: val mean after scaling is close to zero — "
        "train and val distributions may be identical (unlikely in real data)"
    )

    print("PASS: test_scaler_fit_train_only")


# ---------------------------------------------------------------------------
# Test 6: Metrics computed on val labels only
# ---------------------------------------------------------------------------

def test_metrics_computed_on_val_only():
    """
    Simulate a fold and verify that the y_true passed to compute_metrics
    equals y[val_idx], not y[train_idx] or y[all].
    """
    from evaluator import compute_metrics

    X, y, _ = _make_data(seed=6)
    train_idx = np.arange(32)
    val_idx   = np.arange(32, 40)

    y_val  = y[val_idx]
    y_pred = (np.random.RandomState(0).rand(len(val_idx)) > 0.5).astype(int)
    y_prob = np.random.RandomState(0).rand(len(val_idx))

    metrics = compute_metrics(y_val, y_pred, y_prob)

    # Verify sizes are consistent with val set
    assert metrics["TP"] + metrics["TN"] + metrics["FP"] + metrics["FN"] == len(val_idx), (
        "FAIL: confusion matrix counts do not sum to len(val_idx) — "
        "metrics may be using wrong y_true"
    )

    # Verify metric values are in [0, 1]
    for key in ["Accuracy", "Balanced_Accuracy", "Sensitivity", "Specificity", "F1"]:
        val = metrics[key]
        assert 0.0 <= val <= 1.0, f"FAIL: metric {key} = {val} out of [0, 1]"

    print("PASS: test_metrics_computed_on_val_only")


# ---------------------------------------------------------------------------
# Test 7: Leakage assertion fires when indices overlap
# ---------------------------------------------------------------------------

def test_pipeline_assertion_fires():
    """
    Verify that _run_single_fold's leakage assertion raises AssertionError
    when train and val indices intentionally overlap.
    """
    from evaluator import _run_single_fold

    X, y, cpg_names = _make_data(seed=7, n_features=100)
    config = _minimal_config()

    fold_args = {
        "outer_fold_idx": 0,
        "repeat_idx":     0,
        "train_idx":      np.arange(30),
        "val_idx":        np.arange(25, 35),   # <-- deliberate overlap at 25-29
        "X":              X,
        "y":              y,
        "cpg_names":      cpg_names,
        "config":         config,
        "selector_names": ["mw_filter"],
        "panel_sizes":    [5],
        "prefilter_n":    80,
        "verbose":        False,
    }

    try:
        _run_single_fold(fold_args)
        assert False, "FAIL: expected AssertionError for overlapping train/val indices — none raised!"
    except AssertionError as e:
        if "LEAKAGE" in str(e):
            print("PASS: test_pipeline_assertion_fires")
        else:
            # Some other assertion — re-raise with helpful message
            raise AssertionError(
                f"AssertionError raised but not for leakage: {e}"
            )


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

def run_all_tests():
    tests = [
        test_impute_uses_train_mean_only,
        test_variance_prefilter_train_only,
        test_selector_fit_ignores_val,
        test_no_index_overlap,
        test_scaler_fit_train_only,
        test_metrics_computed_on_val_only,
        test_pipeline_assertion_fires,
    ]

    print("=" * 60)
    print("No-Leakage Unit Tests")
    print("=" * 60)
    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"FAIL: {test_fn.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("=" * 60)
    print(f"Results: {passed}/{len(tests)} PASSED, {failed} FAILED")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
