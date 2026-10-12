"""
selectors.py
------------
Seven sklearn-compatible feature selectors for the methylation benchmark.
Each is a BaseEstimator + TransformerMixin that:
  - fit()  on TRAINING data only (no test data ever touches fit)
  - transform() applies the learned mask to any matrix
  - selected_indices_ : array of integer indices into the pre-filtered feature space
  - selected_names_   : list[str] of CpG IDs (set by evaluator after fit)

All selectors work on data AFTER the pre-filter step (typically ~2000-5000 CpGs).

Selectors
---------
1. MannWhitneySelector      — univariate MW-U p-value ranking
2. LassoSelector             — L1 logistic regression, rank by |coef|
3. ElasticNetSelector        — Elastic net logistic regression, rank by |coef|
4. StabilitySelectionSelector— repeated stratified sub-sampling + L1 LR
5. SVMRFESelector            — linear SVM recursive feature elimination
6. MRMRSelector              — greedy max-relevance min-redundancy (numpy)
7. RandomForestSelector      — RF mean-decrease-impurity importance

All use liblinear solver for L1-based methods (no extra packages needed).
"""

import warnings
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from scipy.stats import mannwhitneyu

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _select_topk(scores: np.ndarray, k: int, higher_is_better: bool = True) -> np.ndarray:
    """Return indices of top-k by score. Handles ties by natural index order."""
    if higher_is_better:
        idx = np.argsort(scores)[::-1]
    else:
        idx = np.argsort(scores)
    return np.sort(idx[:k])


# ---------------------------------------------------------------------------
# 1. Mann-Whitney U selector (univariate filter baseline)
# ---------------------------------------------------------------------------

class MannWhitneySelector(BaseEstimator, TransformerMixin):
    """
    Select top-k CpGs by Mann-Whitney U p-value (two-sided).
    Non-parametric, appropriate for bimodal beta values.
    Ranks by ascending p-value (most significant first).
    """

    def __init__(self, k: int = 20):
        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_features = X.shape[1]
        k = min(self.k, n_features)

        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("MannWhitneySelector requires exactly 2 classes.")
        mask0 = y == classes[0]
        mask1 = y == classes[1]

        pvals = np.empty(n_features, dtype=np.float64)
        for j in range(n_features):
            try:
                _, pvals[j] = mannwhitneyu(
                    X[mask0, j], X[mask1, j], alternative="two-sided"
                )
            except ValueError:
                pvals[j] = 1.0  # all ties or all same value

        # rank by p-value ascending
        self.pvalues_ = pvals
        self.selected_indices_ = _select_topk(-pvals, k, higher_is_better=True)
        self.selected_names_: list = []  # filled by evaluator
        self._k_actual = k
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "selected_indices_")
        return X[:, self.selected_indices_]

    def get_scores(self) -> np.ndarray:
        """Return -log10(pvalue) scores for plotting."""
        return -np.log10(np.clip(self.pvalues_, 1e-300, 1.0))


# ---------------------------------------------------------------------------
# 2. Lasso selector (L1 logistic regression)
# ---------------------------------------------------------------------------

class LassoSelector(BaseEstimator, TransformerMixin):
    """
    L1 logistic regression. Sweep over C values; rank all features by |coef|
    from the model where the most non-zero coefficients are at least k.
    If no C gives ≥k non-zero, use the model with the most non-zero features.
    Finally return top-k by |coef|.
    """

    def __init__(self, k: int = 20,
                 C_grid: list = None,
                 max_iter: int = 5000,
                 random_state: int = 42):
        self.k = k
        self.C_grid = C_grid or [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_features = X.shape[1]
        k = min(self.k, n_features)

        # Scale first (required for regularized models)
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)
        self._scaler = scaler

        best_coef = None
        best_C    = None
        best_nnz  = 0

        for C in self.C_grid:
            model = LogisticRegression(
                penalty="l1", C=C, solver="liblinear",
                max_iter=self.max_iter, random_state=self.random_state
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_sc, y)
            coef = np.abs(model.coef_[0])
            nnz  = (coef > 0).sum()
            if best_coef is None or (nnz >= k and (best_nnz < k or C < best_C)):
                # prefer smallest C that gives ≥k non-zero features
                if nnz >= k:
                    best_coef = coef
                    best_C    = C
                    best_nnz  = nnz
            if nnz > best_nnz and nnz < k:
                best_coef = coef
                best_C    = C
                best_nnz  = nnz

        if best_coef is None:
            # fallback: largest C
            model = LogisticRegression(
                penalty="l1", C=self.C_grid[-1], solver="liblinear",
                max_iter=self.max_iter, random_state=self.random_state
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_sc, y)
            best_coef = np.abs(model.coef_[0])
            best_C    = self.C_grid[-1]

        self.coef_abs_ = best_coef
        self.best_C_   = best_C
        self.selected_indices_ = _select_topk(best_coef, k, higher_is_better=True)
        self.selected_names_: list = []
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "selected_indices_")
        return X[:, self.selected_indices_]

    def get_scores(self) -> np.ndarray:
        return self.coef_abs_


# ---------------------------------------------------------------------------
# 3. Elastic Net selector
# ---------------------------------------------------------------------------

class ElasticNetSelector(BaseEstimator, TransformerMixin):
    """
    Elastic net (L1+L2) logistic regression.
    Sweeps over C x l1_ratio grid; selects the (C, l1_ratio) pair that
    gives the most non-zero coefficients ≥ k (preferring sparser models).
    Returns top-k by |coef|.
    Requires sklearn >= 1.0 which supports solver='saga' for elasticnet penalty.
    """

    def __init__(self, k: int = 20,
                 C_grid: list = None,
                 l1_ratio_grid: list = None,
                 max_iter: int = 5000,
                 random_state: int = 42):
        self.k = k
        self.C_grid        = C_grid        or [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
        self.l1_ratio_grid = l1_ratio_grid or [0.3, 0.5, 0.7]
        self.max_iter      = max_iter
        self.random_state  = random_state

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_features = X.shape[1]
        k = min(self.k, n_features)

        scaler = StandardScaler()
        X_sc   = scaler.fit_transform(X)
        self._scaler = scaler

        best_coef     = None
        best_params   = None
        best_nnz      = 0

        for C in self.C_grid:
            for l1r in self.l1_ratio_grid:
                model = LogisticRegression(
                    penalty="elasticnet", C=C, l1_ratio=l1r,
                    solver="saga", max_iter=self.max_iter,
                    random_state=self.random_state
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X_sc, y)
                coef = np.abs(model.coef_[0])
                nnz  = (coef > 0).sum()

                if best_coef is None:
                    best_coef   = coef
                    best_params = (C, l1r)
                    best_nnz    = nnz
                elif nnz >= k and (best_nnz < k or (best_nnz >= k and nnz < best_nnz)):
                    # prefer the setting that gives ≥k but smallest nnz (sparser)
                    best_coef   = coef
                    best_params = (C, l1r)
                    best_nnz    = nnz
                elif nnz > best_nnz and nnz < k:
                    best_coef   = coef
                    best_params = (C, l1r)
                    best_nnz    = nnz

        if best_coef is None:
            best_coef = np.zeros(n_features)

        self.coef_abs_    = best_coef
        self.best_params_ = best_params
        self.selected_indices_ = _select_topk(best_coef, k, higher_is_better=True)
        self.selected_names_: list = []
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "selected_indices_")
        return X[:, self.selected_indices_]

    def get_scores(self) -> np.ndarray:
        return self.coef_abs_


# ---------------------------------------------------------------------------
# 4. Stability Selection
# ---------------------------------------------------------------------------

class StabilitySelectionSelector(BaseEstimator, TransformerMixin):
    """
    Stability selection via repeated stratified sub-sampling + L1 logistic regression.
    Each subsample fits L1 LR with a fixed C and records which features have non-zero coef.
    Features are ranked by their selection frequency across subsamples.
    Top-k by selection frequency are returned.

    Reference: Meinshausen & Bühlmann (2010) J. Royal Stat. Soc. B
    """

    def __init__(self, k: int = 20,
                 n_subsamples: int = 100,
                 subsample_fraction: float = 0.65,
                 C: float = 0.05,
                 max_iter: int = 2000,
                 random_state: int = 42):
        self.k = k
        self.n_subsamples       = n_subsamples
        self.subsample_fraction = subsample_fraction
        self.C                  = C
        self.max_iter           = max_iter
        self.random_state       = random_state

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        k = min(self.k, n_features)

        rng     = np.random.RandomState(self.random_state)
        counts  = np.zeros(n_features, dtype=np.float64)
        n_sub   = max(2, int(n_samples * self.subsample_fraction))

        scaler = StandardScaler()
        X_sc   = scaler.fit_transform(X)

        classes, class_counts = np.unique(y, return_counts=True)
        n_classes = len(classes)

        for i in range(self.n_subsamples):
            # Stratified subsample
            sub_idx = []
            for cls in classes:
                cls_idx  = np.where(y == cls)[0]
                n_take   = max(1, int(len(cls_idx) * self.subsample_fraction))
                chosen   = rng.choice(cls_idx, size=n_take, replace=False)
                sub_idx.extend(chosen)
            sub_idx = np.array(sub_idx)

            X_sub = X_sc[sub_idx]
            y_sub = y[sub_idx]

            # Skip if only one class in subsample
            if len(np.unique(y_sub)) < 2:
                continue

            model = LogisticRegression(
                penalty="l1", C=self.C, solver="liblinear",
                max_iter=self.max_iter, random_state=self.random_state + i
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_sub, y_sub)

            selected = (np.abs(model.coef_[0]) > 0).astype(float)
            counts += selected

        self.selection_frequency_ = counts / self.n_subsamples
        self.selected_indices_ = _select_topk(
            self.selection_frequency_, k, higher_is_better=True
        )
        self.selected_names_: list = []
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "selected_indices_")
        return X[:, self.selected_indices_]

    def get_scores(self) -> np.ndarray:
        return self.selection_frequency_


# ---------------------------------------------------------------------------
# 5. SVM-RFE
# ---------------------------------------------------------------------------

class SVMRFESelector(BaseEstimator, TransformerMixin):
    """
    Linear SVM recursive feature elimination.
    Eliminates step_fraction of the remaining features at each step,
    until k features remain.
    Uses |coef| from LinearSVC as the ranking criterion.
    """

    def __init__(self, k: int = 20,
                 C: float = 1.0,
                 step_fraction: float = 0.2,
                 max_iter: int = 3000,
                 random_state: int = 42):
        self.k             = k
        self.C             = C
        self.step_fraction = step_fraction
        self.max_iter      = max_iter
        self.random_state  = random_state

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        k = min(self.k, n_features)

        scaler = StandardScaler()
        X_sc   = scaler.fit_transform(X)

        # Current active feature indices (into original feature space)
        active = np.arange(n_features)
        X_curr = X_sc.copy()

        while len(active) > k:
            model = LinearSVC(
                C=self.C, max_iter=self.max_iter,
                random_state=self.random_state
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_curr, y)

            importance = np.abs(model.coef_[0])
            n_remove   = max(1, int(len(active) * self.step_fraction))
            # Never remove so many that we go below k
            n_remove   = min(n_remove, len(active) - k)
            # Indices of least important features (in current subspace)
            worst_local = np.argsort(importance)[:n_remove]

            # Build keep mask
            keep_mask = np.ones(len(active), dtype=bool)
            keep_mask[worst_local] = False

            active = active[keep_mask]
            X_curr = X_curr[:, keep_mask]

        # Final ranking within surviving features
        model_final = LinearSVC(
            C=self.C, max_iter=self.max_iter, random_state=self.random_state
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_final.fit(X_curr, y)

        final_importance = np.abs(model_final.coef_[0])
        # sort surviving features by importance descending
        order = np.argsort(final_importance)[::-1]
        active = active[order]

        self.selected_indices_ = np.sort(active[:k])
        self.importance_       = np.zeros(n_features)
        self.importance_[active[:k]] = final_importance[order[:k]]
        self.selected_names_: list = []
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "selected_indices_")
        return X[:, self.selected_indices_]

    def get_scores(self) -> np.ndarray:
        return self.importance_


# ---------------------------------------------------------------------------
# 6. mRMR (Maximum Relevance Minimum Redundancy)
# ---------------------------------------------------------------------------

class MRMRSelector(BaseEstimator, TransformerMixin):
    """
    Greedy maximum-relevance minimum-redundancy feature selection.
    Uses mutual information approximations:
      - Relevance: absolute Pearson correlation with the label (fast proxy for MI)
      - Redundancy: mean absolute Pearson correlation with already-selected features

    Greedy forward selection of k features.
    Operates on the pre-filtered feature space (typically ~2000-5000 features).

    No external package required — pure numpy.
    """

    def __init__(self, k: int = 20, random_state: int = 42):
        self.k = k
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        k = min(self.k, n_features)

        y_float = y.astype(np.float64)

        # Pre-compute all-features correlation with label (relevance)
        # Pearson r, vectorised
        X_mean  = X.mean(axis=0)
        y_mean  = y_float.mean()
        X_c     = X - X_mean[np.newaxis, :]
        y_c     = y_float - y_mean
        num     = (X_c * y_c[:, np.newaxis]).sum(axis=0)
        denom   = np.sqrt((X_c ** 2).sum(axis=0)) * np.sqrt((y_c ** 2).sum())
        with np.errstate(divide="ignore", invalid="ignore"):
            relevance = np.where(denom > 0, np.abs(num / denom), 0.0)

        # Pre-compute feature-feature correlation matrix
        # To avoid O(n_features^2) memory for large n_features, compute lazily
        # For n_features ~ 5000 this is 5000x5000 float32 = 100MB — acceptable
        X_std = X.std(axis=0)
        X_std = np.where(X_std == 0, 1e-10, X_std)
        X_norm = (X - X_mean[np.newaxis, :]) / X_std[np.newaxis, :]
        # corr[i,j] = (X_norm[:,i] @ X_norm[:,j]) / n_samples
        # Compute incrementally to avoid OOM
        # We'll store it lazily as a closure
        _X_norm = X_norm  # captured by closure below

        selected = []
        active = np.ones(n_features, dtype=bool)
        scores = np.zeros(n_features)
        # Keep the running sum of each candidate's absolute correlation with
        # selected probes. This is algebraically the same mean-redundancy
        # score as the former repeated (remaining × selected) matrix product,
        # while avoiding recomputing every old correlation at every step.
        redundancy_sum = np.zeros(n_features, dtype=np.float64)

        for step in range(k):
            if step == 0:
                candidate_scores = relevance.copy()
            else:
                candidate_scores = relevance - redundancy_sum / len(selected)
            candidate_scores[~active] = -np.inf
            best_i = int(np.argmax(candidate_scores))
            scores[best_i] = candidate_scores[best_i]
            selected.append(best_i)
            active[best_i] = False
            corr = (_X_norm.T @ _X_norm[:, best_i]) / n_samples
            redundancy_sum += np.abs(corr)

        self.selected_indices_ = np.array(sorted(selected[:k]))
        self.mrmr_scores_      = scores
        self.selected_names_: list = []
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "selected_indices_")
        return X[:, self.selected_indices_]

    def get_scores(self) -> np.ndarray:
        return self.mrmr_scores_


# ---------------------------------------------------------------------------
# 7. Random Forest importance
# ---------------------------------------------------------------------------

class RandomForestSelector(BaseEstimator, TransformerMixin):
    """
    Random Forest mean-decrease-impurity importance, top k.
    """

    def __init__(self, k: int = 20,
                 n_estimators: int = 200,
                 max_depth: int = 10,
                 min_samples_leaf: int = 2,
                 random_state: int = 42):
        self.k                = k
        self.n_estimators     = n_estimators
        self.max_depth        = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state     = random_state

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_features = X.shape[1]
        k = min(self.k, n_features)

        model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=1,  # outer parallelism is at the fold level
        )
        model.fit(X, y)

        self.feature_importances_ = model.feature_importances_
        self.selected_indices_ = _select_topk(
            self.feature_importances_, k, higher_is_better=True
        )
        self.selected_names_: list = []
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "selected_indices_")
        return X[:, self.selected_indices_]

    def get_scores(self) -> np.ndarray:
        return self.feature_importances_


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def get_selector(name: str, k: int, config: dict, random_state: int = 42):
    """
    Factory function. Returns a fresh (unfitted) selector instance.
    name: one of mw_filter, lasso, elastic_net, stability_selection,
                  svm_rfe, mrmr, rf_importance
    """
    cfg = config.get("selectors", {})

    if name == "mw_filter":
        return MannWhitneySelector(k=k)

    elif name == "lasso":
        c = cfg.get("lasso", {})
        return LassoSelector(
            k=k,
            C_grid=c.get("C_grid", [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]),
            max_iter=c.get("max_iter", 5000),
            random_state=random_state,
        )

    elif name == "elastic_net":
        c = cfg.get("elastic_net", {})
        return ElasticNetSelector(
            k=k,
            C_grid=c.get("C_grid", [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]),
            l1_ratio_grid=c.get("l1_ratio_grid", [0.3, 0.5, 0.7]),
            max_iter=c.get("max_iter", 5000),
            random_state=random_state,
        )

    elif name == "stability_selection":
        c = cfg.get("stability_selection", {})
        return StabilitySelectionSelector(
            k=k,
            n_subsamples=c.get("n_subsamples", 100),
            subsample_fraction=c.get("subsample_fraction", 0.65),
            C=c.get("C", 0.05),
            max_iter=c.get("max_iter", 2000),
            random_state=random_state,
        )

    elif name == "svm_rfe":
        c = cfg.get("svm_rfe", {})
        return SVMRFESelector(
            k=k,
            C=c.get("C", 1.0),
            step_fraction=c.get("step_fraction", 0.2),
            max_iter=3000,
            random_state=random_state,
        )

    elif name == "mrmr":
        return MRMRSelector(k=k, random_state=random_state)

    elif name == "rf_importance":
        c = cfg.get("rf_importance", {})
        return RandomForestSelector(
            k=k,
            n_estimators=c.get("n_estimators", 200),
            max_depth=c.get("max_depth", 10),
            min_samples_leaf=c.get("min_samples_leaf", 2),
            random_state=random_state,
        )

    else:
        raise ValueError(f"Unknown selector: {name!r}. "
                         f"Choose from: mw_filter, lasso, elastic_net, "
                         f"stability_selection, svm_rfe, mrmr, rf_importance")


SELECTOR_NAMES = [
    "mw_filter",
    "lasso",
    "elastic_net",
    "stability_selection",
    "svm_rfe",
    "mrmr",
    "rf_importance",
]

SELECTOR_DISPLAY_NAMES = {
    "mw_filter":           "MW Filter",
    "lasso":               "Lasso (L1)",
    "elastic_net":         "Elastic Net",
    "stability_selection": "Stability Sel.",
    "svm_rfe":             "SVM-RFE",
    "mrmr":                "mRMR",
    "rf_importance":       "RF Importance",
}
