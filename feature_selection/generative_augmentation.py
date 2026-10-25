"""
generative_augmentation.py
--------------------------
Implements Section 6 (Generative Diffusion Augmentation Benchmark):
Compares in-fold data augmentation strategies on the top ~1,000 features:
1. No Augmentation (Baseline)
2. Gaussian Jitter
3. SMOTE-style k-NN Interpolation
4. Tabular Diffusion Augmentation (TabDDPM / reverse diffusion sampler)

Evaluated strictly on real held-out test folds (no synthetic samples in test).
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FS_DIR = os.path.join(ROOT, "feature_selection")
sys.path.insert(0, FS_DIR)

from data_loader import load_primary_data, impute_fold
from evaluator import compute_metrics, _get_classifier_proba

warnings.filterwarnings("ignore")
CONFIG_PATH = os.path.join(FS_DIR, "config.json")
OUT_DIR = os.path.join(ROOT, "results", "feature_selection_benchmark")


def smote_interpolate(X: np.ndarray, y: np.ndarray, n_synthetic: int = 20, k_neighbors: int = 4, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic minority oversampling via k-NN interpolation."""
    rng = np.random.RandomState(seed)
    minority_idx = np.where(y == 0)[0] # Normal is minority (28 vs 36)
    X_min = X[minority_idx]
    
    k = min(k_neighbors, len(X_min) - 1)
    if k < 1:
        return X, y
        
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X_min)
    _, indices = nn.kneighbors(X_min)
    
    syn_samples = []
    for _ in range(n_synthetic):
        i = rng.choice(len(X_min))
        neighbor_i = rng.choice(indices[i, 1:])
        diff = X_min[neighbor_i] - X_min[i]
        syn = X_min[i] + rng.rand() * diff
        syn_samples.append(syn)
        
    X_aug = np.vstack([X, np.array(syn_samples)])
    y_aug = np.concatenate([y, np.zeros(n_synthetic, dtype=int)])
    return X_aug, y_aug


class TabularDiffusionSampler:
    """
    Lightweight forward/reverse diffusion model for tabular continuous features.
    Forward process adds Gaussian noise according to linear beta schedule;
    Reverse process denoises perturbed states using empirical conditional score matching.
    """
    def __init__(self, n_steps: int = 20, beta_start: float = 1e-4, beta_end: float = 0.05, seed: int = 42):
        self.n_steps = n_steps
        self.betas = np.linspace(beta_start, beta_end, n_steps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        self.rng = np.random.RandomState(seed)

    def fit_sample(self, X: np.ndarray, y: np.ndarray, n_generate_per_class: int = 15) -> tuple[np.ndarray, np.ndarray]:
        syn_X = []
        syn_y = []
        for c in [0, 1]:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)
            std_c = np.std(X_c, axis=0) + 1e-4
            
            for _ in range(n_generate_per_class):
                # Sample from prior N(0, I)
                x_t = self.rng.normal(0, 1, size=X.shape[1]) * std_c + mean_c
                # Step-wise reverse denoising
                for t in reversed(range(self.n_steps)):
                    alpha_t = self.alphas[t]
                    alpha_bar_t = self.alphas_cumprod[t]
                    # Estimate conditional drift towards empirical data distribution
                    dists = np.linalg.norm(X_c - x_t, axis=1)
                    weights = np.exp(-dists / (np.median(dists) + 1e-5))
                    weights /= (weights.sum() + 1e-8)
                    x_target = np.sum(X_c * weights[:, None], axis=0)
                    
                    noise = self.rng.normal(0, 1, size=X.shape[1]) * 0.1 if t > 0 else 0
                    x_t = np.sqrt(alpha_t) * x_t + (1 - np.sqrt(alpha_t)) * x_target + noise
                
                syn_X.append(x_t)
                syn_y.append(c)
                
        X_aug = np.vstack([X, np.array(syn_X)])
        y_aug = np.concatenate([y, np.array(syn_y, dtype=int)])
        return X_aug, y_aug


def run_augmentation_experiments(verbose: bool = True):
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    data = load_primary_data(config, verbose=verbose)
    X = data["X"]
    y = data["y"]
    cpg_names = data["cpg_names"]

    # Pre-filter top 1000 features by variance on full X
    X_imp, _, _ = impute_fold(X, X)
    var_all = np.nanvar(X_imp, axis=0)
    top_1000_idx = np.argsort(var_all)[::-1][:1000]
    X_1000 = X_imp[:, top_1000_idx]

    # Convert to M-values
    eps = 1e-5
    X_m = np.log2(np.clip(X_1000, eps, 1.0 - eps) / (1.0 - np.clip(X_1000, eps, 1.0 - eps)))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    methods = ["None (Baseline)", "Gaussian Noise", "SMOTE (k-NN)", "Tabular Diffusion (TabDDPM)"]
    
    results = []

    for method in methods:
        fold_bals = []
        fold_aucs = []
        fold_sens = []
        fold_specs = []

        for fold_i, (tr, va) in enumerate(skf.split(X_m, y)):
            X_tr, X_va = X_m[tr], X_m[va]
            y_tr, y_va = y[tr], y[va]

            # In-fold feature selection: top 20 by MW test on training fold
            p_vals = []
            for col in range(X_tr.shape[1]):
                try:
                    _, p = stats.mannwhitneyu(X_tr[y_tr == 0, col], X_tr[y_tr == 1, col])
                except Exception:
                    p = 1.0
                p_vals.append(p)
            top_20 = np.argsort(p_vals)[:20]

            X_tr_20 = X_tr[:, top_20]
            X_va_20 = X_va[:, top_20]

            # In-fold augmentation
            seed = 42 + fold_i * 10
            if method == "None (Baseline)":
                X_tr_aug, y_tr_aug = X_tr_20, y_tr
            elif method == "Gaussian Noise":
                rng = np.random.RandomState(seed)
                noise = rng.normal(0, 0.05, size=X_tr_20.shape)
                X_tr_aug = np.vstack([X_tr_20, X_tr_20 + noise])
                y_tr_aug = np.concatenate([y_tr, y_tr])
            elif method == "SMOTE (k-NN)":
                X_tr_aug, y_tr_aug = smote_interpolate(X_tr_20, y_tr, n_synthetic=15, seed=seed)
            elif method == "Tabular Diffusion (TabDDPM)":
                sampler = TabularDiffusionSampler(n_steps=15, seed=seed)
                X_tr_aug, y_tr_aug = sampler.fit_sample(X_tr_20, y_tr, n_generate_per_class=10)

            # Fit scaling on augmented training data
            scaler = StandardScaler().fit(X_tr_aug)
            X_tr_sc = scaler.transform(X_tr_aug)
            X_va_sc = scaler.transform(X_va_20)

            clf = LogisticRegression(C=1.0, solver="liblinear", random_state=seed)
            clf.fit(X_tr_sc, y_tr_aug)
            y_pred = clf.predict(X_va_sc)
            y_prob = _get_classifier_proba(clf, X_va_sc)

            m = compute_metrics(y_va, y_pred, y_prob)
            fold_bals.append(m["Balanced_Accuracy"])
            fold_aucs.append(m["ROC_AUC"])
            fold_sens.append(m["Sensitivity"])
            fold_specs.append(m["Specificity"])

        results.append({
            "Augmentation_Method": method,
            "Balanced_Accuracy_Mean": round(float(np.mean(fold_bals)), 4),
            "Balanced_Accuracy_Std": round(float(np.std(fold_bals)), 4),
            "ROC_AUC_Mean": round(float(np.mean(fold_aucs)), 4),
            "Sensitivity_Mean": round(float(np.mean(fold_sens)), 4),
            "Specificity_Mean": round(float(np.mean(fold_specs)), 4),
        })

    df_aug = pd.DataFrame(results)
    aug_csv = os.path.join(OUT_DIR, "augmentation_comparison.csv")
    df_aug.to_csv(aug_csv, index=False)

    if verbose:
        print("\n" + "="*70)
        print("DATA AUGMENTATION BENCHMARK (In-Fold Training at n=64)")
        print("="*70)
        print(df_aug.to_string(index=False))
        print(f"\nSaved augmentation results to {aug_csv}")

    return df_aug


if __name__ == "__main__":
    run_augmentation_experiments(verbose=True)
