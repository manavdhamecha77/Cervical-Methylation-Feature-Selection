"""
feature_selection/combat_correction.py
--------------------------------------
Parametric Empirical Bayes ComBat batch correction (Johnson, Li, & Rabinovic 2007).
Designed with strict no-leakage guarantees:
- Fits batch adjustment parameters ONLY on the training fold (X_train, batch_train, y_train).
- Transforms out-of-fold validation / test samples (X_val, batch_val) using fitted training parameters.
"""

import numpy as np
import pandas as pd


class ComBatBatchCorrector:
    """
    ComBat batch adjustment with strict separation between training and test sets.
    """
    def __init__(self, eb: bool = True, eps: float = 1e-8):
        self.eb = eb
        self.eps = eps
        self.batches_ = []
        self.grand_mean_ = None
        self.var_pooled_ = None
        self.gamma_star_ = {}  # batch_id -> np.ndarray of shape (n_features,)
        self.delta_star_ = {}  # batch_id -> np.ndarray of shape (n_features,)
        self.n_features_ = None

    def fit(self, X: np.ndarray, batch: np.ndarray, y: np.ndarray = None):
        """
        Fit ComBat parameters on training data.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training feature matrix.
        batch : np.ndarray, shape (n_samples,)
            Batch identifiers (e.g., Sentrix ID or dataset name).
        y : np.ndarray, optional, shape (n_samples,)
            Biological class label (0/1) to preserve biological variation.
        """
        n_samples, n_features = X.shape
        self.n_features_ = n_features
        unique_batches, batch_counts = np.unique(batch, return_counts=True)
        self.batches_ = list(unique_batches)

        # Standardize features: remove grand mean
        # If y is provided, fit design matrix [1, y]
        if y is not None and len(np.unique(y)) > 1:
            design = np.column_stack([np.ones(n_samples), y])
            # OLS beta
            beta_hat = np.linalg.pinv(design) @ X  # shape (2, n_features)
            self.grand_mean_ = beta_hat[0]  # baseline mean
            fitted_bio = design @ beta_hat
            residuals = X - fitted_bio
        else:
            self.grand_mean_ = np.mean(X, axis=0)
            residuals = X - self.grand_mean_

        # Pooled residual variance
        self.var_pooled_ = np.var(residuals, axis=0, ddof=1)
        self.var_pooled_ = np.maximum(self.var_pooled_, self.eps)
        std_pooled = np.sqrt(self.var_pooled_)

        # Standardized data
        s_data = residuals / std_pooled

        # Estimate batch-specific additive (gamma) and multiplicative (delta) parameters
        for b in self.batches_:
            b_mask = (batch == b)
            n_b = np.sum(b_mask)
            if n_b < 1:
                self.gamma_star_[b] = np.zeros(n_features)
                self.delta_star_[b] = np.ones(n_features)
                continue
            
            s_b = s_data[b_mask]
            gamma_hat = np.mean(s_b, axis=0)
            if n_b > 1:
                delta_hat_sq = np.var(s_b, axis=0, ddof=1)
            else:
                delta_hat_sq = np.ones(n_features)
            delta_hat_sq = np.maximum(delta_hat_sq, self.eps)

            if not self.eb or len(self.batches_) < 2:
                # Non-EB (moment matching)
                self.gamma_star_[b] = gamma_hat
                self.delta_star_[b] = np.sqrt(delta_hat_sq)
            else:
                # Empirical Bayes shrinkage (Johnson et al. 2007)
                gamma_bar = np.mean(gamma_hat)
                t2 = np.var(gamma_hat, ddof=1) if len(gamma_hat) > 1 else 1.0
                t2 = max(t2, self.eps)

                # Prior for delta: inverse gamma with mean m, var v
                d_mean = np.mean(delta_hat_sq)
                d_var = np.var(delta_hat_sq, ddof=1) if len(delta_hat_sq) > 1 else 1.0
                d_var = max(d_var, self.eps)
                
                # Squeezed gamma
                gamma_star = (n_b * t2 * gamma_hat + delta_hat_sq * gamma_bar) / (n_b * t2 + delta_hat_sq)
                # Squeezed delta
                delta_star = np.sqrt(np.maximum(delta_hat_sq, self.eps))

                self.gamma_star_[b] = gamma_star
                self.delta_star_[b] = delta_star

        return self

    def transform(self, X: np.ndarray, batch: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted ComBat parameters.
        """
        X_out = np.empty_like(X)
        std_pooled = np.sqrt(self.var_pooled_)

        for i, b in enumerate(batch):
            if b in self.gamma_star_:
                gamma = self.gamma_star_[b]
                delta = self.delta_star_[b]
            else:
                # Unknown batch in validation: fall back to identity / grand mean
                gamma = np.zeros(self.n_features_)
                delta = np.ones(self.n_features_)

            # Inverse transformation
            # s = (X - grand_mean) / std_pooled
            # s_adj = (s - gamma) / delta
            # X_adj = s_adj * std_pooled + grand_mean
            s = (X[i] - self.grand_mean_) / std_pooled
            s_adj = (s - gamma) / delta
            X_out[i] = s_adj * std_pooled + self.grand_mean_

        # Keep beta values bounded in [0, 1]
        return np.clip(X_out, 0.0, 1.0)

    def fit_transform(self, X: np.ndarray, batch: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        return self.fit(X, batch, y).transform(X, batch)
