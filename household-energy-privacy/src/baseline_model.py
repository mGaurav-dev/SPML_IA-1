"""Baseline forecasting models: Naive Persistence and Non-Private Ridge Regression."""

from __future__ import annotations

import logging
import time
from typing import Dict, Tuple

import numpy as np
from sklearn.linear_model import Ridge

from .utils import calculate_regression_metrics

logger = logging.getLogger("household_energy_dp")


class PersistenceBaseline:
    """Naive persistence baseline predictor for time series.

    Predicts the next interval's consumption as the most recently observed value:
        y_hat[t] = y[t-1] (lag_1 feature)

    Characteristics:
        - Parameter-free (no fitting phase, zero training runtime).
        - Serves as the operational baseline standard in short-term energy forecasting.
        - Captures short-term autocorrelation but cannot adapt to diurnal shifts or seasonal patterns.
    """

    def __init__(self, lag_feature_index: int = 0) -> None:
        self.lag_feature_index = lag_feature_index
        self.fit_time: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> PersistenceBaseline:
        """Fitting is a no-op for persistence."""
        start = time.perf_counter()
        # Parameter-free
        self.fit_time = time.perf_counter() - start
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts the lag-1 value from the feature matrix."""
        # Returns lag_1 directly
        return X[:, self.lag_feature_index]


class RidgeBaseline:
    """Scikit-Learn Ridge Regression (L2-regularized linear model).

    Models consumption as a linear combination of past lags and calendar features:
        y_hat = X * w + b

    Solves:
        min_w ||Xw - y||_2^2 + alpha * ||w||_2^2
    """

    def __init__(self, alpha: float = 1.0, random_state: int = 42) -> None:
        self.alpha = alpha
        self.random_state = random_state
        self.model = Ridge(alpha=alpha, random_state=random_state)
        self.fit_time: float = 0.0
        self.predict_time: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> RidgeBaseline:
        start = time.perf_counter()
        self.model.fit(X, y)
        self.fit_time = time.perf_counter() - start
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        start = time.perf_counter()
        preds = self.model.predict(X)
        self.predict_time = time.perf_counter() - start
        return preds


def train_and_evaluate_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    lag1_index_unscaled: int = 0,
    X_test_unscaled: np.ndarray | None = None,
    alpha: float = 1.0,
    random_state: int = 42,
) -> Tuple[Dict[str, any], Dict[str, any], np.ndarray, np.ndarray]:
    """Fits and evaluates the Persistence baseline and the Ridge regression baseline.

    Args:
        X_train: Scaled training features.
        y_train: Training targets.
        X_test: Scaled test features.
        y_test: Test targets.
        lag1_index_unscaled: Index of lag_1 feature in feature matrix.
        X_test_unscaled: Unscaled test features (used for persistence baseline so predictions are in kWh).
        alpha: Regularization strength for Ridge.
        random_state: Random seed.

    Returns:
        Tuple of (persistence_metrics, ridge_metrics, persistence_preds, ridge_preds).
    """
    logger.info("Training non-private baseline models...")

    # 1. Persistence Baseline
    persistence = PersistenceBaseline(lag_feature_index=lag1_index_unscaled)
    # If unscaled features are provided, use them so lag_1 is in physical units (kWh)
    eval_X_persist = X_test_unscaled if X_test_unscaled is not None else X_test
    t0 = time.perf_counter()
    y_pred_persist = persistence.predict(eval_X_persist)
    persist_pred_time = time.perf_counter() - t0

    persist_metrics = calculate_regression_metrics(y_test, y_pred_persist)
    persist_metrics.update({
        "model": "Persistence (Lag-1)",
        "fit_time_sec": round(persistence.fit_time, 6),
        "predict_time_sec": round(persist_pred_time, 6),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "privacy_mechanism": "None (Non-private deterministic baseline)",
        "epsilon": float("inf"),
    })

    # 2. Ridge Regression Baseline
    ridge = RidgeBaseline(alpha=alpha, random_state=random_state)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)

    ridge_metrics = calculate_regression_metrics(y_test, y_pred_ridge)
    ridge_metrics.update({
        "model": "Ridge Regression (Non-Private)",
        "fit_time_sec": round(ridge.fit_time, 6),
        "predict_time_sec": round(ridge.predict_time, 6),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "privacy_mechanism": "None (Non-private Empirical Risk Minimization)",
        "epsilon": float("inf"),
        "alpha": alpha,
    })

    logger.info(
        "Baseline results: Persistence RMSE=%.4f, Ridge RMSE=%.4f, Ridge R2=%.4f",
        persist_metrics["rmse"],
        ridge_metrics["rmse"],
        ridge_metrics["r2"],
    )
    return persist_metrics, ridge_metrics, y_pred_persist, y_pred_ridge
