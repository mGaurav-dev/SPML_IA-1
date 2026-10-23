"""Differentially Private Regression using IBM Diffprivlib with rigorous privacy accounting."""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from diffprivlib.models import LinearRegression as DPLinearRegression

from .utils import calculate_regression_metrics, utility_loss_percentage

logger = logging.getLogger("household_energy_dp")


def compute_training_data_bounds(
    X_train: np.ndarray,
    y_train: np.ndarray,
    percentiles: Tuple[float, float] = (0.5, 99.5),
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[float, float]]:
    """Derives feature and target clipping bounds strictly from the training partition.

    Mathematical Assumption:
        Diffprivlib LinearRegression requires known compact bounds on feature domain X and
        target domain y to bound the global sensitivity of the Gram matrix X^T X and vector X^T y.
        To avoid test leakage, these bounds are computed solely on training observations.

    Args:
        X_train: Training feature matrix.
        y_train: Training target array.
        percentiles: Lower and upper percentiles for clipping extreme sensor outliers.

    Returns:
        Tuple of (bounds_X, bounds_y) formatted for Diffprivlib.
    """
    p_low, p_high = percentiles
    x_min = np.percentile(X_train, p_low, axis=0)
    x_max = np.percentile(X_train, p_high, axis=0)
    y_min = float(np.percentile(y_train, p_low))
    y_max = float(np.percentile(y_train, p_high))

    # Guard against identical bounds
    x_max = np.where(x_max <= x_min, x_min + 1.0, x_max)
    if y_max <= y_min:
        y_max = y_min + 1.0

    bounds_X = (x_min, x_max)
    bounds_y = (y_min, y_max)
    return bounds_X, bounds_y


class DifferentiallyPrivateRegressor:
    """Wrapper around IBM Diffprivlib's LinearRegression with clipping and timing instrumentation."""

    def __init__(
        self,
        epsilon: float,
        bounds_X: Tuple[np.ndarray, np.ndarray],
        bounds_y: Tuple[float, float],
        random_state: Optional[int] = None,
    ) -> None:
        self.epsilon = float(epsilon)
        self.bounds_X = bounds_X
        self.bounds_y = bounds_y
        self.random_state = random_state
        self.fit_time: float = 0.0
        self.predict_time: float = 0.0

        self.model = DPLinearRegression(
            epsilon=self.epsilon,
            bounds_X=self.bounds_X,
            bounds_y=self.bounds_y,
            random_state=self.random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> DifferentiallyPrivateRegressor:
        start = time.perf_counter()
        self.model.fit(X, y)
        self.fit_time = time.perf_counter() - start
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        start = time.perf_counter()
        preds = self.model.predict(X)
        self.predict_time = time.perf_counter() - start
        return preds


def evaluate_dp_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epsilon: float,
    bounds_X: Tuple[np.ndarray, np.ndarray],
    bounds_y: Tuple[float, float],
    base_rmse: float,
    n_repeats: int = 5,
    base_seed: int = 42,
) -> Tuple[Dict[str, any], np.ndarray]:
    """Evaluates DP linear regression over multiple stochastic repetitions to measure variance.

    Args:
        X_train: Scaled training features.
        y_train: Training target.
        X_test: Scaled test features.
        y_test: Test target.
        epsilon: Differential privacy budget.
        bounds_X: Input feature bounds.
        bounds_y: Target variable bounds.
        base_rmse: Baseline RMSE used to compute utility loss percentage.
        n_repeats: Number of stochastic trials.
        base_seed: Base random seed.

    Returns:
        Tuple of (aggregated metrics dictionary, representative predictions array).
    """
    maes: List[float] = []
    rmses: List[float] = []
    r2s: List[float] = []
    fit_times: List[float] = []
    last_preds: Optional[np.ndarray] = None

    for i in range(n_repeats):
        seed = base_seed + i * 17
        dp_reg = DifferentiallyPrivateRegressor(
            epsilon=epsilon,
            bounds_X=bounds_X,
            bounds_y=bounds_y,
            random_state=seed,
        )
        dp_reg.fit(X_train, y_train)
        preds = dp_reg.predict(X_test)
        metrics = calculate_regression_metrics(y_test, preds)

        maes.append(metrics["mae"])
        rmses.append(metrics["rmse"])
        r2s.append(metrics["r2"])
        fit_times.append(dp_reg.fit_time)
        last_preds = preds

    mean_rmse = float(np.mean(rmses))
    std_rmse = float(np.std(rmses))
    mean_mae = float(np.mean(maes))
    std_mae = float(np.std(maes))
    mean_r2 = float(np.mean(r2s))
    utility_loss = utility_loss_percentage(base_rmse, mean_rmse)

    summary = {
        "epsilon": float(epsilon),
        "mean_mae": round(mean_mae, 5),
        "std_mae": round(std_mae, 5),
        "mean_rmse": round(mean_rmse, 5),
        "std_rmse": round(std_rmse, 5),
        "mean_r2": round(mean_r2, 5),
        "utility_loss_pct": utility_loss,
        "n_repeats": n_repeats,
        "avg_fit_time_sec": round(float(np.mean(fit_times)), 4),
        "protected_unit": "Single 30-min record (x_i, y_i)",
        "adjacency": "Replace one record",
        "guarantee_scope": "Record-level DP; excludes group household protection",
    }
    return summary, last_preds


def run_privacy_utility_sweep(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epsilons: List[float],
    base_rmse: float,
    n_repeats: int = 5,
    random_state: int = 42,
) -> Tuple[List[Dict[str, any]], Dict[float, np.ndarray]]:
    """Sweeps multiple epsilon values to evaluate the empirical privacy-utility curve.

    Args:
        X_train: Scaled training features.
        y_train: Training target.
        X_test: Scaled test features.
        y_test: Test target.
        epsilons: List of epsilon budgets to evaluate.
        base_rmse: Non-private Ridge baseline RMSE.
        n_repeats: Repetitions per epsilon.
        random_state: Random seed.

    Returns:
        Tuple of (list of metrics per epsilon, dict mapping epsilon to representative predictions).
    """
    logger.info("Computing training bounds for DP mechanism...")
    bounds_X, bounds_y = compute_training_data_bounds(X_train, y_train)

    results: List[Dict[str, any]] = []
    predictions_map: Dict[float, np.ndarray] = {}

    for eps in epsilons:
        logger.info("Evaluating DP Linear Regression at epsilon=%.3f (%d repeats)...", eps, n_repeats)
        metrics, preds = evaluate_dp_regression(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            epsilon=eps,
            bounds_X=bounds_X,
            bounds_y=bounds_y,
            base_rmse=base_rmse,
            n_repeats=n_repeats,
            base_seed=random_state,
        )
        results.append(metrics)
        predictions_map[eps] = preds

    return results, predictions_map
