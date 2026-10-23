"""Utility functions for logging, seeding, directory handling, and metrics calculation."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def setup_logger(log_file: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a thread-safe logger writing to console and optional log file."""
    logger = logging.getLogger("household_energy_dp")
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def ensure_directory(path: Path) -> Path:
    """Ensures that a directory exists, creating parent directories if necessary."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_random_seed(seed: int = 42) -> None:
    """Sets deterministic random seeds for Python and NumPy."""
    random.seed(seed)
    np.random.seed(seed)


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder handling NumPy scalar and array types."""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def save_json(data: Dict[str, Any], file_path: Path) -> Path:
    """Saves dictionary data to a formatted JSON file."""
    ensure_directory(file_path.parent)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, cls=_NumpyEncoder)
    return file_path


def load_json(file_path: Path) -> Dict[str, Any]:
    """Loads a JSON file into a dictionary."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes MAE, RMSE, and R2 regression metrics.

    Args:
        y_true: Ground-truth target values.
        y_pred: Model predictions.

    Returns:
        Dictionary containing MAE, RMSE, and R2.
    """
    y_true_flat = np.asarray(y_true).ravel()
    y_pred_flat = np.asarray(y_pred).ravel()

    # Drop any NaNs that might occur in predictions
    mask = ~np.isnan(y_pred_flat) & ~np.isnan(y_true_flat)
    y_t = y_true_flat[mask]
    y_p = y_pred_flat[mask]

    if len(y_t) == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan"), "samples": 0}

    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
    r2 = float(r2_score(y_t, y_p))

    return {
        "mae": round(mae, 5),
        "rmse": round(rmse, 5),
        "r2": round(r2, 5),
        "samples": int(len(y_t)),
    }


def utility_loss_percentage(base_rmse: float, dp_rmse: float) -> float:
    """Calculates the percentage increase in RMSE (utility degradation) from differential privacy.

    Formula:
        Utility Loss (%) = ((DP_RMSE - Base_RMSE) / Base_RMSE) * 100

    Args:
        base_rmse: Root Mean Squared Error of the non-private baseline.
        dp_rmse: Root Mean Squared Error of the differentially private model.

    Returns:
        Percentage increase in error (positive indicates utility degradation).
    """
    if base_rmse <= 0.0 or np.isnan(base_rmse):
        return 0.0
    return round(((dp_rmse - base_rmse) / base_rmse) * 100.0, 2)
