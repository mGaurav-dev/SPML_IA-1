"""Unit tests for pipeline execution, output creation, missing data failure, and DP validation."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from src.config import DPConfig, ExperimentConfig, ProjectPaths
from src.data_loader import resolve_dataset_path
from src.dp_model import compute_training_data_bounds, DifferentiallyPrivateRegressor
from src.evaluation import export_experiment_artifacts
import pandas as pd


def test_missing_data_resolution_failure() -> None:
    """Verifies that resolve_dataset_path returns None when given a non-existent path."""
    resolved = resolve_dataset_path("non_existent_directory_or_file_12345.csv")
    assert resolved is None


def test_dp_config_validation() -> None:
    """Verifies default DP configuration attributes and parameters."""
    dp_cfg = DPConfig()
    assert 1.0 in dp_cfg.epsilons
    assert dp_cfg.target_epsilon == 1.0
    assert dp_cfg.n_repeats >= 1
    assert "Single 30-minute" in dp_cfg.protected_unit
    assert "Bounded" in dp_cfg.adjacency_definition


def test_dp_bounds_computation_strictly_train() -> None:
    """Verifies that compute_training_data_bounds returns valid min/max tuples for X and y."""
    X_train = np.array([
        [1.0, 10.0],
        [2.0, 20.0],
        [3.0, 30.0],
        [4.0, 40.0],
    ])
    y_train = np.array([0.5, 1.5, 2.5, 3.5])

    bounds_X, bounds_y = compute_training_data_bounds(X_train, y_train, percentiles=(0.0, 100.0))

    assert len(bounds_X) == 2
    assert np.all(bounds_X[0] <= bounds_X[1])
    assert bounds_y[0] < bounds_y[1]
    assert bounds_y[0] == pytest.approx(0.5)
    assert bounds_y[1] == pytest.approx(3.5)


def test_dp_model_fit_and_predict(tmp_path: Path) -> None:
    """Verifies that DifferentiallyPrivateRegressor fits and predicts valid numerical arrays."""
    X_train = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y_train = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
    bounds_X = (np.array([1.0]), np.array([5.0]))
    bounds_y = (1.0, 6.0)

    dp_reg = DifferentiallyPrivateRegressor(epsilon=1.0, bounds_X=bounds_X, bounds_y=bounds_y, random_state=42)
    dp_reg.fit(X_train, y_train)

    X_test = np.array([[2.5], [3.5]])
    preds = dp_reg.predict(X_test)

    assert preds.shape == (2,)
    assert not np.isnan(preds).any()
    assert dp_reg.fit_time >= 0.0
    assert dp_reg.predict_time >= 0.0


def test_output_file_creation(tmp_path: Path) -> None:
    """Verifies that export_experiment_artifacts creates all required CSV and JSON output files."""
    paths = ProjectPaths(root_dir=tmp_path)
    data_quality = {"status": "ok", "rows": 100}
    baseline_metrics = [{"model": "Persistence", "rmse": 0.25}]
    dp_results = [{"epsilon": 1.0, "mean_rmse": 0.28, "std_rmse": 0.02, "utility_loss_pct": 12.0}]
    ldp_df = pd.DataFrame([{"epsilon_ldp": 1.0, "mae_proportion_error": 0.01}])
    agg_df = pd.DataFrame([{"resolution": "30min", "record_count": 100}])
    run_cfg = {"tested": True}

    artifacts = export_experiment_artifacts(
        paths=paths,
        data_quality_report=data_quality,
        baseline_metrics=baseline_metrics,
        dp_results=dp_results,
        ldp_df=ldp_df,
        aggregation_df=agg_df,
        run_config=run_cfg,
    )

    for name, path in artifacts.items():
        assert path.exists(), f"Expected artifact {name} at {path} does not exist!"
        assert path.stat().st_size > 0, f"Artifact {name} at {path} is empty!"
