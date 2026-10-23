"""Evaluation routines, plot generation, and export of results tables and figures."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
# Headless backend for headless automated environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import ProjectPaths
from .utils import ensure_directory, save_json

logger = logging.getLogger("household_energy_dp")


def plot_prediction_comparison(
    y_test: np.ndarray,
    baseline_preds: Dict[str, np.ndarray],
    dp_preds: Dict[float, np.ndarray],
    output_path: Path,
    sample_window: int = 144,  # Exactly 3 days (144 half-hours)
) -> Path:
    """Plots actual versus predicted half-hourly electricity consumption over a test window.

    Args:
        y_test: Ground-truth target array.
        baseline_preds: Dictionary of baseline model predictions {'Persistence': array, 'Ridge': array}.
        dp_preds: Dictionary of DP model predictions {epsilon: array}.
        output_path: Output PNG file path.
        sample_window: Number of intervals to visualize.

    Returns:
        Path to the saved figure.
    """
    ensure_directory(output_path.parent)
    window = min(sample_window, len(y_test))
    x_indices = np.arange(window)

    fig, ax = plt.subplots(figsize=(12, 5), dpi=300)

    # Ground truth
    ax.plot(x_indices, y_test[:window], label="Ground Truth (Actual)", color="black", linewidth=1.5, alpha=0.9)

    # Baselines
    if "Persistence" in baseline_preds:
        ax.plot(
            x_indices,
            baseline_preds["Persistence"][:window],
            label="Persistence (Lag-1)",
            color="#2ca02c",
            linestyle="--",
            alpha=0.7,
        )

    if "Ridge" in baseline_preds:
        ax.plot(
            x_indices,
            baseline_preds["Ridge"][:window],
            label="Ridge Baseline (Non-Private)",
            color="#1f77b4",
            linestyle="-.",
            alpha=0.8,
        )

    # DP models: show eps=5.0 and eps=2.0 if available, or available models
    plotted_epsilons = [e for e in [5.0, 2.0, 1.0] if e in dp_preds]
    colors = {5.0: "#9467bd", 2.0: "#d62728", 1.0: "#e377c2"}

    # Determine upper visual clipping limit to preserve plot readability
    y_max_plot = float(np.percentile(y_test[:window], 99.5)) * 1.5

    for eps in plotted_epsilons:
        preds_window = np.clip(dp_preds[eps][:window], 0.0, y_max_plot)
        ax.plot(
            x_indices,
            preds_window,
            label=f"DP Linear Regression (ε = {eps})",
            color=colors.get(eps, "#d62728"),
            linewidth=1.2,
            alpha=0.85,
        )

    ax.set_ylim(-0.05, y_max_plot)

    ax.set_title("Short-Term Household Electricity Consumption: Actual vs Baseline vs DP Predictions", fontsize=12, fontweight="bold")
    ax.set_xlabel("Half-Hourly Time Intervals (3-day evaluation window)", fontsize=10)
    ax.set_ylabel("Electricity Consumption (kWh / 30 min)", fontsize=10)
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved prediction comparison plot to: %s", output_path)
    return output_path


def plot_privacy_utility_tradeoff(
    dp_results: List[Dict[str, any]],
    base_ridge_rmse: float,
    output_path: Path,
) -> Path:
    """Plots empirical privacy-utility trade-off curve (RMSE vs Epsilon) with error bars.

    Args:
        dp_results: List of measured DP experiment summaries.
        base_ridge_rmse: Baseline non-private Ridge RMSE.
        output_path: Output PNG file path.

    Returns:
        Path to the saved figure.
    """
    ensure_directory(output_path.parent)

    epsilons = [r["epsilon"] for r in dp_results]
    mean_rmses = [r["mean_rmse"] for r in dp_results]
    std_rmses = [r["std_rmse"] for r in dp_results]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

    # DP error curve with error bars
    ax.errorbar(
        epsilons,
        mean_rmses,
        yerr=std_rmses,
        fmt="-o",
        color="#d62728",
        ecolor="#ff9896",
        elinewidth=2,
        capsize=4,
        label="DP Linear Regression (Mean ± Std)",
        linewidth=1.8,
    )

    # Non-private reference line
    ax.axhline(
        base_ridge_rmse,
        color="#1f77b4",
        linestyle="--",
        linewidth=1.5,
        label=f"Non-Private Ridge Baseline (RMSE = {base_ridge_rmse:.4f})",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Differential Privacy Privacy–Utility Trade-off Curve", fontsize=12, fontweight="bold")
    ax.set_xlabel("Privacy Budget Epsilon (ε) [log scale]", fontsize=10)
    ax.set_ylabel("Root Mean Squared Error (RMSE in kWh) [log scale]", fontsize=10)
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    ax.grid(True, which="both", linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved privacy-utility trade-off plot to: %s", output_path)
    return output_path


def plot_aggregation_comparison(
    half_hourly_df: pd.DataFrame,
    hourly_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    output_path: Path,
    days_to_plot: int = 14,
) -> Path:
    """Visualizes multi-resolution temporal profiles to illustrate data minimization and smoothing.

    Args:
        half_hourly_df: DataFrame with 30-min data.
        hourly_df: DataFrame with hourly data.
        daily_df: DataFrame with daily data.
        output_path: Output PNG path.
        days_to_plot: Number of days to display.

    Returns:
        Path to the saved figure.
    """
    ensure_directory(output_path.parent)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), dpi=300, sharex=False)

    # Window slices
    hh_slice = half_hourly_df.iloc[: days_to_plot * 48]
    h_slice = hourly_df.iloc[: days_to_plot * 24]
    d_slice = daily_df.iloc[:days_to_plot]

    # Subplot 1: 30-min native
    axes[0].plot(hh_slice["timestamp"], hh_slice["energy_kwh"], color="#1f77b4", linewidth=1.0)
    axes[0].set_title(f"Native 30-Minute Resolution (Full Temporal Detail, {len(hh_slice)} points)", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("kWh / 30min", fontsize=9)
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # Subplot 2: Hourly sum
    axes[1].plot(h_slice["timestamp"], h_slice["energy_kwh"], color="#ff7f0e", linewidth=1.2)
    axes[1].set_title(f"Aggregated Hourly Resolution (Moderate Smoothing, {len(h_slice)} points)", fontsize=10, fontweight="bold")
    axes[1].set_ylabel("kWh / hour", fontsize=9)
    axes[1].grid(True, linestyle=":", alpha=0.6)

    # Subplot 3: Daily sum
    axes[2].bar(d_slice["timestamp"], d_slice["energy_kwh"], color="#2ca02c", width=0.8, alpha=0.8)
    axes[2].set_title(f"Aggregated Daily Resolution (Coarse Totals, {len(d_slice)} points)", fontsize=10, fontweight="bold")
    axes[2].set_ylabel("kWh / day", fontsize=9)
    axes[2].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved aggregation comparison plot to: %s", output_path)
    return output_path


def export_experiment_artifacts(
    paths: ProjectPaths,
    data_quality_report: Dict[str, any],
    baseline_metrics: List[Dict[str, any]],
    dp_results: List[Dict[str, any]],
    ldp_df: pd.DataFrame,
    aggregation_df: pd.DataFrame,
    run_config: Dict[str, any],
) -> Dict[str, Path]:
    """Exports all experiment tables and configurations to standardized results paths.

    Returns:
        Dictionary mapping artifact name to its resolved Path.
    """
    artifacts: Dict[str, Path] = {}

    # 1. Data Quality Report (JSON)
    dq_path = paths.results_tables / "data_quality_report.json"
    save_json(data_quality_report, dq_path)
    artifacts["data_quality_report"] = dq_path

    # 2. Baseline Metrics (CSV)
    base_df = pd.DataFrame(baseline_metrics)
    base_path = paths.results_tables / "baseline_metrics.csv"
    base_df.to_csv(base_path, index=False)
    artifacts["baseline_metrics"] = base_path

    # 3. Privacy Comparison (CSV)
    dp_df = pd.DataFrame(dp_results)
    dp_path = paths.results_tables / "privacy_comparison.csv"
    dp_df.to_csv(dp_path, index=False)
    artifacts["privacy_comparison"] = dp_path

    # 4. LDP Results (CSV)
    ldp_path = paths.results_tables / "ldp_results.csv"
    ldp_df.to_csv(ldp_path, index=False)
    artifacts["ldp_results"] = ldp_path

    # 5. Aggregation Comparison (CSV)
    agg_path = paths.results_tables / "aggregation_comparison.csv"
    aggregation_df.to_csv(agg_path, index=False)
    artifacts["aggregation_comparison"] = agg_path

    # 6. Run Configuration (JSON)
    cfg_path = paths.results_tables / "run_config.json"
    save_json(run_config, cfg_path)
    artifacts["run_config"] = cfg_path

    logger.info("All experimental artifacts successfully exported to %s", paths.results_tables)
    return artifacts
