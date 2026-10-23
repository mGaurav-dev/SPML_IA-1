"""Main entry point for running the Household Energy Differential Privacy ML experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from src.aggregation_demo import compute_temporal_aggregations
from src.baseline_model import train_and_evaluate_baselines
from src.config import ExperimentConfig, ProjectPaths
from src.data_loader import (
    generate_synthetic_smoke_data,
    load_dataset,
    resolve_dataset_path,
)
from src.dp_model import run_privacy_utility_sweep
from src.evaluation import (
    export_experiment_artifacts,
    plot_aggregation_comparison,
    plot_prediction_comparison,
    plot_privacy_utility_tradeoff,
)
from src.ldp_demo import run_ldp_demonstration
from src.preprocessing import (
    clean_and_validate_time_series,
    construct_features,
    scale_features_strictly_train,
    split_chronologically,
)
from src.utils import set_random_seed, setup_logger


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for experiment configuration."""
    parser = argparse.ArgumentParser(
        description="Evaluating Privacy-Utility Trade-offs in ML on Household Energy Consumption Data Using Differential Privacy."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Optional path to the London Smart Meter CSV file (e.g. data/raw/LCL-June2015v2_0.csv).",
    )
    parser.add_argument(
        "--household-id",
        type=str,
        default="MAC000002",
        help="Target household identifier to analyze (default: MAC000002).",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=25000,
        help="Maximum rows to load for training/evaluation (default: 25000).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run quick synthetic smoke test pipeline for software testing only.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip generating matplotlib figure PNGs (faster CLI execution).",
    )
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
        help="List of epsilon privacy budgets to evaluate for DP regression.",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=5,
        help="Number of stochastic trials per epsilon value (default: 5).",
    )
    return parser.parse_args()


def run_pipeline(config: ExperimentConfig) -> int:
    """Executes the complete experimental pipeline across all 5 syllabus modules."""
    paths = config.paths
    log_file = paths.results_logs / "run.log"
    logger = setup_logger(log_file=log_file)

    logger.info("=" * 80)
    logger.info("IA-1: PRIVACY-UTILITY TRADE-OFFS IN ML ON HOUSEHOLD ENERGY CONSUMPTION")
    logger.info("Course: Security and Privacy for Machine Learning")
    logger.info("=" * 80)

    set_random_seed(config.model.random_state)

    # -------------------------------------------------------------------------
    # STAGE 1: Dataset Identification & Preliminary Loading
    # -------------------------------------------------------------------------
    raw_df: pd.DataFrame
    load_stats: dict

    if config.smoke_test:
        logger.warning("! RUNNING IN SMOKE TEST MODE USING SYNTHETIC DATA !")
        raw_df, load_stats = generate_synthetic_smoke_data(
            n_rows=1500,
            household_id="SYNTHETIC_HH_01",
            random_seed=config.model.random_state,
        )
    else:
        resolved_path = resolve_dataset_path(config.data.data_path, paths)
        if resolved_path is None or not resolved_path.is_file():
            logger.error("Dataset not found! Looked in candidate directories.")
            logger.error(
                "Please place 'LCL-June2015v2_0.csv' in 'data/raw/' or provide '--data-path <path>'."
            )
            logger.error(
                "Download the official London Datastore dataset from: "
                "https://data.london.gov.uk/dataset/smartmeter-energy-consumption-data-in-london-households-vqm0d"
            )
            logger.info("Alternatively, run with '--smoke-test' to verify software functionality.")
            return 1

        raw_df, load_stats = load_dataset(
            file_path=resolved_path,
            target_household=config.data.target_household,
            max_rows=config.data.subset_size,
            chunk_size=config.data.chunk_size,
        )

    # -------------------------------------------------------------------------
    # STAGE 2: Preprocessing, Cadence Verification, and Chronological Split
    # -------------------------------------------------------------------------
    logger.info("--- STAGE 2: Preprocessing and Chronological Splitting ---")
    clean_df, data_quality = clean_and_validate_time_series(raw_df, expected_cadence_minutes=30)
    data_quality.update(load_stats)

    # Feature engineering (strictly past-looking)
    feature_df = construct_features(clean_df, feature_names=config.model.features)

    # Save a sample of processed data
    processed_sample_path = paths.data_processed / "processed_features_sample.csv"
    feature_df.head(1000).to_csv(processed_sample_path, index=False)
    logger.info("Saved processed sample snapshot to %s", processed_sample_path)

    # Chronological split
    train_df, val_df, test_df = split_chronologically(
        feature_df,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
    )

    feature_cols = config.model.features
    target_col = config.model.target

    X_train_raw = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_val_raw = val_df[feature_cols].values
    y_val = val_df[target_col].values
    X_test_raw = test_df[feature_cols].values
    y_test = test_df[target_col].values

    # Feature scaling (fit ONLY on train)
    X_train, X_val, X_test, scaler = scale_features_strictly_train(
        X_train_raw, X_val_raw, X_test_raw
    )

    # -------------------------------------------------------------------------
    # STAGE 3: Baseline Experiments (Persistence and Non-Private Ridge)
    # -------------------------------------------------------------------------
    logger.info("--- STAGE 3: Baseline Model Training ---")
    lag1_idx = feature_cols.index("lag_1")
    persist_metrics, ridge_metrics, y_pred_persist, y_pred_ridge = train_and_evaluate_baselines(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        lag1_index_unscaled=lag1_idx,
        X_test_unscaled=X_test_raw,
        alpha=config.model.ridge_alpha,
        random_state=config.model.random_state,
    )

    baseline_metrics_list = [persist_metrics, ridge_metrics]
    base_ridge_rmse = ridge_metrics["rmse"]

    # -------------------------------------------------------------------------
    # STAGE 4: Differential Privacy ML Experiment (IBM Diffprivlib)
    # -------------------------------------------------------------------------
    logger.info("--- STAGE 4: Differential Privacy Regression Experiment ---")
    dp_results, dp_predictions_map = run_privacy_utility_sweep(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        epsilons=config.dp.epsilons,
        base_rmse=base_ridge_rmse,
        n_repeats=config.dp.n_repeats,
        random_state=config.dp.random_state,
    )

    # -------------------------------------------------------------------------
    # STAGE 5: Local Differential Privacy (LDP) Demonstration
    # -------------------------------------------------------------------------
    logger.info("--- STAGE 5: Local Differential Privacy (k-RR) Demonstration ---")
    ldp_df, ldp_summary = run_ldp_demonstration(
        y_train=y_train,
        y_test=y_test,
        num_bins=config.ldp.num_bins,
        bin_labels=config.ldp.bin_labels,
        epsilons=config.ldp.epsilons,
        random_state=config.ldp.random_state,
    )

    # -------------------------------------------------------------------------
    # STAGE 6: Privacy-Preserving Data Mining (PPDM) Aggregation Demo
    # -------------------------------------------------------------------------
    logger.info("--- STAGE 6: Privacy-Preserving Data Mining Aggregation Demo ---")
    hh_agg, h_agg, d_agg, aggregation_metrics_df = compute_temporal_aggregations(clean_df)

    # -------------------------------------------------------------------------
    # STAGE 7: Plot Generation & Artifact Export
    # -------------------------------------------------------------------------
    logger.info("--- STAGE 7: Generating Figures and Exporting Artifacts ---")

    if not config.skip_plots:
        # Plot 1: Actual vs Predictions
        plot_prediction_comparison(
            y_test=y_test,
            baseline_preds={"Persistence": y_pred_persist, "Ridge": y_pred_ridge},
            dp_preds=dp_predictions_map,
            output_path=paths.results_figures / "prediction_comparison.png",
        )

        # Plot 2: Privacy-Utility Trade-off Curve
        plot_privacy_utility_tradeoff(
            dp_results=dp_results,
            base_ridge_rmse=base_ridge_rmse,
            output_path=paths.results_figures / "privacy_utility_plot.png",
        )

        # Plot 3: Aggregation Comparison
        plot_aggregation_comparison(
            half_hourly_df=hh_agg,
            hourly_df=h_agg,
            daily_df=d_agg,
            output_path=paths.results_figures / "aggregation_comparison.png",
        )

    # Export CSV and JSON artifacts
    run_config_dict = {
        "dataset_path": str(config.data.data_path or "auto-detected"),
        "target_household": config.data.target_household,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "features": config.model.features,
        "target": config.model.target,
        "dp_epsilons": config.dp.epsilons,
        "dp_repeats": config.dp.n_repeats,
        "protected_unit": config.dp.protected_unit,
        "adjacency_definition": config.dp.adjacency_definition,
        "formal_guarantee": config.dp.formal_guarantee,
        "smoke_test": config.smoke_test,
    }

    artifacts = export_experiment_artifacts(
        paths=paths,
        data_quality_report=data_quality,
        baseline_metrics=baseline_metrics_list,
        dp_results=dp_results,
        ldp_df=ldp_df,
        aggregation_df=aggregation_metrics_df,
        run_config=run_config_dict,
    )

    logger.info("=" * 80)
    logger.info("EXPERIMENT RUN COMPLETE. SUMMARY OF RESULTS:")
    logger.info("=" * 80)
    logger.info("Baseline Persistence RMSE: %.4f", persist_metrics["rmse"])
    logger.info("Baseline Ridge RMSE:       %.4f", ridge_metrics["rmse"])
    for r in dp_results:
        logger.info(
            "DP Linear Regression (ε = %-5.2f): RMSE = %.4f ± %.4f (ΔRMSE = %+6.2f%%)",
            r["epsilon"],
            r["mean_rmse"],
            r["std_rmse"],
            r["utility_loss_pct"],
        )
    logger.info("=" * 80)
    logger.info("All tables and figures saved to: %s", paths.results_tables.parent)
    return 0


def main() -> int:
    args = parse_arguments()
    config = ExperimentConfig()

    if args.data_path:
        config.data.data_path = args.data_path
    if args.household_id:
        config.data.target_household = args.household_id
    if args.subset_size:
        config.data.subset_size = args.subset_size
    if args.smoke_test:
        config.smoke_test = True
    if args.skip_plots:
        config.skip_plots = True
    if args.epsilons:
        config.dp.epsilons = args.epsilons
    if args.n_repeats:
        config.dp.n_repeats = args.n_repeats

    return run_pipeline(config)


if __name__ == "__main__":
    sys.exit(main())
