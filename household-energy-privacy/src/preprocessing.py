"""Preprocessing, validation, chronological splitting, and leakage-free feature engineering."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("household_energy_dp")


def clean_and_validate_time_series(
    df: pd.DataFrame,
    expected_cadence_minutes: int = 30,
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """Cleans raw time series data, handles duplicates, validates cadence, and reports quality.

    Steps:
        1. Count initial records and missing values.
        2. Drop invalid timestamps (NaT).
        3. Sort by household_id and timestamp chronologically.
        4. Detect and remove duplicate timestamps per household (keeps first).
        5. Verify cadence (computes timedelta between consecutive readings).
        6. Impute short missing value gaps using forward-fill (limit=2), logging counts.

    Args:
        df: Input DataFrame with ['household_id', 'timestamp', 'energy_kwh'].
        expected_cadence_minutes: Expected interval duration in minutes (default 30).

    Returns:
        Tuple of (cleaned DataFrame, quality report dictionary).
    """
    initial_rows = len(df)
    initial_nulls = int(df["energy_kwh"].isna().sum())
    initial_nat_timestamps = int(df["timestamp"].isna().sum())

    # Step 1: Remove invalid timestamps
    cleaned = df.dropna(subset=["timestamp"]).copy()
    rows_dropped_bad_timestamps = initial_rows - len(cleaned)

    # Step 2: Sort chronologically within each household
    cleaned = cleaned.sort_values(by=["household_id", "timestamp"]).reset_index(drop=True)

    # Step 3: Deduplication
    duplicate_mask = cleaned.duplicated(subset=["household_id", "timestamp"], keep="first")
    num_duplicates = int(duplicate_mask.sum())
    if num_duplicates > 0:
        logger.warning("Detected %d duplicate timestamps. Removing duplicates keeping first occurrence.", num_duplicates)
        cleaned = cleaned[~duplicate_mask].reset_index(drop=True)

    # Step 4: Verify Cadence
    cleaned["time_delta"] = cleaned.groupby("household_id")["timestamp"].diff()
    expected_delta = pd.Timedelta(minutes=expected_cadence_minutes)
    valid_cadence_mask = cleaned["time_delta"] == expected_delta
    # The first row of each household naturally has NaN time_delta
    first_rows_per_hh = cleaned.groupby("household_id").head(1).index
    cadence_checked = cleaned.drop(index=first_rows_per_hh)
    cadence_conformity_ratio = (
        float((cadence_checked["time_delta"] == expected_delta).mean())
        if len(cadence_checked) > 0
        else 1.0
    )
    cleaned = cleaned.drop(columns=["time_delta"])

    # Step 5: Missing Value Handling
    # Short gaps (sensor dropout <= 2 intervals) are forward-filled per household
    missing_before_impute = int(cleaned["energy_kwh"].isna().sum())
    cleaned["energy_kwh"] = (
        cleaned.groupby("household_id")["energy_kwh"]
        .transform(lambda s: s.ffill(limit=2))
    )
    # Any residual missing values (e.g. at the start of series) use median fallback
    residual_missing = int(cleaned["energy_kwh"].isna().sum())
    if residual_missing > 0:
        global_median = float(cleaned["energy_kwh"].median())
        cleaned["energy_kwh"] = cleaned["energy_kwh"].fillna(global_median)

    quality_report = {
        "initial_raw_rows": initial_rows,
        "rows_dropped_invalid_timestamp": rows_dropped_bad_timestamps,
        "duplicate_timestamps_removed": num_duplicates,
        "initial_null_energy_values": initial_nulls,
        "missing_energy_values_imputed_ffill": missing_before_impute - residual_missing,
        "residual_missing_values_imputed_median": residual_missing,
        "final_cleaned_rows": len(cleaned),
        "cadence_minutes": expected_cadence_minutes,
        "cadence_conformity_ratio": round(cadence_conformity_ratio, 4),
        "min_timestamp": str(cleaned["timestamp"].min()),
        "max_timestamp": str(cleaned["timestamp"].max()),
        "energy_kwh_min": round(float(cleaned["energy_kwh"].min()), 4),
        "energy_kwh_median": round(float(cleaned["energy_kwh"].median()), 4),
        "energy_kwh_max": round(float(cleaned["energy_kwh"].max()), 4),
    }

    logger.info(
        "Data validation complete: %d rows retained. Cadence conformity: %.2f%%. Imputed nulls: %d.",
        len(cleaned),
        cadence_conformity_ratio * 100,
        missing_before_impute,
    )
    return cleaned, quality_report


def construct_features(
    df: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Constructs strictly past-looking lag and calendar features without lookahead leakage.

    Features constructed per household:
        - lag_1: Consumption at t-1 (previous 30 minutes)
        - lag_2: Consumption at t-2 (1 hour prior)
        - lag_48: Consumption at t-48 (exactly 24 hours prior)
        - rolling_mean_4: Rolling average of past 4 intervals [t-4 to t-1]
        - hour: Hour of interval start (0 to 23)
        - day_of_week: Day of week (0=Monday to 6=Sunday)
        - is_weekend: Binary indicator (1 for Sat/Sun, 0 otherwise)
        - month: Month of year (1 to 12)

    Target:
        - energy_kwh: Consumption at interval t.

    Args:
        df: Cleaned DataFrame with ['household_id', 'timestamp', 'energy_kwh'].
        feature_names: Optional explicit list of feature names to keep.

    Returns:
        DataFrame with feature columns and target column, with initial lag NaNs dropped.
    """
    df_feat = df.copy()

    # Lags must be computed strictly WITHIN household groups to avoid boundary contamination
    hh_grouped = df_feat.groupby("household_id")["energy_kwh"]

    df_feat["lag_1"] = hh_grouped.shift(1)
    df_feat["lag_2"] = hh_grouped.shift(2)
    df_feat["lag_48"] = hh_grouped.shift(48)
    # Rolling mean over past 4 readings (t-4 to t-1), strictly excluding current value t
    df_feat["rolling_mean_4"] = (
        hh_grouped.shift(1).rolling(window=4, min_periods=1).mean()
    )

    # Valid time features derived from timestamp
    df_feat["hour"] = df_feat["timestamp"].dt.hour
    df_feat["day_of_week"] = df_feat["timestamp"].dt.dayofweek
    df_feat["is_weekend"] = (df_feat["day_of_week"] >= 5).astype(int)
    df_feat["month"] = df_feat["timestamp"].dt.month

    # Drop rows with NaN caused by lag construction (e.g. first 48 half-hours)
    rows_before_drop = len(df_feat)
    df_feat = df_feat.dropna().reset_index(drop=True)
    rows_dropped_lags = rows_before_drop - len(df_feat)
    logger.info("Constructed features. Dropped %d initial boundary rows with undefined lags.", rows_dropped_lags)

    return df_feat


def split_chronologically(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Partitions time series chronologically into train, validation, and test splits.

    Guarantees:
        max(train.timestamp) < min(val.timestamp) <= max(val.timestamp) < min(test.timestamp)
        Zero temporal leakage or random shuffling.

    Args:
        df: Feature DataFrame with 'timestamp'.
        train_ratio: Proportion of data allocated to training.
        val_ratio: Proportion of data allocated to validation.
        test_ratio: Proportion of data allocated to test.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-5:
        raise ValueError("Train, validation, and test ratios must sum to 1.0")

    # Ensure chronological order
    df_sorted = df.sort_values(by="timestamp").reset_index(drop=True)
    n = len(df_sorted)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df_sorted.iloc[:train_end].copy().reset_index(drop=True)
    val_df = df_sorted.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = df_sorted.iloc[val_end:].copy().reset_index(drop=True)

    # Verify chronological separation
    assert train_df["timestamp"].max() < val_df["timestamp"].min(), "Train and validation splits overlap!"
    assert val_df["timestamp"].max() < test_df["timestamp"].min(), "Validation and test splits overlap!"

    logger.info(
        "Chronological split complete: Train=%d (%s to %s), Val=%d (%s to %s), Test=%d (%s to %s)",
        len(train_df),
        train_df["timestamp"].min(),
        train_df["timestamp"].max(),
        len(val_df),
        val_df["timestamp"].min(),
        val_df["timestamp"].max(),
        len(test_df),
        test_df["timestamp"].min(),
        test_df["timestamp"].max(),
    )
    return train_df, val_df, test_df


def scale_features_strictly_train(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Fits StandardScaler ONLY on training features and transforms validation/test sets.

    Guarantees:
        No statistics from validation or test sets are used in feature normalization.

    Args:
        X_train: Training feature matrix.
        X_val: Validation feature matrix.
        X_test: Test feature matrix.

    Returns:
        Tuple of (X_train_scaled, X_val_scaled, X_test_scaled, fitted scaler).
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler
