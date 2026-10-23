"""Unit tests for chronological splitting and leakage-free feature scaling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import scale_features_strictly_train, split_chronologically


def test_chronological_splits_integrity() -> None:
    """Verifies that splits are strictly chronological with zero lookahead leakage."""
    timestamps = pd.date_range("2013-01-01 00:00:00", periods=1000, freq="30min")
    df = pd.DataFrame({
        "household_id": "H1",
        "timestamp": timestamps,
        "energy_kwh": np.random.uniform(0.1, 2.0, 1000),
    })

    train_df, val_df, test_df = split_chronologically(
        df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15
    )

    assert len(train_df) == 700
    assert len(val_df) == 150
    assert len(test_df) == 150

    # Strict chronological boundary check
    assert train_df["timestamp"].max() < val_df["timestamp"].min()
    assert val_df["timestamp"].max() < test_df["timestamp"].min()


def test_split_chronologically_invalid_ratio_raises() -> None:
    """Verifies that non-unity ratio sums raise ValueError."""
    df = pd.DataFrame({"timestamp": pd.date_range("2013-01-01", periods=10, freq="30min")})
    with pytest.raises(ValueError, match="must sum to 1.0"):
        split_chronologically(df, train_ratio=0.5, val_ratio=0.2, test_ratio=0.2)


def test_scale_features_no_test_leakage() -> None:
    """Verifies that StandardScaler is fit strictly on training data."""
    X_train = np.array([[10.0], [20.0], [30.0]])  # mean = 20.0
    X_val = np.array([[100.0]])
    X_test = np.array([[1000.0]])

    X_train_s, X_val_s, X_test_s, scaler = scale_features_strictly_train(X_train, X_val, X_test)

    # Scaler mean must match X_train mean exactly
    assert scaler.mean_[0] == pytest.approx(20.0)
    # X_train scaled should have mean 0
    assert np.mean(X_train_s) == pytest.approx(0.0)
    # X_test scaled must be transformed with train mean (1000 - 20) / std
    expected_test_scaled = (1000.0 - scaler.mean_[0]) / scaler.scale_[0]
    assert X_test_s[0, 0] == pytest.approx(expected_test_scaled)
