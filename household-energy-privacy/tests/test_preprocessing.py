"""Unit tests for preprocessing, schema adaptation, cadence verification, and lag boundaries."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_loader import adapt_schema
from src.preprocessing import (
    clean_and_validate_time_series,
    construct_features,
)


def test_schema_adaptation_valid() -> None:
    """Verifies that adapt_schema correctly renames variations of column names."""
    df_raw = pd.DataFrame({
        "LCLid": ["MAC000002", "MAC000002"],
        "DateTime": ["2013-01-01 00:00:00", "2013-01-01 00:30:00"],
        "KWH/hh (per half hour) ": [" 0.12 ", "Null"],
    })
    adapted = adapt_schema(df_raw)
    assert list(adapted.columns) == ["household_id", "timestamp", "energy_kwh"]
    assert adapted["household_id"].iloc[0] == "MAC000002"
    assert adapted["energy_kwh"].iloc[0] == pytest.approx(0.12)
    assert np.isnan(adapted["energy_kwh"].iloc[1])


def test_schema_adaptation_missing_column_raises() -> None:
    """Verifies that adapt_schema raises ValueError when an essential column is missing."""
    df_invalid = pd.DataFrame({
        "household": ["H1"],
        "energy": [1.5],
    })
    with pytest.raises(ValueError, match="Missing required canonical fields"):
        adapt_schema(df_invalid)


def test_clean_and_validate_deduplication() -> None:
    """Verifies that duplicate timestamps within a household are removed keeping the first occurrence."""
    df = pd.DataFrame({
        "household_id": ["H1", "H1", "H1"],
        "timestamp": pd.to_datetime(["2013-01-01 00:00:00", "2013-01-01 00:00:00", "2013-01-01 00:30:00"]),
        "energy_kwh": [0.1, 0.99, 0.2],
    })
    cleaned, report = clean_and_validate_time_series(df, expected_cadence_minutes=30)
    assert len(cleaned) == 2
    assert report["duplicate_timestamps_removed"] == 1
    # First occurrence retained
    assert cleaned["energy_kwh"].iloc[0] == 0.1


def test_clean_and_validate_imputation() -> None:
    """Verifies that short missing gaps are forward-filled without silent large-scale drops."""
    df = pd.DataFrame({
        "household_id": ["H1", "H1", "H1", "H1"],
        "timestamp": pd.date_range("2013-01-01 00:00:00", periods=4, freq="30min"),
        "energy_kwh": [0.25, np.nan, np.nan, 0.40],
    })
    cleaned, report = clean_and_validate_time_series(df, expected_cadence_minutes=30)
    assert cleaned["energy_kwh"].isna().sum() == 0
    assert cleaned["energy_kwh"].iloc[1] == pytest.approx(0.25)
    assert cleaned["energy_kwh"].iloc[2] == pytest.approx(0.25)


def test_construct_features_household_boundaries() -> None:
    """Verifies that lag features DO NOT leak across household boundaries."""
    # Two distinct households with 50 readings each
    times = pd.date_range("2013-01-01 00:00:00", periods=50, freq="30min")
    df_h1 = pd.DataFrame({"household_id": "H1", "timestamp": times, "energy_kwh": 1.0})
    df_h2 = pd.DataFrame({"household_id": "H2", "timestamp": times, "energy_kwh": 9.0})
    combined = pd.concat([df_h1, df_h2], ignore_index=True)

    feats = construct_features(combined)
    # The first row of H2 in feats must have lags derived ONLY from H2 (9.0), NEVER from H1 (1.0)
    h2_rows = feats[feats["household_id"] == "H2"]
    assert len(h2_rows) > 0
    assert (h2_rows["lag_1"] == 9.0).all()
    assert (h2_rows["lag_48"] == 9.0).all()
    assert not (h2_rows["lag_1"] == 1.0).any()
