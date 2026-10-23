"""Data loading, schema adaptation, chunked reading, and smoke dataset generation."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import ProjectPaths

logger = logging.getLogger("household_energy_dp")


# Canonical schema dictionary mapping canonical names to known alias patterns
SCHEMA_ALIASES = {
    "household_id": ["lclid", "household_id", "customer_id", "id", "meter_id", "household"],
    "timestamp": ["datetime", "timestamp", "date", "date_time", "local_time", "time"],
    "energy_kwh": [
        "kwh/hh (per half hour)",
        "kwh/hh (per half hour) ",
        "energy_kwh",
        "kwh",
        "consumption_kwh",
        "usage_kwh",
        "energy",
        "consumption",
    ],
}


def _clean_column_name(col: str) -> str:
    """Strips whitespace and non-alphanumeric characters for robust schema matching."""
    return re.sub(r"[^a-z0-9]+", "", str(col).strip().lower())


def resolve_dataset_path(
    user_path: Optional[str | Path] = None,
    paths: Optional[ProjectPaths] = None,
) -> Optional[Path]:
    """Resolves the dataset path by inspecting candidate locations in order of priority.

    If an explicit user_path is provided, it is checked directly.
    Otherwise, standard project directories and workspace locations are searched.

    Args:
        user_path: Explicit file path provided by the user.
        paths: Project directory structure object.

    Returns:
        Resolved Path object if a valid CSV is found, otherwise None.
    """
    if user_path:
        p = Path(user_path)
        if p.is_file() and p.suffix.lower() == ".csv":
            return p.resolve()
        if p.is_dir():
            for child in sorted(p.glob("*.csv")):
                if child.is_file():
                    return child.resolve()
        return None

    candidates: List[Path] = []
    if paths is not None:
        candidates.extend([
            paths.data_raw / "LCL-June2015v2_0.csv",
            paths.data_raw,
            paths.root_dir.parent / "LCL-June2015v2_0.csv",
            paths.root_dir / "LCL-June2015v2_0.csv",
        ])

    # Direct workspace paths
    candidates.extend([
        Path("e:/SPML/IA1/LCL-June2015v2_0.csv"),
        Path("../LCL-June2015v2_0.csv"),
        Path("LCL-June2015v2_0.csv"),
    ])

    for candidate in candidates:
        if candidate.is_file() and candidate.suffix.lower() == ".csv":
            return candidate.resolve()
        if candidate.is_dir():
            for child in sorted(candidate.glob("*.csv")):
                if child.is_file():
                    return child.resolve()

    return None


def adapt_schema(
    df: pd.DataFrame,
    required_canonical: Sequence[str] = ("household_id", "timestamp", "energy_kwh"),
) -> pd.DataFrame:
    """Adapts raw source column names to canonical internal schema names.

    Args:
        df: Raw DataFrame chunk.
        required_canonical: Required canonical columns.

    Returns:
        DataFrame with standardized column names and parsed datatypes.

    Raises:
        ValueError: If any required canonical column cannot be matched.
    """
    # Clean original column names by stripping trailing whitespace
    cleaned_cols = {col: col.strip() for col in df.columns}
    df = df.rename(columns=cleaned_cols)

    rename_map: dict[str, str] = {}
    normalized_cols = {_clean_column_name(c): c for c in df.columns}

    for canonical, aliases in SCHEMA_ALIASES.items():
        matched = False
        for alias in aliases:
            cleaned_alias = _clean_column_name(alias)
            if cleaned_alias in normalized_cols:
                rename_map[normalized_cols[cleaned_alias]] = canonical
                matched = True
                break
        if not matched:
            for clean_col, orig_col in normalized_cols.items():
                if any(_clean_column_name(alias) in clean_col for alias in aliases):
                    rename_map[orig_col] = canonical
                    matched = True
                    break

    missing = [c for c in required_canonical if c not in rename_map.values()]
    if missing:
        raise ValueError(
            f"Schema adapter failed: Missing required canonical fields: {missing}. "
            f"Available source columns: {list(df.columns)}"
        )

    df_renamed = df.rename(columns=rename_map)

    # Ensure canonical types
    if "household_id" in df_renamed.columns:
        df_renamed["household_id"] = df_renamed["household_id"].astype(str).str.strip()

    if "timestamp" in df_renamed.columns:
        df_renamed["timestamp"] = pd.to_datetime(df_renamed["timestamp"], errors="coerce")

    if "energy_kwh" in df_renamed.columns:
        # Coerce strings like 'Null', blanks, or errors to NaN
        df_renamed["energy_kwh"] = pd.to_numeric(
            df_renamed["energy_kwh"].astype(str).str.strip(), errors="coerce"
        )

    return df_renamed[list(required_canonical)]


def load_dataset(
    file_path: Path,
    target_household: Optional[str] = "MAC000002",
    max_rows: Optional[int] = 25000,
    chunk_size: int = 50000,
) -> Tuple[pd.DataFrame, dict]:
    """Loads and filters energy consumption data using memory-efficient chunked streaming.

    Args:
        file_path: Path to the dataset CSV file.
        target_household: Optional household ID to filter (e.g. 'MAC000002').
        max_rows: Maximum number of rows to collect.
        chunk_size: Number of lines per chunk read from disk.

    Returns:
        Tuple of (combined DataFrame, raw loading statistics dict).
    """
    logger.info("Streaming dataset from: %s (chunk_size=%d)", file_path, chunk_size)

    chunks: List[pd.DataFrame] = []
    total_raw_rows_scanned = 0
    total_null_entries_found = 0
    households_encountered = set()

    # Read CSV in chunks
    for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False):
        total_raw_rows_scanned += len(chunk)
        adapted_chunk = adapt_schema(chunk)

        # Track invalid/null values before filtering
        null_count = int(adapted_chunk["energy_kwh"].isna().sum())
        total_null_entries_found += null_count
        households_encountered.update(adapted_chunk["household_id"].unique())

        # Filter by target household if requested
        if target_household is not None:
            adapted_chunk = adapted_chunk[adapted_chunk["household_id"] == target_household]

        if not adapted_chunk.empty:
            chunks.append(adapted_chunk)
            current_count = sum(len(c) for c in chunks)
            if max_rows is not None and current_count >= max_rows:
                logger.info("Reached target row count (%d rows). Terminating chunk stream.", current_count)
                break

    if not chunks:
        raise ValueError(
            f"No matching records found for household '{target_household}' in {file_path}. "
            f"Households found in stream: {list(households_encountered)[:10]}"
        )

    combined_df = pd.concat(chunks, ignore_index=True)
    if max_rows is not None and len(combined_df) > max_rows:
        combined_df = combined_df.iloc[:max_rows].copy()

    stats = {
        "raw_file": str(file_path),
        "total_raw_rows_scanned": total_raw_rows_scanned,
        "selected_rows_count": len(combined_df),
        "target_household_filter": target_household,
        "distinct_households_in_data": int(combined_df["household_id"].nunique()),
        "raw_null_or_invalid_values_encountered": total_null_entries_found,
    }
    logger.info("Data loading complete. Selected %d rows across %d household(s).",
                len(combined_df), combined_df["household_id"].nunique())
    return combined_df, stats


def generate_synthetic_smoke_data(
    n_rows: int = 1500,
    household_id: str = "SYNTHETIC_HH_01",
    start_date: str = "2023-01-01 00:00:00",
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, dict]:
    """Generates synthetic half-hourly energy consumption data for smoke testing the pipeline.

    Notice:
        This dataset is solely intended for automated integration and regression testing.
        It must NEVER be represented as empirical observations from real households.

    Args:
        n_rows: Number of half-hourly time intervals.
        household_id: Synthetic household identifier.
        start_date: Starting ISO timestamp.
        random_seed: Random state seed.

    Returns:
        Tuple of (synthetic DataFrame, metadata dict).
    """
    logger.warning("GENERATING SYNTHETIC SMOKE DATA - FOR CODE TESTING PURPOSES ONLY!")
    rng = np.random.default_rng(random_seed)
    timestamps = pd.date_range(start=start_date, periods=n_rows, freq="30min")

    # Generate synthetic diurnal load pattern: baseload + morning/evening peaks + noise
    hour = (timestamps.hour + timestamps.minute / 60.0).to_numpy()
    diurnal_pattern = 0.15 + 0.25 * np.sin((hour - 6) / 24 * 2 * np.pi) ** 2 + 0.35 * np.exp(-((hour - 19) ** 2) / 8)
    noise = rng.normal(0, 0.05, size=n_rows)
    energy_kwh = np.clip(diurnal_pattern + noise, 0.01, 3.5).astype(float)

    # Intentionally insert a few missing values and duplicates to test data cleaning
    energy_kwh[10] = np.nan
    energy_kwh[50] = np.nan

    df = pd.DataFrame({
        "household_id": household_id,
        "timestamp": timestamps,
        "energy_kwh": energy_kwh,
    })

    # Add duplicate timestamp row to test deduplication
    duplicate_row = df.iloc[[100]].copy()
    df = pd.concat([df, duplicate_row], ignore_index=True).sort_values("timestamp").reset_index(drop=True)

    stats = {
        "raw_file": "SYNTHETIC_SMOKE_DATA",
        "total_raw_rows_scanned": len(df),
        "selected_rows_count": len(df),
        "target_household_filter": household_id,
        "distinct_households_in_data": 1,
        "raw_null_or_invalid_values_encountered": 2,
    }
    return df, stats
