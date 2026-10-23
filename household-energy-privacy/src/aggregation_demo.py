"""Privacy-Preserving Data Mining (PPDM) demonstration: temporal aggregation vs disclosure risk."""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("household_energy_dp")


def compute_temporal_aggregations(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generates multi-resolution temporal summaries (30-min, 1-hour, 1-day) and compares utility metrics.

    Important Privacy Note:
        Aggregation is a conventional data-minimization technique, NOT a formal differential privacy guarantee.
        Without randomized perturbation, fine-grained readings remain vulnerable to differencing attacks
        and auxiliary side-channel linkage.

    Args:
        df: DataFrame with 'timestamp' and 'energy_kwh'.

    Returns:
        Tuple of (half_hourly_df, hourly_df, daily_df, comparison_metrics_df).
    """
    df_sorted = df.sort_values(by="timestamp").set_index("timestamp").copy()

    # 1. Native half-hourly resolution
    s_30m = df_sorted["energy_kwh"]

    # 2. Hourly aggregation (sum of energy consumed per hour)
    s_1h = s_30m.resample("1h").sum()

    # 3. Daily aggregation (sum of energy consumed per day)
    s_1d = s_30m.resample("1D").sum()

    # Compute comparative descriptive statistics
    def _calc_stats(series: pd.Series, label: str, duration_desc: str) -> Dict[str, any]:
        s_clean = series.dropna()
        return {
            "resolution": label,
            "duration": duration_desc,
            "record_count": len(s_clean),
            "mean_energy": round(float(s_clean.mean()), 4),
            "std_energy": round(float(s_clean.std()), 4),
            "min_energy": round(float(s_clean.min()), 4),
            "max_energy": round(float(s_clean.max()), 4),
            "peak_to_average_ratio": round(float(s_clean.max() / s_clean.mean()), 3) if s_clean.mean() > 0 else 0,
            "temporal_detail_retained": "Full" if label == "30min" else ("Moderate" if label == "1h" else "Coarse"),
            "residual_disclosure_risk": (
                "High (reveals exact appliance activations & occupancy bursts)"
                if label == "30min"
                else (
                    "Moderate (smooths micro-events; daily schedules still detectable)"
                    if label == "1h"
                    else "Low-to-Moderate (reveals total daily usage, vacation periods, baseline)"
                )
            ),
            "formal_dp_guarantee": "None (Heuristic aggregation only; vulnerable to differencing attacks)",
        }

    records = [
        _calc_stats(s_30m, "30min", "Half-Hourly (Native)"),
        _calc_stats(s_1h, "1h", "Hourly Sum"),
        _calc_stats(s_1d, "1d", "Daily Sum"),
    ]

    metrics_df = pd.DataFrame(records)

    half_hourly_df = s_30m.reset_index()
    hourly_df = s_1h.reset_index()
    daily_df = s_1d.reset_index()

    logger.info("Computed temporal aggregations: 30min=%d rows, 1h=%d rows, 1d=%d rows",
                len(half_hourly_df), len(hourly_df), len(daily_df))

    return half_hourly_df, hourly_df, daily_df, metrics_df
