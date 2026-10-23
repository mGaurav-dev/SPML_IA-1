"""Configuration dataclasses and project path management."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional


@dataclass
class ProjectPaths:
    """Manages relative and resolved project directory paths."""
    root_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)

    @property
    def data_raw(self) -> Path:
        return self.root_dir / "data" / "raw"

    @property
    def data_processed(self) -> Path:
        return self.root_dir / "data" / "processed"

    @property
    def results_figures(self) -> Path:
        return self.root_dir / "results" / "figures"

    @property
    def results_tables(self) -> Path:
        return self.root_dir / "results" / "tables"

    @property
    def results_logs(self) -> Path:
        return self.root_dir / "results" / "logs"

    @property
    def report_dir(self) -> Path:
        return self.root_dir / "report"


@dataclass
class DataConfig:
    """Data loading, filtering, and splitting configuration."""
    data_path: Optional[str] = None
    target_household: Optional[str] = "MAC000002"
    subset_size: int = 25000
    chunk_size: int = 50000
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    resample_cadence: str = "30min"
    required_raw_columns: Tuple[str, ...] = ("LCLid", "DateTime", "KWH/hh (per half hour)")


@dataclass
class ModelConfig:
    """Machine learning feature and baseline model configuration."""
    features: List[str] = field(
        default_factory=lambda: [
            "lag_1",
            "lag_2",
            "lag_48",
            "rolling_mean_4",
            "hour",
            "day_of_week",
            "is_weekend",
            "month",
        ]
    )
    target: str = "energy_kwh"
    random_state: int = 42
    ridge_alpha: float = 1.0


@dataclass
class DPConfig:
    """Differential Privacy experiment configuration."""
    epsilons: List[float] = field(
        default_factory=lambda: [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    )
    target_epsilon: float = 1.0
    n_repeats: int = 5
    random_state: int = 42
    clipping_percentiles: Tuple[float, float] = (0.5, 99.5)
    protected_unit: str = "Single 30-minute observation record (x_i, y_i)"
    adjacency_definition: str = (
        "Bounded differential privacy: two datasets D and D' differ by replacing "
        "at most one record (x, y) with (x', y'), where |D| = |D'|."
    )
    formal_guarantee: str = (
        "Record-level (epsilon, 0)-Differential Privacy via Diffprivlib LinearRegression "
        "(covariance matrix perturbation). Note: this protects individual half-hourly "
        "readings, not entire multi-year household time series."
    )


@dataclass
class LDPConfig:
    """Local Differential Privacy demonstration configuration."""
    num_bins: int = 3
    bin_labels: List[str] = field(default_factory=lambda: ["Low", "Medium", "High"])
    epsilons: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0, 4.0])
    random_state: int = 42


@dataclass
class AggregationConfig:
    """Privacy-preserving data mining (PPDM) aggregation configuration."""
    intervals: List[str] = field(default_factory=lambda: ["30min", "1h", "1d"])


@dataclass
class ExperimentConfig:
    """Top-level pipeline configuration."""
    paths: ProjectPaths = field(default_factory=ProjectPaths)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    dp: DPConfig = field(default_factory=DPConfig)
    ldp: LDPConfig = field(default_factory=LDPConfig)
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    smoke_test: bool = False
    skip_plots: bool = False
