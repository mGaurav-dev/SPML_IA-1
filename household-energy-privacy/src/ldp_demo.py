"""Local Differential Privacy (LDP) demonstration using k-ary Randomized Response (k-RR)."""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("household_energy_dp")


def compute_discrete_bin_thresholds(
    y_train: np.ndarray,
    num_bins: int = 3,
) -> np.ndarray:
    """Computes quantile bin edges strictly on the training partition.

    Args:
        y_train: Continuous energy consumption target from training set.
        num_bins: Number of discrete bins (e.g. 3 for Low, Medium, High).

    Returns:
        Array of internal threshold values (length = num_bins - 1).
    """
    percentiles = np.linspace(0, 100, num_bins + 1)[1:-1]
    thresholds = np.percentile(y_train, percentiles)
    return thresholds


def discretize_feature(
    values: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """Discretizes continuous energy values into integer category indices [0, ..., k-1].

    Args:
        values: Continuous consumption values.
        thresholds: Pre-computed quantile thresholds from training data.

    Returns:
        1D integer array with category indices in range [0, len(thresholds)].
    """
    return np.digitize(values, thresholds)


class KRandomizedResponse:
    """k-ary Randomized Response (k-RR) mechanism for Local Differential Privacy.

    Guarantees epsilon-LDP for any individual report before it leaves the client/meter.

    Probabilities for domain of size k:
        P(Y_tilde = i | Y = i) = p = exp(eps) / (exp(eps) + k - 1)
        P(Y_tilde = j | Y = i) = q = 1 / (exp(eps) + k - 1) for j != i
    """

    def __init__(self, epsilon: float, k: int, random_state: int = 42) -> None:
        if epsilon <= 0:
            raise ValueError(f"Privacy budget epsilon must be strictly positive, got {epsilon}")
        if k < 2:
            raise ValueError(f"Domain size k must be at least 2, got {k}")

        self.epsilon = float(epsilon)
        self.k = int(k)
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

        # Mechanism transition probabilities
        denom = np.exp(self.epsilon) + self.k - 1
        self.p = float(np.exp(self.epsilon) / denom)
        self.q = float(1.0 / denom)

    def perturb(self, true_categories: np.ndarray) -> np.ndarray:
        """Perturbs true discrete categories using the k-RR mechanism.

        Args:
            true_categories: Integer array with values in [0, k-1].

        Returns:
            Perturbed integer array with values in [0, k-1].
        """
        n = len(true_categories)
        perturbed = np.empty(n, dtype=int)

        # Uniform random draw in [0, 1)
        draws = self.rng.random(size=n)
        keep_mask = draws < self.p

        # Keep true category when draw < p
        perturbed[keep_mask] = true_categories[keep_mask]

        # For the remaining cases, draw uniformly at random from other k - 1 categories
        flip_indices = np.where(~keep_mask)[0]
        for idx in flip_indices:
            true_val = true_categories[idx]
            other_categories = [c for c in range(self.k) if c != true_val]
            perturbed[idx] = self.rng.choice(other_categories)

        return perturbed

    def estimate_proportions(self, perturbed_categories: np.ndarray) -> np.ndarray:
        """Computes the mathematically unbiased population frequency estimator:

        Formula:
            hat_pi_i = (tilde_pi_i - q) / (p - q)

        Args:
            perturbed_categories: Array of privatized category reports.

        Returns:
            Array of estimated true proportions for categories 0 to k-1.
        """
        n = len(perturbed_categories)
        counts = np.bincount(perturbed_categories, minlength=self.k)
        empirical_proportions = counts / n

        # Unbiased inversion of the k-RR transition matrix
        unbiased_estimates = (empirical_proportions - self.q) / (self.p - self.q)
        return unbiased_estimates


def run_ldp_demonstration(
    y_train: np.ndarray,
    y_test: np.ndarray,
    num_bins: int = 3,
    bin_labels: List[str] | None = None,
    epsilons: List[float] | None = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """Runs end-to-end Local Differential Privacy demonstration on energy consumption categories.

    Notice:
        This LDP demonstration operates on local feature privatization. It does not
        automatically provide formal differential privacy guarantees to the centralized
        regression model unless trained directly on perturbed features.

    Args:
        y_train: Continuous target array from training set.
        y_test: Continuous target array from test set.
        num_bins: Number of discretized consumption bins (default 3).
        bin_labels: Human-readable names for each bin.
        epsilons: List of LDP epsilon parameters.
        random_state: Base random seed.

    Returns:
        Tuple of (results DataFrame, summary dict).
    """
    if bin_labels is None:
        bin_labels = [f"Bin_{i}" for i in range(num_bins)]
    if epsilons is None:
        epsilons = [0.5, 1.0, 2.0, 4.0]

    # Step 1: Discretize feature using thresholds strictly from training data
    thresholds = compute_discrete_bin_thresholds(y_train, num_bins=num_bins)
    true_test_categories = discretize_feature(y_test, thresholds)

    # True ground-truth test proportions
    true_counts = np.bincount(true_test_categories, minlength=num_bins)
    true_proportions = true_counts / len(true_test_categories)

    results_records: List[Dict[str, any]] = []

    for eps in epsilons:
        mechanism = KRandomizedResponse(epsilon=eps, k=num_bins, random_state=random_state)
        perturbed_reports = mechanism.perturb(true_test_categories)
        estimated_proportions = mechanism.estimate_proportions(perturbed_reports)

        # Metrics: Mean Absolute Error and Total Variation Distance (TVD)
        abs_errors = np.abs(estimated_proportions - true_proportions)
        mae_proportions = float(np.mean(abs_errors))
        tvd = float(0.5 * np.sum(abs_errors))

        record = {
            "epsilon_ldp": float(eps),
            "k_categories": num_bins,
            "p_true": round(mechanism.p, 4),
            "q_other": round(mechanism.q, 4),
            "mae_proportion_error": round(mae_proportions, 5),
            "total_variation_distance": round(tvd, 5),
        }
        for i, label in enumerate(bin_labels):
            record[f"true_prop_{label}"] = round(float(true_proportions[i]), 4)
            record[f"est_prop_{label}"] = round(float(estimated_proportions[i]), 4)

        results_records.append(record)

    results_df = pd.DataFrame(results_records)
    summary = {
        "num_bins": num_bins,
        "bin_labels": bin_labels,
        "thresholds": [round(float(t), 4) for t in thresholds],
        "test_sample_count": len(y_test),
        "tested_epsilons": epsilons,
    }
    return results_df, summary
