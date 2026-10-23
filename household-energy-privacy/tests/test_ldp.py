"""Unit tests for Local Differential Privacy (k-RR) mechanism and aggregate proportion estimation."""

from __future__ import annotations

import numpy as np
import pytest

from src.ldp_demo import (
    KRandomizedResponse,
    compute_discrete_bin_thresholds,
    discretize_feature,
)


def test_ldp_parameter_validation() -> None:
    """Verifies that non-positive epsilon or domain k < 2 raises ValueError."""
    with pytest.raises(ValueError, match="strictly positive"):
        KRandomizedResponse(epsilon=-0.5, k=3)
    with pytest.raises(ValueError, match="(?i)domain size k must be at least 2"):
        KRandomizedResponse(epsilon=1.0, k=1)


def test_ldp_probability_axioms() -> None:
    """Verifies that transition probabilities p and q sum to 1 over all k choices."""
    k = 4
    for eps in [0.1, 0.5, 1.0, 2.5, 5.0]:
        mech = KRandomizedResponse(epsilon=eps, k=k)
        assert mech.p > mech.q
        assert mech.p + (k - 1) * mech.q == pytest.approx(1.0)


def test_ldp_perturbation_domain_bounds() -> None:
    """Verifies that all perturbed outputs lie strictly in [0, k-1]."""
    k = 3
    mech = KRandomizedResponse(epsilon=1.0, k=k, random_state=42)
    true_cats = np.random.randint(0, k, size=500)
    perturbed = mech.perturb(true_cats)

    assert perturbed.shape == (500,)
    assert set(perturbed).issubset({0, 1, 2})


def test_ldp_unbiased_frequency_estimation() -> None:
    """Verifies that the aggregate proportion estimator is asymptotically unbiased."""
    k = 3
    true_proportions = np.array([0.20, 0.50, 0.30])
    n = 60000

    # Use independent random seeds for dataset generation and privacy mechanism
    data_rng = np.random.default_rng(101)
    categories = data_rng.choice([0, 1, 2], size=n, p=true_proportions)

    mech = KRandomizedResponse(epsilon=2.0, k=k, random_state=202)
    perturbed = mech.perturb(categories)
    estimated_props = mech.estimate_proportions(perturbed)

    # With n=60000, eps=2.0, estimated proportions should match true within 0.03
    assert np.allclose(estimated_props, true_proportions, atol=0.03)


def test_discretize_feature_thresholds() -> None:
    """Verifies discretization using training quantile thresholds."""
    y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    thresholds = compute_discrete_bin_thresholds(y_train, num_bins=3)
    assert len(thresholds) == 2

    # Thresholds are approx [2.67, 4.33]
    y_test = np.array([0.5, 3.5, 5.5])
    cats = discretize_feature(y_test, thresholds)
    assert list(cats) == [0, 1, 2]
