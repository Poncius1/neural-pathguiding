"""PDF utilities for discrete neural path guiding.

This module converts discrete bin probabilities into continuous directional PDFs.

It provides:
- probability validation
- uniform mixing
- PDF evaluation in solid-angle measure
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from neural_path_guiding.core.bins import HemisphereBins


FloatArray = NDArray[np.float64]


def normalize_probabilities(probabilities: FloatArray, expected_size: int) -> FloatArray:
    """Validate and normalize a finite, non-negative probability vector."""
    if isinstance(expected_size, bool) or not isinstance(
        expected_size,
        (int, np.integer),
    ):
        raise TypeError("expected_size must be an integer.")

    if expected_size <= 0:
        raise ValueError("expected_size must be greater than zero.")

    probabilities = np.asarray(probabilities, dtype=np.float64)

    if probabilities.shape != (expected_size,):
        raise ValueError(
            f"probabilities must have shape ({expected_size},), "
            f"got {probabilities.shape}."
        )

    if not bool(np.all(np.isfinite(probabilities))):
        raise ValueError("probabilities must contain only finite values.")

    if bool(np.any(probabilities < 0.0)):
        raise ValueError("probabilities must be non-negative.")

    maximum_probability = float(probabilities.max())

    if maximum_probability <= 0.0:
        raise ValueError("probabilities must sum to a positive value.")

    # Scaling first prevents overflow when several valid values are near float64 max.
    scaled_probabilities = probabilities / maximum_probability
    total_probability = float(scaled_probabilities.sum(dtype=np.float64))

    if not np.isfinite(total_probability) or total_probability <= 0.0:
        raise ValueError("probabilities must sum to a positive value.")

    normalized = scaled_probabilities / total_probability

    if not bool(np.all(np.isfinite(normalized))):
        raise ValueError("normalized probabilities must contain only finite values.")

    return normalized


def mix_with_uniform(
    probabilities: FloatArray,
    alpha: float,
    expected_size: int,
) -> FloatArray:
    # Mixes probabilities with a uniform distribution to avoid zero-probability bins.
    if not np.isfinite(alpha):
        raise ValueError("alpha must be finite.")

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1].")

    probabilities = normalize_probabilities(probabilities, expected_size)

    uniform = np.full(expected_size, 1.0 / expected_size, dtype=np.float64)
    mixed = (1.0 - alpha) * probabilities + alpha * uniform

    return normalize_probabilities(mixed, expected_size)


def evaluate_pdf_from_probabilities(
    bins: HemisphereBins,
    probabilities: FloatArray,
    omega_local: FloatArray,
) -> float:
    # Converts discrete bin probability into continuous solid-angle PDF.
    probabilities = normalize_probabilities(
        probabilities=probabilities,
        expected_size=bins.n_bins,
    )

    bin_index = bins.find_bin_from_direction(omega_local)

    return float(probabilities[bin_index] / bins.bin_solid_angle)
