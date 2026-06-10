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
    # Validates and normalizes a probability vector.
    probabilities = np.asarray(probabilities, dtype=np.float64)

    if probabilities.shape != (expected_size,):
        raise ValueError(
            f"probabilities must have shape ({expected_size},), "
            f"got {probabilities.shape}."
        )

    if bool(np.any(probabilities < 0.0)):
        raise ValueError("probabilities must be non-negative.")

    total_probability = float(probabilities.sum())

    if total_probability <= 0.0:
        raise ValueError("probabilities must sum to a positive value.")

    return probabilities / total_probability


def mix_with_uniform(
    probabilities: FloatArray,
    alpha: float,
    expected_size: int,
) -> FloatArray:
    # Mixes probabilities with uniform distribution to avoid zero bins.
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1].")

    probabilities = normalize_probabilities(probabilities, expected_size)

    uniform = np.full(expected_size, 1.0 / expected_size, dtype=np.float64)
    mixed = (1.0 - alpha) * probabilities + alpha * uniform

    return mixed / float(mixed.sum())


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