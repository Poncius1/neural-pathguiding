"""Target distribution utilities.

This module converts per-bin contribution estimates into probability targets.

It does not depend on Mitsuba.
"""

from __future__ import annotations

import numpy as np

from neural_path_guiding.core.features import FloatArray
from neural_path_guiding.core.pdf import normalize_probabilities


def contributions_to_target_distribution(
    contributions: FloatArray,
    smoothing: float,
) -> FloatArray:
    # Converts non-negative per-bin contributions into a probability distribution.
    if not np.isfinite(smoothing):
        raise ValueError("smoothing must be finite.")

    if smoothing < 0.0:
        raise ValueError("smoothing must be non-negative.")

    values = np.asarray(contributions, dtype=np.float64)

    if values.ndim != 1:
        raise ValueError(f"contributions must be a 1D array, got {values.shape}.")

    if values.size == 0:
        raise ValueError("contributions must not be empty.")

    if not np.all(np.isfinite(values)):
        raise ValueError("contributions contains non-finite values.")

    if bool(np.any(values < 0.0)):
        raise ValueError("contributions must be non-negative.")

    if not bool(np.any(values > 0.0)) and smoothing == 0.0:
        raise ValueError(
            "contributions must contain a positive value when smoothing is zero."
        )

    # Scale both terms before adding so finite values cannot overflow.
    scale = max(float(values.max()), float(smoothing))
    scaled_smoothed = values / scale + smoothing / scale

    return normalize_probabilities(
        probabilities=scaled_smoothed,
        expected_size=values.size,
    )
