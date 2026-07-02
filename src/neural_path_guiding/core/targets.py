"""Target distribution utilities.

This module converts per-bin contribution estimates into probability targets.

It does not depend on Mitsuba.
"""

from __future__ import annotations

import numpy as np

from neural_path_guiding.core.features import FloatArray


def contributions_to_target_distribution(
    contributions: FloatArray,
    smoothing: float,
) -> FloatArray:
    # Converts non-negative per-bin contributions into a probability distribution.
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

    smoothed = values + smoothing
    total = float(smoothed.sum())

    if total <= 0.0:
        return np.full(values.shape, 1.0 / values.size, dtype=np.float64)

    return smoothed / total