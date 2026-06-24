"""Image quality metrics.

This module computes numerical metrics between two rendered images.

It provides:
- MSE
- RMSE
- MAE
- PSNR
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def mean_squared_error(reference: FloatArray, candidate: FloatArray) -> float:
    # Average squared difference over all pixels and channels.
    reference, candidate = validate_image_pair(reference, candidate)

    difference = reference - candidate
    return float(np.mean(difference * difference))


def root_mean_squared_error(reference: FloatArray, candidate: FloatArray) -> float:
    # Square root of MSE.
    mse = mean_squared_error(reference, candidate)
    return math.sqrt(mse)


def mean_absolute_error(reference: FloatArray, candidate: FloatArray) -> float:
    # Average absolute difference over all pixels and channels.
    reference, candidate = validate_image_pair(reference, candidate)

    return float(np.mean(np.abs(reference - candidate)))


def peak_signal_to_noise_ratio(mse: float, peak_value: float) -> float:
    # PSNR in decibels.
    if mse < 0.0:
        raise ValueError("mse must be non-negative.")

    if peak_value <= 0.0:
        raise ValueError("peak_value must be greater than zero.")

    if mse == 0.0:
        return math.inf

    return 20.0 * math.log10(peak_value) - 10.0 * math.log10(mse)


def automatic_peak_value(reference: FloatArray) -> float:
    # Uses the brightest reference value as peak, with 1.0 as minimum.
    peak = float(np.max(reference))

    return max(peak, 1.0)


def validate_image_pair(
    reference: FloatArray,
    candidate: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    # Checks that both images can be compared safely.
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)

    if reference.shape != candidate.shape:
        raise ValueError(
            "Images must have the same shape. "
            f"Reference: {reference.shape}, candidate: {candidate.shape}"
        )

    if reference.size == 0:
        raise ValueError("Images must not be empty.")

    if not np.all(np.isfinite(reference)):
        raise ValueError("Reference image contains non-finite values.")

    if not np.all(np.isfinite(candidate)):
        raise ValueError("Candidate image contains non-finite values.")

    return reference, candidate