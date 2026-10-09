"""Renderer-independent multi-technique MIS estimator.

Samples are continuous, non-delta directions with PDFs expressed in the same
measure (usually inverse steradians). Delta transport needs a separate path.
"""
from __future__ import annotations

from collections.abc import Sequence
import numpy as np

from neural_path_guiding.core.mis import power_heuristic_weights


def combine_mis_samples(
    contributions: Sequence[np.ndarray],
    cross_pdfs: Sequence[np.ndarray],
    *,
    exponent: float = 2.0,
) -> np.ndarray:
    """Combine samples from K strategies using the multi-sample MIS estimator.

    contributions[i]: (n_i, 3) RGB integrand values F(x), NOT F(x)/p_i.
    cross_pdfs[i]: (n_i, K), evaluated at samples from strategy i;
                   column j contains p_j(x) in a common measure.
    Each active technique contributes (1/n_i) * sum_s w_i(x_s)F(x_s)/p_i(x_s).
    A technique with n_i=0 must have empty arrays and receives zero weight.
    Zero-PDF samples from their *own* technique are invalid.
    """
    k = len(contributions)
    if k < 2 or len(cross_pdfs) != k:
        raise ValueError("Expected matching arrays for at least two techniques")
    counts = []
    rgb_arrays = []
    pdf_arrays = []
    for i, (rgb, pdf) in enumerate(zip(contributions, cross_pdfs)):
        rgb = np.asarray(rgb, dtype=np.float64)
        pdf = np.asarray(pdf, dtype=np.float64)
        if rgb.ndim != 2 or rgb.shape[1] != 3:
            raise ValueError(f"contributions[{i}] must be (n_i, 3)")
        if pdf.shape != (rgb.shape[0], k):
            raise ValueError(f"cross_pdfs[{i}] must be (n_i, {k})")
        if not np.all(np.isfinite(rgb)):
            raise ValueError("Contributions must be finite")
        if not np.all(np.isfinite(pdf)) or np.any(pdf < 0):
            raise ValueError("PDF values must be finite and nonnegative")
        if rgb.shape[0] and np.any(pdf[:, i] <= 0):
            raise ValueError(f"Sampled technique {i} has a non-positive own PDF")
        counts.append(rgb.shape[0])
        rgb_arrays.append(rgb)
        pdf_arrays.append(pdf)
    if not any(counts):
        raise ValueError("At least one strategy must have samples")

    output = np.zeros(3, dtype=np.float64)
    for i, (rgb, pdf, n) in enumerate(zip(rgb_arrays, pdf_arrays, counts)):
        if n == 0:
            continue
        for f, densities in zip(rgb, pdf):
            weights = power_heuristic_weights(
                densities, sample_counts=counts, exponent=exponent
            )
            output += weights[i] * f / (n * densities[i])
    return output
