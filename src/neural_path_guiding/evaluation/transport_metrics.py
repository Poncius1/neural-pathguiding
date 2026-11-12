"""Experiment-specific RGB image metrics for the single-loop transport study.

Does not depend on Mitsuba; image loading belongs to evaluation.image_io.
The result key names deliberately preserve transport manifest schema v1.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from neural_path_guiding.evaluation.metrics import (
    mean_absolute_error,
    mean_squared_error,
    validate_image_pair,
)

LUMINANCE_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)


def file_sha256(path: Path) -> str:
    """Hash a file in bounded chunks, avoiding a scripts -> scripts import."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_rgb_image(image: np.ndarray, *, name: str = "image") -> np.ndarray:
    """Accept only finite HxWx3 linear RGB images."""
    rgb = np.asarray(image, dtype=np.float64)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.shape[0] == 0 or rgb.shape[1] == 0:
        raise ValueError(f"{name} must have shape (height, width, 3), got {rgb.shape}")
    if not np.all(np.isfinite(rgb)):
        raise ValueError(f"{name} contains non-finite values")
    return rgb


def summarize_transport_image(
    image: np.ndarray, *, seconds: float, reference: np.ndarray | None = None
) -> dict[str, object]:
    """Return backward-compatible per-mode manifest fields.

    RGB MSE is error against a finite reference, not Monte Carlo variance.
    """
    rgb = validate_rgb_image(image)
    if not np.isfinite(seconds) or seconds < 0:
        raise ValueError("seconds must be finite and nonnegative")
    luminance = rgb @ LUMINANCE_WEIGHTS
    result: dict[str, object] = {
        "seconds": float(seconds),
        "mean_luminance": float(np.mean(luminance)),
        "max_luminance": float(np.max(luminance)),
        "zero_pixels": int(np.sum(np.all(rgb == 0.0, axis=-1))),
    }
    if reference is not None:
        ref = validate_rgb_image(reference, name="reference")
        # validate_image_pair checks matching shape and finite data.
        ref, rgb = validate_image_pair(ref, rgb)
        result.update({
            "rgb_mse": mean_squared_error(ref, rgb),
            "rgb_mae": mean_absolute_error(ref, rgb),
            "mean_rgb_error": np.mean(rgb - ref, axis=(0, 1)).tolist(),
        })
    return result
