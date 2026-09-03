"""Color utilities used by neural path guiding targets.

The directional neural model predicts one scalar probability per bin.
RGB radiance and contribution values therefore need to be reduced to
a non-negative scalar before constructing a target distribution.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


_REC709_LUMINANCE_WEIGHTS = np.array(
    [0.2126, 0.7152, 0.0722],
    dtype=np.float64,
)


def rgb_luminance(rgb: ArrayLike) -> float:
    """Convert a finite non-negative RGB value to luminance.
    Values greater than one are allowed because rendered radiance is
    high-dynamic-range data.
    """

    rgb_array = np.asarray(
        rgb,
        dtype=np.float64,
    )

    if rgb_array.shape != (3,):
        raise ValueError(
            "rgb must have shape (3,), "
            f"got {rgb_array.shape}."
        )

    if not bool(
        np.all(
            np.isfinite(rgb_array)
        )
    ):
        raise ValueError(
            "rgb must contain only finite values."
        )

    if bool(
        np.any(
            rgb_array < 0.0
        )
    ):
        raise ValueError(
            "rgb values must be non-negative."
        )

    return float(
        np.dot(
            rgb_array,
            _REC709_LUMINANCE_WEIGHTS,
        )
    )