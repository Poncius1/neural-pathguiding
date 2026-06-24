"""Image I/O utilities for evaluation.

This module loads rendered images as NumPy arrays.

It currently uses Mitsuba Bitmap because the project renders EXR images
through Mitsuba.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mitsuba as mi
import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]

_mi: Any = mi


def load_image_as_array(image_path: Path) -> FloatArray:
    # Loads an image file and returns a float64 NumPy array.
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    bitmap = _mi.Bitmap(str(image_path))
    image = np.asarray(bitmap, dtype=np.float64)

    if image.ndim == 2:
        image = image[:, :, np.newaxis]

    if image.ndim != 3:
        raise ValueError(f"Expected image with 2 or 3 dimensions, got {image.shape}")

    if image.shape[2] > 3:
        image = image[:, :, :3]

    return image