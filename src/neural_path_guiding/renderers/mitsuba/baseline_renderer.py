"""Baseline rendering utilities for Mitsuba.

This module renders Mitsuba scenes without neural path guiding.

It provides:
- image rendering
- output directory creation
- render timing
- render result metadata
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import mitsuba as mi


_mi: Any = mi


# Basic information produced by a render call.
@dataclass(frozen=True)
class RenderResult:
    output_path: Path  # Image file written to disk.
    spp: int  # Samples per pixel.
    seed: int  # Random seed used by Mitsuba.
    elapsed_seconds: float  # Total render time in seconds.


def render_scene(
    scene: Any,
    output_path: Path,
    spp: int,
    seed: int,
) -> RenderResult:
    # Renders a Mitsuba scene and writes the image to disk.
    if spp <= 0:
        raise ValueError("spp must be greater than zero.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    image = _mi.render(scene, spp=spp, seed=seed)
    elapsed_seconds = time.perf_counter() - start_time

    _mi.Bitmap(image).write(str(output_path))

    return RenderResult(
        output_path=output_path,
        spp=spp,
        seed=seed,
        elapsed_seconds=elapsed_seconds,
    )