"""Incident-radiance estimation utilities for physical teachers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mitsuba as mi
import numpy as np

from neural_path_guiding.core.features import FloatArray


_mi: Any = mi


@dataclass(frozen=True)
class IncidentRadianceRuntime:
    """Objects required to estimate incident radiance repeatedly."""

    integrator: Any
    sampler: Any
    radiance_samples: int


def create_incident_radiance_runtime(
    *,
    radiance_samples: int,
    max_depth: int,
    rr_depth: int,
    seed: int,
) -> IncidentRadianceRuntime:
    """Create a path integrator and sampler for a physical teacher."""
    _validate_positive_integer(
        radiance_samples,
        "radiance_samples",
    )
    _validate_positive_integer(
        max_depth,
        "max_depth",
    )
    _validate_non_negative_integer(
        rr_depth,
        "rr_depth",
    )
    _validate_non_negative_integer(
        seed,
        "seed",
    )

    if rr_depth > max_depth:
        raise ValueError(
            "rr_depth must be less than or equal to max_depth."
        )

    integrator = _mi.load_dict(
        {
            "type": "path",
            "max_depth": max_depth,
            "rr_depth": rr_depth,
        }
    )

    sampler = _mi.load_dict(
        {
            "type": "independent",
            "sample_count": radiance_samples,
        }
    )

    sampler.seed(
        seed,
        wavefront_size=1,
    )

    return IncidentRadianceRuntime(
        integrator=integrator,
        sampler=sampler,
        radiance_samples=radiance_samples,
    )


def estimate_incident_radiance(
    *,
    runtime: IncidentRadianceRuntime,
    scene: Any,
    ray: Any,
) -> FloatArray:
    """Estimate mean RGB incident radiance along one ray."""
    accumulated_radiance = np.zeros(
        3,
        dtype=np.float64,
    )

    for _ in range(runtime.radiance_samples):
        radiance, _, _ = runtime.integrator.sample(
            scene,
            runtime.sampler,
            ray,
        )

        accumulated_radiance += mitsuba_spectrum_to_rgb(
            radiance
        )

        runtime.sampler.schedule_state()

    return accumulated_radiance / runtime.radiance_samples


def mitsuba_spectrum_to_rgb(
    spectrum: Any,
) -> FloatArray:
    """Convert a Mitsuba RGB spectrum into a validated NumPy array."""
    rgb = np.array(
        [
            float(spectrum[0]),
            float(spectrum[1]),
            float(spectrum[2]),
        ],
        dtype=np.float64,
    )

    if not bool(np.all(np.isfinite(rgb))):
        raise ValueError(
            "Incident radiance must contain only finite values."
        )

    if bool(np.any(rgb < 0.0)):
        raise ValueError(
            "Incident radiance must be non-negative."
        )

    return rgb


def _validate_positive_integer(
    value: int,
    name: str,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
    ):
        raise TypeError(
            f"{name} must be an integer."
        )

    if value <= 0:
        raise ValueError(
            f"{name} must be greater than zero."
        )


def _validate_non_negative_integer(
    value: int,
    name: str,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
    ):
        raise TypeError(
            f"{name} must be an integer."
        )

    if value < 0:
        raise ValueError(
            f"{name} must be non-negative."
        )