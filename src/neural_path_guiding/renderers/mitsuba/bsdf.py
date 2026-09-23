"""Mitsuba BSDF evaluation utilities for physical teachers and MIS.

Mitsuba's BSDF.eval() returns the BSDF value already multiplied by
the cosine foreshortening term. Callers must not multiply by cosine
again.
"""

from __future__ import annotations

from typing import Any

import mitsuba as mi
import numpy as np

from neural_path_guiding.core.features import FloatArray
from neural_path_guiding.renderers.mitsuba.adapters import (
    normalize_numpy_vector,
)


_mi: Any = mi


def evaluate_bsdf_times_cosine(
    *,
    surface_interaction: Any,
    world_direction: FloatArray,
) -> FloatArray:
    """Evaluate the smooth BSDF multiplied by its cosine term."""

    bsdf, context, local_direction = _prepare_bsdf_query(
        surface_interaction=surface_interaction,
        world_direction=world_direction,
    )

    value = bsdf.eval(
        context,
        surface_interaction,
        local_direction,
    )

    return mitsuba_bsdf_to_rgb(value)


def evaluate_bsdf_pdf(
    *,
    surface_interaction: Any,
    world_direction: FloatArray,
) -> float:
    """Evaluate the BSDF sampling PDF for a world-space direction.

    The returned density is with respect to solid angle, like the neural
    guiding PDF. Both proposals can therefore be compared by MIS as long
    as they are evaluated for the same world-space direction.
    """

    bsdf, context, local_direction = _prepare_bsdf_query(
        surface_interaction=surface_interaction,
        world_direction=world_direction,
    )

    value = float(
        bsdf.pdf(
            context,
            surface_interaction,
            local_direction,
        )
    )

    if not np.isfinite(value) or value < 0.0:
        raise ValueError(
            "BSDF PDF must be finite and non-negative."
        )

    return value


def _prepare_bsdf_query(
    *,
    surface_interaction: Any,
    world_direction: FloatArray,
) -> tuple[Any, Any, Any]:
    """Normalize a world direction and prepare a Mitsuba BSDF query."""

    normalized_direction = normalize_numpy_vector(
        world_direction
    )

    mitsuba_direction = _mi.Vector3f(
        *normalized_direction
    )
    local_direction = surface_interaction.to_local(
        mitsuba_direction
    )

    bsdf = surface_interaction.bsdf()

    if bsdf is None:
        raise ValueError(
            "Surface interaction does not contain a BSDF."
        )

    context = _mi.BSDFContext()

    return bsdf, context, local_direction


def mitsuba_bsdf_to_rgb(
    value: Any,
) -> FloatArray:
    """Convert a Mitsuba RGB BSDF value to a validated array."""

    rgb = np.array(
        [
            float(value[0]),
            float(value[1]),
            float(value[2]),
        ],
        dtype=np.float64,
    )

    if not bool(np.all(np.isfinite(rgb))):
        raise ValueError(
            "BSDF evaluation must contain only finite values."
        )

    if bool(np.any(rgb < 0.0)):
        raise ValueError(
            "BSDF evaluation must be non-negative."
        )

    return rgb