"""Evaluate emitter PDFs for arbitrary world-space directions.

Target:
    Mitsuba 3.8.0
    scalar_rgb

Used by Neural + BSDF + Light MIS.

The visibility-safe ray determines which emitter is reached.

For area emitters, a second ray from the original shading
position is used only when its intersection is consistent
with the visibility ray.

If the exact-origin ray suffers a self-intersection, the
visibility-safe record is used instead.

Limitations:
    - Scalar Mitsuba variants.
    - Opaque surfaces.
    - No participating media.
    - No null-transmission chains.
    - Continuous, non-delta emitter PDFs only.

The fallback can have a small geometric PDF discrepancy.
This must be measured before claiming unbiased rendering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mitsuba as mi
import numpy as np

from neural_path_guiding.renderers.mitsuba.adapters import (
    mitsuba_vector_to_numpy,
    normalize_numpy_vector,
)


@dataclass(frozen=True)
class EmitterPdfQuery:
    direction_world: np.ndarray
    pdf: float
    hit_emitter: bool
    environment: bool

    # Diagnostic information.
    used_geometric_hit: bool = False
    used_visibility_fallback: bool = False


def _validate_pdf(value: Any) -> float:
    pdf = float(value)

    if not np.isfinite(pdf):
        raise ValueError(
            "Emitter PDF must be finite."
        )

    if pdf < 0.0:
        raise ValueError(
            "Emitter PDF must be non-negative."
        )

    return pdf


def _zero_result(
    direction: np.ndarray,
) -> EmitterPdfQuery:
    return EmitterPdfQuery(
        direction_world=direction,
        pdf=0.0,
        hit_emitter=False,
        environment=False,
    )


def _position(interaction: Any) -> np.ndarray:
    position = mitsuba_vector_to_numpy(
        interaction.p
    )

    if not np.all(np.isfinite(position)):
        raise ValueError(
            "Non-finite surface position."
        )

    return position


def _same_shape(
    first: Any,
    second: Any,
) -> bool:
    if not bool(first.is_valid()):
        return False

    if not bool(second.is_valid()):
        return False

    if first.shape is None or second.shape is None:
        return False

    return bool(
        first.shape == second.shape
    )


def _compatible_emitter_intersections(
    *,
    original_interaction: Any,
    visibility_hit: Any,
    geometric_hit: Any,
    position_tolerance: float,
) -> bool:
    """Check if two hits describe the same visible emitter.

    Returns False for inconsistent or self-intersecting
    geometric rays. Never assumes that a nearby hit is
    automatically the same emitter.
    """

    if not _same_shape(
        visibility_hit,
        geometric_hit,
    ):
        return False

    p_visibility = _position(
        visibility_hit
    )

    p_geometric = _position(
        geometric_hit
    )

    p_origin = _position(
        original_interaction
    )

    difference = float(
        np.linalg.norm(
            p_visibility - p_geometric
        )
    )

    distance = float(
        np.linalg.norm(
            p_geometric - p_origin
        )
    )

    allowed_difference = (
        position_tolerance
        * max(1.0, distance)
    )

    return difference <= allowed_difference


def _direction_sample_pdf(
    *,
    scene: Any,
    interaction: Any,
    emitter_hit: Any,
) -> float:
    """Build a DirectionSample3f and evaluate its PDF."""

    direction_sample = mi.DirectionSample3f(
        scene,
        si=emitter_hit,
        ref=interaction,
    )

    return _validate_pdf(
        scene.pdf_emitter_direction(
            interaction,
            direction_sample,
            True,
        )
    )


def evaluate_emitter_pdf(
    *,
    scene: Any,
    interaction: Any,
    world_direction: np.ndarray,
    position_tolerance: float = 1e-3,
) -> EmitterPdfQuery:
    """Evaluate emitter PDF in solid angle.

    Strategy:

    1. Trace a visibility-safe ray.
    2. Return zero if the first surface is non-emissive.
    3. For environment rays, use the visibility-safe miss.
    4. For area emitters, attempt exact-origin reconstruction.
    5. Use exact geometry only when both hits agree.
    6. Otherwise fall back to the visibility-safe emitter hit.

    The visibility ray is authoritative for identifying
    which emitter is reached.

    A fallback is recorded explicitly for later diagnostics.
    """

    if (
        not np.isfinite(position_tolerance)
        or position_tolerance <= 0.0
    ):
        raise ValueError(
            "position_tolerance must be finite and positive."
        )

    direction = normalize_numpy_vector(
        world_direction
    )

    mi_direction = mi.Vector3f(
        float(direction[0]),
        float(direction[1]),
        float(direction[2]),
    )

    # --------------------------------------------------
    # 1. Trace a visibility-safe ray.
    # --------------------------------------------------

    visibility_ray = interaction.spawn_ray(
        mi_direction
    )

    visibility_hit = scene.ray_intersect(
        visibility_ray
    )

    visibility_valid = bool(
        visibility_hit.is_valid()
    )

    # --------------------------------------------------
    # 2. Handle environment directions.
    # --------------------------------------------------

    if not visibility_valid:

        environment = scene.environment()

        if environment is None:
            return _zero_result(direction)

        # No second geometric ray is necessary.
        # The visibility-safe ray already established
        # that the direction reaches the environment.
        pdf = _direction_sample_pdf(
            scene=scene,
            interaction=interaction,
            emitter_hit=visibility_hit,
        )

        return EmitterPdfQuery(
            direction_world=direction,
            pdf=pdf,
            hit_emitter=True,
            environment=True,
            used_geometric_hit=False,
            used_visibility_fallback=False,
        )

    # --------------------------------------------------
    # 3. Check the first visible surface.
    # --------------------------------------------------

    visibility_emitter = visibility_hit.emitter(
        scene
    )

    if visibility_emitter is None:
        # The direction reaches an opaque,
        # non-emissive surface first.
        return _zero_result(direction)

    # --------------------------------------------------
    # 4. Attempt more precise geometric reconstruction.
    # --------------------------------------------------

    geometric_ray = mi.Ray3f(
        interaction.p,
        mi_direction,
        interaction.time,
    )

    geometric_hit = scene.ray_intersect(
        geometric_ray
    )

    # --------------------------------------------------
    # 5. Use geometric hit only when consistent.
    # --------------------------------------------------

    compatible = _compatible_emitter_intersections(
        original_interaction=interaction,
        visibility_hit=visibility_hit,
        geometric_hit=geometric_hit,
        position_tolerance=position_tolerance,
    )

    if compatible:

        geometric_emitter = geometric_hit.emitter(
            scene
        )

        if geometric_emitter is not None:

            pdf = _direction_sample_pdf(
                scene=scene,
                interaction=interaction,
                emitter_hit=geometric_hit,
            )

            return EmitterPdfQuery(
                direction_world=direction,
                pdf=pdf,
                hit_emitter=True,
                environment=False,
                used_geometric_hit=True,
                used_visibility_fallback=False,
            )

    # --------------------------------------------------
    # 6. Fall back to the visibility-safe emitter record.
    # --------------------------------------------------

    # We do not discard the sample and do not assume
    # that the emitter PDF is zero.
    #
    # This avoids a hard failure but may introduce
    # a small numerical PDF discrepancy because the
    # emitter intersection came from an offset ray.
    pdf = _direction_sample_pdf(
        scene=scene,
        interaction=interaction,
        emitter_hit=visibility_hit,
    )

    return EmitterPdfQuery(
        direction_world=direction,
        pdf=pdf,
        hit_emitter=True,
        environment=False,
        used_geometric_hit=False,
        used_visibility_fallback=True,
    )