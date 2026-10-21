"""Evaluate and diagnose emitter PDFs in Mitsuba 3.8.0.

Designed for scalar_rgb and fixed-point direct-light MIS.

The visibility-safe ray is authoritative for identifying
the visible emitter.

An exact-origin geometric ray is used for area-light PDF
reconstruction only when its intersection is consistent.

Fallback diagnostics are collected without modifying
the estimated PDF, sample contribution, or MIS weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mitsuba as mi
import numpy as np

from neural_path_guiding.renderers.mitsuba.adapters import (
    mitsuba_vector_to_numpy,
    normalize_numpy_vector,
)


FALLBACK_REASONS = (
    "geometric_miss",
    "different_shape",
    "position_mismatch",
    "non_emissive_geometric_hit",
    "unspecified",
)


@dataclass(frozen=True)
class EmitterPdfQuery:
    direction_world: np.ndarray
    pdf: float
    hit_emitter: bool
    environment: bool
    used_geometric_hit: bool = False
    used_visibility_fallback: bool = False

    # Optional diagnostics.
    fallback_reason: str | None = None
    geometric_hit_distance: float | None = None
    position_difference: float | None = None
    diagnostic: dict[str, Any] | None = None


@dataclass
class EmitterPdfStats:
    """Diagnostics for one estimator run, without global state."""

    total_queries: int = 0
    zero_pdf_queries: int = 0
    blocked_queries: int = 0
    environment_queries: int = 0
    geometric_queries: int = 0
    fallback_queries: int = 0

    fallback_reasons: dict[str, int] = field(
        default_factory=lambda: {
            reason: 0 for reason in FALLBACK_REASONS
        }
    )

    max_diagnostic_examples: int = 20

    fallback_examples: list[dict[str, Any]] = field(
        default_factory=list
    )

    def record(self, query: EmitterPdfQuery) -> None:
        """Record one successfully evaluated emitter PDF."""

        if query.used_geometric_hit and query.used_visibility_fallback:
            raise ValueError(
                "A query cannot use both geometric and fallback hits."
            )

        if not query.hit_emitter:
            category = "blocked"

        elif query.environment:
            if (
                query.used_geometric_hit
                or query.used_visibility_fallback
            ):
                raise ValueError(
                    "Environment queries cannot use area-hit modes."
                )

            category = "environment"

        elif query.used_geometric_hit:
            category = "geometric"

        elif query.used_visibility_fallback:
            category = "fallback"

        else:
            raise ValueError(
                "Unclassified finite-emitter PDF query."
            )

        reason = None

        if category == "fallback":
            reason = query.fallback_reason or "unspecified"

            if reason not in self.fallback_reasons:
                raise ValueError(
                    f"Unknown fallback reason: {reason}"
                )

        self.total_queries += 1

        if query.pdf == 0.0:
            self.zero_pdf_queries += 1

        if category == "blocked":
            self.blocked_queries += 1

        elif category == "environment":
            self.environment_queries += 1

        elif category == "geometric":
            self.geometric_queries += 1

        else:
            self.fallback_queries += 1
            assert reason is not None

            self.fallback_reasons[reason] += 1

            if (
                query.diagnostic is not None
                and len(self.fallback_examples)
                < self.max_diagnostic_examples
            ):
                self.fallback_examples.append(
                    dict(query.diagnostic)
                )

    def to_dict(self) -> dict[str, Any]:
        """Export JSON-compatible diagnostic information."""

        classified = (
            self.blocked_queries
            + self.environment_queries
            + self.geometric_queries
            + self.fallback_queries
        )

        if classified != self.total_queries:
            raise ValueError(
                "Emitter PDF diagnostic counters are inconsistent."
            )

        total_area = (
            self.geometric_queries
            + self.fallback_queries
        )

        return {
            "total_queries": self.total_queries,
            "zero_pdf_queries": self.zero_pdf_queries,
            "blocked_queries": self.blocked_queries,
            "environment_queries": self.environment_queries,
            "geometric_queries": self.geometric_queries,
            "fallback_queries": self.fallback_queries,
            "fallback_rate": (
                self.fallback_queries / self.total_queries
                if self.total_queries
                else 0.0
            ),
            "fallback_rate_among_area_hits": (
                self.fallback_queries / total_area
                if total_area
                else 0.0
            ),
            "fallback_reasons": dict(self.fallback_reasons),
            "fallback_examples": list(self.fallback_examples),
        }


def _validate_pdf(value: Any) -> float:
    pdf = float(value)

    if not np.isfinite(pdf):
        raise ValueError("Emitter PDF must be finite.")

    if pdf < 0.0:
        raise ValueError("Emitter PDF must be non-negative.")

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
    position = np.asarray(
        mitsuba_vector_to_numpy(interaction.p),
        dtype=np.float64,
    )

    if not np.all(np.isfinite(position)):
        raise ValueError("Non-finite surface position.")

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

    return bool(first.shape == second.shape)


def _compatible_emitter_intersections(
    *,
    original_interaction: Any,
    visibility_hit: Any,
    geometric_hit: Any,
    position_tolerance: float,
) -> bool:
    """Preserve the established geometric compatibility test."""

    if not _same_shape(visibility_hit, geometric_hit):
        return False

    p_visibility = _position(visibility_hit)
    p_geometric = _position(geometric_hit)
    p_origin = _position(original_interaction)

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
        position_tolerance * max(1.0, distance)
    )

    return difference <= allowed_difference


def _fallback_reason(
    *,
    original_interaction: Any,
    visibility_hit: Any,
    geometric_hit: Any,
    position_tolerance: float,
) -> tuple[str | None, float | None, float | None]:
    """Classify geometry disagreement without changing ray behavior.

    Returns:
        reason
        geometric_hit_distance
        position_difference

    A None reason means geometrically compatible intersections.
    """

    if not bool(geometric_hit.is_valid()):
        return "geometric_miss", None, None

    origin = _position(original_interaction)
    geometric_position = _position(geometric_hit)

    geometric_distance = float(
        np.linalg.norm(geometric_position - origin)
    )

    if not _same_shape(visibility_hit, geometric_hit):
        return "different_shape", geometric_distance, None

    visibility_position = _position(visibility_hit)

    position_difference = float(
        np.linalg.norm(
            visibility_position - geometric_position
        )
    )

    allowed_difference = (
        position_tolerance
        * max(1.0, geometric_distance)
    )

    if position_difference > allowed_difference:
        return (
            "position_mismatch",
            geometric_distance,
            position_difference,
        )

    return None, geometric_distance, position_difference


def _object_id(value: Any) -> str:
    """Extract an object identifier for diagnostics."""

    if value is None:
        return "none"

    method = getattr(value, "id", None)

    if callable(method):
        identifier = str(method())

        if identifier:
            return identifier

    return type(value).__name__


def _direction_sample_pdf(
    *,
    scene: Any,
    interaction: Any,
    emitter_hit: Any,
) -> float:
    ds = mi.DirectionSample3f(
        scene,
        si=emitter_hit,
        ref=interaction,
    )

    return _validate_pdf(
        scene.pdf_emitter_direction(
            interaction,
            ds,
            True,
        )
    )


def _fallback_diagnostic(
    *,
    direction: np.ndarray,
    original_interaction: Any,
    visibility_hit: Any,
    geometric_hit: Any,
    reason: str,
    pdf: float,
    geometric_distance: float | None,
    position_difference: float | None,
) -> dict[str, Any]:
    """Prepare a small JSON-safe record for a fallback."""

    geometric_valid = bool(geometric_hit.is_valid())

    return {
        "reason": reason,
        "direction_world": [
            float(value) for value in direction
        ],
        "pdf_used": float(pdf),
        "origin_position": _position(
            original_interaction
        ).tolist(),
        "visibility_position": _position(
            visibility_hit
        ).tolist(),
        "geometric_valid": geometric_valid,
        "geometric_position": (
            _position(geometric_hit).tolist()
            if geometric_valid
            else None
        ),
        "visibility_shape": _object_id(
            visibility_hit.shape
        ),
        "geometric_shape": (
            _object_id(geometric_hit.shape)
            if geometric_valid
            else "none"
        ),
        "geometric_hit_distance": geometric_distance,
        "position_difference": position_difference,
    }


def evaluate_emitter_pdf(
    *,
    scene: Any,
    interaction: Any,
    world_direction: np.ndarray,
    position_tolerance: float = 1e-3,
    stats: EmitterPdfStats | None = None,
) -> EmitterPdfQuery:
    """Evaluate the emitter PDF with optional fallback diagnostics."""

    if (
        not np.isfinite(position_tolerance)
        or position_tolerance <= 0.0
    ):
        raise ValueError(
            "position_tolerance must be finite and positive."
        )

    direction = normalize_numpy_vector(world_direction)

    mi_direction = mi.Vector3f(
        float(direction[0]),
        float(direction[1]),
        float(direction[2]),
    )

    visibility_ray = interaction.spawn_ray(
        mi_direction
    )

    visibility_hit = scene.ray_intersect(
        visibility_ray
    )

    if not bool(visibility_hit.is_valid()):
        environment = scene.environment()

        if environment is None:
            result = _zero_result(direction)

        else:
            pdf = _direction_sample_pdf(
                scene=scene,
                interaction=interaction,
                emitter_hit=visibility_hit,
            )

            result = EmitterPdfQuery(
                direction_world=direction,
                pdf=pdf,
                hit_emitter=True,
                environment=True,
            )

    else:
        visibility_emitter = visibility_hit.emitter(
            scene
        )

        if visibility_emitter is None:
            result = _zero_result(direction)

        else:
            geometric_ray = mi.Ray3f(
                interaction.p,
                mi_direction,
                interaction.time,
            )

            geometric_hit = scene.ray_intersect(
                geometric_ray
            )

            (
                reason,
                geometric_distance,
                position_difference,
            ) = _fallback_reason(
                original_interaction=interaction,
                visibility_hit=visibility_hit,
                geometric_hit=geometric_hit,
                position_tolerance=position_tolerance,
            )

            # Same logic as the previous version:
            # accept exact geometry only if it agrees
            # and the geometric hit is emissive.
            if (
                reason is None
                and geometric_hit.emitter(scene) is not None
            ):
                pdf = _direction_sample_pdf(
                    scene=scene,
                    interaction=interaction,
                    emitter_hit=geometric_hit,
                )

                result = EmitterPdfQuery(
                    direction_world=direction,
                    pdf=pdf,
                    hit_emitter=True,
                    environment=False,
                    used_geometric_hit=True,
                    geometric_hit_distance=geometric_distance,
                    position_difference=position_difference,
                )

            else:
                if reason is None:
                    reason = "non_emissive_geometric_hit"

                pdf = _direction_sample_pdf(
                    scene=scene,
                    interaction=interaction,
                    emitter_hit=visibility_hit,
                )

                diagnostic = None

                if stats is not None:
                    diagnostic = _fallback_diagnostic(
                        direction=direction,
                        original_interaction=interaction,
                        visibility_hit=visibility_hit,
                        geometric_hit=geometric_hit,
                        reason=reason,
                        pdf=pdf,
                        geometric_distance=geometric_distance,
                        position_difference=position_difference,
                    )

                result = EmitterPdfQuery(
                    direction_world=direction,
                    pdf=pdf,
                    hit_emitter=True,
                    environment=False,
                    used_visibility_fallback=True,
                    fallback_reason=reason,
                    geometric_hit_distance=geometric_distance,
                    position_difference=position_difference,
                    diagnostic=diagnostic,
                )

    if stats is not None:
        stats.record(result)

    return result