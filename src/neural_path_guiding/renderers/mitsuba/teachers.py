"""Mitsuba teacher estimators for directional guiding targets.

Direction convention:

- normal is expressed in world space
- outgoing_direction points away from the surface
- world_direction is the candidate incoming-light direction

Targets:

- T0: visibility/cosine heuristic
- T1: incident-radiance luminance
- T2: incident radiance times BSDF times cosine
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import mitsuba as mi
import numpy as np

from neural_path_guiding.core.bins import HemisphereBins
from neural_path_guiding.core.color import rgb_luminance
from neural_path_guiding.core.features import FloatArray
from neural_path_guiding.core.frames import (
    build_frame_from_normal,
    local_to_world,
)
from neural_path_guiding.renderers.mitsuba.adapters import (
    normalize_numpy_vector,
)
from neural_path_guiding.renderers.mitsuba.bsdf import (
    evaluate_bsdf_times_cosine,
)
from neural_path_guiding.renderers.mitsuba.radiance import (
    IncidentRadianceRuntime,
    create_incident_radiance_runtime,
    estimate_incident_radiance,
)


_mi: Any = mi

TeacherEstimator = Callable[..., float]


VISIBILITY_COSINE_TARGET = "visibility_cosine_v0"
INCIDENT_RADIANCE_TARGET = "incident_radiance_v1"
PRODUCT_INTEGRAND_TARGET = "product_integrand_v1"


TEACHER_TARGET_DESCRIPTIONS = {
    VISIBILITY_COSINE_TARGET: (
        "T0: heuristic visibility multiplied by cosine"
    ),
    INCIDENT_RADIANCE_TARGET: (
        "T1: incident radiance luminance"
    ),
    PRODUCT_INTEGRAND_TARGET: (
        "T2: incident radiance times BSDF times cosine"
    ),
}


DEFINED_TARGET_TYPES = frozenset(
    TEACHER_TARGET_DESCRIPTIONS
)


PHYSICAL_TARGET_TYPES = frozenset(
    {
        INCIDENT_RADIANCE_TARGET,
        PRODUCT_INTEGRAND_TARGET,
    }
)


# Every currently defined target has an implementation.
SUPPORTED_TARGET_TYPES = frozenset(
    {
        VISIBILITY_COSINE_TARGET,
        INCIDENT_RADIANCE_TARGET,
        PRODUCT_INTEGRAND_TARGET,
    }
)


@dataclass(frozen=True)
class TeacherContext:
    """Information shared by directional teacher evaluations."""

    scene: Any
    surface_interaction: Any
    normal: FloatArray
    outgoing_direction: FloatArray


@dataclass(frozen=True)
class VisibilityCosineTeacherSettings:
    """Configuration for the T0 teacher."""

    target_type: str
    samples_per_bin: int
    smoothing: float
    max_distance: float
    environment_weight: float
    emitter_weight: float
    occluded_weight: float


@dataclass(frozen=True)
class PhysicalTeacherSettings:
    """Configuration shared by T1 and T2."""

    target_type: str
    samples_per_bin: int
    radiance_samples: int
    smoothing: float
    max_depth: int
    rr_depth: int


TeacherSettings = (
    VisibilityCosineTeacherSettings
    | PhysicalTeacherSettings
)

TeacherRuntime = IncidentRadianceRuntime | None


def validate_teacher_target_type(
    target_type: str,
) -> None:
    """Reject unknown or unsupported targets."""
    if target_type in SUPPORTED_TARGET_TYPES:
        return

    if target_type in DEFINED_TARGET_TYPES:
        raise ValueError(
            "Teacher target_type is defined but not "
            f"implemented yet: {target_type}."
        )

    supported = ", ".join(
        sorted(SUPPORTED_TARGET_TYPES)
    )

    raise ValueError(
        f"Unsupported teacher target_type: {target_type}. "
        f"Supported values: {supported}."
    )


def validate_physical_teacher_settings(
    teacher: PhysicalTeacherSettings,
) -> None:
    """Validate parameters shared by T1 and T2."""
    if teacher.target_type not in PHYSICAL_TARGET_TYPES:
        raise ValueError(
            "PhysicalTeacherSettings requires either "
            f"{INCIDENT_RADIANCE_TARGET} or "
            f"{PRODUCT_INTEGRAND_TARGET}."
        )

    _validate_positive_integer(
        teacher.samples_per_bin,
        "samples_per_bin",
    )
    _validate_positive_integer(
        teacher.radiance_samples,
        "radiance_samples",
    )
    _validate_positive_integer(
        teacher.max_depth,
        "max_depth",
    )
    _validate_non_negative_integer(
        teacher.rr_depth,
        "rr_depth",
    )

    if teacher.rr_depth > teacher.max_depth:
        raise ValueError(
            "rr_depth must be less than or equal to max_depth."
        )

    _validate_finite_non_negative(
        teacher.smoothing,
        "smoothing",
    )


def validate_visibility_cosine_teacher_settings(
    teacher: VisibilityCosineTeacherSettings,
) -> None:
    """Validate parameters belonging to T0."""
    if teacher.target_type != VISIBILITY_COSINE_TARGET:
        raise ValueError(
            "VisibilityCosineTeacherSettings requires "
            f"target_type='{VISIBILITY_COSINE_TARGET}'."
        )

    _validate_positive_integer(
        teacher.samples_per_bin,
        "samples_per_bin",
    )
    _validate_finite_non_negative(
        teacher.smoothing,
        "smoothing",
    )
    _validate_finite_positive(
        teacher.max_distance,
        "max_distance",
    )
    _validate_finite_non_negative(
        teacher.environment_weight,
        "environment_weight",
    )
    _validate_finite_non_negative(
        teacher.emitter_weight,
        "emitter_weight",
    )
    _validate_finite_non_negative(
        teacher.occluded_weight,
        "occluded_weight",
    )


def validate_teacher_settings(
    teacher: TeacherSettings,
) -> None:
    """Validate any recognized teacher configuration."""
    match teacher:
        case VisibilityCosineTeacherSettings():
            validate_visibility_cosine_teacher_settings(
                teacher
            )

        case PhysicalTeacherSettings():
            validate_physical_teacher_settings(
                teacher
            )

        case _:
            raise TypeError(
                "teacher must be VisibilityCosineTeacherSettings "
                "or PhysicalTeacherSettings."
            )


def build_teacher_metadata(
    teacher: TeacherSettings,
) -> dict[str, object]:
    """Build serializable teacher metadata."""
    common: dict[str, object] = {
        "target_type": teacher.target_type,
        "samples_per_bin": teacher.samples_per_bin,
        "smoothing": teacher.smoothing,
    }

    match teacher:
        case VisibilityCosineTeacherSettings():
            return {
                **common,
                "max_distance": teacher.max_distance,
                "environment_weight": teacher.environment_weight,
                "emitter_weight": teacher.emitter_weight,
                "occluded_weight": teacher.occluded_weight,
            }

        case PhysicalTeacherSettings():
            return {
                **common,
                "radiance_samples": teacher.radiance_samples,
                "max_depth": teacher.max_depth,
                "rr_depth": teacher.rr_depth,
            }

        case _:
            raise TypeError(
                "Cannot build metadata for an unknown teacher."
            )


def create_teacher_runtime(
    teacher: TeacherSettings,
    seed: int,
) -> TeacherRuntime:
    """Create reusable runtime state for one dataset generation."""
    validate_teacher_settings(teacher)
    validate_teacher_target_type(teacher.target_type)

    match teacher:
        case VisibilityCosineTeacherSettings():
            return None

        case PhysicalTeacherSettings():
            return create_incident_radiance_runtime(
                radiance_samples=teacher.radiance_samples,
                max_depth=teacher.max_depth,
                rr_depth=teacher.rr_depth,
                seed=seed,
            )

        case _:
            raise TypeError(
                "Cannot create runtime for an unknown teacher."
            )


def estimate_bin_contributions(
    context: TeacherContext,
    bins: HemisphereBins,
    teacher: TeacherSettings,
    rng: np.random.Generator,
    runtime: TeacherRuntime = None,
) -> FloatArray:
    """Estimate one unnormalized contribution per directional bin."""
    contributions = np.zeros(
        bins.n_bins,
        dtype=np.float64,
    )

    frame = build_frame_from_normal(
        context.normal
    )

    estimator = resolve_teacher_estimator(
        teacher.target_type
    )

    for bin_index in range(bins.n_bins):
        accumulated = 0.0

        for _ in range(teacher.samples_per_bin):
            local_direction = bins.sample_direction_in_bin(
                bin_index=bin_index,
                u_mu=float(rng.random()),
                u_phi=float(rng.random()),
            )

            world_direction = normalize_numpy_vector(
                local_to_world(
                    frame,
                    local_direction,
                )
            )

            accumulated += estimator(
                context=context,
                world_direction=world_direction,
                teacher=teacher,
                runtime=runtime,
            )

        contributions[bin_index] = (
            accumulated
            / teacher.samples_per_bin
        )

    return contributions


def estimate_visibility_cosine_contribution(
    context: TeacherContext,
    world_direction: FloatArray,
    teacher: TeacherSettings,
    runtime: TeacherRuntime = None,
) -> float:
    """Evaluate the T0 sanity-check teacher."""
    del runtime

    if not isinstance(
        teacher,
        VisibilityCosineTeacherSettings,
    ):
        raise TypeError(
            "T0 requires VisibilityCosineTeacherSettings."
        )

    normal = normalize_numpy_vector(
        context.normal
    )

    direction = normalize_numpy_vector(
        world_direction
    )

    cosine = max(
        0.0,
        float(np.dot(normal, direction)),
    )

    if cosine <= 0.0:
        return 0.0

    ray = context.surface_interaction.spawn_ray(
        _mi.Vector3f(*direction)
    )

    ray.maxt = teacher.max_distance
    hit = context.scene.ray_intersect(ray)

    if not bool(hit.is_valid()):
        return teacher.environment_weight * cosine

    if surface_hit_has_emitter(
        hit,
        context.scene,
    ):
        return teacher.emitter_weight * cosine

    return teacher.occluded_weight * cosine


def estimate_incident_radiance_contribution(
    context: TeacherContext,
    world_direction: FloatArray,
    teacher: TeacherSettings,
    runtime: TeacherRuntime = None,
) -> float:
    """Evaluate T1 as luminance of Li(x, wi)."""
    if not isinstance(
        teacher,
        PhysicalTeacherSettings,
    ):
        raise TypeError(
            "T1 requires PhysicalTeacherSettings."
        )

    if teacher.target_type != INCIDENT_RADIANCE_TARGET:
        raise ValueError(
            "The incident-radiance estimator requires "
            f"target_type='{INCIDENT_RADIANCE_TARGET}'."
        )

    if runtime is None:
        raise ValueError(
            "T1 requires an IncidentRadianceRuntime."
        )

    normal = normalize_numpy_vector(
        context.normal
    )

    direction = normalize_numpy_vector(
        world_direction
    )

    if float(np.dot(normal, direction)) <= 0.0:
        return 0.0

    ray = context.surface_interaction.spawn_ray(
        _mi.Vector3f(*direction)
    )

    incident_radiance = estimate_incident_radiance(
        runtime=runtime,
        scene=context.scene,
        ray=ray,
    )

    return float(
        rgb_luminance(incident_radiance)
    )


def estimate_product_integrand_contribution(
    context: TeacherContext,
    world_direction: FloatArray,
    teacher: TeacherSettings,
    runtime: TeacherRuntime = None,
) -> float:
    """Evaluate T2 as luminance of Li times BSDF times cosine.

    evaluate_bsdf_times_cosine() already includes the cosine term
    returned by Mitsuba BSDF.eval(). It must not be multiplied again.
    """
    if not isinstance(
        teacher,
        PhysicalTeacherSettings,
    ):
        raise TypeError(
            "T2 requires PhysicalTeacherSettings."
        )

    if teacher.target_type != PRODUCT_INTEGRAND_TARGET:
        raise ValueError(
            "The product-integrand estimator requires "
            f"target_type='{PRODUCT_INTEGRAND_TARGET}'."
        )

    if runtime is None:
        raise ValueError(
            "T2 requires an IncidentRadianceRuntime."
        )

    normal = normalize_numpy_vector(
        context.normal
    )

    direction = normalize_numpy_vector(
        world_direction
    )

    if float(np.dot(normal, direction)) <= 0.0:
        return 0.0

    ray = context.surface_interaction.spawn_ray(
        _mi.Vector3f(*direction)
    )

    incident_radiance = estimate_incident_radiance(
        runtime=runtime,
        scene=context.scene,
        ray=ray,
    )

    bsdf_times_cosine = evaluate_bsdf_times_cosine(
        surface_interaction=context.surface_interaction,
        world_direction=direction,
    )

    product = (
        incident_radiance
        * bsdf_times_cosine
    )

    return float(
        rgb_luminance(product)
    )


def surface_hit_has_emitter(
    surface_interaction: Any,
    scene: Any,
) -> bool:
    """Return whether an interaction belongs to an emitter."""
    return (
        surface_interaction.emitter(scene)
        is not None
    )


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


def _validate_finite_positive(
    value: float,
    name: str,
) -> None:
    if (
        not np.isfinite(value)
        or value <= 0.0
    ):
        raise ValueError(
            f"{name} must be finite and greater than zero."
        )


def _validate_finite_non_negative(
    value: float,
    name: str,
) -> None:
    if (
        not np.isfinite(value)
        or value < 0.0
    ):
        raise ValueError(
            f"{name} must be finite and non-negative."
        )


_TEACHER_ESTIMATORS: dict[str, TeacherEstimator] = {
    VISIBILITY_COSINE_TARGET: (
        estimate_visibility_cosine_contribution
    ),
    INCIDENT_RADIANCE_TARGET: (
        estimate_incident_radiance_contribution
    ),
    PRODUCT_INTEGRAND_TARGET: (
        estimate_product_integrand_contribution
    ),
}


def resolve_teacher_estimator(
    target_type: str,
) -> TeacherEstimator:
    """Return the estimator associated with a supported target."""
    validate_teacher_target_type(target_type)

    return _TEACHER_ESTIMATORS[target_type]