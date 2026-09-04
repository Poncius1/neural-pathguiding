"""Build Mitsuba dataset-generation settings from versioned configuration."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from neural_path_guiding.config import (
    optional_float,
    optional_int,
    require_float,
    require_float_sequence,
    require_int,
    require_mapping,
    require_string,
    resolve_project_path,
)
from neural_path_guiding.core.bins import HemisphereBins
from neural_path_guiding.renderers.mitsuba.dataset_generator import (
    DatasetCamera,
    DatasetGenerationSettings,
    DatasetProvenance,
)
from neural_path_guiding.renderers.mitsuba.scene_loader import (
    DEFAULT_VARIANT,
    configure_mitsuba,
    load_scene_from_config,
    validate_windows_llvm_runtime,
)
from neural_path_guiding.renderers.mitsuba.teachers import (
    INCIDENT_RADIANCE_TARGET,
    PRODUCT_INTEGRAND_TARGET,
    VISIBILITY_COSINE_TARGET,
    PhysicalTeacherSettings,
    TeacherSettings,
    VisibilityCosineTeacherSettings,
    validate_teacher_settings,
    validate_teacher_target_type,
)


TeacherBuilder = Callable[
    [dict[str, Any], str],
    TeacherSettings,
]


def build_settings_from_config(
    config: dict[str, Any],
    config_path: Path,
    project_root: Path,
) -> DatasetGenerationSettings:
    """Convert YAML data into validated generator settings.

    Configuration validation happens before Mitsuba is configured and before
    the scene is loaded. This keeps configuration errors inexpensive and easy
    to understand.
    """
    mitsuba_config = require_mapping(config, "mitsuba")
    scene_config = require_mapping(config, "scene")
    camera_config = require_mapping(config, "camera")
    bins_config = require_mapping(config, "bins")
    dataset_config = require_mapping(config, "dataset")
    teacher_config = require_mapping(config, "teacher")

    experiment_name = _read_experiment_name(
        config,
        config_path,
    )

    target_type = require_string(
        teacher_config,
        "target_type",
    )

    _reject_removed_teacher_options(teacher_config)

    variant = _read_variant(mitsuba_config)
    camera = _build_camera(camera_config)
    bins = _build_bins(bins_config)

    teacher = build_teacher_settings(
        teacher_config=teacher_config,
        target_type=target_type,
    )

    # T1 and T2 are already recognized configuration types, but they must
    # remain blocked until their physical estimators are implemented.
    validate_teacher_target_type(target_type)

    output_path = resolve_project_path(
        require_string(
            dataset_config,
            "output_path",
        ),
        project_root,
    )

    validate_windows_llvm_runtime()
    configure_mitsuba(variant)

    scene = load_scene_from_config(
        scene_config=scene_config,
        project_root=project_root,
    )

    return DatasetGenerationSettings(
        scene=scene,
        camera=camera,
        bins=bins,
        teacher=teacher,
        num_shading_points=require_int(
            dataset_config,
            "num_shading_points",
        ),
        max_sampling_attempts=require_int(
            dataset_config,
            "max_sampling_attempts",
        ),
        seed=require_int(
            dataset_config,
            "seed",
        ),
        output_path=output_path,
        experiment_name=experiment_name,
        provenance=DatasetProvenance(
            project_root=project_root,
            config_path=config_path,
            config_snapshot=dict(config),
            scene_config=dict(scene_config),
            mitsuba_variant=variant,
        ),
    )


def build_teacher_settings(
    teacher_config: dict[str, Any],
    target_type: str,
) -> TeacherSettings:
    """Build and validate settings for a known teacher type."""
    builder = TEACHER_BUILDERS.get(target_type)

    if builder is None:
        known_targets = ", ".join(
            sorted(TEACHER_BUILDERS)
        )

        raise ValueError(
            f"Unknown teacher target_type: {target_type}. "
            f"Known values: {known_targets}."
        )

    teacher = builder(
        teacher_config,
        target_type,
    )

    validate_teacher_settings(teacher)

    return teacher


def _read_experiment_name(
    config: dict[str, Any],
    config_path: Path,
) -> str:
    experiment_name = config.get(
        "experiment_name",
        config_path.stem,
    )

    if (
        not isinstance(experiment_name, str)
        or experiment_name.strip() == ""
    ):
        raise ValueError(
            "experiment_name must be a non-empty string."
        )

    return experiment_name


def _read_variant(
    mitsuba_config: dict[str, Any],
) -> str:
    variant = mitsuba_config.get(
        "variant",
        DEFAULT_VARIANT,
    )

    if (
        not isinstance(variant, str)
        or variant.strip() == ""
    ):
        raise ValueError(
            "mitsuba.variant must be a non-empty string."
        )

    return variant


def _reject_removed_teacher_options(
    teacher_config: dict[str, Any],
) -> None:
    if "ray_epsilon" in teacher_config:
        raise ValueError(
            "teacher.ray_epsilon was removed because "
            "Mitsuba spawn_ray() already applies the "
            "required ray-origin offset."
        )


def _build_camera(
    camera_config: dict[str, Any],
) -> DatasetCamera:
    return DatasetCamera(
        origin=np.array(
            require_float_sequence(
                camera_config,
                "origin",
                3,
            ),
            dtype=np.float64,
        ),
        target=np.array(
            require_float_sequence(
                camera_config,
                "target",
                3,
            ),
            dtype=np.float64,
        ),
        up=np.array(
            require_float_sequence(
                camera_config,
                "up",
                3,
            ),
            dtype=np.float64,
        ),
        fov_degrees=require_float(
            camera_config,
            "fov_degrees",
        ),
        image_width=require_int(
            camera_config,
            "image_width",
        ),
        image_height=require_int(
            camera_config,
            "image_height",
        ),
    )


def _build_bins(
    bins_config: dict[str, Any],
) -> HemisphereBins:
    return HemisphereBins(
        n_mu=require_int(
            bins_config,
            "n_mu",
        ),
        n_phi=require_int(
            bins_config,
            "n_phi",
        ),
    )


def _build_visibility_teacher(
    teacher_config: dict[str, Any],
    target_type: str,
) -> TeacherSettings:
    """Build the T0 visibility/cosine teacher."""
    return VisibilityCosineTeacherSettings(
        target_type=target_type,
        samples_per_bin=require_int(
            teacher_config,
            "samples_per_bin",
        ),
        smoothing=optional_float(
            teacher_config,
            "smoothing",
            default=1e-6,
        ),
        max_distance=optional_float(
            teacher_config,
            "max_distance",
            default=1000.0,
        ),
        environment_weight=optional_float(
            teacher_config,
            "environment_weight",
            default=1.0,
        ),
        emitter_weight=optional_float(
            teacher_config,
            "emitter_weight",
            default=8.0,
        ),
        occluded_weight=optional_float(
            teacher_config,
            "occluded_weight",
            default=0.02,
        ),
    )


def _build_physical_teacher(
    teacher_config: dict[str, Any],
    target_type: str,
) -> TeacherSettings:
    """Build settings shared by the T1 and T2 physical teachers."""
    return PhysicalTeacherSettings(
        target_type=target_type,
        samples_per_bin=require_int(
            teacher_config,
            "samples_per_bin",
        ),
        radiance_samples=optional_int(
            teacher_config,
            "radiance_samples",
            default=1,
        ),
        smoothing=optional_float(
            teacher_config,
            "smoothing",
            default=1e-6,
        ),
        max_depth=optional_int(
            teacher_config,
            "max_depth",
            default=6,
        ),
        rr_depth=optional_int(
            teacher_config,
            "rr_depth",
            default=3,
        ),
    )


TEACHER_BUILDERS: dict[str, TeacherBuilder] = {
    VISIBILITY_COSINE_TARGET: _build_visibility_teacher,
    INCIDENT_RADIANCE_TARGET: _build_physical_teacher,
    PRODUCT_INTEGRAND_TARGET: _build_physical_teacher,
}