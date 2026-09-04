"""Mitsuba dataset generation utilities.

This module is responsible for orchestrating dataset generation:

- sample visible shading points using camera rays
- extract neural-guiding features
- delegate directional contribution estimation to a teacher
- construct normalized target distributions
- save arrays and reproducibility metadata

The physical meaning of each target is implemented in teachers.py.
This module should not contain teacher-specific estimation logic.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
from typing import Any, cast

import mitsuba as mi
import numpy as np

from neural_path_guiding.core.bins import HemisphereBins
from neural_path_guiding.core.features import (
    FEATURE_DIMENSION,
    FEATURE_SCHEMA_VERSION,
    FloatArray,
)
from neural_path_guiding.core.targets import (
    contributions_to_target_distribution,
)
from neural_path_guiding.data.dataset import save_dataset
from neural_path_guiding.data.schema import (
    DATASET_FORMAT_VERSION,
    NeuralGuidingDataset,
)
from neural_path_guiding.renderers.mitsuba.adapters import (
    normalize_numpy_vector,
    surface_interaction_to_features,
)
from neural_path_guiding.renderers.mitsuba.teachers import (
    TeacherContext,
    TeacherSettings,
    build_teacher_metadata,
    create_teacher_runtime,
    estimate_bin_contributions,
    validate_teacher_settings,
    validate_teacher_target_type,
)

_mi: Any = mi


@dataclass(frozen=True)
class DatasetCamera:
    """Simple pinhole camera used to collect visible shading points."""

    origin: FloatArray
    target: FloatArray
    up: FloatArray
    fov_degrees: float
    image_width: int
    image_height: int


@dataclass(frozen=True)
class DatasetProvenance:
    """Source information required to reproduce a generated dataset."""

    project_root: Path
    config_path: Path
    config_snapshot: Mapping[str, Any]
    scene_config: Mapping[str, Any]
    mitsuba_variant: str


@dataclass(frozen=True)
class DatasetGenerationSettings:
    """Complete settings required to generate one dataset."""

    scene: Any
    camera: DatasetCamera
    bins: HemisphereBins
    teacher: TeacherSettings
    num_shading_points: int
    max_sampling_attempts: int
    seed: int
    output_path: Path
    experiment_name: str
    provenance: DatasetProvenance


@dataclass(frozen=True)
class DatasetGenerationResult:
    """Small summary returned after dataset generation."""

    output_path: Path
    num_samples: int
    feature_dimension: int
    num_bins: int


def generate_dataset(
    settings: DatasetGenerationSettings,
) -> DatasetGenerationResult:
    """Generate and save one neural-guiding dataset."""
    validate_dataset_settings(settings)

    rng = np.random.default_rng(settings.seed)
    teacher_runtime = create_teacher_runtime(
    teacher=settings.teacher,
    seed=settings.seed,
)

    features_list: list[FloatArray] = []
    targets_list: list[FloatArray] = []
    mean_contribution_list: list[FloatArray] = []
    position_list: list[FloatArray] = []
    normal_list: list[FloatArray] = []
    pixel_list: list[tuple[int, int]] = []

    attempts = 0

    while len(features_list) < settings.num_shading_points:
        attempts += 1

        if attempts > settings.max_sampling_attempts:
            raise RuntimeError(
                "Could not collect enough visible shading points. "
                f"Collected {len(features_list)} / "
                f"{settings.num_shading_points}."
            )

        pixel_x, pixel_y, ray = sample_camera_ray(
            camera=settings.camera,
            rng=rng,
        )

        surface_interaction = settings.scene.ray_intersect(ray)

        if not bool(surface_interaction.is_valid()):
            continue

        ray_direction = mitsuba_vector_to_numpy(ray.d)
        outgoing_direction = -ray_direction

        shading_features = surface_interaction_to_features(
            surface_interaction=surface_interaction,
            outgoing_direction=outgoing_direction,
            bounce_depth=0,
        )

        feature_vector = shading_features.to_array()

        teacher_context = TeacherContext(
            scene=settings.scene,
            surface_interaction=surface_interaction,
            normal=shading_features.normal,
            outgoing_direction=outgoing_direction,
        )

        contributions = estimate_bin_contributions(
            context=teacher_context,
            bins=settings.bins,
            teacher=settings.teacher,
            rng=rng,
            runtime=teacher_runtime,
        )

        target_distribution = contributions_to_target_distribution(
            contributions=contributions,
            smoothing=settings.teacher.smoothing,
        )

        features_list.append(feature_vector)
        targets_list.append(target_distribution)
        mean_contribution_list.append(contributions)
        position_list.append(shading_features.position)
        normal_list.append(shading_features.normal)
        pixel_list.append((pixel_x, pixel_y))

    features = np.asarray(features_list, dtype=np.float32)
    targets = np.asarray(targets_list, dtype=np.float32)
    mean_contributions = np.asarray(
        mean_contribution_list,
        dtype=np.float32,
    )
    positions = np.asarray(position_list, dtype=np.float32)
    normals = np.asarray(normal_list, dtype=np.float32)
    pixels = np.asarray(pixel_list, dtype=np.int32)

    metadata = build_metadata(settings)

    dataset = NeuralGuidingDataset(
        features=features,
        targets=targets,
        mean_contributions=mean_contributions,
        positions=positions,
        normals=normals,
        pixels=pixels,
        metadata=metadata,
    )

    save_dataset(settings.output_path, dataset)

    return DatasetGenerationResult(
        output_path=settings.output_path,
        num_samples=int(features.shape[0]),
        feature_dimension=int(features.shape[1]),
        num_bins=int(targets.shape[1]),
    )


def sample_camera_ray(
    camera: DatasetCamera,
    rng: np.random.Generator,
) -> tuple[int, int, Any]:
    """Sample one image position and construct its camera ray."""
    pixel_x = int(rng.integers(0, camera.image_width))
    pixel_y = int(rng.integers(0, camera.image_height))

    u = (
        pixel_x + float(rng.random())
    ) / camera.image_width

    v = (
        pixel_y + float(rng.random())
    ) / camera.image_height

    direction = compute_pinhole_camera_direction(
        camera=camera,
        u=u,
        v=v,
    )

    ray = _mi.Ray3f(
        _mi.Point3f(*camera.origin),
        _mi.Vector3f(*direction),
    )

    return pixel_x, pixel_y, ray


def compute_pinhole_camera_direction(
    camera: DatasetCamera,
    u: float,
    v: float,
) -> FloatArray:
    """Convert normalized image coordinates into a world direction."""
    forward = normalize_numpy_vector(
        camera.target - camera.origin
    )

    right = normalize_numpy_vector(
        np.cross(forward, camera.up)
    )

    true_up = normalize_numpy_vector(
        np.cross(right, forward)
    )

    aspect_ratio = (
        camera.image_width / camera.image_height
    )

    fov_scale = math.tan(
        math.radians(camera.fov_degrees) * 0.5
    )

    screen_x = (
        (2.0 * u - 1.0)
        * aspect_ratio
        * fov_scale
    )

    screen_y = (
        (1.0 - 2.0 * v)
        * fov_scale
    )

    direction = (
        forward
        + screen_x * right
        + screen_y * true_up
    )

    return normalize_numpy_vector(direction)


def build_metadata(
    settings: DatasetGenerationSettings,
) -> dict[str, Any]:
    """Build reproducibility metadata for a generated dataset."""
    provenance = settings.provenance

    return {
        "dataset_format_version": DATASET_FORMAT_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "experiment_name": settings.experiment_name,
        "target_type": settings.teacher.target_type,
        "feature_dimension": FEATURE_DIMENSION,
        "num_bins": settings.bins.n_bins,
        "n_mu": settings.bins.n_mu,
        "n_phi": settings.bins.n_phi,
        "num_shading_points": settings.num_shading_points,
        "samples_per_bin": settings.teacher.samples_per_bin,
        "seed": settings.seed,
        "output_path": _display_path(
            settings.output_path,
            provenance.project_root,
        ),
        "source": {
            "config_path": _display_path(
                provenance.config_path,
                provenance.project_root,
            ),
            "config_sha256": _sha256_file(
                provenance.config_path
            ),
            "config_snapshot": dict(
                provenance.config_snapshot
            ),
        },
        "scene": _build_scene_metadata(provenance),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "mitsuba": str(
                getattr(_mi, "__version__", "unknown")
            ),
            "mitsuba_variant": provenance.mitsuba_variant,
            "platform": platform.platform(),
        },
        "array_schema": {
            "features": [
                "num_samples",
                FEATURE_DIMENSION,
            ],
            "targets": [
                "num_samples",
                settings.bins.n_bins,
            ],
            "mean_contributions": [
                "num_samples",
                settings.bins.n_bins,
            ],
            "positions": [
                "num_samples",
                3,
            ],
            "normals": [
                "num_samples",
                3,
            ],
            "pixels": [
                "num_samples",
                2,
            ],
        },
        "teacher": build_teacher_metadata(
            settings.teacher
        ),
        "camera": {
            "origin": settings.camera.origin.tolist(),
            "target": settings.camera.target.tolist(),
            "up": settings.camera.up.tolist(),
            "fov_degrees": (
                settings.camera.fov_degrees
            ),
            "image_width": (
                settings.camera.image_width
            ),
            "image_height": (
                settings.camera.image_height
            ),
        },
    }


def validate_dataset_settings(
    settings: DatasetGenerationSettings,
) -> None:
    """Validate settings before performing expensive Mitsuba work."""
    _validate_positive_integer(
        settings.num_shading_points,
        "num_shading_points",
    )

    _validate_positive_integer(
        settings.max_sampling_attempts,
        "max_sampling_attempts",
    )

    if (
        settings.max_sampling_attempts
        < settings.num_shading_points
    ):
        raise ValueError(
            "max_sampling_attempts must be "
            ">= num_shading_points."
        )

    if (
        isinstance(settings.seed, bool)
        or not isinstance(
            settings.seed,
            (int, np.integer),
        )
    ):
        raise TypeError("seed must be an integer.")

    if settings.seed < 0:
        raise ValueError(
            "seed must be non-negative."
        )

    # Validate the parameters belonging to the selected
    # teacher implementation.
    validate_teacher_settings(settings.teacher)

    # A target may be defined in the research roadmap while
    # its estimator is not implemented yet. Only implemented
    # targets are allowed to generate datasets.
    validate_teacher_target_type(
        settings.teacher.target_type
    )

    _validate_positive_integer(
        settings.camera.image_width,
        "image_width",
    )

    _validate_positive_integer(
        settings.camera.image_height,
        "image_height",
    )

    _validate_finite_positive(
        settings.camera.fov_degrees,
        "fov_degrees",
    )

    if settings.camera.fov_degrees >= 180.0:
        raise ValueError(
            "fov_degrees must be less than 180."
        )

    origin = _validate_vector3(
        settings.camera.origin,
        "camera.origin",
    )

    target = _validate_vector3(
        settings.camera.target,
        "camera.target",
    )

    up = _validate_vector3(
        settings.camera.up,
        "camera.up",
    )

    forward = normalize_numpy_vector(
        target - origin
    )

    normalized_up = normalize_numpy_vector(up)

    if (
        float(
            np.linalg.norm(
                np.cross(forward, normalized_up)
            )
        )
        <= 1e-12
    ):
        raise ValueError(
            "camera.up must not be parallel "
            "to the viewing direction."
        )

    if settings.scene is None:
        raise ValueError(
            "scene must not be None."
        )

    if (
        not isinstance(settings.experiment_name, str)
        or settings.experiment_name.strip() == ""
    ):
        raise ValueError(
            "experiment_name must be "
            "a non-empty string."
        )

    if settings.output_path.suffix.lower() != ".npz":
        raise ValueError(
            "output_path must use the .npz extension."
        )

    provenance = settings.provenance

    if not provenance.project_root.is_dir():
        raise ValueError(
            "project_root is not a directory: "
            f"{provenance.project_root}"
        )

    if not provenance.config_path.is_file():
        raise FileNotFoundError(
            "Config file not found: "
            f"{provenance.config_path}"
        )

    if (
        not isinstance(provenance.mitsuba_variant, str)
        or provenance.mitsuba_variant.strip() == ""
    ):
        raise ValueError(
            "mitsuba_variant must be "
            "a non-empty string."
        )

    try:
        json.dumps(
            dict(provenance.config_snapshot)
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "config_snapshot must be JSON serializable."
        ) from error

    _build_scene_metadata(provenance)


def _build_scene_metadata(
    provenance: DatasetProvenance,
) -> dict[str, Any]:
    """Build metadata for XML or built-in scenes."""
    scene_type = provenance.scene_config.get("type")

    if scene_type == "xml":
        path_value = provenance.scene_config.get("path")

        if (
            not isinstance(path_value, str)
            or path_value.strip() == ""
        ):
            raise ValueError(
                "XML scene config requires "
                "a non-empty path."
            )

        scene_path = Path(path_value)

        if not scene_path.is_absolute():
            scene_path = (
                provenance.project_root / scene_path
            )

        if not scene_path.is_file():
            raise FileNotFoundError(
                "Mitsuba scene file not found: "
                f"{scene_path}"
            )

        return {
            "type": "xml",
            "path": _display_path(
                scene_path,
                provenance.project_root,
            ),
            "sha256": _sha256_file(scene_path),
        }

    if scene_type == "builtin":
        scene_name = provenance.scene_config.get("name")

        if (
            not isinstance(scene_name, str)
            or scene_name.strip() == ""
        ):
            raise ValueError(
                "Built-in scene config requires "
                "a non-empty name."
            )

        return {
            "type": "builtin",
            "name": scene_name,
            "sha256": None,
        }

    raise ValueError(
        "Unsupported scene type in provenance: "
        f"{scene_type}"
    )


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def _display_path(
    path: Path,
    project_root: Path,
) -> str:
    """Prefer project-relative paths in metadata."""
    resolved_path = path.resolve()
    resolved_root = project_root.resolve()

    try:
        return str(
            resolved_path.relative_to(resolved_root)
        )
    except ValueError:
        return str(resolved_path)


def mitsuba_vector_to_numpy(
    vector: Any,
) -> FloatArray:
    """Convert a three-component Mitsuba vector to NumPy."""
    return np.array(
        [
            float(vector[0]),
            float(vector[1]),
            float(vector[2]),
        ],
        dtype=np.float64,
    )


def _validate_positive_integer(
    value: int,
    name: str,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(
            value,
            (int, np.integer),
        )
    ):
        raise TypeError(
            f"{name} must be an integer."
        )

    if value <= 0:
        raise ValueError(
            f"{name} must be greater than zero."
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
            f"{name} must be finite "
            "and greater than zero."
        )


def _validate_vector3(
    vector: FloatArray,
    name: str,
) -> FloatArray:
    vector = np.asarray(
        vector,
        dtype=np.float64,
    )

    if vector.shape != (3,):
        raise ValueError(
            f"{name} must have shape (3,), "
            f"got {vector.shape}."
        )

    if not bool(np.all(np.isfinite(vector))):
        raise ValueError(
            f"{name} must contain only "
            "finite values."
        )

    return vector