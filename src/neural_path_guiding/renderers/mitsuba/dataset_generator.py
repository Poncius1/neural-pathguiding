"""Mitsuba dataset generation utilities.

This module generates an initial dataset for neural path guiding.

Dataset v0:
- samples visible shading points using camera rays
- extracts feature vectors from Mitsuba surface intersections
- estimates a simple per-bin visibility/cosine target
- saves the dataset as a compressed NPZ file

This is a first validation dataset. It is not the final physical teacher yet.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
from typing import Any

import mitsuba as mi
import numpy as np

from neural_path_guiding.core.bins import HemisphereBins
from neural_path_guiding.core.features import FEATURE_DIMENSION, FloatArray
from neural_path_guiding.core.frames import build_frame_from_normal, local_to_world
from neural_path_guiding.core.targets import contributions_to_target_distribution
from neural_path_guiding.data.dataset import save_dataset
from neural_path_guiding.data.schema import (
    DATASET_FORMAT_VERSION,
    NeuralGuidingDataset,
)
from neural_path_guiding.renderers.mitsuba.adapters import (
    normalize_numpy_vector,
    surface_interaction_to_features,
)


_mi: Any = mi

TeacherEstimator = Callable[..., float]

VISIBILITY_COSINE_TARGET = "visibility_cosine_v0"
SUPPORTED_TARGET_TYPES = frozenset({VISIBILITY_COSINE_TARGET})


def validate_teacher_target_type(target_type: str) -> None:
    if target_type not in SUPPORTED_TARGET_TYPES:
        supported = ", ".join(sorted(SUPPORTED_TARGET_TYPES))
        raise ValueError(
            f"Unsupported teacher target_type: {target_type}. "
            f"Supported values: {supported}."
        )


# Simple pinhole camera used for visible-point sampling.
@dataclass(frozen=True)
class DatasetCamera:
    origin: FloatArray
    target: FloatArray
    up: FloatArray
    fov_degrees: float
    image_width: int
    image_height: int


# Settings for the initial visibility/cosine target.
@dataclass(frozen=True)
class VisibilityCosineTeacherSettings:
    target_type: str
    samples_per_bin: int
    smoothing: float
    max_distance: float
    environment_weight: float
    emitter_weight: float
    occluded_weight: float


# Source information needed to reproduce a generated dataset.
@dataclass(frozen=True)
class DatasetProvenance:
    project_root: Path
    config_path: Path
    config_snapshot: Mapping[str, Any]
    scene_config: Mapping[str, Any]
    mitsuba_variant: str


# Full settings needed to generate one dataset.
@dataclass(frozen=True)
class DatasetGenerationSettings:
    scene: Any
    camera: DatasetCamera
    bins: HemisphereBins
    teacher: VisibilityCosineTeacherSettings
    num_shading_points: int
    max_sampling_attempts: int
    seed: int
    output_path: Path
    experiment_name: str
    provenance: DatasetProvenance


# Summary returned after dataset generation.
@dataclass(frozen=True)
class DatasetGenerationResult:
    output_path: Path
    num_samples: int
    feature_dimension: int
    num_bins: int


def generate_dataset(settings: DatasetGenerationSettings) -> DatasetGenerationResult:
    # Generates features, targets and metadata, then writes them to disk.
    validate_dataset_settings(settings)

    rng = np.random.default_rng(settings.seed)

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
                f"Collected {len(features_list)} / {settings.num_shading_points}."
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

        contributions = estimate_bin_contributions(
            scene=settings.scene,
            surface_interaction=surface_interaction,
            normal=shading_features.normal,
            bins=settings.bins,
            teacher=settings.teacher,
            rng=rng,
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
    mean_contributions = np.asarray(mean_contribution_list, dtype=np.float32)
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
    # Samples one random pixel and creates a world-space camera ray.
    pixel_x = int(rng.integers(0, camera.image_width))
    pixel_y = int(rng.integers(0, camera.image_height))

    u = (pixel_x + float(rng.random())) / camera.image_width
    v = (pixel_y + float(rng.random())) / camera.image_height

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
    # Converts normalized image coordinates into a world-space ray direction.
    forward = normalize_numpy_vector(camera.target - camera.origin)
    right = normalize_numpy_vector(np.cross(forward, camera.up))
    true_up = normalize_numpy_vector(np.cross(right, forward))

    aspect_ratio = camera.image_width / camera.image_height
    fov_scale = math.tan(math.radians(camera.fov_degrees) * 0.5)

    screen_x = (2.0 * u - 1.0) * aspect_ratio * fov_scale
    screen_y = (1.0 - 2.0 * v) * fov_scale

    direction = forward + screen_x * right + screen_y * true_up

    return normalize_numpy_vector(direction)


def estimate_bin_contributions(
    scene: Any,
    surface_interaction: Any,
    normal: FloatArray,
    bins: HemisphereBins,
    teacher: VisibilityCosineTeacherSettings,
    rng: np.random.Generator,
) -> FloatArray:
    # Estimates one target contribution value per hemisphere bin.
    contributions = np.zeros(bins.n_bins, dtype=np.float64)
    frame = build_frame_from_normal(normal)
    teacher_estimator = resolve_teacher_estimator(teacher.target_type)

    for bin_index in range(bins.n_bins):
        accumulated = 0.0

        for _ in range(teacher.samples_per_bin):
            local_direction = bins.sample_direction_in_bin(
                bin_index=bin_index,
                u_mu=float(rng.random()),
                u_phi=float(rng.random()),
            )

            world_direction = normalize_numpy_vector(
                local_to_world(frame, local_direction)
            )

            accumulated += teacher_estimator(
                scene=scene,
                surface_interaction=surface_interaction,
                world_direction=world_direction,
                normal=normal,
                teacher=teacher,
            )

        contributions[bin_index] = accumulated / teacher.samples_per_bin

    return contributions


def resolve_teacher_estimator(target_type: str) -> TeacherEstimator:
    # Resolves configuration once, outside the directional sampling loops.
    validate_teacher_target_type(target_type)

    if target_type == VISIBILITY_COSINE_TARGET:
        return estimate_visibility_cosine_contribution

    raise AssertionError("Validated teacher target_type has no implementation.")


def estimate_visibility_cosine_contribution(
    scene: Any,
    surface_interaction: Any,
    world_direction: FloatArray,
    normal: FloatArray,
    teacher: VisibilityCosineTeacherSettings,
) -> float:
    # Simple target: cosine times a visibility/emitter heuristic.
    normal = normalize_numpy_vector(normal)
    world_direction = normalize_numpy_vector(world_direction)

    cosine = max(0.0, float(np.dot(normal, world_direction)))

    if cosine <= 0.0:
        return 0.0

    ray = surface_interaction.spawn_ray(_mi.Vector3f(*world_direction))
    ray.maxt = teacher.max_distance

    hit = scene.ray_intersect(ray)

    if not bool(hit.is_valid()):
        return teacher.environment_weight * cosine

    if surface_hit_has_emitter(hit, scene):
        return teacher.emitter_weight * cosine

    return teacher.occluded_weight * cosine


def surface_hit_has_emitter(surface_interaction: Any, scene: Any) -> bool:
    # Checks whether the hit surface is an emitter.
    return surface_interaction.emitter(scene) is not None


def build_metadata(settings: DatasetGenerationSettings) -> dict[str, Any]:
    # Stores the information needed to reproduce the dataset.
    provenance = settings.provenance

    return {
        "dataset_format_version": DATASET_FORMAT_VERSION,
        "experiment_name": settings.experiment_name,
        "target_type": settings.teacher.target_type,
        "feature_dimension": FEATURE_DIMENSION,
        "num_bins": settings.bins.n_bins,
        "n_mu": settings.bins.n_mu,
        "n_phi": settings.bins.n_phi,
        "num_shading_points": settings.num_shading_points,
        "samples_per_bin": settings.teacher.samples_per_bin,
        "seed": settings.seed,
        "output_path": _display_path(settings.output_path, provenance.project_root),
        "source": {
            "config_path": _display_path(
                provenance.config_path,
                provenance.project_root,
            ),
            "config_sha256": _sha256_file(provenance.config_path),
            "config_snapshot": dict(provenance.config_snapshot),
        },
        "scene": _build_scene_metadata(provenance),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "mitsuba": str(getattr(_mi, "__version__", "unknown")),
            "mitsuba_variant": provenance.mitsuba_variant,
            "platform": platform.platform(),
        },
        "array_schema": {
            "features": ["num_samples", FEATURE_DIMENSION],
            "targets": ["num_samples", settings.bins.n_bins],
            "mean_contributions": ["num_samples", settings.bins.n_bins],
            "positions": ["num_samples", 3],
            "normals": ["num_samples", 3],
            "pixels": ["num_samples", 2],
        },
        "teacher": {
            "smoothing": settings.teacher.smoothing,
            "max_distance": settings.teacher.max_distance,
            "environment_weight": settings.teacher.environment_weight,
            "emitter_weight": settings.teacher.emitter_weight,
            "occluded_weight": settings.teacher.occluded_weight,
        },
        "camera": {
            "origin": settings.camera.origin.tolist(),
            "target": settings.camera.target.tolist(),
            "up": settings.camera.up.tolist(),
            "fov_degrees": settings.camera.fov_degrees,
            "image_width": settings.camera.image_width,
            "image_height": settings.camera.image_height,
        },
    }


def _build_scene_metadata(provenance: DatasetProvenance) -> dict[str, Any]:
    scene_type = provenance.scene_config.get("type")

    if scene_type == "xml":
        path_value = provenance.scene_config.get("path")
        if not isinstance(path_value, str) or path_value.strip() == "":
            raise ValueError("XML scene config requires a non-empty path.")

        scene_path = Path(path_value)
        if not scene_path.is_absolute():
            scene_path = provenance.project_root / scene_path

        if not scene_path.is_file():
            raise FileNotFoundError(f"Mitsuba scene file not found: {scene_path}")

        return {
            "type": "xml",
            "path": _display_path(scene_path, provenance.project_root),
            "sha256": _sha256_file(scene_path),
        }

    if scene_type == "builtin":
        scene_name = provenance.scene_config.get("name")
        if not isinstance(scene_name, str) or scene_name.strip() == "":
            raise ValueError("Built-in scene config requires a non-empty name.")

        return {
            "type": "builtin",
            "name": scene_name,
            "sha256": None,
        }

    raise ValueError(f"Unsupported scene type in provenance: {scene_type}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _display_path(path: Path, project_root: Path) -> str:
    resolved_path = path.resolve()
    resolved_root = project_root.resolve()

    try:
        return str(resolved_path.relative_to(resolved_root))
    except ValueError:
        return str(resolved_path)


def mitsuba_vector_to_numpy(vector: Any) -> FloatArray:
    # Converts a Mitsuba vector or point to a NumPy vector.
    return np.array(
        [
            float(vector[0]),
            float(vector[1]),
            float(vector[2]),
        ],
        dtype=np.float64,
    )


def validate_dataset_settings(settings: DatasetGenerationSettings) -> None:
    # Validates dataset generation parameters before doing expensive work.
    _validate_positive_integer(settings.num_shading_points, "num_shading_points")
    _validate_positive_integer(settings.max_sampling_attempts, "max_sampling_attempts")
    _validate_positive_integer(settings.teacher.samples_per_bin, "samples_per_bin")

    if settings.max_sampling_attempts < settings.num_shading_points:
        raise ValueError("max_sampling_attempts must be >= num_shading_points.")

    if isinstance(settings.seed, bool) or not isinstance(settings.seed, (int, np.integer)):
        raise TypeError("seed must be an integer.")

    if settings.seed < 0:
        raise ValueError("seed must be non-negative.")

    validate_teacher_target_type(settings.teacher.target_type)

    _validate_finite_non_negative(settings.teacher.smoothing, "smoothing")
    _validate_finite_positive(settings.teacher.max_distance, "max_distance")
    _validate_finite_non_negative(
        settings.teacher.environment_weight,
        "environment_weight",
    )
    _validate_finite_non_negative(settings.teacher.emitter_weight, "emitter_weight")
    _validate_finite_non_negative(
        settings.teacher.occluded_weight,
        "occluded_weight",
    )

    _validate_positive_integer(settings.camera.image_width, "image_width")
    _validate_positive_integer(settings.camera.image_height, "image_height")
    _validate_finite_positive(settings.camera.fov_degrees, "fov_degrees")

    if settings.camera.fov_degrees >= 180.0:
        raise ValueError("fov_degrees must be less than 180.")

    origin = _validate_vector3(settings.camera.origin, "camera.origin")
    target = _validate_vector3(settings.camera.target, "camera.target")
    up = _validate_vector3(settings.camera.up, "camera.up")
    forward = normalize_numpy_vector(target - origin)
    normalized_up = normalize_numpy_vector(up)

    if float(np.linalg.norm(np.cross(forward, normalized_up))) <= 1e-12:
        raise ValueError("camera.up must not be parallel to the viewing direction.")

    if settings.scene is None:
        raise ValueError("scene must not be None.")

    if not isinstance(settings.experiment_name, str) or settings.experiment_name.strip() == "":
        raise ValueError("experiment_name must be a non-empty string.")

    if settings.output_path.suffix.lower() != ".npz":
        raise ValueError("output_path must use the .npz extension.")

    provenance = settings.provenance
    if not provenance.project_root.is_dir():
        raise ValueError(f"project_root is not a directory: {provenance.project_root}")

    if not provenance.config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {provenance.config_path}")

    if not isinstance(provenance.mitsuba_variant, str) or provenance.mitsuba_variant.strip() == "":
        raise ValueError("mitsuba_variant must be a non-empty string.")

    try:
        json.dumps(dict(provenance.config_snapshot))
    except (TypeError, ValueError) as error:
        raise ValueError("config_snapshot must be JSON serializable.") from error

    _build_scene_metadata(provenance)


def _validate_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")

    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")


def _validate_finite_positive(value: float, name: str) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and greater than zero.")


def _validate_finite_non_negative(value: float, name: str) -> None:
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")


def _validate_vector3(vector: FloatArray, name: str) -> FloatArray:
    vector = np.asarray(vector, dtype=np.float64)

    if vector.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {vector.shape}.")

    if not bool(np.all(np.isfinite(vector))):
        raise ValueError(f"{name} must contain only finite values.")

    return vector
