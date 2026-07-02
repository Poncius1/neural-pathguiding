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

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import mitsuba as mi
import numpy as np
from numpy.typing import NDArray

from neural_path_guiding.core.bins import HemisphereBins
from neural_path_guiding.core.features import FEATURE_DIMENSION, FloatArray
from neural_path_guiding.core.frames import build_frame_from_normal, local_to_world
from neural_path_guiding.core.targets import contributions_to_target_distribution
from neural_path_guiding.renderers.mitsuba.adapters import (
    normalize_numpy_vector,
    surface_interaction_to_features,
)


_mi: Any = mi

DatasetFloatArray = NDArray[np.float32]
DatasetIntArray = NDArray[np.int32]


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
    samples_per_bin: int
    smoothing: float
    ray_epsilon: float
    max_distance: float
    environment_weight: float
    emitter_weight: float
    occluded_weight: float


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
    contribution_list: list[FloatArray] = []
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
        contribution_list.append(contributions)
        position_list.append(shading_features.position)
        normal_list.append(shading_features.normal)
        pixel_list.append((pixel_x, pixel_y))

    features = np.asarray(features_list, dtype=np.float32)
    targets = np.asarray(targets_list, dtype=np.float32)
    contributions = np.asarray(contribution_list, dtype=np.float32)
    positions = np.asarray(position_list, dtype=np.float32)
    normals = np.asarray(normal_list, dtype=np.float32)
    pixels = np.asarray(pixel_list, dtype=np.int32)

    metadata = build_metadata(settings)

    write_dataset(
        output_path=settings.output_path,
        features=features,
        targets=targets,
        contributions=contributions,
        positions=positions,
        normals=normals,
        pixels=pixels,
        metadata=metadata,
    )

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

            accumulated += estimate_visibility_cosine_contribution(
                scene=scene,
                surface_interaction=surface_interaction,
                world_direction=world_direction,
                normal=normal,
                teacher=teacher,
            )

        contributions[bin_index] = accumulated / teacher.samples_per_bin

    return contributions


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
    try:
        emitter = surface_interaction.emitter(scene)
    except Exception:
        return False

    return emitter is not None


def write_dataset(
    output_path: Path,
    features: DatasetFloatArray,
    targets: DatasetFloatArray,
    contributions: DatasetFloatArray,
    positions: DatasetFloatArray,
    normals: DatasetFloatArray,
    pixels: DatasetIntArray,
    metadata: dict[str, Any],
) -> None:
    # Writes the dataset to a compressed NPZ file.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        features=features,
        targets=targets,
        contribution_sums=contributions,
        positions=positions,
        normals=normals,
        pixels=pixels,
        metadata_json=json.dumps(metadata, indent=2),
    )


def build_metadata(settings: DatasetGenerationSettings) -> dict[str, Any]:
    # Stores the information needed to reproduce the dataset.
    return {
        "experiment_name": settings.experiment_name,
        "target_type": "visibility_cosine_v0",
        "feature_dimension": FEATURE_DIMENSION,
        "num_bins": settings.bins.n_bins,
        "n_mu": settings.bins.n_mu,
        "n_phi": settings.bins.n_phi,
        "num_shading_points": settings.num_shading_points,
        "samples_per_bin": settings.teacher.samples_per_bin,
        "seed": settings.seed,
        "teacher": {
            "smoothing": settings.teacher.smoothing,
            "ray_epsilon": settings.teacher.ray_epsilon,
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
    if settings.num_shading_points <= 0:
        raise ValueError("num_shading_points must be greater than zero.")

    if settings.max_sampling_attempts < settings.num_shading_points:
        raise ValueError("max_sampling_attempts must be >= num_shading_points.")

    if settings.teacher.samples_per_bin <= 0:
        raise ValueError("samples_per_bin must be greater than zero.")

    if settings.teacher.smoothing < 0.0:
        raise ValueError("smoothing must be non-negative.")

    if settings.teacher.max_distance <= 0.0:
        raise ValueError("max_distance must be greater than zero.")

    if settings.camera.image_width <= 0:
        raise ValueError("image_width must be greater than zero.")

    if settings.camera.image_height <= 0:
        raise ValueError("image_height must be greater than zero.")

    if settings.camera.fov_degrees <= 0.0:
        raise ValueError("fov_degrees must be greater than zero.")