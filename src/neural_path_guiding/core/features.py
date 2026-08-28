"""Feature representation for neural path guiding.

This module defines the numerical input used by the future MLP.

It does not depend on Mitsuba.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]

FEATURE_DIMENSION = 14


FEATURE_NAMES = (
    "position_x",
    "position_y",
    "position_z",
    "normal_x",
    "normal_y",
    "normal_z",
    "outgoing_x",
    "outgoing_y",
    "outgoing_z",
    "roughness",
    "albedo_r",
    "albedo_g",
    "albedo_b",
    "bounce_depth",
)


# Features extracted at a shading point.
@dataclass(frozen=True)
class ShadingFeatures:
    position: FloatArray  # World-space surface position.
    normal: FloatArray  # World-space surface normal.
    outgoing_direction: FloatArray  # Direction from surface toward camera/previous vertex.
    roughness: float  # Surface roughness approximation.
    albedo: FloatArray  # Base color approximation.
    bounce_depth: int  # Current path depth.

    def to_array(self) -> FloatArray:
        # Converts features to a flat vector for the MLP.
        position = validate_vector3(self.position, "position")
        normal = normalize_vector3(self.normal, "normal")
        outgoing = normalize_vector3(self.outgoing_direction, "outgoing_direction")
        albedo = validate_vector3(self.albedo, "albedo")

        if not np.isfinite(self.roughness):
            raise ValueError("roughness must be finite.")

        if not 0.0 <= self.roughness <= 1.0:
            raise ValueError("roughness must be in [0, 1].")

        if bool(np.any((albedo < 0.0) | (albedo > 1.0))):
            raise ValueError("albedo components must be in [0, 1].")

        if isinstance(self.bounce_depth, bool) or not isinstance(
            self.bounce_depth,
            (int, np.integer),
        ):
            raise TypeError("bounce_depth must be an integer.")

        if self.bounce_depth < 0:
            raise ValueError("bounce_depth must be non-negative.")

        feature_vector = np.array(
            [
                position[0],
                position[1],
                position[2],
                normal[0],
                normal[1],
                normal[2],
                outgoing[0],
                outgoing[1],
                outgoing[2],
                float(self.roughness),
                albedo[0],
                albedo[1],
                albedo[2],
                float(self.bounce_depth),
            ],
            dtype=np.float64,
        )

        return validate_feature_vector(feature_vector)


def validate_feature_vector(features: FloatArray) -> FloatArray:
    # Checks that a feature vector has the expected shape.
    features = np.asarray(features, dtype=np.float64)

    if features.shape != (FEATURE_DIMENSION,):
        raise ValueError(
            f"features must have shape ({FEATURE_DIMENSION},), "
            f"got {features.shape}."
        )

    if not np.all(np.isfinite(features)):
        raise ValueError("features contains non-finite values.")

    return features


def validate_vector3(vector: FloatArray, name: str) -> FloatArray:
    # Checks that a value is a finite 3D vector.
    vector = np.asarray(vector, dtype=np.float64)

    if vector.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {vector.shape}.")

    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} contains non-finite values.")

    return vector


def normalize_vector3(vector: FloatArray, name: str) -> FloatArray:
    # Normalizes a finite 3D vector.
    vector = validate_vector3(vector, name)

    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError(f"{name} must be non-zero.")

    return vector / norm