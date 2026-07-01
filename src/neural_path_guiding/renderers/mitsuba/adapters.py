"""Adapters between Mitsuba objects and project data structures.

This module converts Mitsuba surface information into project-level features.

The goal is to keep Mitsuba-specific code outside of core/.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from neural_path_guiding.core.features import FloatArray, ShadingFeatures


def surface_interaction_to_features(
    surface_interaction: Any,
    outgoing_direction: FloatArray,
    bounce_depth: int,
    roughness: float = 1.0,
    albedo: FloatArray | None = None,
) -> ShadingFeatures:
    # Converts a Mitsuba SurfaceInteraction into ShadingFeatures.
    if albedo is None:
        albedo = np.array([1.0, 1.0, 1.0], dtype=np.float64)

    return ShadingFeatures(
        position=mitsuba_vector_to_numpy(surface_interaction.p),
        normal=mitsuba_vector_to_numpy(surface_interaction.n),
        outgoing_direction=np.asarray(outgoing_direction, dtype=np.float64),
        roughness=roughness,
        albedo=np.asarray(albedo, dtype=np.float64),
        bounce_depth=bounce_depth,
    )


def mitsuba_vector_to_numpy(vector: Any) -> FloatArray:
    # Converts a scalar Mitsuba vector or point to a NumPy vector.
    return np.array(
        [
            float(vector[0]),
            float(vector[1]),
            float(vector[2]),
        ],
        dtype=np.float64,
    )


def normalize_numpy_vector(vector: FloatArray) -> FloatArray:
    # Normalizes a NumPy 3D vector.
    vector = np.asarray(vector, dtype=np.float64)

    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError("Cannot normalize a zero vector.")

    return vector / norm