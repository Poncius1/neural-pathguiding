"""Local/world frame utilities.

This module builds an orthonormal frame from a surface normal.

It is used to convert directions between:
- local shading space, where +Z is the surface normal
- world space, where Mitsuba traces rays
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neural_path_guiding.core.features import (
    FloatArray,
    normalize_vector3,
    validate_vector3,
)


# Local shading frame.
@dataclass(frozen=True)
class OrthonormalFrame:
    tangent: FloatArray  # Local +X axis in world space.
    bitangent: FloatArray  # Local +Y axis in world space.
    normal: FloatArray  # Local +Z axis in world space.


def build_frame_from_normal(normal: FloatArray) -> OrthonormalFrame:
    # Builds a stable local frame from a world-space normal.
    n = normalize_vector3(normal, "normal")

    if abs(float(n[2])) < 0.999:
        helper = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    tangent = normalize_vector3(np.cross(helper, n), "tangent")
    bitangent = normalize_vector3(np.cross(n, tangent), "bitangent")

    return OrthonormalFrame(
        tangent=tangent,
        bitangent=bitangent,
        normal=n,
    )


def local_to_world(frame: OrthonormalFrame, local_direction: FloatArray) -> FloatArray:
    # Converts a local direction to world space.
    local = validate_vector3(local_direction, "local_direction")

    world = (
        local[0] * frame.tangent
        + local[1] * frame.bitangent
        + local[2] * frame.normal
    )

    return np.asarray(world, dtype=np.float64)


def world_to_local(frame: OrthonormalFrame, world_direction: FloatArray) -> FloatArray:
    # Converts a world-space direction to local shading space.
    world = validate_vector3(world_direction, "world_direction")

    return np.array(
        [
            float(np.dot(world, frame.tangent)),
            float(np.dot(world, frame.bitangent)),
            float(np.dot(world, frame.normal)),
        ],
        dtype=np.float64,
    )