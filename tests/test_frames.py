"""Tests for local/world frame conversions."""

from __future__ import annotations

import unittest

import numpy as np

from neural_path_guiding.core.frames import (
    build_frame_from_normal,
    local_to_world,
    world_to_local,
)


class TestOrthonormalFrame(unittest.TestCase):
    def test_frame_axes_are_orthonormal(self) -> None:
        frame = build_frame_from_normal(np.array([0.0, 1.0, 0.0]))

        self.assert_close(float(np.linalg.norm(frame.tangent)), 1.0)
        self.assert_close(float(np.linalg.norm(frame.bitangent)), 1.0)
        self.assert_close(float(np.linalg.norm(frame.normal)), 1.0)

        self.assert_close(float(np.dot(frame.tangent, frame.bitangent)), 0.0)
        self.assert_close(float(np.dot(frame.tangent, frame.normal)), 0.0)
        self.assert_close(float(np.dot(frame.bitangent, frame.normal)), 0.0)

    def test_frame_is_right_handed(self) -> None:
        normal = np.array([0.0, 1.0, 0.0])
        frame = build_frame_from_normal(normal)

        recovered_normal = np.cross(frame.tangent, frame.bitangent)

        self.assert_array_close(recovered_normal, frame.normal)

    def test_local_z_axis_maps_to_normal(self) -> None:
        normal = np.array([0.2, 1.0, 0.4])
        frame = build_frame_from_normal(normal)

        world_direction = local_to_world(
            frame=frame,
            local_direction=np.array([0.0, 0.0, 1.0]),
        )

        expected_normal = normal / float(np.linalg.norm(normal))

        self.assert_array_close(world_direction, expected_normal)

    def test_world_normal_maps_to_local_z_axis(self) -> None:
        normal = np.array([0.2, 1.0, 0.4])
        frame = build_frame_from_normal(normal)

        local_direction = world_to_local(
            frame=frame,
            world_direction=frame.normal,
        )

        self.assert_array_close(local_direction, np.array([0.0, 0.0, 1.0]))

    def test_local_world_round_trip(self) -> None:
        normal = np.array([0.2, 1.0, 0.4])
        frame = build_frame_from_normal(normal)

        local_direction = np.array([0.3, 0.4, 0.8660254])
        world_direction = local_to_world(frame, local_direction)
        recovered_local = world_to_local(frame, world_direction)

        self.assert_array_close(recovered_local, local_direction, tolerance=1e-7)

    def test_axis_aligned_normals_are_supported(self) -> None:
        normals = [
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, -1.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
        ]

        for normal in normals:
            frame = build_frame_from_normal(normal)

            self.assert_close(float(np.linalg.norm(frame.tangent)), 1.0)
            self.assert_close(float(np.linalg.norm(frame.bitangent)), 1.0)
            self.assert_close(float(np.linalg.norm(frame.normal)), 1.0)

    def test_zero_normal_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_frame_from_normal(np.array([0.0, 0.0, 0.0]))

    def assert_close(
        self,
        actual: float,
        expected: float,
        tolerance: float = 1e-12,
    ) -> None:
        difference = abs(float(actual) - float(expected))

        self.assertLessEqual(
            difference,
            tolerance,
            msg=f"Expected {expected}, got {actual}. Difference: {difference}",
        )

    def assert_array_close(
        self,
        actual: np.ndarray,
        expected: np.ndarray,
        tolerance: float = 1e-12,
    ) -> None:
        self.assertTrue(
            bool(np.allclose(actual, expected, atol=tolerance, rtol=0.0)),
            msg=f"Expected {expected}, got {actual}",
        )


if __name__ == "__main__":
    unittest.main()