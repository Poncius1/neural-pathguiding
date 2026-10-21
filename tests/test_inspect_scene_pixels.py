"""Tests for deterministic scene pixel inspection helpers."""

from __future__ import annotations

import unittest

from scripts.inspect_scene_pixels import (
    grid_coordinates,
    select_candidates,
)


class TestScenePixelInspector(unittest.TestCase):

    def test_grid_includes_image_edges(self):
        coords = grid_coordinates(512, 9)

        self.assertEqual(coords[0], 0)
        self.assertEqual(coords[-1], 511)
        self.assertEqual(len(coords), 9)

    def test_grid_has_unique_coordinates(self):
        coords = grid_coordinates(3, 20)

        self.assertEqual(coords, [0, 1, 2])

    def test_invalid_grid_rejected(self):
        with self.assertRaises(ValueError):
            grid_coordinates(0, 5)

    def test_prefers_distinct_shapes(self):
        records = [
            {
                "pixel_x": 10,
                "pixel_y": 10,
                "valid": True,
                "shape": "floor",
                "bsdf": "diffuse",
                "emissive": False,
            },
            {
                "pixel_x": 20,
                "pixel_y": 10,
                "valid": True,
                "shape": "floor",
                "bsdf": "diffuse",
                "emissive": False,
            },
            {
                "pixel_x": 30,
                "pixel_y": 10,
                "valid": True,
                "shape": "sphere",
                "bsdf": "diffuse",
                "emissive": False,
            },
        ]

        selected = select_candidates(
            records,
            maximum=2,
        )

        self.assertEqual(len(selected), 2)
        self.assertEqual(
            {item["shape"] for item in selected},
            {"floor", "sphere"},
        )

    def test_excludes_misses_and_emitters(self):
        records = [
            {
                "pixel_x": 1,
                "pixel_y": 1,
                "valid": False,
                "shape": "none",
                "bsdf": "none",
                "emissive": False,
            },
            {
                "pixel_x": 2,
                "pixel_y": 2,
                "valid": True,
                "shape": "ceiling_light",
                "bsdf": "none",
                "emissive": True,
            },
        ]

        self.assertEqual(
            select_candidates(records),
            [],
        )


if __name__ == "__main__":
    unittest.main()