"""Tests for color and luminance utilities."""
import unittest
import numpy as np
from neural_path_guiding.core.color import rgb_luminance


class TestRgbLuminance(unittest.TestCase):
    """Validate RGB-to-luminance conversion."""

    def test_uses_rec709_primary_weights(self) -> None:
        cases = (
            (
                np.array([1.0, 0.0, 0.0]),
                0.2126,
            ),
            (
                np.array([0.0, 1.0, 0.0]),
                0.7152,
            ),
            (
                np.array([0.0, 0.0, 1.0]),
                0.0722,
            ),
        )

        for rgb, expected in cases:
            with self.subTest(rgb=rgb):
                self.assertAlmostEqual(
                    rgb_luminance(rgb),
                    expected,
                    places=7,
                )

    def test_preserves_neutral_hdr_values(self) -> None:
        self.assertAlmostEqual(
            rgb_luminance([1.0, 1.0, 1.0]),
            1.0,
            places=7,
        )
        self.assertAlmostEqual(
            rgb_luminance([4.0, 4.0, 4.0]),
            4.0,
            places=7,
        )

    def test_rejects_invalid_shape(self) -> None:
        invalid_values = (
            [1.0, 2.0],
            [1.0, 2.0, 3.0, 4.0],
            [[1.0, 2.0, 3.0]],
        )

        for rgb in invalid_values:
            with self.subTest(rgb=rgb):
                with self.assertRaises(ValueError):
                    rgb_luminance(rgb)

    def test_rejects_invalid_values(self) -> None:
        invalid_values = (
            [1.0, float("nan"), 1.0],
            [1.0, float("inf"), 1.0],
            [1.0, -0.1, 1.0],
        )

        for rgb in invalid_values:
            with self.subTest(rgb=rgb):
                with self.assertRaises(ValueError):
                    rgb_luminance(rgb)


if __name__ == "__main__":
    unittest.main()