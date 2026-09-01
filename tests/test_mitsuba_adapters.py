"""Tests for NumPy-side Mitsuba adapter utilities."""

from __future__ import annotations

import unittest

import numpy as np

from neural_path_guiding.renderers.mitsuba.adapters import normalize_numpy_vector


class TestMitsubaAdapters(unittest.TestCase):
    def test_large_finite_vector_is_normalized_without_overflow(self) -> None:
        maximum = np.finfo(np.float64).max

        normalized = normalize_numpy_vector(np.array([maximum, maximum, 0.0]))

        self.assertTrue(bool(np.all(np.isfinite(normalized))))
        self.assertAlmostEqual(float(np.linalg.norm(normalized)), 1.0)

    def test_wrong_vector_shape_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize_numpy_vector(np.zeros(2))

    def test_non_finite_vector_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize_numpy_vector(np.array([0.0, np.nan, 1.0]))


if __name__ == "__main__":
    unittest.main()
