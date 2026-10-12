"""Unit tests for emitter PDF geometric consistency.

Tests the current implementation of emitter_pdf.py.

These tests validate:
- PDF numerical inputs.
- Geometric intersection compatibility.
- Small ray-origin offsets.
- Different shapes.
- Invalid geometric intersections.
"""

from __future__ import annotations

import unittest

import numpy as np

from neural_path_guiding.renderers.mitsuba.emitter_pdf import (
    _compatible_emitter_intersections,
    _validate_pdf,
)


class FakeShape:
    """Simple shape identity for unit testing."""

    pass


class FakeInteraction:
    """Minimal fake Mitsuba surface interaction."""

    def __init__(
        self,
        position,
        *,
        valid=True,
        shape=None,
    ):
        self.p = np.asarray(
            position,
            dtype=np.float64,
        )
        self._valid = valid
        self.shape = shape

    def is_valid(self):
        return self._valid


class TestEmitterPdfValidation(unittest.TestCase):

    def setUp(self):
        self.shape = FakeShape()

        self.origin = FakeInteraction(
            [0.0, 0.0, 0.0],
        )

    def compatible(
        self,
        visibility_hit,
        geometric_hit,
        tolerance=1e-3,
    ):
        return _compatible_emitter_intersections(
            original_interaction=self.origin,
            visibility_hit=visibility_hit,
            geometric_hit=geometric_hit,
            position_tolerance=tolerance,
        )

    def test_identical_intersections(self):
        visibility = FakeInteraction(
            [0.0, 0.0, 2.0],
            shape=self.shape,
        )

        geometric = FakeInteraction(
            [0.0, 0.0, 2.0],
            shape=self.shape,
        )

        self.assertTrue(
            self.compatible(visibility, geometric)
        )

    def test_small_offset_is_accepted(self):
        visibility = FakeInteraction(
            [0.0, 0.0, 2.0001],
            shape=self.shape,
        )

        geometric = FakeInteraction(
            [0.0, 0.0, 2.0],
            shape=self.shape,
        )

        self.assertTrue(
            self.compatible(visibility, geometric)
        )

    def test_large_offset_is_rejected(self):
        visibility = FakeInteraction(
            [0.0, 0.0, 2.1],
            shape=self.shape,
        )

        geometric = FakeInteraction(
            [0.0, 0.0, 2.0],
            shape=self.shape,
        )

        self.assertFalse(
            self.compatible(visibility, geometric)
        )

    def test_different_shapes_are_rejected(self):
        visibility = FakeInteraction(
            [0.0, 0.0, 2.0],
            shape=FakeShape(),
        )

        geometric = FakeInteraction(
            [0.0, 0.0, 2.0],
            shape=FakeShape(),
        )

        self.assertFalse(
            self.compatible(visibility, geometric)
        )

    def test_invalid_geometric_intersection(self):
        visibility = FakeInteraction(
            [0.0, 0.0, 2.0],
            shape=self.shape,
        )

        geometric = FakeInteraction(
            [0.0, 0.0, 0.0],
            valid=False,
        )

        self.assertFalse(
            self.compatible(visibility, geometric)
        )

    def test_invalid_visibility_intersection(self):
        visibility = FakeInteraction(
            [0.0, 0.0, 0.0],
            valid=False,
        )

        geometric = FakeInteraction(
            [0.0, 0.0, 2.0],
            shape=self.shape,
        )

        self.assertFalse(
            self.compatible(visibility, geometric)
        )

    def test_zero_pdf(self):
        self.assertEqual(
            _validate_pdf(0.0),
            0.0,
        )

    def test_positive_pdf(self):
        self.assertEqual(
            _validate_pdf(0.25),
            0.25,
        )

    def test_negative_pdf_is_rejected(self):
        with self.assertRaises(ValueError):
            _validate_pdf(-0.1)

    def test_nan_pdf_is_rejected(self):
        with self.assertRaises(ValueError):
            _validate_pdf(float("nan"))

    def test_infinite_pdf_is_rejected(self):
        with self.assertRaises(ValueError):
            _validate_pdf(float("inf"))


if __name__ == "__main__":
    unittest.main()