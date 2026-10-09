"""Unit tests for emitter PDF geometric consistency."""

from __future__ import annotations

import unittest

import numpy as np

from neural_path_guiding.renderers.mitsuba.emitter_pdf import (
    _validate_intersections,
    _validate_pdf,
)


class FakeShape:
    pass


class FakeInteraction:
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

    def validate(self, visibility_hit, geometric_hit):
        _validate_intersections(
            original_interaction=self.origin,
            visibility_hit=visibility_hit,
            geometric_hit=geometric_hit,
            position_tolerance=1e-3,
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

        self.validate(
            visibility,
            geometric,
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

        self.validate(
            visibility,
            geometric,
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

        with self.assertRaisesRegex(
            RuntimeError,
            "geometric tolerance",
        ):
            self.validate(
                visibility,
                geometric,
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

        with self.assertRaisesRegex(
            RuntimeError,
            "different shapes",
        ):
            self.validate(
                visibility,
                geometric,
            )

    def test_different_hit_status_is_rejected(self):
        visibility = FakeInteraction(
            [0.0, 0.0, 2.0],
            valid=True,
            shape=self.shape,
        )

        geometric = FakeInteraction(
            [0.0, 0.0, 0.0],
            valid=False,
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "disagree",
        ):
            self.validate(
                visibility,
                geometric,
            )

    def test_both_rays_miss(self):
        visibility = FakeInteraction(
            [0.0, 0.0, 0.0],
            valid=False,
        )

        geometric = FakeInteraction(
            [0.0, 0.0, 0.0],
            valid=False,
        )

        self.validate(
            visibility,
            geometric,
        )

    def test_valid_pdf(self):
        self.assertEqual(
            _validate_pdf(0.25),
            0.25,
        )

    def test_zero_pdf(self):
        self.assertEqual(
            _validate_pdf(0.0),
            0.0,
        )

    def test_negative_pdf_is_rejected(self):
        with self.assertRaises(ValueError):
            _validate_pdf(-0.1)

    def test_nonfinite_pdf_is_rejected(self):
        with self.assertRaises(ValueError):
            _validate_pdf(float("nan"))


if __name__ == "__main__":
    unittest.main()