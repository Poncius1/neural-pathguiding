"""Tests for target distribution utilities."""

from __future__ import annotations

import unittest

import numpy as np

from neural_path_guiding.core.targets import contributions_to_target_distribution


class TestTargetDistribution(unittest.TestCase):
    def test_target_distribution_sums_to_one(self) -> None:
        contributions = np.array([1.0, 2.0, 3.0], dtype=np.float64)

        target = contributions_to_target_distribution(
            contributions=contributions,
            smoothing=0.0,
        )

        self.assertAlmostEqual(float(target.sum()), 1.0)

    def test_zero_contributions_with_smoothing_become_uniform(self) -> None:
        contributions = np.zeros(4, dtype=np.float64)

        target = contributions_to_target_distribution(
            contributions=contributions,
            smoothing=1e-6,
        )

        expected = np.full(4, 0.25, dtype=np.float64)

        self.assertTrue(bool(np.allclose(target, expected)))

    def test_negative_contributions_are_rejected(self) -> None:
        contributions = np.array([1.0, -1.0], dtype=np.float64)

        with self.assertRaises(ValueError):
            contributions_to_target_distribution(
                contributions=contributions,
                smoothing=0.0,
            )

    def test_non_finite_contributions_are_rejected(self) -> None:
        for invalid in (np.nan, np.inf, -np.inf):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    contributions_to_target_distribution(
                        contributions=np.array([1.0, invalid]),
                        smoothing=0.0,
                    )

    def test_non_finite_smoothing_is_rejected(self) -> None:
        contributions = np.array([1.0, 2.0])

        for smoothing in (np.nan, np.inf, -np.inf):
            with self.subTest(smoothing=smoothing):
                with self.assertRaises(ValueError):
                    contributions_to_target_distribution(
                        contributions=contributions,
                        smoothing=smoothing,
                    )

    def test_zero_contributions_without_smoothing_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            contributions_to_target_distribution(
                contributions=np.zeros(4),
                smoothing=0.0,
            )

    def test_large_contributions_are_normalized_without_overflow(self) -> None:
        contributions = np.array([np.finfo(np.float64).max] * 2)

        target = contributions_to_target_distribution(
            contributions=contributions,
            smoothing=0.0,
        )

        self.assertTrue(bool(np.all(np.isfinite(target))))
        self.assertTrue(bool(np.allclose(target, np.array([0.5, 0.5]))))

    def test_large_smoothing_is_applied_without_overflow(self) -> None:
        maximum = np.finfo(np.float64).max

        target = contributions_to_target_distribution(
            contributions=np.array([maximum, 0.0]),
            smoothing=maximum,
        )

        expected = np.array([2.0 / 3.0, 1.0 / 3.0])
        self.assertTrue(bool(np.all(np.isfinite(target))))
        self.assertTrue(bool(np.allclose(target, expected)))


if __name__ == "__main__":
    unittest.main()
