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


if __name__ == "__main__":
    unittest.main()