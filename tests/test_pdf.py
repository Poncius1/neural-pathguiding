"""Tests for PDF utilities.

These tests validate the probability part of the method:

- discrete probabilities are normalized correctly
- uniform mixing produces positive probabilities
- discrete bin probabilities produce the correct continuous PDF
- Monte Carlo estimates match known hemisphere integrals
"""

from __future__ import annotations

from collections.abc import Callable
import math
import unittest

import numpy as np
from numpy.typing import NDArray

from neural_path_guiding.core.bins import HemisphereBins
from neural_path_guiding.core.pdf import (
    evaluate_pdf_from_probabilities,
    mix_with_uniform,
    normalize_probabilities,
)


FloatArray = NDArray[np.float64]


class TestPdfUtilities(unittest.TestCase):
    # Shared bin setup for all tests.
    def setUp(self) -> None:
        self.bins = HemisphereBins(n_mu=4, n_phi=8)

    def test_normalize_probabilities(self) -> None:
        probabilities = np.ones(self.bins.n_bins)

        normalized = normalize_probabilities(
            probabilities=probabilities,
            expected_size=self.bins.n_bins,
        )

        self.assert_close(float(normalized.sum()), 1.0)

    def test_normalize_probabilities_rejects_non_finite_values(self) -> None:
        invalid_values = [
            np.array([1.0, np.nan]),
            np.array([1.0, np.inf]),
            np.array([1.0, -np.inf]),
        ]

        for probabilities in invalid_values:
            with self.subTest(probabilities=probabilities):
                with self.assertRaises(ValueError):
                    normalize_probabilities(probabilities, expected_size=2)

    def test_normalize_probabilities_rejects_negative_values(self) -> None:
        with self.assertRaises(ValueError):
            normalize_probabilities(np.array([1.0, -0.1]), expected_size=2)

    def test_normalize_probabilities_rejects_zero_sum(self) -> None:
        with self.assertRaises(ValueError):
            normalize_probabilities(np.zeros(2), expected_size=2)

    def test_normalize_probabilities_rejects_empty_vector(self) -> None:
        with self.assertRaises(ValueError):
            normalize_probabilities(np.array([]), expected_size=0)

    def test_normalize_probabilities_avoids_sum_overflow(self) -> None:
        probabilities = np.array([np.finfo(np.float64).max] * 2)

        normalized = normalize_probabilities(probabilities, expected_size=2)

        self.assertTrue(bool(np.all(np.isfinite(normalized))))
        self.assertTrue(bool(np.allclose(normalized, np.array([0.5, 0.5]))))

    def test_mix_with_uniform_rejects_non_finite_alpha(self) -> None:
        probabilities = np.ones(self.bins.n_bins)

        for alpha in (np.nan, np.inf, -np.inf):
            with self.subTest(alpha=alpha):
                with self.assertRaises(ValueError):
                    mix_with_uniform(
                        probabilities=probabilities,
                        alpha=alpha,
                        expected_size=self.bins.n_bins,
                    )

    def test_uniform_probabilities_produce_uniform_pdf(self) -> None:
        probabilities = np.full(self.bins.n_bins, 1.0 / self.bins.n_bins)

        omega = self.bins.sample_direction_in_bin(
            bin_index=5,
            u_mu=0.25,
            u_phi=0.75,
        )

        pdf = evaluate_pdf_from_probabilities(
            bins=self.bins,
            probabilities=probabilities,
            omega_local=omega,
        )

        expected_pdf = 1.0 / (2.0 * math.pi)

        self.assert_close(pdf, expected_pdf)

    def test_mix_with_uniform_produces_valid_distribution(self) -> None:
        probabilities = np.zeros(self.bins.n_bins)
        probabilities[0] = 1.0

        mixed = mix_with_uniform(
            probabilities=probabilities,
            alpha=0.05,
            expected_size=self.bins.n_bins,
        )

        self.assertEqual(mixed.shape, (self.bins.n_bins,))
        self.assert_close(float(mixed.sum()), 1.0)
        self.assertTrue(bool(np.all(mixed > 0.0)))

    def test_integral_of_constant_over_hemisphere(self) -> None:
        probabilities = np.full(self.bins.n_bins, 1.0 / self.bins.n_bins)

        estimate = self.estimate_integral(
            probabilities=probabilities,
            integrand=lambda omega: 1.0,
            sample_count=20_000,
        )

        expected = 2.0 * math.pi

        self.assert_close(estimate, expected, tolerance=1e-10)

    def test_integral_of_cosine_over_hemisphere(self) -> None:
        probabilities = np.full(self.bins.n_bins, 1.0 / self.bins.n_bins)

        estimate = self.estimate_integral(
            probabilities=probabilities,
            integrand=lambda omega: float(omega[2]),
            sample_count=50_000,
        )

        expected = math.pi

        self.assert_close(estimate, expected, tolerance=0.03)

    def assert_close(
        self,
        actual: float,
        expected: float,
        tolerance: float = 1e-12,
    ) -> None:
        # Avoids Pylance overload issues with assertAlmostEqual and NumPy floats.
        difference = abs(float(actual) - float(expected))

        self.assertLessEqual(
            difference,
            tolerance,
            msg=f"Expected {expected}, got {actual}. Difference: {difference}",
        )

    def estimate_integral(
        self,
        probabilities: FloatArray,
        integrand: Callable[[FloatArray], float],
        sample_count: int,
    ) -> float:
        # Monte Carlo estimate of an integral over the hemisphere.
        rng = np.random.default_rng(seed=42)
        accumulated = 0.0

        for _ in range(sample_count):
            bin_index = int(rng.choice(self.bins.n_bins, p=probabilities))

            omega = self.bins.sample_direction_in_bin(
                bin_index=bin_index,
                u_mu=float(rng.random()),
                u_phi=float(rng.random()),
            )

            pdf = evaluate_pdf_from_probabilities(
                bins=self.bins,
                probabilities=probabilities,
                omega_local=omega,
            )

            accumulated += float(integrand(omega)) / pdf

        return accumulated / sample_count


if __name__ == "__main__":
    unittest.main()
