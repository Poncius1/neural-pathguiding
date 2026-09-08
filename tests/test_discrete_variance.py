"""Tests for discrete importance-sampling variance analysis."""

from __future__ import annotations

import unittest

import numpy as np

from neural_path_guiding.evaluation.discrete_variance import (
    calculate_discrete_estimator_statistics,
    simulate_discrete_estimates,
)


class TestDiscreteVariance(unittest.TestCase):
    def test_uniform_proposal_has_expected_variance(
        self,
    ) -> None:
        contributions = np.array(
            [1.0, 3.0],
            dtype=np.float64,
        )
        probabilities = np.array(
            [0.5, 0.5],
            dtype=np.float64,
        )

        statistics = (
            calculate_discrete_estimator_statistics(
                contributions,
                probabilities,
                bin_solid_angle=0.5,
                sample_count=4,
            )
        )

        self.assertAlmostEqual(
            statistics.exact_integral,
            2.0,
        )
        self.assertAlmostEqual(
            statistics.single_sample_variance,
            1.0,
        )
        self.assertAlmostEqual(
            statistics.estimator_variance,
            0.25,
        )
        self.assertAlmostEqual(
            statistics.standard_error,
            0.5,
        )
        self.assertAlmostEqual(
            statistics.relative_standard_error,
            0.25,
        )

    def test_optimal_proposal_has_zero_variance(
        self,
    ) -> None:
        contributions = np.array(
            [1.0, 3.0],
            dtype=np.float64,
        )
        probabilities = np.array(
            [0.25, 0.75],
            dtype=np.float64,
        )

        statistics = (
            calculate_discrete_estimator_statistics(
                contributions,
                probabilities,
                bin_solid_angle=0.5,
                sample_count=1,
            )
        )

        self.assertAlmostEqual(
            statistics.exact_integral,
            2.0,
        )
        self.assertAlmostEqual(
            statistics.single_sample_variance,
            0.0,
            places=14,
        )

    def test_simulation_is_reproducible(
        self,
    ) -> None:
        contributions = np.array(
            [1.0, 2.0, 4.0],
            dtype=np.float64,
        )
        probabilities = np.array(
            [0.2, 0.3, 0.5],
            dtype=np.float64,
        )

        first = simulate_discrete_estimates(
            contributions,
            probabilities,
            bin_solid_angle=0.25,
            samples_per_estimate=8,
            repetitions=20,
            rng=np.random.default_rng(42),
        )
        second = simulate_discrete_estimates(
            contributions,
            probabilities,
            bin_solid_angle=0.25,
            samples_per_estimate=8,
            repetitions=20,
            rng=np.random.default_rng(42),
        )

        np.testing.assert_array_equal(
            first,
            second,
        )

    def test_simulation_mean_approaches_exact_integral(
        self,
    ) -> None:
        contributions = np.array(
            [1.0, 3.0],
            dtype=np.float64,
        )
        probabilities = np.array(
            [0.5, 0.5],
            dtype=np.float64,
        )

        estimates = simulate_discrete_estimates(
            contributions,
            probabilities,
            bin_solid_angle=0.5,
            samples_per_estimate=64,
            repetitions=5000,
            rng=np.random.default_rng(7),
        )

        self.assertAlmostEqual(
            float(np.mean(estimates)),
            2.0,
            delta=0.02,
        )

    def test_rejects_missing_proposal_support(
        self,
    ) -> None:
        contributions = np.array(
            [1.0, 2.0],
            dtype=np.float64,
        )
        probabilities = np.array(
            [1.0, 0.0],
            dtype=np.float64,
        )

        with self.assertRaisesRegex(
            ValueError,
            "positive sampling probability",
        ):
            calculate_discrete_estimator_statistics(
                contributions,
                probabilities,
                bin_solid_angle=1.0,
                sample_count=1,
            )

    def test_rejects_invalid_values(
        self,
    ) -> None:
        valid_contributions = np.array(
            [1.0, 2.0],
            dtype=np.float64,
        )
        valid_probabilities = np.array(
            [0.5, 0.5],
            dtype=np.float64,
        )

        invalid_cases = (
            (
                np.array([1.0, np.nan]),
                valid_probabilities,
            ),
            (
                np.array([1.0, -1.0]),
                valid_probabilities,
            ),
            (
                valid_contributions,
                np.array([0.5, np.inf]),
            ),
            (
                valid_contributions,
                np.array([0.5, -0.5]),
            ),
        )

        for contributions, probabilities in invalid_cases:
            with self.subTest(
                contributions=contributions,
                probabilities=probabilities,
            ):
                with self.assertRaises(ValueError):
                    calculate_discrete_estimator_statistics(
                        contributions,
                        probabilities,
                        bin_solid_angle=1.0,
                        sample_count=1,
                    )


if __name__ == "__main__":
    unittest.main()