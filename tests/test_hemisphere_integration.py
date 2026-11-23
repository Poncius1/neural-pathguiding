"""Analytical Monte Carlo validation of hemisphere sampling."""

from __future__ import annotations

import math
import unittest

import numpy as np

from neural_path_guiding.core.bins import HemisphereBins
from neural_path_guiding.core.directional_distribution import (
    DiscreteDirectionalDistribution,
)


class TestHemisphereIntegration(unittest.TestCase):

    def setUp(self) -> None:
        self.bins = HemisphereBins(n_mu=4, n_phi=8)
        self.sample_count = 4096

    def test_uniform_distribution_pdf(self) -> None:
        distribution = DiscreteDirectionalDistribution(
            bins=self.bins,
            probabilities=np.ones(self.bins.n_bins),
        )

        rng = np.random.default_rng(42)
        expected_pdf = 1.0 / (2.0 * math.pi)

        for _ in range(128):
            sample = distribution.sample(rng)

            self.assertAlmostEqual(
                sample.pdf,
                expected_pdf,
                places=12,
            )

            self.assertAlmostEqual(
                distribution.pdf(sample.direction_local),
                sample.pdf,
                places=12,
            )

    def test_equal_solid_angle_partition(self) -> None:
        total = sum(
            self.bins.get_bin_bounds(index).solid_angle
            for index in range(self.bins.n_bins)
        )

        self.assertAlmostEqual(
            total,
            2.0 * math.pi,
            places=12,
        )

    def test_uniform_hemisphere_moments(self) -> None:
        distribution = DiscreteDirectionalDistribution(
            bins=self.bins,
            probabilities=np.ones(self.bins.n_bins),
        )

        rng = np.random.default_rng(43)

        directions = np.array([
            distribution.sample(rng).direction_local
            for _ in range(self.sample_count)
        ])

        mu = directions[:, 2]

        # Analytical moments for mu ~ Uniform(0, 1).
        self.assertLess(
            abs(float(mu.mean()) - 0.5),
            0.025,
        )

        self.assertLess(
            abs(float(np.mean(mu**2)) - 1.0 / 3.0),
            0.025,
        )

    def test_constant_integrand(self) -> None:
        distribution = DiscreteDirectionalDistribution(
            bins=self.bins,
            probabilities=np.ones(self.bins.n_bins),
        )

        rng = np.random.default_rng(44)

        estimates = np.array([
            1.0 / distribution.sample(rng).pdf
            for _ in range(128)
        ])

        np.testing.assert_allclose(
            estimates,
            2.0 * math.pi,
            rtol=0.0,
            atol=1e-12,
        )

    def test_uniform_cosine_integrals(self) -> None:
        distribution = DiscreteDirectionalDistribution(
            bins=self.bins,
            probabilities=np.ones(self.bins.n_bins),
        )

        self.check_analytical_integrals(distribution, seed=45)

    def test_nonuniform_cosine_integrals(self) -> None:
        distribution = DiscreteDirectionalDistribution(
            bins=self.bins,
            probabilities=np.linspace(
                0.5,
                2.0,
                self.bins.n_bins,
            ),
            uniform_mix=0.05,
        )

        self.check_analytical_integrals(distribution, seed=46)

    def check_analytical_integrals(
        self,
        distribution: DiscreteDirectionalDistribution,
        *,
        seed: int,
    ) -> None:
        rng = np.random.default_rng(seed)

        estimates = np.empty(
            (self.sample_count, 3),
            dtype=np.float64,
        )

        for index in range(self.sample_count):
            sample = distribution.sample(rng)
            mu = float(sample.direction_local[2])

            estimates[index] = (
                np.array([1.0, mu, mu**2]) / sample.pdf
            )

        expected = np.array([
            2.0 * math.pi,
            math.pi,
            2.0 * math.pi / 3.0,
        ])

        means = estimates.mean(axis=0)

        standard_errors = (
            estimates.std(axis=0, ddof=1)
            / math.sqrt(self.sample_count)
        )

        for index in range(3):
            with self.subTest(integrand=index):
                error = abs(means[index] - expected[index])
                tolerance = 5.0 * standard_errors[index] + 1e-10

                self.assertLessEqual(
                    float(error),
                    float(tolerance),
                    msg=(
                        f"Expected {expected[index]}, "
                        f"obtained {means[index]}"
                    ),
                )


if __name__ == "__main__":
    unittest.main()