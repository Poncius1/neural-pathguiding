"""Tests for the discrete directional distribution interface."""

from __future__ import annotations

from importlib.metadata import distribution
import unittest

import numpy as np

from neural_path_guiding.core.bins import (
    HemisphereBins,
)
from neural_path_guiding.core.directional_distribution import (
    DiscreteDirectionalDistribution,
    softmax_logits,
)


class TestDiscreteDirectionalDistribution(
    unittest.TestCase
):
    def setUp(self) -> None:
        self.bins = HemisphereBins(
            n_mu=4,
            n_phi=8,
        )

    def test_softmax_is_stable_for_large_logits(
        self,
    ) -> None:
        logits = np.full(
            self.bins.n_bins,
            -1000.0,
        )
        logits[7] = 1000.0

        probabilities = softmax_logits(
            logits=logits,
            expected_size=self.bins.n_bins,
        )

        self.assertTrue(
            bool(
                np.all(
                    np.isfinite(probabilities)
                )
            )
        )
        self.assertAlmostEqual(
            float(probabilities.sum()),
            1.0,
        )
        self.assertAlmostEqual(
            float(probabilities[7]),
            1.0,
        )

    def test_distribution_normalizes_probabilities(
        self,
    ) -> None:
        probabilities = np.arange(
            1,
            self.bins.n_bins + 1,
            dtype=np.float64,
        )

        distribution = (
            DiscreteDirectionalDistribution(
                bins=self.bins,
                probabilities=probabilities,
            )
        )

        self.assertAlmostEqual(
            float(
                distribution
                .probabilities
                .sum()
            ),
            1.0,
        )
        self.assertFalse(
            distribution
            .probabilities
            .flags
            .writeable
        )

    def test_uniform_mix_provides_support(
        self,
    ) -> None:
        probabilities = np.zeros(
            self.bins.n_bins,
            dtype=np.float64,
        )
        probabilities[5] = 1.0

        alpha = 0.2

        distribution = (
            DiscreteDirectionalDistribution(
                bins=self.bins,
                probabilities=probabilities,
                uniform_mix=alpha,
            )
        )

        minimum_expected = (
            alpha / self.bins.n_bins
        )

        zero_probability_bins = (
    probabilities == 0.0
        )

        np.testing.assert_allclose(
            distribution.probabilities[
                zero_probability_bins
            ],
            minimum_expected,
            rtol=0.0,
            atol=1e-15,
        )

        self.assertTrue(
            bool(
                np.all(
                    distribution.probabilities > 0.0
                )
            )
        )
        self.assertAlmostEqual(
            float(
                distribution
                .probabilities
                .sum()
            ),
            1.0,
        )

    def test_sample_pdf_matches_evaluation(
        self,
    ) -> None:
        logits = np.linspace(
            -2.0,
            2.0,
            self.bins.n_bins,
        )

        distribution = (
            DiscreteDirectionalDistribution
            .from_logits(
                bins=self.bins,
                logits=logits,
                uniform_mix=0.05,
            )
        )

        rng = np.random.default_rng(42)

        for _ in range(200):
            sample = distribution.sample(rng)

            evaluated_pdf = distribution.pdf(
                sample.direction_local
            )

            self.assertAlmostEqual(
                sample.pdf,
                evaluated_pdf,
            )

            self.assertEqual(
                distribution
                .probability_for_bin(
                    sample.bin_index
                ),
                distribution
                .probabilities[
                    sample.bin_index
                ],
            )

    def test_pdf_integrates_to_one(
        self,
    ) -> None:
        probabilities = np.arange(
            1,
            self.bins.n_bins + 1,
            dtype=np.float64,
        )

        distribution = (
            DiscreteDirectionalDistribution(
                bins=self.bins,
                probabilities=probabilities,
                uniform_mix=0.05,
            )
        )

        integral = 0.0

        for bin_index in range(
            self.bins.n_bins
        ):
            direction = (
                self.bins
                .sample_direction_in_bin(
                    bin_index=bin_index,
                    u_mu=0.5,
                    u_phi=0.5,
                )
            )

            integral += (
                distribution.pdf(direction)
                * self.bins.bin_solid_angle
            )

        self.assertAlmostEqual(
            integral,
            1.0,
            places=12,
        )

    def test_empirical_histogram_matches_pmf(
        self,
    ) -> None:
        probabilities = np.arange(
            1,
            self.bins.n_bins + 1,
            dtype=np.float64,
        )

        distribution = (
            DiscreteDirectionalDistribution(
                bins=self.bins,
                probabilities=probabilities,
                uniform_mix=0.1,
            )
        )

        rng = np.random.default_rng(2026)

        counts = np.zeros(
            self.bins.n_bins,
            dtype=np.int64,
        )

        sample_count = 10_000

        for _ in range(sample_count):
            sample = distribution.sample(rng)
            counts[sample.bin_index] += 1

        empirical = (
            counts.astype(np.float64)
            / sample_count
        )

        self.assertTrue(
            bool(
                np.allclose(
                    empirical,
                    distribution.probabilities,
                    atol=0.015,
                    rtol=0.0,
                )
            )
        )

    def test_rejects_invalid_inputs(
        self,
    ) -> None:
        invalid_logits = (
            np.zeros(
                self.bins.n_bins - 1
            ),
            np.full(
                self.bins.n_bins,
                np.nan,
            ),
            np.full(
                self.bins.n_bins,
                np.inf,
            ),
        )

        for logits in invalid_logits:
            with self.subTest(logits=logits):
                with self.assertRaises(
                    ValueError
                ):
                    (
                        DiscreteDirectionalDistribution
                        .from_logits(
                            bins=self.bins,
                            logits=logits,
                        )
                    )

        with self.assertRaises(ValueError):
            DiscreteDirectionalDistribution(
                bins=self.bins,
                probabilities=np.ones(
                    self.bins.n_bins
                ),
                uniform_mix=-0.1,
            )

        distribution = (
            DiscreteDirectionalDistribution(
                bins=self.bins,
                probabilities=np.ones(
                    self.bins.n_bins
                ),
            )
        )

        with self.assertRaises(ValueError):
            distribution.probability_for_bin(
                self.bins.n_bins
            )


if __name__ == "__main__":
    unittest.main()