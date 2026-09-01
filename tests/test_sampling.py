"""Tests for discrete directional sampling and sample/PDF consistency."""

from __future__ import annotations

import unittest

import numpy as np

from neural_path_guiding.core.bins import HemisphereBins
from neural_path_guiding.core.pdf import (
    evaluate_pdf_from_probabilities,
    mix_with_uniform,
)
from neural_path_guiding.core.sampling import DirectionalSample, sample_direction


class TestDirectionalSampling(unittest.TestCase):
    def setUp(self) -> None:
        self.bins = HemisphereBins(n_mu=4, n_phi=8)

    def test_sample_has_valid_direction_bin_and_pdf(self) -> None:
        probabilities = np.arange(1, self.bins.n_bins + 1, dtype=np.float64)
        sample = sample_direction(
            bins=self.bins,
            probabilities=probabilities,
            rng=np.random.default_rng(42),
        )

        self.assertEqual(sample.direction_local.shape, (3,))
        self.assertTrue(bool(np.all(np.isfinite(sample.direction_local))))
        self.assertAlmostEqual(float(np.linalg.norm(sample.direction_local)), 1.0)
        self.assertGreaterEqual(float(sample.direction_local[2]), 0.0)
        self.assertGreaterEqual(sample.bin_index, 0)
        self.assertLess(sample.bin_index, self.bins.n_bins)
        self.assertGreater(sample.pdf, 0.0)
        self.assertFalse(sample.direction_local.flags.writeable)

        recovered_bin = self.bins.find_bin_from_direction(sample.direction_local)
        self.assertEqual(recovered_bin, sample.bin_index)

    def test_returned_pdf_matches_independent_evaluation(self) -> None:
        probabilities = np.arange(1, self.bins.n_bins + 1, dtype=np.float64)
        alpha = 0.15
        effective_probabilities = mix_with_uniform(
            probabilities=probabilities,
            alpha=alpha,
            expected_size=self.bins.n_bins,
        )
        sample = sample_direction(
            bins=self.bins,
            probabilities=probabilities,
            rng=np.random.default_rng(7),
            alpha=alpha,
        )

        evaluated_pdf = evaluate_pdf_from_probabilities(
            bins=self.bins,
            probabilities=effective_probabilities,
            omega_local=sample.direction_local,
        )

        self.assertAlmostEqual(sample.pdf, evaluated_pdf)

    def test_concentrated_distribution_always_selects_its_bin(self) -> None:
        selected_bin = 11
        probabilities = np.zeros(self.bins.n_bins, dtype=np.float64)
        probabilities[selected_bin] = 1.0
        rng = np.random.default_rng(123)

        for _ in range(200):
            sample = sample_direction(self.bins, probabilities, rng)
            self.assertEqual(sample.bin_index, selected_bin)
            self.assertEqual(
                self.bins.find_bin_from_direction(sample.direction_local),
                selected_bin,
            )

    def test_same_seed_reproduces_complete_sample_sequence(self) -> None:
        probabilities = np.arange(1, self.bins.n_bins + 1, dtype=np.float64)
        first_rng = np.random.default_rng(2026)
        second_rng = np.random.default_rng(2026)

        first_sequence = [
            sample_direction(self.bins, probabilities, first_rng)
            for _ in range(20)
        ]
        second_sequence = [
            sample_direction(self.bins, probabilities, second_rng)
            for _ in range(20)
        ]

        for first, second in zip(first_sequence, second_sequence, strict=True):
            self.assertEqual(first.bin_index, second.bin_index)
            self.assertEqual(first.pdf, second.pdf)
            self.assertTrue(
                bool(np.array_equal(first.direction_local, second.direction_local))
            )

    def test_invalid_probability_vectors_are_rejected(self) -> None:
        invalid_probabilities = [
            np.zeros(self.bins.n_bins),
            np.full(self.bins.n_bins, np.nan),
            np.full(self.bins.n_bins, np.inf),
            -np.ones(self.bins.n_bins),
            np.ones(self.bins.n_bins - 1),
        ]

        for probabilities in invalid_probabilities:
            with self.subTest(probabilities=probabilities):
                with self.assertRaises(ValueError):
                    sample_direction(
                        self.bins,
                        probabilities,
                        np.random.default_rng(42),
                    )

    def test_invalid_rng_is_rejected(self) -> None:
        probabilities = np.ones(self.bins.n_bins, dtype=np.float64)

        with self.assertRaises(TypeError):
            sample_direction(self.bins, probabilities, rng=None)  # type: ignore[arg-type]

    def test_invalid_uniform_mix_alpha_is_rejected(self) -> None:
        probabilities = np.ones(self.bins.n_bins, dtype=np.float64)

        for alpha in (-0.1, 1.1, np.nan, np.inf):
            with self.subTest(alpha=alpha):
                with self.assertRaises(ValueError):
                    sample_direction(
                        self.bins,
                        probabilities,
                        np.random.default_rng(42),
                        alpha=alpha,
                    )

    def test_directional_sample_rejects_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            DirectionalSample(
                direction_local=np.array([0.0, 0.0, 2.0]),
                bin_index=0,
                pdf=1.0,
            )

        with self.assertRaises(ValueError):
            DirectionalSample(
                direction_local=np.array([0.0, 0.0, 1.0]),
                bin_index=0,
                pdf=0.0,
            )

    def test_uniform_pmf_matches_empirical_histogram(self) -> None:
        probabilities = np.ones(self.bins.n_bins, dtype=np.float64)
        empirical = self.estimate_bin_frequencies(
            probabilities=probabilities,
            sample_count=15_000,
            seed=101,
        )
        expected = np.full(self.bins.n_bins, 1.0 / self.bins.n_bins)

        self.assertTrue(bool(np.allclose(empirical, expected, atol=0.01, rtol=0.0)))

    def test_random_pmf_matches_empirical_histogram(self) -> None:
        probability_rng = np.random.default_rng(55)
        probabilities = probability_rng.random(self.bins.n_bins)
        expected = probabilities / float(probabilities.sum())
        empirical = self.estimate_bin_frequencies(
            probabilities=probabilities,
            sample_count=25_000,
            seed=202,
        )

        self.assertTrue(bool(np.allclose(empirical, expected, atol=0.01, rtol=0.0)))

    def test_uniform_mix_matches_empirical_histogram(self) -> None:
        probabilities = np.zeros(self.bins.n_bins, dtype=np.float64)
        probabilities[0] = 1.0
        alpha = 0.2
        expected = mix_with_uniform(
            probabilities=probabilities,
            alpha=alpha,
            expected_size=self.bins.n_bins,
        )
        empirical = self.estimate_bin_frequencies(
            probabilities=probabilities,
            sample_count=20_000,
            seed=303,
            alpha=alpha,
        )

        self.assertTrue(bool(np.allclose(empirical, expected, atol=0.01, rtol=0.0)))

    def estimate_bin_frequencies(
        self,
        probabilities: np.ndarray,
        sample_count: int,
        seed: int,
        alpha: float = 0.0,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        counts = np.zeros(self.bins.n_bins, dtype=np.int64)

        for _ in range(sample_count):
            sample = sample_direction(
                self.bins,
                probabilities,
                rng,
                alpha=alpha,
            )
            counts[sample.bin_index] += 1

        return counts.astype(np.float64) / sample_count


if __name__ == "__main__":
    unittest.main()
