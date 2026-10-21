"""Unit tests for MIS convergence analysis."""

from __future__ import annotations

import unittest

import numpy as np

from scripts.validate_mis_convergence import (
    compare_with_reference,
    luminance_samples,
    parse_pixel,
    sample_statistics,
    summarize_rows,
)


class TestConvergenceValidation(unittest.TestCase):

    def test_parse_pixel(self):
        self.assertEqual(parse_pixel("481,440"), (481, 440))

    def test_luminance_grayscale(self):
        rgb = np.array(
            [
                [0.25, 0.25, 0.25],
                [0.75, 0.75, 0.75],
            ]
        )

        luminance = luminance_samples(rgb)

        np.testing.assert_allclose(
            luminance,
            [0.25, 0.75],
        )

    def test_luminance_preserves_channel_correlation(self):
        rgb = np.array([
            [0.1, 0.2, 0.3],
            [0.2, 0.4, 0.6],
            [0.3, 0.6, 0.9],
        ])

        result = luminance_samples(rgb)

        expected = rgb @ np.array(
            [0.2126, 0.7152, 0.0722]
        )

        np.testing.assert_allclose(result, expected)

    def test_sample_statistics(self):
        values = np.array([1.0, 2.0, 3.0])

        result = sample_statistics(values)

        self.assertAlmostEqual(result["mean"], 2.0)
        self.assertAlmostEqual(result["variance"], 1.0)
        self.assertAlmostEqual(
            result["standard_error"],
            np.sqrt(1.0 / 3.0),
        )

    def test_standard_error_decreases_with_samples(self):
        base = np.array([0.0, 2.0])

        small = sample_statistics(base)
        large = sample_statistics(np.tile(base, 100))

        self.assertLess(
            large["standard_error"],
            small["standard_error"],
        )

    def test_identical_means_have_zero_z(self):
        values = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ])

        result = compare_with_reference(
            values,
            values.copy(),
        )

        difference = result["difference_luminance"]
        z_score = result["z_score"]

        self.assertIsInstance(difference, float)
        self.assertIsInstance(z_score, float)

        assert isinstance(difference, float)
        assert isinstance(z_score, float)

        self.assertAlmostEqual(difference, 0.0)
        self.assertAlmostEqual(z_score, 0.0)

        self.assertTrue(
            result["within_1_96_combined_se"]
        )
        
    def test_different_means(self):
        estimate = np.array([
            [1.0, 1.0, 1.0],
            [3.0, 3.0, 3.0],
        ])

        reference = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 2.0, 2.0],
        ])

        result = compare_with_reference(
            estimate,
            reference,
        )

        difference = result["difference_luminance"]

        assert isinstance(difference, float)

        self.assertAlmostEqual(
            difference,
            1.0,
        )
        
    def test_nonfinite_samples_rejected(self):
        with self.assertRaises(ValueError):
            luminance_samples(
                np.array([
                    [np.nan, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ])
            )

    def test_summary_aggregates_seeds(self):
        base = {
            "pixel_x": 481,
            "pixel_y": 440,
            "budget": 100,
            "estimator": "neural_bsdf_light_mis_3",
            "standard_error_luminance": 0.1,
            "seconds_per_estimate": 0.001,
            "variance_times_seconds": 0.00001,
            "within_1_96_combined_se": True,
            "fallback_queries": 0,
        }

        first = {
            **base,
            "seed": 42,
            "variance_luminance": 0.01,
            "absolute_difference_luminance": 0.02,
            "difference_luminance": 0.02,
            "z_score": 0.5,
        }

        second = {
            **base,
            "seed": 123,
            "variance_luminance": 0.03,
            "absolute_difference_luminance": 0.04,
            "difference_luminance": -0.04,
            "z_score": -1.0,
        }

        summary = summarize_rows([first, second])

        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["seed_count"], 2)
        self.assertAlmostEqual(
            summary[0]["mean_variance"],
            0.02,
        )
        self.assertEqual(
            summary[0]["fallback_queries_total"],
            0,
        )


if __name__ == "__main__":
    unittest.main()