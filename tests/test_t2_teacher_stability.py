"""Tests for T2 teacher stability metrics and experiment helpers."""

from __future__ import annotations

import unittest

import numpy as np

from scripts.validate_t2_teacher_stability import (
    compare_distributions,
    jensen_shannon_divergence,
    make_independent_seeds,
    parse_pixel,
    summarize_comparisons,
    total_variation_distance,
    validate_distribution,
)


class TestDistributionMetrics(unittest.TestCase):

    def test_identical_distributions_have_zero_tvd(
        self,
    ) -> None:
        p = np.array([0.2, 0.3, 0.5])

        result = total_variation_distance(p, p)

        self.assertAlmostEqual(result, 0.0)

    def test_identical_distributions_have_zero_jsd(
        self,
    ) -> None:
        p = np.array([0.2, 0.3, 0.5])

        result = jensen_shannon_divergence(p, p)

        self.assertAlmostEqual(result, 0.0)

    def test_disjoint_distributions(self) -> None:
        p = np.array([1.0, 0.0])
        q = np.array([0.0, 1.0])

        self.assertAlmostEqual(
            total_variation_distance(p, q),
            1.0,
        )

        self.assertAlmostEqual(
            jensen_shannon_divergence(p, q),
            np.log(2.0),
        )

    def test_metrics_are_symmetric(self) -> None:
        p = np.array([0.8, 0.2])
        q = np.array([0.3, 0.7])

        self.assertAlmostEqual(
            total_variation_distance(p, q),
            total_variation_distance(q, p),
        )

        self.assertAlmostEqual(
            jensen_shannon_divergence(p, q),
            jensen_shannon_divergence(q, p),
        )

    def test_comparison_returns_two_metrics(
        self,
    ) -> None:
        p = np.array([0.5, 0.5])
        q = np.array([0.75, 0.25])

        result = compare_distributions(p, q)

        self.assertAlmostEqual(
            result["tvd"],
            0.25,
        )

        self.assertGreater(
            result["jsd"],
            0.0,
        )

    def test_rejects_bad_normalization(self) -> None:
        with self.assertRaises(ValueError):
            validate_distribution(
                np.array([0.2, 0.2])
            )

    def test_rejects_negative_probabilities(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            validate_distribution(
                np.array([1.1, -0.1])
            )

    def test_rejects_mismatched_shapes(self) -> None:
        with self.assertRaises(ValueError):
            total_variation_distance(
                np.array([0.5, 0.5]),
                np.array([0.2, 0.3, 0.5]),
            )

    def test_zero_probabilities_are_supported(
        self,
    ) -> None:
        p = np.array([0.0, 1.0, 0.0])
        q = np.array([0.5, 0.5, 0.0])

        result = jensen_shannon_divergence(p, q)

        self.assertTrue(np.isfinite(result))
        self.assertGreaterEqual(result, 0.0)


class TestExperimentHelpers(unittest.TestCase):

    def test_parse_pixel(self) -> None:
        self.assertEqual(
            parse_pixel("224,224"),
            (224, 224),
        )

    def test_rejects_invalid_pixel(self) -> None:
        with self.assertRaises(Exception):
            parse_pixel("abc")

    def test_independent_seeds_are_reproducible(
        self,
    ) -> None:
        first = make_independent_seeds(
            42,
            10,
        )

        second = make_independent_seeds(
            42,
            10,
        )

        self.assertEqual(first, second)
        self.assertEqual(len(set(first)), 10)

    def test_summary_aggregation(self) -> None:
        records = [
            {
                "pixel_x": 0,
                "pixel_y": 0,
                "comparison": "low_replica_difference",
                "tvd": 0.2,
                "jsd": 0.04,
            },
            {
                "pixel_x": 1,
                "pixel_y": 1,
                "comparison": "low_replica_difference",
                "tvd": 0.4,
                "jsd": 0.08,
            },
        ]

        summary = summarize_comparisons(records)

        group = summary["low_replica_difference"]

        self.assertEqual(group["count"], 2)

        self.assertAlmostEqual(
            group["mean_tvd"],
            0.3,
        )

        self.assertAlmostEqual(
            group["max_tvd"],
            0.4,
        )

        self.assertAlmostEqual(
            group["mean_jsd"],
            0.06,
        )


if __name__ == "__main__":
    unittest.main()