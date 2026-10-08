"""Tests for multiple importance sampling heuristics."""

from __future__ import annotations

import unittest
import numpy as np
from neural_path_guiding.core.mis import (
    balance_heuristic,
    combine_two_technique_samples,
    power_heuristic,
)
from neural_path_guiding.core.mis import power_heuristic_weights

class TestMISHeuristics(unittest.TestCase):
    def test_equal_densities_receive_equal_balance_weights(
        self,
    ) -> None:
        weight_a = balance_heuristic(
            0.25,
            0.25,
        )
        weight_b = balance_heuristic(
            0.25,
            0.25,
        )

        self.assertAlmostEqual(weight_a, 0.5)
        self.assertAlmostEqual(weight_a + weight_b, 1.0)

    def test_balance_heuristic_accounts_for_sample_counts(
        self,
    ) -> None:
        weight = balance_heuristic(
            0.2,
            0.4,
            samples_a=2,
            samples_b=1,
        )

        self.assertAlmostEqual(weight, 0.5)

    def test_power_heuristic_favors_higher_weighted_density(
        self,
    ) -> None:
        weight = power_heuristic(
            0.8,
            0.2,
        )

        self.assertGreater(weight, 0.5)
        self.assertAlmostEqual(
            weight,
            0.8**2 / (0.8**2 + 0.2**2),
        )

    def test_weights_for_opposite_techniques_sum_to_one(
        self,
    ) -> None:
        weight_a = power_heuristic(
            0.3,
            0.7,
            samples_a=2,
            samples_b=3,
        )
        weight_b = power_heuristic(
            0.7,
            0.3,
            samples_a=3,
            samples_b=2,
        )

        self.assertAlmostEqual(
            weight_a + weight_b,
            1.0,
        )

    def test_zero_density_gets_zero_weight(
        self,
    ) -> None:
        self.assertEqual(
            power_heuristic(0.0, 1.0),
            0.0,
        )
        self.assertEqual(
            power_heuristic(1.0, 0.0),
            1.0,
        )

    def test_zero_sample_count_gets_zero_weight(
        self,
    ) -> None:
        self.assertEqual(
            balance_heuristic(
                1.0,
                1.0,
                samples_a=0,
                samples_b=1,
            ),
            0.0,
        )

    def test_heuristic_remains_finite_for_extreme_pdfs(
        self,
    ) -> None:
        weight = power_heuristic(
            1e300,
            1e-300,
        )

        self.assertEqual(weight, 1.0)

    def test_rejects_both_zero_densities(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "zero weighted density",
        ):
            balance_heuristic(0.0, 0.0)

    def test_rejects_invalid_pdfs(
        self,
    ) -> None:
        invalid_values = (
            -1.0,
            float("nan"),
            float("inf"),
            True,
        )

        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises(
                    (TypeError, ValueError)
                ):
                    power_heuristic(value, 1.0)

    def test_rejects_invalid_sample_counts(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            balance_heuristic(
                1.0,
                1.0,
                samples_a=-1,
            )

        with self.assertRaises(TypeError):
            balance_heuristic(
                1.0,
                1.0,
                samples_a=True,
            )

    def test_rejects_invalid_exponent(
        self,
    ) -> None:
        for exponent in (
            0.0,
            -1.0,
            float("nan"),
            float("inf"),
        ):
            with self.subTest(exponent=exponent):
                with self.assertRaises(ValueError):
                    power_heuristic(
                        1.0,
                        1.0,
                        exponent=exponent,
                    )

    def test_combines_two_rgb_samples_with_power_weights(self) -> None:
            contribution_a = np.array([2.0, 1.0, 0.5])
            contribution_b = np.array([1.0, 3.0, 2.0])

            result = combine_two_technique_samples(
                contribution_a=contribution_a,
                pdf_a_at_a=0.8,
                pdf_b_at_a=0.2,
                contribution_b=contribution_b,
                pdf_a_at_b=0.25,
                pdf_b_at_b=0.75,
            )

            weight_a = 0.8**2 / (0.8**2 + 0.2**2)
            weight_b = 0.75**2 / (0.75**2 + 0.25**2)
            expected = (
                weight_a * contribution_a / 0.8
                + weight_b * contribution_b / 0.75
            )

            np.testing.assert_allclose(result, expected)

    def test_combiner_rejects_zero_density_for_sampled_technique(self) -> None:
        with self.assertRaisesRegex(ValueError, "pdf_a_at_a"):
            combine_two_technique_samples(
                contribution_a=np.ones(3),
                pdf_a_at_a=0.0,
                pdf_b_at_a=1.0,
                contribution_b=np.ones(3),
                pdf_a_at_b=1.0,
                pdf_b_at_b=1.0,
            )

    def test_combiner_rejects_invalid_rgb_contribution(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            combine_two_technique_samples(
                contribution_a=np.ones(2),
                pdf_a_at_a=1.0,
                pdf_b_at_a=1.0,
                contribution_b=np.ones(3),
                pdf_a_at_b=1.0,
                pdf_b_at_b=1.0,
            )

class TestMultiTechniquePowerHeuristic(unittest.TestCase):
    def test_equal_pdfs_receive_equal_weights(self) -> None:
        weights = power_heuristic_weights([0.25, 0.25, 0.25])

        np.testing.assert_allclose(
            weights,
            [1.0 / 3.0] * 3,
        )

    def test_zero_pdf_receives_zero_weight(self) -> None:
        weights = power_heuristic_weights([0.5, 0.0, 0.25])

        self.assertEqual(weights[1], 0.0)
        np.testing.assert_allclose(weights.sum(), 1.0)

    def test_sample_counts_affect_weights(self) -> None:
        weights = power_heuristic_weights(
            [0.5, 0.5],
            sample_counts=[1, 2],
        )

        np.testing.assert_allclose(weights, [0.2, 0.8])

    def test_extreme_pdfs_produce_finite_weights(self) -> None:
        weights = power_heuristic_weights(
            [1e-250, 1e250, 0.0],
        )

        self.assertTrue(np.all(np.isfinite(weights)))
        np.testing.assert_allclose(weights, [0.0, 1.0, 0.0])

    def test_rejects_invalid_pdf_values(self) -> None:
        invalid_inputs = (
            [0.5],
            [0.5, np.nan],
            [0.5, np.inf],
            [0.5, -0.1],
            [0.0, 0.0],
        )

        for pdfs in invalid_inputs:
            with self.subTest(pdfs=pdfs):
                with self.assertRaises(ValueError):
                    power_heuristic_weights(pdfs)

    def test_rejects_invalid_sample_counts(self) -> None:
        with self.assertRaises(ValueError):
            power_heuristic_weights(
                [0.5, 0.5],
                sample_counts=[1],
            )

        with self.assertRaises(ValueError):
            power_heuristic_weights(
                [0.5, 0.5],
                sample_counts=[0, 0],
            )

    def test_rejects_invalid_exponent(self) -> None:
        with self.assertRaises(ValueError):
            power_heuristic_weights(
                [0.5, 0.5],
                exponent=0.0,
            )
if __name__ == "__main__":
    unittest.main()