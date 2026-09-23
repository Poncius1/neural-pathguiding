"""Tests for multiple importance sampling heuristics."""

from __future__ import annotations

import unittest

from neural_path_guiding.core.mis import (
    balance_heuristic,
    power_heuristic,
)


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


if __name__ == "__main__":
    unittest.main()