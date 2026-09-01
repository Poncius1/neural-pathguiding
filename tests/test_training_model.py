"""Tests for the minimal neural path guiding MLP."""

from __future__ import annotations

import unittest

import torch

from neural_path_guiding.training.model import GuidingMLP


class TestGuidingMLP(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)

    def test_forward_produces_expected_logit_shape(self) -> None:
        model = GuidingMLP(
            feature_dimension=14,
            num_bins=32,
            hidden_dimension=64,
        )
        features = torch.zeros(
            (4, 14),
            dtype=torch.float32,
        )

        logits = model(features)

        self.assertEqual(logits.shape, (4, 32))
        self.assertTrue(
            bool(torch.all(torch.isfinite(logits)))
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )
        probability_sums = probabilities.sum(dim=1)

        self.assertTrue(
            torch.allclose(
                probability_sums,
                torch.ones_like(probability_sums),
                atol=1e-6,
                rtol=0.0,
            )
        )

    def test_num_bins_is_dynamic(self) -> None:
        model = GuidingMLP(
            feature_dimension=14,
            num_bins=8,
            hidden_dimension=16,
        )
        features = torch.zeros(
            (3, 14),
            dtype=torch.float32,
        )

        logits = model(features)

        self.assertEqual(logits.shape, (3, 8))
        self.assertEqual(model.feature_dimension, 14)
        self.assertEqual(model.hidden_dimension, 16)
        self.assertEqual(model.num_bins, 8)

    def test_gradients_reach_every_parameter(self) -> None:
        model = GuidingMLP(
            feature_dimension=14,
            num_bins=32,
        )
        features = torch.randn(
            (4, 14),
            dtype=torch.float32,
        )

        logits = model(features)
        loss = logits.square().mean()
        loss.backward()

        for name, parameter in model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)

                if parameter.grad is not None:
                    self.assertTrue(
                        bool(
                            torch.all(
                                torch.isfinite(parameter.grad)
                            )
                        )
                    )

    def test_wrong_feature_rank_is_rejected(self) -> None:
        model = GuidingMLP(
            feature_dimension=14,
            num_bins=32,
        )
        features = torch.zeros(
            14,
            dtype=torch.float32,
        )

        with self.assertRaises(ValueError):
            model(features)

    def test_wrong_feature_dimension_is_rejected(self) -> None:
        model = GuidingMLP(
            feature_dimension=14,
            num_bins=32,
        )
        features = torch.zeros(
            (4, 13),
            dtype=torch.float32,
        )

        with self.assertRaises(ValueError):
            model(features)

    def test_invalid_model_dimensions_are_rejected(self) -> None:
        invalid_arguments = [
            {
                "feature_dimension": 0,
                "num_bins": 32,
                "hidden_dimension": 64,
            },
            {
                "feature_dimension": 14,
                "num_bins": 0,
                "hidden_dimension": 64,
            },
            {
                "feature_dimension": 14,
                "num_bins": 32,
                "hidden_dimension": 0,
            },
        ]

        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    GuidingMLP(**arguments)


if __name__ == "__main__":
    unittest.main()