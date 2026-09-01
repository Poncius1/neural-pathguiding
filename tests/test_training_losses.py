"""Tests for neural path guiding training losses."""

from __future__ import annotations

import math
import unittest

import torch

from neural_path_guiding.training.losses import (
    soft_target_cross_entropy,
)


class TestSoftTargetCrossEntropy(unittest.TestCase):
    def test_uniform_prediction_has_expected_loss(self) -> None:
        logits = torch.zeros(
            (2, 4),
            dtype=torch.float32,
        )
        targets = torch.full(
            (2, 4),
            0.25,
            dtype=torch.float32,
        )

        loss = soft_target_cross_entropy(
            logits=logits,
            targets=targets,
        )

        self.assertAlmostEqual(
            float(loss.item()),
            math.log(4.0),
            places=6,
        )

    def test_matching_distribution_reaches_target_entropy(self) -> None:
        targets = torch.tensor(
            [
                [0.70, 0.20, 0.10],
                [0.20, 0.30, 0.50],
            ],
            dtype=torch.float32,
        )

        # softmax(log(P)) = P when every probability is positive.
        logits = torch.log(targets)

        loss = soft_target_cross_entropy(
            logits=logits,
            targets=targets,
        )

        expected_entropy = -torch.sum(
            targets * torch.log(targets),
            dim=1,
        ).mean()

        self.assertTrue(
            torch.allclose(
                loss,
                expected_entropy,
                atol=1e-6,
                rtol=0.0,
            )
        )

    def test_incorrect_prediction_has_higher_loss(self) -> None:
        targets = torch.tensor(
            [[0.90, 0.05, 0.05]],
            dtype=torch.float32,
        )

        matching_logits = torch.log(targets)

        incorrect_logits = torch.tensor(
            [[-5.0, -5.0, 5.0]],
            dtype=torch.float32,
        )

        matching_loss = soft_target_cross_entropy(
            logits=matching_logits,
            targets=targets,
        )
        incorrect_loss = soft_target_cross_entropy(
            logits=incorrect_logits,
            targets=targets,
        )

        self.assertLess(
            float(matching_loss.item()),
            float(incorrect_loss.item()),
        )

    def test_extreme_logits_remain_finite(self) -> None:
        logits = torch.tensor(
            [
                [1000.0, -1000.0],
                [-1000.0, 1000.0],
            ],
            dtype=torch.float32,
        )
        targets = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        )

        loss = soft_target_cross_entropy(
            logits=logits,
            targets=targets,
        )

        self.assertTrue(
            bool(torch.isfinite(loss))
        )
        self.assertAlmostEqual(
            float(loss.item()),
            0.0,
            places=6,
        )

    def test_loss_produces_finite_gradients(self) -> None:
        torch.manual_seed(42)

        logits = torch.randn(
            (4, 8),
            dtype=torch.float32,
            requires_grad=True,
        )
        targets = torch.full(
            (4, 8),
            1.0 / 8.0,
            dtype=torch.float32,
        )

        loss = soft_target_cross_entropy(
            logits=logits,
            targets=targets,
        )
        loss.backward()

        self.assertIsNotNone(logits.grad)

        if logits.grad is not None:
            self.assertTrue(
                bool(
                    torch.all(
                        torch.isfinite(logits.grad)
                    )
                )
            )

    def test_invalid_shapes_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            soft_target_cross_entropy(
                logits=torch.zeros(4),
                targets=torch.full((4,), 0.25),
            )

        with self.assertRaises(ValueError):
            soft_target_cross_entropy(
                logits=torch.zeros((2, 4)),
                targets=torch.full((2, 3), 1.0 / 3.0),
            )

        with self.assertRaises(ValueError):
            soft_target_cross_entropy(
                logits=torch.zeros((0, 4)),
                targets=torch.zeros((0, 4)),
            )

    def test_invalid_values_and_dtypes_are_rejected(self) -> None:
        valid_logits = torch.zeros(
            (1, 3),
            dtype=torch.float32,
        )

        invalid_targets = [
            torch.tensor(
                [[-0.1, 0.5, 0.6]],
                dtype=torch.float32,
            ),
            torch.tensor(
                [[0.2, 0.2, 0.2]],
                dtype=torch.float32,
            ),
            torch.tensor(
                [[float("nan"), 0.5, 0.5]],
                dtype=torch.float32,
            ),
        ]

        for targets in invalid_targets:
            with self.subTest(targets=targets):
                with self.assertRaises(ValueError):
                    soft_target_cross_entropy(
                        logits=valid_logits,
                        targets=targets,
                    )

        with self.assertRaises(TypeError):
            soft_target_cross_entropy(
                logits=torch.zeros(
                    (1, 3),
                    dtype=torch.int64,
                ),
                targets=torch.tensor(
                    [[0.2, 0.3, 0.5]],
                    dtype=torch.float32,
                ),
            )

        with self.assertRaises(TypeError):
            soft_target_cross_entropy(
                logits=valid_logits,
                targets=torch.tensor(
                    [[0, 0, 1]],
                    dtype=torch.int64,
                ),
            )


if __name__ == "__main__":
    unittest.main()