"""Tests for the minimal full-batch trainer."""

from __future__ import annotations

import unittest

import torch
from torch.nn import functional as F

from neural_path_guiding.training.model import GuidingMLP
from neural_path_guiding.training.trainer import (
    train_full_batch,
)


class TestFullBatchTrainer(unittest.TestCase):
    def make_training_problem(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.tensor(
            [
                [-2.0, -1.0, 0.0, 1.0],
                [-1.5, -0.5, 0.0, 1.0],
                [-1.0, -2.0, 1.0, 0.0],
                [-0.5, -1.0, 1.0, 0.0],
                [-2.0, 1.0, 0.0, 1.0],
                [-1.0, 0.5, 0.0, 1.0],
                [-0.5, 2.0, 1.0, 0.0],
                [-0.2, 1.0, 1.0, 0.0],
                [0.2, -1.0, 0.0, 1.0],
                [0.5, -2.0, 0.0, 1.0],
                [1.0, -0.5, 1.0, 0.0],
                [2.0, -1.0, 1.0, 0.0],
                [0.5, 1.0, 0.0, 1.0],
                [1.0, 2.0, 0.0, 1.0],
                [1.5, 0.5, 1.0, 0.0],
                [2.0, 1.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )

        labels = (
            features[:, 0] > 0.0
        ).to(torch.int64)

        targets = F.one_hot(
            labels,
            num_classes=2,
        ).to(torch.float32)

        return features, targets

    def test_training_reduces_loss_and_updates_parameters(
        self,
    ) -> None:
        torch.manual_seed(42)

        features, targets = self.make_training_problem()

        model = GuidingMLP(
            feature_dimension=4,
            num_bins=2,
            hidden_dimension=16,
        )

        initial_parameters = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }

        result = train_full_batch(
            model=model,
            features=features,
            targets=targets,
            num_steps=150,
            learning_rate=0.02,
        )

        self.assertEqual(result.num_steps, 150)
        self.assertEqual(len(result.losses), 150)

        self.assertLess(
            result.final_loss,
            result.initial_loss,
        )
        self.assertLess(
            result.final_loss,
            0.1,
        )

        parameters_changed = any(
            not torch.equal(
                initial_parameters[name],
                parameter.detach(),
            )
            for name, parameter in model.named_parameters()
        )

        self.assertTrue(parameters_changed)
        self.assertFalse(model.training)

    def test_same_seed_reproduces_training(self) -> None:
        features, targets = self.make_training_problem()

        torch.manual_seed(123)
        first_model = GuidingMLP(
            feature_dimension=4,
            num_bins=2,
            hidden_dimension=16,
        )

        torch.manual_seed(123)
        second_model = GuidingMLP(
            feature_dimension=4,
            num_bins=2,
            hidden_dimension=16,
        )

        first_result = train_full_batch(
            model=first_model,
            features=features,
            targets=targets,
            num_steps=50,
            learning_rate=0.01,
        )
        second_result = train_full_batch(
            model=second_model,
            features=features,
            targets=targets,
            num_steps=50,
            learning_rate=0.01,
        )

        self.assertTrue(
            torch.allclose(
                torch.tensor(first_result.losses),
                torch.tensor(second_result.losses),
                atol=0.0,
                rtol=0.0,
            )
        )
        self.assertEqual(
            first_result.final_loss,
            second_result.final_loss,
        )

        for (
            first_parameter,
            second_parameter,
        ) in zip(
            first_model.parameters(),
            second_model.parameters(),
            strict=True,
        ):
            self.assertTrue(
                torch.equal(
                    first_parameter,
                    second_parameter,
                )
            )

    def test_invalid_training_settings_are_rejected(
        self,
    ) -> None:
        features, targets = self.make_training_problem()

        invalid_settings = [
            {
                "num_steps": 0,
                "learning_rate": 0.01,
                "exception": ValueError,
            },
            {
                "num_steps": -1,
                "learning_rate": 0.01,
                "exception": ValueError,
            },
            {
                "num_steps": 10,
                "learning_rate": 0.0,
                "exception": ValueError,
            },
            {
                "num_steps": 10,
                "learning_rate": float("nan"),
                "exception": ValueError,
            },
            {
                "num_steps": 1.5,
                "learning_rate": 0.01,
                "exception": TypeError,
            },
            {
                "num_steps": 10,
                "learning_rate": "invalid",
                "exception": TypeError,
            },
        ]

        for settings in invalid_settings:
            with self.subTest(settings=settings):
                model = GuidingMLP(
                    feature_dimension=4,
                    num_bins=2,
                )

                with self.assertRaises(
                    settings["exception"]
                ):
                    train_full_batch(
                        model=model,
                        features=features,
                        targets=targets,
                        num_steps=settings["num_steps"],
                        learning_rate=settings[
                            "learning_rate"
                        ],
                    )

    def test_device_mismatch_is_rejected(self) -> None:
        features, targets = self.make_training_problem()

        model = GuidingMLP(
            feature_dimension=4,
            num_bins=2,
        )

        meta_features = features.to("meta")

        with self.assertRaises(ValueError):
            train_full_batch(
                model=model,
                features=meta_features,
                targets=targets,
                num_steps=1,
                learning_rate=0.01,
            )


if __name__ == "__main__":
    unittest.main()