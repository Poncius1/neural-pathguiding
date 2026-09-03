"""Tests for the minimal MLP training script."""

import sys
import unittest
from math import log
from pathlib import Path
from unittest.mock import patch

import torch

from neural_path_guiding.training.model import GuidingMLP
from scripts import train_mlp as train_mlp_script


class TestTrainMlpScript(unittest.TestCase):
    """Validate command-line parsing and training helpers."""

    def test_parse_arguments_uses_expected_defaults(self) -> None:
        command_line = [
            "train_mlp.py",
            "--dataset",
            "data/example.npz",
            "--output",
            "checkpoints/example.pt",
        ]

        with patch.object(sys, "argv", command_line):
            arguments = train_mlp_script.parse_arguments()

        self.assertEqual(
            arguments.dataset,
            Path("data/example.npz"),
        )
        self.assertEqual(
            arguments.output,
            Path("checkpoints/example.pt"),
        )
        self.assertEqual(
            arguments.experiment,
            "minimal_mlp_overfit",
        )
        self.assertEqual(arguments.sample_count, 32)
        self.assertEqual(arguments.steps, 2000)
        self.assertEqual(arguments.learning_rate, 1e-3)
        self.assertEqual(arguments.hidden_dimension, 64)
        self.assertEqual(arguments.seed, 42)
        self.assertEqual(arguments.device, "cpu")

    def test_parse_arguments_accepts_custom_values(self) -> None:
        command_line = [
            "train_mlp.py",
            "--dataset",
            "data/example.npz",
            "--output",
            "checkpoints/example.pt",
            "--experiment",
            "custom_experiment",
            "--sample-count",
            "16",
            "--steps",
            "25",
            "--learning-rate",
            "0.01",
            "--hidden-dimension",
            "32",
            "--seed",
            "9",
            "--device",
            "cuda",
        ]

        with patch.object(sys, "argv", command_line):
            arguments = train_mlp_script.parse_arguments()

        self.assertEqual(
            arguments.experiment,
            "custom_experiment",
        )
        self.assertEqual(arguments.sample_count, 16)
        self.assertEqual(arguments.steps, 25)
        self.assertEqual(arguments.learning_rate, 0.01)
        self.assertEqual(arguments.hidden_dimension, 32)
        self.assertEqual(arguments.seed, 9)
        self.assertEqual(arguments.device, "cuda")

    def test_resolve_device_rejects_unavailable_cuda(self) -> None:
        with patch.object(
            torch.cuda,
            "is_available",
            return_value=False,
        ):
            with self.assertRaises(RuntimeError):
                train_mlp_script.resolve_device("cuda")

    def test_calculate_metrics_returns_expected_values(self) -> None:
        model = GuidingMLP(
            feature_dimension=2,
            hidden_dimension=4,
            num_bins=4,
        )

        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()

        features = torch.zeros(
            (3, 2),
            dtype=torch.float32,
        )
        targets = torch.tensor(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )

        entropy, kl_divergence, accuracy = (
            train_mlp_script.calculate_metrics(
                model,
                features,
                targets,
            )
        )

        self.assertAlmostEqual(entropy, 0.0, places=6)
        self.assertAlmostEqual(
            kl_divergence,
            log(4.0),
            places=6,
        )
        self.assertAlmostEqual(accuracy, 0.0, places=6)
        self.assertFalse(model.training)

    def test_build_checkpoint_metadata_includes_directional_schema(
        self,
    ) -> None:
        metadata = train_mlp_script.build_checkpoint_metadata(
            experiment="test_experiment",
            dataset_path=Path("data/example.npz"),
            sample_count=32,
            device=torch.device("cpu"),
            final_loss=3.125,
            n_mu=4,
            n_phi=8,
            target_type="visibility_cosine_v0",
            dataset_format_version=1,
        )

        self.assertEqual(
            metadata["experiment"],
            "test_experiment",
        )
        self.assertEqual(metadata["dataset"], "example.npz")
        self.assertEqual(metadata["sample_count"], 32)
        self.assertEqual(metadata["n_mu"], 4)
        self.assertEqual(metadata["n_phi"], 8)
        self.assertEqual(
            metadata["target_type"],
            "visibility_cosine_v0",
        )
        self.assertEqual(
            metadata["dataset_format_version"],
            1,
        )
        self.assertEqual(
            metadata["feature_schema_version"],
            1,
        )
        self.assertEqual(metadata["device"], "cpu")
        self.assertEqual(metadata["final_loss"], 3.125)


if __name__ == "__main__":
    unittest.main()