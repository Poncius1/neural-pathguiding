"""Tests for training checkpoint serialization."""

import tempfile
import unittest
from pathlib import Path
from typing import Any, cast
import torch

from neural_path_guiding.training.checkpoint import (
    load_checkpoint,
    save_checkpoint,
)
from neural_path_guiding.training.model import GuidingMLP


class TestTrainingCheckpoint(unittest.TestCase):
    """Validate saving and loading of guiding models."""

    def setUp(self) -> None:
        torch.manual_seed(7)

        self.model = GuidingMLP(
            feature_dimension=3,
            hidden_dimension=8,
            num_bins=4,
        )

        self.feature_mean = torch.tensor(
            [1.0, 2.0, 3.0],
            dtype=torch.float32,
        )
        self.feature_scale = torch.tensor(
            [2.0, 3.0, 4.0],
            dtype=torch.float32,
        )

    def test_round_trip_preserves_predictions_and_information(self) -> None:
        raw_features = torch.tensor(
            [
                [2.0, 5.0, 7.0],
                [0.0, 2.0, 3.0],
            ],
            dtype=torch.float32,
        )

        normalized_features = (
            raw_features - self.feature_mean
        ) / self.feature_scale

        self.model.eval()

        with torch.inference_mode():
            expected_logits = self.model(normalized_features)

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = (
                Path(temporary_directory) / "guiding_model.pt"
            )

            saved_path = save_checkpoint(
                checkpoint_path,
                self.model,
                self.feature_mean,
                self.feature_scale,
                seed=42,
                num_steps=2000,
                learning_rate=1e-3,
                metadata={
                    "dataset": "simple_diffuse_dataset_v0.npz",
                    "sample_count": 32,
                },
            )

            loaded = load_checkpoint(saved_path)

        with torch.inference_mode():
            actual_logits = loaded.model(normalized_features)

        torch.testing.assert_close(
            actual_logits,
            expected_logits,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            loaded.feature_mean,
            self.feature_mean,
        )
        torch.testing.assert_close(
            loaded.feature_scale,
            self.feature_scale,
        )

        self.assertFalse(loaded.model.training)
        self.assertEqual(loaded.seed, 42)
        self.assertEqual(loaded.num_steps, 2000)
        self.assertEqual(loaded.learning_rate, 1e-3)
        self.assertEqual(
            loaded.metadata["dataset"],
            "simple_diffuse_dataset_v0.npz",
        )
        self.assertEqual(loaded.metadata["sample_count"], 32)

    def test_save_creates_parent_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = (
                Path(temporary_directory)
                / "checkpoints"
                / "experiment"
                / "model.pt"
            )

            saved_path = save_checkpoint(
                checkpoint_path,
                self.model,
                self.feature_mean,
                self.feature_scale,
                seed=1,
                num_steps=10,
                learning_rate=1e-3,
            )

            self.assertEqual(saved_path, checkpoint_path)
            self.assertTrue(checkpoint_path.is_file())

    def test_rejects_invalid_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "model.pt"

            with self.assertRaises(ValueError):
                save_checkpoint(
                    checkpoint_path,
                    self.model,
                    torch.tensor([1.0, float("nan"), 3.0]),
                    self.feature_scale,
                    seed=1,
                    num_steps=10,
                    learning_rate=1e-3,
                )

            with self.assertRaises(ValueError):
                save_checkpoint(
                    checkpoint_path,
                    self.model,
                    self.feature_mean,
                    torch.tensor([1.0, 0.0, 1.0]),
                    seed=1,
                    num_steps=10,
                    learning_rate=1e-3,
                )

            with self.assertRaises(ValueError):
                save_checkpoint(
                    checkpoint_path,
                    self.model,
                    torch.ones((1, 3)),
                    self.feature_scale,
                    seed=1,
                    num_steps=10,
                    learning_rate=1e-3,
                )

    def test_rejects_invalid_training_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "model.pt"

            with self.assertRaises(ValueError):
                save_checkpoint(
                    checkpoint_path,
                    self.model,
                    self.feature_mean,
                    self.feature_scale,
                    seed=-1,
                    num_steps=10,
                    learning_rate=1e-3,
                )

            with self.assertRaises(ValueError):
                save_checkpoint(
                    checkpoint_path,
                    self.model,
                    self.feature_mean,
                    self.feature_scale,
                    seed=1,
                    num_steps=0,
                    learning_rate=1e-3,
                )

            with self.assertRaises(ValueError):
                save_checkpoint(
                    checkpoint_path,
                    self.model,
                    self.feature_mean,
                    self.feature_scale,
                    seed=1,
                    num_steps=10,
                    learning_rate=float("inf"),
                )

    def test_rejects_invalid_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "model.pt"

            with self.assertRaises(TypeError):
                save_checkpoint(
                    checkpoint_path,
                    self.model,
                    self.feature_mean,
                    self.feature_scale,
                    seed=1,
                    num_steps=10,
                    learning_rate=1e-3,
                    metadata=cast(Any, {"values": [1, 2, 3]}),
                )

            with self.assertRaises(TypeError):
                save_checkpoint(
                    checkpoint_path,
                    self.model,
                    self.feature_mean,
                    self.feature_scale,
                    seed=1,
                    num_steps=10,
                    learning_rate=1e-3,
                    metadata=cast(Any, {1: "invalid"}),
                )

    def test_missing_checkpoint_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            missing_path = Path(temporary_directory) / "missing.pt"

            with self.assertRaises(FileNotFoundError):
                load_checkpoint(missing_path)

    def test_unsupported_format_version_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "model.pt"

            torch.save(
                {
                    "format_version": 999,
                },
                checkpoint_path,
            )

            with self.assertRaises(ValueError):
                load_checkpoint(checkpoint_path)


if __name__ == "__main__":
    unittest.main()