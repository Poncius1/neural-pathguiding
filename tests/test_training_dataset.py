"""Tests for PyTorch training subset preparation."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from neural_path_guiding.data.dataset import save_dataset
from neural_path_guiding.data.schema import (
    DATASET_FORMAT_VERSION,
    NeuralGuidingDataset,
)
from neural_path_guiding.training.dataset import load_training_subset


class TestTrainingDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.dataset_path = (
            Path(self.temporary_directory.name)
            / "training_dataset.npz"
        )

        save_dataset(
            dataset_path=self.dataset_path,
            dataset=self.make_dataset(),
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def make_dataset(self) -> NeuralGuidingDataset:
        num_samples = 8
        num_bins = 8

        features = np.arange(
            num_samples * 14,
            dtype=np.float32,
        ).reshape(num_samples, 14)

        # These columns imitate currently constant material/path features.
        features[:, 9] = 1.0
        features[:, 10:13] = 1.0
        features[:, 13] = 0.0

        base_target = np.arange(
            1,
            num_bins + 1,
            dtype=np.float32,
        )
        base_target /= base_target.sum()

        targets = np.tile(
            base_target,
            (num_samples, 1),
        ).astype(np.float32)

        normals = np.tile(
            np.array(
                [[0.0, 1.0, 0.0]],
                dtype=np.float32,
            ),
            (num_samples, 1),
        )

        pixels = np.column_stack(
            (
                np.arange(num_samples),
                np.arange(num_samples),
            )
        ).astype(np.int32)

        return NeuralGuidingDataset(
            features=features,
            targets=targets,
            mean_contributions=np.ones(
                (num_samples, num_bins),
                dtype=np.float32,
            ),
            positions=np.zeros(
                (num_samples, 3),
                dtype=np.float32,
            ),
            normals=normals,
            pixels=pixels,
            metadata={
                "dataset_format_version": DATASET_FORMAT_VERSION,
                "experiment_name": "training_dataset_test",
                "target_type": "visibility_cosine_v0",
                "feature_dimension": 14,
                "num_bins": num_bins,
                "num_shading_points": num_samples,
                "n_mu": 2,
                "n_phi": 4,
            },
        )

    def test_subset_has_expected_shapes_and_dtypes(self) -> None:
        subset = load_training_subset(
            dataset_path=self.dataset_path,
            sample_count=4,
            seed=42,
        )

        self.assertEqual(subset.features.shape, (4, 14))
        self.assertEqual(subset.targets.shape, (4, 8))
        self.assertEqual(subset.feature_mean.shape, (14,))
        self.assertEqual(subset.feature_scale.shape, (14,))
        self.assertEqual(subset.indices.shape, (4,))

        self.assertEqual(subset.features.dtype, torch.float32)
        self.assertEqual(subset.targets.dtype, torch.float32)

        self.assertEqual(subset.num_samples, 4)
        self.assertEqual(subset.feature_dimension, 14)
        self.assertEqual(subset.num_bins, 8)
        self.assertGreater(subset.n_mu, 0)
        self.assertGreater(subset.n_phi, 0)
        self.assertEqual(
            subset.n_mu * subset.n_phi,
            subset.num_bins,
        )
        self.assertTrue(subset.target_type)
        self.assertEqual(subset.dataset_format_version, 1)

        feature_means = subset.features.mean(dim=0)

        self.assertTrue(
            torch.allclose(
                feature_means,
                torch.zeros_like(feature_means),
                atol=1e-6,
                rtol=0.0,
            )
        )

        target_sums = subset.targets.sum(dim=1)

        self.assertTrue(
            torch.allclose(
                target_sums,
                torch.ones_like(target_sums),
                atol=1e-6,
                rtol=0.0,
            )
        )

        # Constant columns must become zero after normalization.
        self.assertTrue(
            torch.equal(
                subset.features[:, 9:],
                torch.zeros_like(subset.features[:, 9:]),
            )
        )

        # Constant columns use scale 1 instead of dividing by zero.
        self.assertTrue(
            torch.equal(
                subset.feature_scale[9:],
                torch.ones_like(subset.feature_scale[9:]),
            )
        )

    def test_same_seed_selects_the_same_subset(self) -> None:
        first = load_training_subset(
            dataset_path=self.dataset_path,
            sample_count=4,
            seed=123,
        )
        second = load_training_subset(
            dataset_path=self.dataset_path,
            sample_count=4,
            seed=123,
        )

        self.assertTrue(
            bool(np.array_equal(first.indices, second.indices))
        )
        self.assertTrue(
            torch.equal(first.features, second.features)
        )
        self.assertTrue(
            torch.equal(first.targets, second.targets)
        )

    def test_invalid_sample_counts_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            load_training_subset(
                dataset_path=self.dataset_path,
                sample_count=0,
                seed=42,
            )

        with self.assertRaises(ValueError):
            load_training_subset(
                dataset_path=self.dataset_path,
                sample_count=9,
                seed=42,
            )

    def test_negative_seed_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            load_training_subset(
                dataset_path=self.dataset_path,
                sample_count=4,
                seed=-1,
            )


if __name__ == "__main__":
    unittest.main()