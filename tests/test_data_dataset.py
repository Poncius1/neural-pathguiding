"""Tests for the canonical dataset schema, validation and NPZ persistence."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from neural_path_guiding.data.dataset import load_dataset, save_dataset
from neural_path_guiding.data.schema import (
    DATASET_FORMAT_VERSION,
    NeuralGuidingDataset,
)
from neural_path_guiding.data.validation import (
    validate_dataset,
    validate_training_arrays,
)
from neural_path_guiding.evaluation.dataset_inspection import inspect_dataset_file


class TestDataDataset(unittest.TestCase):
    def make_dataset(self) -> NeuralGuidingDataset:
        num_samples = 2
        num_bins = 8

        return NeuralGuidingDataset(
            features=np.ones((num_samples, 14), dtype=np.float32),
            targets=np.full(
                (num_samples, num_bins),
                1.0 / num_bins,
                dtype=np.float32,
            ),
            mean_contributions=np.ones(
                (num_samples, num_bins),
                dtype=np.float32,
            ),
            positions=np.zeros((num_samples, 3), dtype=np.float32),
            normals=np.array(
                [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            ),
            pixels=np.array([[0, 1], [2, 3]], dtype=np.int32),
            metadata={
                "dataset_format_version": DATASET_FORMAT_VERSION,
                "experiment_name": "test_dataset",
                "target_type": "visibility_cosine_v0",
                "feature_dimension": 14,
                "num_bins": num_bins,
                "num_shading_points": num_samples,
                "n_mu": 2,
                "n_phi": 4,
            },
        )

    def test_valid_dataset_round_trip(self) -> None:
        expected = self.make_dataset()

        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_path = Path(temporary_directory) / "dataset.npz"
            save_dataset(dataset_path, expected)
            actual = load_dataset(dataset_path)
            inspection = inspect_dataset_file(dataset_path)

        self.assertTrue(bool(np.array_equal(actual.features, expected.features)))
        self.assertTrue(bool(np.array_equal(actual.targets, expected.targets)))
        self.assertTrue(
            bool(
                np.array_equal(
                    actual.mean_contributions,
                    expected.mean_contributions,
                )
            )
        )
        self.assertEqual(actual.metadata, expected.metadata)
        self.assertEqual(actual.num_samples, 2)
        self.assertEqual(actual.feature_dimension, 14)
        self.assertEqual(actual.num_bins, 8)
        self.assertEqual(inspection["num_samples"], 2)
        self.assertEqual(inspection["feature_dimension"], 14)
        self.assertEqual(inspection["num_bins"], 8)

    def test_missing_required_array_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_path = Path(temporary_directory) / "incomplete.npz"
            np.savez_compressed(
                dataset_path,
                features=np.ones((1, 14), dtype=np.float32),
            )

            with self.assertRaisesRegex(ValueError, "missing required array"):
                load_dataset(dataset_path)

    def test_metadata_must_match_array_dimensions(self) -> None:
        dataset = self.make_dataset()
        invalid_metadata = dict(dataset.metadata)
        invalid_metadata["num_bins"] = 32

        with self.assertRaises(ValueError):
            validate_dataset(replace(dataset, metadata=invalid_metadata))

    def test_diagnostic_array_shape_is_validated(self) -> None:
        dataset = self.make_dataset()
        invalid = replace(
            dataset,
            mean_contributions=np.ones((2, 7), dtype=np.float32),
        )

        with self.assertRaises(ValueError):
            validate_dataset(invalid)

    def test_normals_must_have_unit_length(self) -> None:
        dataset = self.make_dataset()
        invalid = replace(
            dataset,
            normals=np.ones((2, 3), dtype=np.float32),
        )

        with self.assertRaises(ValueError):
            validate_dataset(invalid)

    def test_training_arrays_require_floating_point_dtype(self) -> None:
        features = np.ones((1, 14), dtype=np.int32)
        targets = np.ones((1, 4), dtype=np.float32) / 4.0

        with self.assertRaises(ValueError):
            validate_training_arrays(features, targets)

    def test_dataset_path_requires_npz_extension(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_path = Path(temporary_directory) / "dataset.npy"

            with self.assertRaises(ValueError):
                save_dataset(dataset_path, self.make_dataset())


if __name__ == "__main__":
    unittest.main()
