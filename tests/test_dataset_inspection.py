"""Tests for dataset inspection."""

from __future__ import annotations

import unittest

import numpy as np

from neural_path_guiding.evaluation.dataset_inspection import (
    compute_target_entropy,
    inspect_dataset_arrays,
)


class TestDatasetInspection(unittest.TestCase):
    def test_valid_dataset_is_inspected(self) -> None:
        features = np.ones((3, 14), dtype=np.float32)
        targets = np.array(
            [
                [0.25, 0.25, 0.25, 0.25],
                [0.70, 0.10, 0.10, 0.10],
                [0.10, 0.70, 0.10, 0.10],
            ],
            dtype=np.float32,
        )

        report = inspect_dataset_arrays(
            features=features,
            targets=targets,
            metadata={"experiment_name": "test"},
            top_k_bins=2,
        )

        self.assertEqual(report["num_samples"], 3)
        self.assertEqual(report["feature_dimension"], 14)
        self.assertEqual(report["num_bins"], 4)
        self.assertEqual(len(report["top_average_bins"]), 2)
        self.assertEqual(report["warnings"], [])

    def test_concentrated_target_has_lower_entropy(self) -> None:
        uniform = np.array([[0.25, 0.25, 0.25, 0.25]], dtype=np.float32)
        concentrated = np.array([[0.97, 0.01, 0.01, 0.01]], dtype=np.float32)

        uniform_entropy = compute_target_entropy(uniform)
        concentrated_entropy = compute_target_entropy(concentrated)

        self.assertGreater(float(uniform_entropy[0]), float(concentrated_entropy[0]))

    def test_invalid_feature_shape_is_rejected(self) -> None:
        features = np.ones((14,), dtype=np.float32)
        targets = np.ones((1, 4), dtype=np.float32)

        with self.assertRaises(ValueError):
            inspect_dataset_arrays(features=features, targets=targets)

    def test_negative_targets_are_rejected(self) -> None:
        features = np.ones((1, 14), dtype=np.float32)
        targets = np.array([[1.0, -0.1]], dtype=np.float32)

        with self.assertRaises(ValueError):
            inspect_dataset_arrays(features=features, targets=targets)

    def test_non_normalized_targets_create_warning(self) -> None:
        features = np.ones((1, 14), dtype=np.float32)
        targets = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)

        report = inspect_dataset_arrays(features=features, targets=targets)

        self.assertEqual(len(report["warnings"]), 1)


if __name__ == "__main__":
    unittest.main()