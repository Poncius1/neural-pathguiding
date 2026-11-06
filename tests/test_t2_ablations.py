
"""Unit tests for ablation masking and experiment bookkeeping."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from scripts.run_t2_ablations import (
    FEATURE_GROUPS,
    _write_summary,
    mask_features,
)


class TestFeatureAblations(unittest.TestCase):

    def setUp(self):
        self.features = (
            torch.arange(
                28,
                dtype=torch.float32,
            ).reshape(2, 14)
            + 1
        )

    def test_group_definitions(self):
        self.assertEqual(
            FEATURE_GROUPS["no_position"],
            (0, 1, 2),
        )

        self.assertEqual(
            FEATURE_GROUPS["no_outgoing"],
            (6, 7, 8),
        )

        self.assertEqual(
            FEATURE_GROUPS["no_albedo"],
            (10, 11, 12),
        )

    def test_full_does_not_change_features(self):
        result = mask_features(
            self.features,
            "full",
        )

        torch.testing.assert_close(
            result,
            self.features,
        )

        self.assertIsNot(
            result,
            self.features,
        )

    def test_masks_exact_columns(self):
        for name, columns in FEATURE_GROUPS.items():
            with self.subTest(name=name):

                result = mask_features(
                    self.features,
                    name,
                )

                expected = self.features.clone()

                if columns:
                    expected[:, list(columns)] = 0.0

                torch.testing.assert_close(
                    result,
                    expected,
                )

    def test_inputs_not_mutated(self):
        original = self.features.clone()

        mask_features(
            self.features,
            "no_albedo",
        )

        torch.testing.assert_close(
            self.features,
            original,
        )

    def test_invalid_ablation_rejected(self):
        with self.assertRaises(ValueError):
            mask_features(
                self.features,
                "no_normals",
            )

    def test_invalid_shape_rejected(self):
        with self.assertRaises(ValueError):
            mask_features(
                torch.zeros((2, 13)),
                "no_albedo",
            )

    def test_zero_is_train_mean_in_normalized_space(self):
        raw = torch.tensor([
            [4.0, 10.0],
            [6.0, 20.0],
            [8.0, 30.0],
        ])

        mean = raw.mean(dim=0)

        scale = raw.std(
            dim=0,
            unbiased=False,
        )

        standardized = (
            raw - mean
        ) / scale

        standardized[:, 0] = 0.0

        restored = (
            standardized * scale
        ) + mean

        torch.testing.assert_close(
            restored[:, 0],
            mean[0].expand(3),
        )

    def test_partial_summary_has_no_fake_std(self):
        with TemporaryDirectory() as folder:

            output_dir = Path(folder)

            results = [
                {
                    "ablation": "no_albedo",
                    "seed": 7,
                    "best_epoch": 10,
                    "ce": 3.2,
                    "kl": 0.2,
                    "tvd": 0.1,
                    "jsd": 0.04,
                    "dataset_sha256": "x",
                    "splits_sha256": "y",
                }
            ]

            _write_summary(
                output_dir,
                results,
            )

            report = json.loads(
                (
                    output_dir / "ablation_summary.json"
                ).read_text(
                    encoding="utf-8"
                )
            )

            self.assertIsNone(
                report["summary"][
                    "no_albedo"
                ]["ce"]["std_sample"]
            )

            self.assertFalse(
                report["test_evaluated"]
            )


if __name__ == "__main__":
    unittest.main()
