"""Fast unit tests for Phase 3.2.3 validation diagnostics."""
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from scripts.evaluate_t2_validation import (
    identify_material, row_metrics, summarize, validate_checkpoint,
)


class TestMetrics(unittest.TestCase):
    def test_identical_distributions(self):
        p = np.array([[0.1, 0.9], [0.0, 1.0]])
        m = row_metrics(p, np.array([[0.1, 0.9], [1e-12, 1 - 1e-12]]))
        self.assertLess(m["tvd"].max(), 1e-10)
        self.assertLess(m["jsd"].max(), 1e-10)
        self.assertLess(m["kl"].max(), 1e-10)

    def test_orthogonal_prediction_is_bad(self):
        m = row_metrics(np.array([[1., 0.]]), np.array([[0.01, 0.99]]))
        self.assertGreater(m["tvd"][0], 0.98)
        self.assertGreater(m["kl"][0], 4)

    def test_rejects_invalid_probabilities(self):
        with self.assertRaises(ValueError):
            row_metrics(np.array([[0.2, 0.2]]), np.array([[0.5, 0.5]]))
        with self.assertRaises(ValueError):
            row_metrics(np.array([[1., 0.]]), np.array([[0., 1.]]))

    def test_rejects_shape_mismatch(self):
        with self.assertRaises(ValueError):
            row_metrics(np.ones((2, 2)), np.ones((2, 3)))

    def test_group_summary(self):
        m = row_metrics(np.array([[1., 0.], [0., 1.]]), np.array([[0.9, 0.1], [0.1, 0.9]]))
        summary = summarize(m, np.array([True, False]))
        self.assertAlmostEqual(summary["tvd"], 0.1)

    def test_unknown_material_rejected(self):
        with self.assertRaises(ValueError):
            identify_material(np.array([0.1, 0.2, 0.3]))

    def test_known_material_tolerant(self):
        self.assertEqual(identify_material(np.array([0.250001, 0.38, 0.85])), "blue")


class TestCheckpointContract(unittest.TestCase):
    def setUp(self):
        self.data = SimpleNamespace(
            metadata={"n_mu": 4, "n_phi": 8}, dataset_sha256="datahash",
            splits_sha256="splithash", feature_dimension=14, num_bins=32,
            feature_mean=torch.zeros(14), feature_scale=torch.ones(14),
        )
        self.meta = {
            "feature_schema_version": 2, "material_feature_mode": "lambertian_reflectance_v1",
            "target_type": "product_integrand_v1", "n_mu": 4, "n_phi": 8,
            "dataset_sha256": "datahash", "splits_sha256": "splithash",
            "selection_metric": "validation_cross_entropy",
        }
        self.ckpt = SimpleNamespace(metadata=self.meta.copy(),
                                    model=SimpleNamespace(feature_dimension=14, num_bins=32),
                                    feature_mean=torch.zeros(14), feature_scale=torch.ones(14))

    def test_accepts_matching(self):
        validate_checkpoint(self.ckpt, self.data)

    def test_rejects_wrong_dataset(self):
        self.ckpt.metadata["dataset_sha256"] = "other"
        with self.assertRaisesRegex(ValueError, "dataset_sha256"):
            validate_checkpoint(self.ckpt, self.data)

    def test_rejects_wrong_normalizer(self):
        self.ckpt.feature_mean[0] = 1.0
        with self.assertRaisesRegex(ValueError, "feature_mean"):
            validate_checkpoint(self.ckpt, self.data)


if __name__ == "__main__":
    unittest.main()
