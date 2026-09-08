"""Tests for checkpoint-based guiding model inference."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from neural_path_guiding.core.features import FEATURE_SCHEMA_VERSION
from neural_path_guiding.training.checkpoint import (
    MetadataValue,
    save_checkpoint,
)
from neural_path_guiding.training.inference import (
    GuidingInferenceRuntime,
)
from neural_path_guiding.training.model import GuidingMLP


class TestGuidingInferenceRuntime(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(
            self.temporary_directory.cleanup
        )

        self.root = Path(
            self.temporary_directory.name
        )
        self.checkpoint_path = (
            self.root / "guiding_model.pt"
        )

        torch.manual_seed(17)

        self.model = GuidingMLP(
            feature_dimension=14,
            hidden_dimension=16,
            num_bins=8,
        )

        self.feature_mean = torch.linspace(
            -1.0,
            1.0,
            steps=14,
        )
        self.feature_scale = torch.linspace(
            0.5,
            1.8,
            steps=14,
        )

        self.features = np.linspace(
            -2.0,
            2.0,
            num=14,
            dtype=np.float64,
        )

        self.metadata: dict[str, MetadataValue] = {
            "experiment": "inference_test",
            "n_mu": 2,
            "n_phi": 4,
            "target_type": "product_integrand_v1",
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
        }

        self._save_checkpoint(
            self.checkpoint_path,
            self.metadata,
        )

    def _save_checkpoint(
        self,
        path: Path,
        metadata: dict[str, MetadataValue],
    ) -> None:
        save_checkpoint(
            path,
            self.model,
            self.feature_mean,
            self.feature_scale,
            seed=17,
            num_steps=10,
            learning_rate=1e-3,
            metadata=metadata,
        )

    def test_loaded_runtime_reproduces_model_logits(
        self,
    ) -> None:
        runtime = GuidingInferenceRuntime.from_checkpoint(
            self.checkpoint_path
        )

        feature_tensor = torch.as_tensor(
            self.features,
            dtype=self.feature_mean.dtype,
        )

        normalized_features = (
            feature_tensor - self.feature_mean
        ) / self.feature_scale

        self.model.eval()

        with torch.inference_mode():
            expected_logits = self.model(
                normalized_features.unsqueeze(0)
            )[0].numpy()

        actual_logits = runtime.predict_logits(
            self.features
        )

        np.testing.assert_allclose(
            actual_logits,
            expected_logits,
            rtol=1e-6,
            atol=1e-7,
        )

    def test_predicts_normalized_distribution(
        self,
    ) -> None:
        runtime = GuidingInferenceRuntime.from_checkpoint(
            self.checkpoint_path,
            uniform_mix=0.1,
        )

        distribution = runtime.predict_distribution(
            self.features
        )

        self.assertEqual(
            distribution.bins.n_bins,
            8,
        )
        self.assertEqual(
            distribution.probabilities.shape,
            (8,),
        )

        self.assertAlmostEqual(
            float(
                np.sum(
                    distribution.probabilities
                )
            ),
            1.0,
            places=12,
        )

        self.assertTrue(
            bool(
                np.all(
                    distribution.probabilities > 0.0
                )
            )
        )

    def test_exposes_checkpoint_information(
        self,
    ) -> None:
        runtime = GuidingInferenceRuntime.from_checkpoint(
            self.checkpoint_path
        )

        self.assertEqual(
            runtime.feature_dimension,
            14,
        )
        self.assertEqual(
            runtime.num_bins,
            8,
        )
        self.assertEqual(
            runtime.bins.n_mu,
            2,
        )
        self.assertEqual(
            runtime.bins.n_phi,
            4,
        )
        self.assertEqual(
            runtime.target_type,
            "product_integrand_v1",
        )
        self.assertEqual(
            runtime.device.type,
            "cpu",
        )

    def test_rejects_invalid_feature_vectors(
        self,
    ) -> None:
        runtime = GuidingInferenceRuntime.from_checkpoint(
            self.checkpoint_path
        )

        invalid_vectors = (
            np.zeros(13),
            np.zeros((1, 14)),
            np.full(14, np.nan),
            np.full(14, np.inf),
        )

        for features in invalid_vectors:
            with self.subTest(
                shape=features.shape
            ):
                with self.assertRaises(ValueError):
                    runtime.predict_logits(features)

    def test_rejects_incompatible_bin_metadata(
        self,
    ) -> None:
        invalid_path = (
            self.root / "invalid_bins.pt"
        )

        invalid_metadata = dict(
            self.metadata
        )
        invalid_metadata["n_phi"] = 3

        self._save_checkpoint(
            invalid_path,
            invalid_metadata,
        )

        with self.assertRaisesRegex(
            ValueError,
            "does not match",
        ):
            GuidingInferenceRuntime.from_checkpoint(
                invalid_path
            )

    def test_rejects_incompatible_feature_schema(
        self,
    ) -> None:
        invalid_path = (
            self.root / "invalid_schema.pt"
        )

        invalid_metadata = dict(
            self.metadata
        )
        invalid_metadata[
            "feature_schema_version"
        ] = FEATURE_SCHEMA_VERSION + 1

        self._save_checkpoint(
            invalid_path,
            invalid_metadata,
        )

        with self.assertRaisesRegex(
            ValueError,
            "feature schema version",
        ):
            GuidingInferenceRuntime.from_checkpoint(
                invalid_path
            )

    def test_inference_does_not_create_gradients(
        self,
    ) -> None:
        runtime = GuidingInferenceRuntime.from_checkpoint(
            self.checkpoint_path
        )

        runtime.predict_logits(
            self.features
        )

        self.assertFalse(
            runtime.checkpoint.model.training
        )

        for parameter in (
            runtime.checkpoint.model.parameters()
        ):
            self.assertIsNone(
                parameter.grad
            )


if __name__ == "__main__":
    unittest.main()