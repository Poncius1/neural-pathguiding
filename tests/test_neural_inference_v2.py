"""Tests of the material-aware Mitsuba inference bridge."""
import tempfile
import unittest
from pathlib import Path

import mitsuba as mi
import numpy as np
import torch

from neural_path_guiding.core.bins import HemisphereBins
from neural_path_guiding.renderers.mitsuba.neural_inference_v2 import NeuralMitsubaRuntimeV2
from neural_path_guiding.training.checkpoint import load_checkpoint, save_checkpoint
from neural_path_guiding.training.model import GuidingMLP

mi.set_variant("scalar_rgb")


class TestNeuralInferenceV2(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "model.pt"
        model = GuidingMLP(feature_dimension=14, num_bins=32, hidden_dimension=16)
        save_checkpoint(
            self.path, model, torch.zeros(14), torch.ones(14),
            seed=42, num_steps=1, learning_rate=0.001,
            metadata={"n_mu": 4, "n_phi": 8,
                      "feature_schema_version": 2,
                      "target_type": "product_integrand_v1"},
        )
        self.scene = getattr(mi, "load_dict")({
            "type": "scene",
            "sphere": {"type": "sphere", "bsdf": {
                "type": "diffuse", "reflectance": {
                    "type": "rgb", "value": [0.25, 0.38, 0.85]}}},
        })
        ray = mi.Ray3f(mi.Point3f(0, 0, 3), mi.Vector3f(0, 0, -1))
        self.interaction = self.scene.ray_intersect(ray)
        self.ray = ray

    def tearDown(self):
        self.directory.cleanup()

    def test_loads_schema_v2(self):
        runtime = NeuralMitsubaRuntimeV2.from_checkpoint(self.path)
        self.assertEqual(runtime.bins, HemisphereBins(4, 8))

    def test_distribution_normalized(self):
        runtime = NeuralMitsubaRuntimeV2.from_checkpoint(self.path)
        dist = runtime.distribution_from_camera_ray(self.interaction, self.ray)
        self.assertEqual(dist.probabilities.shape, (32,))
        self.assertAlmostEqual(float(np.sum(dist.probabilities)), 1.0, places=10)
        self.assertTrue(np.all(dist.probabilities > 0.0))

    def test_sampling_and_pdf_world_consistent(self):
        runtime = NeuralMitsubaRuntimeV2.from_checkpoint(self.path)
        dist = runtime.distribution_from_camera_ray(self.interaction, self.ray)
        rng = np.random.default_rng(42)
        for _ in range(100):
            sample = runtime.sample_world(self.interaction, dist, rng)
            self.assertGreater(sample.local[2], 0.0)
            self.assertAlmostEqual(float(np.linalg.norm(sample.world)), 1.0, places=6)
            self.assertAlmostEqual(
                sample.pdf_solid_angle,
                runtime.pdf_world(self.interaction, dist, sample.world),
                places=8,
            )

    def test_rejects_wrong_schema(self):
        checkpoint = load_checkpoint(self.path)
        checkpoint.metadata["feature_schema_version"] = 1
        with self.assertRaises(ValueError):
            NeuralMitsubaRuntimeV2(checkpoint, HemisphereBins(4, 8))

    def test_rejects_ablated_checkpoint(self):
        checkpoint = load_checkpoint(self.path)
        checkpoint.metadata["feature_ablation"] = "no_albedo"
        with self.assertRaises(ValueError):
            NeuralMitsubaRuntimeV2(checkpoint, HemisphereBins(4, 8))

    def test_rejects_other_bounces(self):
        runtime = NeuralMitsubaRuntimeV2.from_checkpoint(self.path)
        with self.assertRaises(ValueError):
            runtime.distribution_at(self.interaction,
                outgoing_world=np.array([0, 0, 1.0]), bounce_depth=1)

    def test_rejects_zero_uniform_support(self):
        with self.assertRaises(ValueError):
            NeuralMitsubaRuntimeV2.from_checkpoint(self.path, uniform_mix=0.0)


if __name__ == "__main__":
    unittest.main()
