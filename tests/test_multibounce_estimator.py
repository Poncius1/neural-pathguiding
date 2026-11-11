
"""Smoke tests for fixed-point multibounce estimators."""

import unittest
from typing import Callable, cast

import mitsuba as mi
import numpy as np

from neural_path_guiding.core.bins import HemisphereBins
from neural_path_guiding.core.directional_distribution import (
    DiscreteDirectionalDistribution,
)
from neural_path_guiding.core.frames import build_frame_from_normal
from neural_path_guiding.renderers.mitsuba.adapters import (
    mitsuba_vector_to_numpy,
)
from neural_path_guiding.renderers.mitsuba.guiding_adapter import (
    GuidingProposal,
)
from neural_path_guiding.renderers.mitsuba.multibounce_estimator import (
    MultibounceEvaluator,
    estimate_bsdf,
    estimate_neural,
    estimate_neural_bsdf_mis,
)

mi.set_variant("scalar_rgb")


class TestMultibounceEstimator(unittest.TestCase):

    def setUp(self):
        load_dict = cast(Callable[..., mi.Scene], mi.load_dict)
        self.scene = load_dict({
            "type": "scene",
            "sphere": {
                "type": "sphere",
                "bsdf": {
                    "type": "diffuse",
                    "reflectance": {
                        "type": "rgb",
                        "value": [0.5, 0.5, 0.5],
                    },
                },
            },
            "environment": {
                "type": "constant",
                "radiance": {
                    "type": "rgb",
                    "value": [1.0, 1.0, 1.0],
                },
            },
        })

        self.ray = mi.Ray3f(
            mi.Point3f(0, 0, 3),
            mi.Vector3f(0, 0, -1),
        )
        self.interaction = self.scene.ray_intersect(self.ray)

        self.assertTrue(bool(self.interaction.is_valid()))

        bins = HemisphereBins(4, 8)

        distribution = DiscreteDirectionalDistribution(
            bins=bins,
            probabilities=np.full(32, 1 / 32, dtype=np.float64),
            uniform_mix=0.05,
        )

        self.proposal = GuidingProposal(
            frame=build_frame_from_normal(
                mitsuba_vector_to_numpy(self.interaction.n)
            ),
            distribution=distribution,
        )

        self.evaluator = MultibounceEvaluator.create(
            self.scene,
            radiance_samples=2,
            max_depth=4,
            rr_depth=3,
            seed=42,
        )

        self.rng = np.random.default_rng(42)

    def test_neural_returns_finite_rgb(self):
        value = estimate_neural(
            self.interaction,
            self.proposal,
            self.evaluator,
            self.rng,
            2,
        )
        self.assertEqual(value.shape, (3,))
        self.assertTrue(np.all(np.isfinite(value)))

    def test_bsdf_returns_finite_rgb(self):
        value = estimate_bsdf(
            self.interaction,
            self.evaluator,
            self.rng,
            2,
        )
        self.assertEqual(value.shape, (3,))
        self.assertTrue(np.all(np.isfinite(value)))

    def test_mis_returns_finite_rgb(self):
        value = estimate_neural_bsdf_mis(
            self.interaction,
            self.proposal,
            self.evaluator,
            self.rng,
        )
        self.assertEqual(value.shape, (3,))
        self.assertTrue(np.all(np.isfinite(value)))

    def test_rejects_zero_sample_count(self):
        with self.assertRaises(ValueError):
            estimate_neural(
                self.interaction,
                self.proposal,
                self.evaluator,
                self.rng,
                0,
            )

        with self.assertRaises(ValueError):
            estimate_bsdf(
                self.interaction,
                self.evaluator,
                self.rng,
                0,
            )

    def test_uniform_pdf_and_world_frame(self):
        sample = self.proposal.sample(self.rng)
        self.assertAlmostEqual(
            self.proposal.pdf(sample.direction_world),
            sample.pdf,
            places=7,
        )
        self.assertAlmostEqual(
            sample.pdf,
            1.0 / (2.0 * np.pi),
            places=7,
        )


if __name__ == "__main__":
    unittest.main()
