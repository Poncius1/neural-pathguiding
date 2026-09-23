"""Tests for the Mitsuba neural-guiding adapter."""

from __future__ import annotations

from typing import cast
import unittest

import numpy as np

from neural_path_guiding.core.directional_distribution import (
    DiscreteDirectionalDistribution,
)
from neural_path_guiding.core.features import FEATURE_DIMENSION
from neural_path_guiding.core.bins import HemisphereBins
from neural_path_guiding.renderers.mitsuba.guiding_adapter import (
    MitsubaGuidingAdapter,
)
from neural_path_guiding.training.inference import (
    GuidingInferenceRuntime,
)


class FakeSurfaceInteraction:
    """Minimal stand-in for the Mitsuba interaction fields we consume."""

    def __init__(
        self,
        position: tuple[float, float, float] = (1.0, 2.0, 3.0),
        normal: tuple[float, float, float] = (0.0, 0.0, 1.0),
    ) -> None:
        self.p = position
        self.n = normal


class FakeInferenceRuntime:
    """Small runtime substitute that records the features it receives."""

    def __init__(self) -> None:
        self.bins = HemisphereBins(
            n_mu=2,
            n_phi=4,
        )
        self.last_features: np.ndarray | None = None

    def predict_distribution(
        self,
        features: np.ndarray,
    ) -> DiscreteDirectionalDistribution:
        self.last_features = np.array(
            features,
            dtype=np.float64,
            copy=True,
        )

        probabilities = np.arange(
            1,
            self.bins.n_bins + 1,
            dtype=np.float64,
        )
        probabilities /= np.sum(probabilities)

        return DiscreteDirectionalDistribution(
            bins=self.bins,
            probabilities=probabilities,
        )


class TestMitsubaGuidingAdapter(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime = FakeInferenceRuntime()
        self.adapter = MitsubaGuidingAdapter(
            runtime=cast(
                GuidingInferenceRuntime,
                self.runtime,
            )
        )
        self.interaction = FakeSurfaceInteraction()

    def make_proposal(
        self,
        *,
        normal: tuple[float, float, float] = (
            0.0,
            0.0,
            1.0,
        ),
    ):
        interaction = FakeSurfaceInteraction(
            normal=normal
        )

        return self.adapter.prepare_proposal(
            surface_interaction=interaction,
            outgoing_direction=np.array(
                [0.0, 0.0, 2.0],
                dtype=np.float64,
            ),
            bounce_depth=1,
        )

    def test_extracts_model_features_from_interaction(
        self,
    ) -> None:
        self.make_proposal()

        self.assertIsNotNone(
            self.runtime.last_features
        )
        assert self.runtime.last_features is not None

        self.assertEqual(
            self.runtime.last_features.shape,
            (FEATURE_DIMENSION,),
        )
        np.testing.assert_allclose(
            self.runtime.last_features[:3],
            [1.0, 2.0, 3.0],
        )
        np.testing.assert_allclose(
            self.runtime.last_features[3:6],
            [0.0, 0.0, 1.0],
        )
        np.testing.assert_allclose(
            self.runtime.last_features[6:9],
            [0.0, 0.0, 1.0],
        )
        self.assertEqual(
            self.runtime.last_features[9],
            1.0,
        )
        np.testing.assert_allclose(
            self.runtime.last_features[10:13],
            [1.0, 1.0, 1.0],
        )
        self.assertEqual(
            self.runtime.last_features[13],
            1.0,
        )

    def test_sample_returns_unit_world_direction(
        self,
    ) -> None:
        proposal = self.make_proposal()
        sample = proposal.sample(
            np.random.default_rng(42)
        )

        self.assertAlmostEqual(
            float(np.linalg.norm(sample.direction_world)),
            1.0,
            places=12,
        )
        self.assertGreaterEqual(
            sample.direction_local[2],
            0.0,
        )

    def test_sample_pdf_matches_world_pdf_evaluation(
        self,
    ) -> None:
        proposal = self.make_proposal()
        sample = proposal.sample(
            np.random.default_rng(7)
        )

        evaluated_pdf = proposal.pdf(
            sample.direction_world
        )

        self.assertAlmostEqual(
            sample.pdf,
            evaluated_pdf,
            places=12,
        )

    def test_world_transform_works_for_rotated_normal(
        self,
    ) -> None:
        proposal = self.make_proposal(
            normal=(0.0, 1.0, 0.0)
        )
        sample = proposal.sample(
            np.random.default_rng(13)
        )

        self.assertAlmostEqual(
            float(np.linalg.norm(sample.direction_world)),
            1.0,
            places=12,
        )
        self.assertGreaterEqual(
            float(sample.direction_world[1]),
            -1e-12,
        )
        self.assertAlmostEqual(
            proposal.pdf(sample.direction_world),
            sample.pdf,
            places=12,
        )

    def test_pdf_is_zero_below_guiding_hemisphere(
        self,
    ) -> None:
        proposal = self.make_proposal()

        pdf = proposal.pdf(
            np.array(
                [0.0, 0.0, -1.0],
                dtype=np.float64,
            )
        )

        self.assertEqual(pdf, 0.0)

    def test_rejects_invalid_outgoing_direction(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            self.adapter.prepare_proposal(
                surface_interaction=self.interaction,
                outgoing_direction=np.zeros(3),
                bounce_depth=0,
            )


if __name__ == "__main__":
    unittest.main()