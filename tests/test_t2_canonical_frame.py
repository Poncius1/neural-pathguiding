
"""Regression tests for the canonical T2 directional frame."""

import unittest

import mitsuba as mi
import numpy as np

from neural_path_guiding.core.bins import HemisphereBins
from neural_path_guiding.core.directional_distribution import (
    DiscreteDirectionalDistribution,
)
from neural_path_guiding.core.frames import (
    build_frame_from_normal,
    local_to_world,
)
from neural_path_guiding.renderers.mitsuba.adapters import (
    mitsuba_vector_to_numpy,
)
from neural_path_guiding.renderers.mitsuba.neural_inference_v2 import (
    NeuralMitsubaRuntimeV2,
)


mi.set_variant("scalar_rgb")


class FakeInteraction:
    def __init__(self, normal):
        self.n = mi.Vector3f(*normal)


class TestT2CanonicalFrame(unittest.TestCase):
    def setUp(self):
        self.bins = HemisphereBins(4, 8)

        # Deliberately nonuniform: a frame rotation must not be
        # hidden by the rotational symmetry of a uniform PDF.
        probabilities = np.arange(
            1, 33, dtype=np.float64
        )
        probabilities /= probabilities.sum()

        self.distribution = DiscreteDirectionalDistribution(
            bins=self.bins,
            probabilities=probabilities,
            uniform_mix=0.05,
        )

    def test_sampling_matches_teacher_frame(self):
        normals = (
            (0.0, 0.0, 1.0),
            (0.0, 1.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.3, 0.5, 0.8),
        )

        for normal in normals:
            with self.subTest(normal=normal):
                interaction = FakeInteraction(normal)
                rng = np.random.default_rng(42)

                sample = NeuralMitsubaRuntimeV2.sample_world(
                    interaction,
                    self.distribution,
                    rng,
                )

                canonical = build_frame_from_normal(
                    mitsuba_vector_to_numpy(interaction.n)
                )

                expected_world = local_to_world(
                    canonical,
                    sample.local,
                )

                np.testing.assert_allclose(
                    sample.world,
                    expected_world,
                    rtol=0.0,
                    atol=1e-12,
                )

    def test_sample_pdf_matches_world_pdf(self):
        interaction = FakeInteraction(
            (0.3, 0.5, 0.8)
        )

        rng = np.random.default_rng(123)

        for _ in range(100):
            sample = NeuralMitsubaRuntimeV2.sample_world(
                interaction,
                self.distribution,
                rng,
            )

            evaluated = NeuralMitsubaRuntimeV2.pdf_world(
                interaction,
                self.distribution,
                sample.world,
            )

            self.assertAlmostEqual(
                sample.pdf_solid_angle,
                evaluated,
                places=10,
            )

    def test_pdf_is_zero_below_hemisphere(self):
        normal = np.array(
            [0.0, 1.0, 0.0],
            dtype=np.float64,
        )
        interaction = FakeInteraction(normal)

        pdf = NeuralMitsubaRuntimeV2.pdf_world(
            interaction,
            self.distribution,
            -normal,
        )

        self.assertEqual(pdf, 0.0)

    def test_runtime_and_direct_proposal_agree(self):
        interaction = FakeInteraction(
            (0.3, 0.5, 0.8)
        )

        proposal = (
            NeuralMitsubaRuntimeV2.proposal_from_distribution(
                interaction,
                self.distribution,
            )
        )

        rng = np.random.default_rng(2026)

        for _ in range(50):
            sample = proposal.sample(rng)

            pdf = NeuralMitsubaRuntimeV2.pdf_world(
                interaction,
                self.distribution,
                sample.direction_world,
            )

            self.assertAlmostEqual(
                sample.pdf,
                pdf,
                places=10,
            )


if __name__ == "__main__":
    unittest.main()
