"""Tests for Mitsuba BSDF evaluation utilities."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from neural_path_guiding.renderers.mitsuba import bsdf
from neural_path_guiding.renderers.mitsuba.bsdf import (
    evaluate_bsdf_times_cosine,
    mitsuba_bsdf_to_rgb,
)


class FakeBSDF:
    def __init__(
        self,
        value: list[float],
    ) -> None:
        self.value = value
        self.received_context: object | None = None
        self.received_interaction: object | None = None
        self.received_direction: object | None = None

    def eval(
        self,
        context: object,
        surface_interaction: object,
        local_direction: object,
    ) -> list[float]:
        self.received_context = context
        self.received_interaction = surface_interaction
        self.received_direction = local_direction

        return self.value


class FakeSurfaceInteraction:
    def __init__(
        self,
        fake_bsdf: FakeBSDF | None,
        local_direction: np.ndarray,
    ) -> None:
        self.fake_bsdf = fake_bsdf
        self.local_direction = local_direction
        self.received_world_direction: object | None = None

    def to_local(
        self,
        world_direction: object,
    ) -> np.ndarray:
        self.received_world_direction = world_direction

        return self.local_direction

    def bsdf(self) -> FakeBSDF | None:
        return self.fake_bsdf


class TestBsdf(unittest.TestCase):
    def test_evaluates_bsdf_in_local_coordinates(
        self,
    ) -> None:
        fake_bsdf = FakeBSDF(
            [0.1, 0.2, 0.3]
        )

        local_direction = np.array(
            [0.0, 0.0, 1.0],
            dtype=np.float64,
        )

        surface_interaction = FakeSurfaceInteraction(
            fake_bsdf=fake_bsdf,
            local_direction=local_direction,
        )

        fake_context = object()

        fake_mitsuba = SimpleNamespace(
            Vector3f=lambda *values: np.array(
                values,
                dtype=np.float64,
            ),
            BSDFContext=lambda: fake_context,
        )

        with patch.object(
            bsdf,
            "_mi",
            fake_mitsuba,
        ):
            result = evaluate_bsdf_times_cosine(
                surface_interaction=surface_interaction,
                world_direction=np.array(
                    [0.0, 0.0, 2.0],
                    dtype=np.float64,
                ),
            )

        np.testing.assert_allclose(
            result,
            np.array([0.1, 0.2, 0.3]),
        )

        received_world_direction = (
            surface_interaction.received_world_direction
        )

        self.assertIsNotNone(
            received_world_direction
        )
        assert received_world_direction is not None

        np.testing.assert_allclose(
            np.asarray(
                received_world_direction,
                dtype=np.float64,
            ),
            np.array([0.0, 0.0, 1.0]),
        )

        self.assertIs(
            fake_bsdf.received_context,
            fake_context,
        )
        self.assertIs(
            fake_bsdf.received_interaction,
            surface_interaction,
        )
        self.assertIs(
            fake_bsdf.received_direction,
            local_direction,
        )

    def test_does_not_multiply_cosine_twice(
        self,
    ) -> None:
        fake_bsdf = FakeBSDF(
            [0.4, 0.5, 0.6]
        )

        surface_interaction = FakeSurfaceInteraction(
            fake_bsdf=fake_bsdf,
            local_direction=np.array(
                [0.0, 0.0, 0.25],
                dtype=np.float64,
            ),
        )

        fake_mitsuba = SimpleNamespace(
            Vector3f=lambda *values: values,
            BSDFContext=lambda: object(),
        )

        with patch.object(
            bsdf,
            "_mi",
            fake_mitsuba,
        ):
            result = evaluate_bsdf_times_cosine(
                surface_interaction=surface_interaction,
                world_direction=np.array(
                    [0.0, 0.0, 1.0],
                    dtype=np.float64,
                ),
            )

        # The helper returns exactly what BSDF.eval() produced.
        # It must not multiply by local_direction.z again.
        np.testing.assert_allclose(
            result,
            np.array([0.4, 0.5, 0.6]),
        )

    def test_rejects_missing_bsdf(self) -> None:
        surface_interaction = FakeSurfaceInteraction(
            fake_bsdf=None,
            local_direction=np.array(
                [0.0, 0.0, 1.0],
                dtype=np.float64,
            ),
        )

        fake_mitsuba = SimpleNamespace(
            Vector3f=lambda *values: values,
        )

        with (
            patch.object(
                bsdf,
                "_mi",
                fake_mitsuba,
            ),
            self.assertRaisesRegex(
                ValueError,
                "does not contain a BSDF",
            ),
        ):
            evaluate_bsdf_times_cosine(
                surface_interaction=surface_interaction,
                world_direction=np.array(
                    [0.0, 0.0, 1.0],
                    dtype=np.float64,
                ),
            )

    def test_rejects_invalid_bsdf_values(
        self,
    ) -> None:
        invalid_values = (
            [np.nan, 0.0, 0.0],
            [np.inf, 0.0, 0.0],
            [-0.1, 0.0, 0.0],
        )

        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    mitsuba_bsdf_to_rgb(value)


if __name__ == "__main__":
    unittest.main()