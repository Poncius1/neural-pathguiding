"""Tests for Mitsuba BSDF PDF evaluation."""

from __future__ import annotations

import importlib
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np


try:
    import mitsuba  # noqa: F401
except ModuleNotFoundError:
    mitsuba_stub = types.ModuleType("mitsuba")
    sys.modules["mitsuba"] = mitsuba_stub
    try:
        bsdf_module = importlib.import_module(
            "neural_path_guiding.renderers.mitsuba.bsdf"
        )
    finally:
        sys.modules.pop("mitsuba", None)
else:
    bsdf_module = importlib.import_module(
        "neural_path_guiding.renderers.mitsuba.bsdf"
    )


class FakeMitsuba:
    class BSDFContext:
        pass

    @staticmethod
    def Vector3f(
        x: float,
        y: float,
        z: float,
    ) -> np.ndarray:
        return np.array(
            [x, y, z],
            dtype=np.float64,
        )


class FakeBSDF:
    def __init__(self, pdf_value: float) -> None:
        self.pdf_value = pdf_value
        self.received_context = None
        self.received_interaction = None
        self.received_direction = None

    def pdf(
        self,
        context: object,
        surface_interaction: object,
        local_direction: np.ndarray,
    ) -> float:
        self.received_context = context
        self.received_interaction = surface_interaction
        self.received_direction = local_direction
        return self.pdf_value


class FakeSurfaceInteraction:
    def __init__(self, bsdf: FakeBSDF | None) -> None:
        self._bsdf = bsdf
        self.received_world_direction = None

    def to_local(
        self,
        world_direction: np.ndarray,
    ) -> np.ndarray:
        self.received_world_direction = world_direction
        return world_direction

    def bsdf(self) -> FakeBSDF | None:
        return self._bsdf


class TestEvaluateBSDFPdf(unittest.TestCase):
    def test_evaluates_pdf_for_normalized_world_direction(
        self,
    ) -> None:
        expected_pdf = 0.25
        fake_bsdf = FakeBSDF(expected_pdf)
        interaction = FakeSurfaceInteraction(fake_bsdf)

        with patch.object(
            bsdf_module,
            "_mi",
            FakeMitsuba,
        ):
            actual_pdf = bsdf_module.evaluate_bsdf_pdf(
                surface_interaction=interaction,
                world_direction=np.array(
                    [0.0, 0.0, 2.0],
                    dtype=np.float64,
                ),
            )

        world_direction = interaction.received_world_direction
        local_direction = fake_bsdf.received_direction

        if world_direction is None:
            self.fail("Mitsuba direction was not recorded.")

        if local_direction is None:
            self.fail("BSDF direction was not recorded.")

        np.testing.assert_allclose(
            world_direction,
            [0.0, 0.0, 1.0],
        )
        np.testing.assert_allclose(
            local_direction,
            [0.0, 0.0, 1.0],
        )
        self.assertIs(
                    fake_bsdf.received_interaction,
                    interaction,
                )

    def test_zero_pdf_is_valid(
        self,
    ) -> None:
        interaction = FakeSurfaceInteraction(
            FakeBSDF(0.0)
        )

        with patch.object(
            bsdf_module,
            "_mi",
            FakeMitsuba,
        ):
            pdf = bsdf_module.evaluate_bsdf_pdf(
                surface_interaction=interaction,
                world_direction=np.array(
                    [1.0, 0.0, 1.0],
                    dtype=np.float64,
                ),
            )

        self.assertEqual(pdf, 0.0)

    def test_rejects_negative_pdf(
        self,
    ) -> None:
        interaction = FakeSurfaceInteraction(
            FakeBSDF(-0.1)
        )

        with patch.object(
            bsdf_module,
            "_mi",
            FakeMitsuba,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "finite and non-negative",
            ):
                bsdf_module.evaluate_bsdf_pdf(
                    surface_interaction=interaction,
                    world_direction=np.array(
                        [0.0, 0.0, 1.0],
                        dtype=np.float64,
                    ),
                )

    def test_rejects_non_finite_pdf(
        self,
    ) -> None:
        interaction = FakeSurfaceInteraction(
            FakeBSDF(float("nan"))
        )

        with patch.object(
            bsdf_module,
            "_mi",
            FakeMitsuba,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "finite and non-negative",
            ):
                bsdf_module.evaluate_bsdf_pdf(
                    surface_interaction=interaction,
                    world_direction=np.array(
                        [0.0, 0.0, 1.0],
                        dtype=np.float64,
                    ),
                )

    def test_rejects_surface_without_bsdf(
        self,
    ) -> None:
        interaction = FakeSurfaceInteraction(None)

        with patch.object(
            bsdf_module,
            "_mi",
            FakeMitsuba,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "does not contain a BSDF",
            ):
                bsdf_module.evaluate_bsdf_pdf(
                    surface_interaction=interaction,
                    world_direction=np.array(
                        [0.0, 0.0, 1.0],
                        dtype=np.float64,
                    ),
                )


if __name__ == "__main__":
    unittest.main()