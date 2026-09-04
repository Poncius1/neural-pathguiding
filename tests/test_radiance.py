"""Tests for incident-radiance estimation utilities."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

import numpy as np

from neural_path_guiding.renderers.mitsuba import radiance
from neural_path_guiding.renderers.mitsuba.radiance import (
    IncidentRadianceRuntime,
    create_incident_radiance_runtime,
    estimate_incident_radiance,
    mitsuba_spectrum_to_rgb,
)


class FakeIntegrator:
    def __init__(
        self,
        radiance_values: list[list[float]],
    ) -> None:
        self._radiance_values = iter(radiance_values)

    def sample(
        self,
        scene: object,
        sampler: object,
        ray: object,
    ) -> tuple[np.ndarray, bool, list]:
        value = next(self._radiance_values)

        return (
            np.asarray(value, dtype=np.float64),
            True,
            [],
        )


class FakeSampler:
    def __init__(self) -> None:
        self.seed_calls: list[tuple[int, int]] = []
        self.schedule_count = 0

    def seed(
        self,
        seed: int,
        wavefront_size: int,
    ) -> None:
        self.seed_calls.append(
            (seed, wavefront_size)
        )

    def schedule_state(self) -> None:
        self.schedule_count += 1


class TestRadiance(unittest.TestCase):
    def test_creates_integrator_and_sampler(self) -> None:
        fake_integrator = object()
        fake_sampler = FakeSampler()

        fake_mitsuba = SimpleNamespace(
            load_dict=Mock(
                side_effect=[
                    fake_integrator,
                    fake_sampler,
                ]
            )
        )

        with patch.object(
            radiance,
            "_mi",
            fake_mitsuba,
        ):
            runtime = create_incident_radiance_runtime(
                radiance_samples=4,
                max_depth=6,
                rr_depth=3,
                seed=42,
            )

        self.assertIs(
            runtime.integrator,
            fake_integrator,
        )
        self.assertIs(
            runtime.sampler,
            fake_sampler,
        )
        self.assertEqual(
            runtime.radiance_samples,
            4,
        )
        self.assertEqual(
            fake_sampler.seed_calls,
            [(42, 1)],
        )

        self.assertEqual(
            fake_mitsuba.load_dict.call_args_list,
            [
                call(
                    {
                        "type": "path",
                        "max_depth": 6,
                        "rr_depth": 3,
                    }
                ),
                call(
                    {
                        "type": "independent",
                        "sample_count": 4,
                    }
                ),
            ],
        )

    def test_averages_incident_radiance_samples(self) -> None:
        sampler = FakeSampler()

        runtime = IncidentRadianceRuntime(
            integrator=FakeIntegrator(
                [
                    [1.0, 2.0, 3.0],
                    [3.0, 4.0, 5.0],
                ]
            ),
            sampler=sampler,
            radiance_samples=2,
        )

        result = estimate_incident_radiance(
            runtime=runtime,
            scene=object(),
            ray=object(),
        )

        np.testing.assert_allclose(
            result,
            np.array([2.0, 3.0, 4.0]),
        )

        self.assertEqual(
            sampler.schedule_count,
            2,
        )

    def test_converts_valid_spectrum_to_rgb(self) -> None:
        result = mitsuba_spectrum_to_rgb(
            [0.25, 0.5, 1.0]
        )

        np.testing.assert_allclose(
            result,
            np.array([0.25, 0.5, 1.0]),
        )

    def test_rejects_non_finite_radiance(self) -> None:
        invalid_values = (
            [np.nan, 0.0, 0.0],
            [np.inf, 0.0, 0.0],
        )

        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError,
                    "finite",
                ):
                    mitsuba_spectrum_to_rgb(value)

    def test_rejects_negative_radiance(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "non-negative",
        ):
            mitsuba_spectrum_to_rgb(
                [1.0, -0.1, 2.0]
            )

    def test_rejects_invalid_runtime_settings(self) -> None:
        invalid_settings = (
            {
                "radiance_samples": 0,
                "max_depth": 6,
                "rr_depth": 3,
                "seed": 42,
            },
            {
                "radiance_samples": 1,
                "max_depth": 0,
                "rr_depth": 0,
                "seed": 42,
            },
            {
                "radiance_samples": 1,
                "max_depth": 4,
                "rr_depth": 5,
                "seed": 42,
            },
            {
                "radiance_samples": 1,
                "max_depth": 6,
                "rr_depth": 3,
                "seed": -1,
            },
        )

        for settings in invalid_settings:
            with self.subTest(settings=settings):
                with self.assertRaises(
                    (TypeError, ValueError)
                ):
                    create_incident_radiance_runtime(
                        **settings
                    )


if __name__ == "__main__":
    unittest.main()