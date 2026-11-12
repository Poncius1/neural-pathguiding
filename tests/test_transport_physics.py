"""Analytical transport regressions for Mitsuba 3 scalar_rgb.

These tests complement the image comparisons. They are not proofs of
unbiasedness or support for every Mitsuba BSDF/emitter configuration.
"""
from __future__ import annotations

import unittest
from typing import Any

import mitsuba as mi
import numpy as np

mi.set_variant("scalar_rgb")

# Import registers npg_transport with Mitsuba.
from neural_path_guiding.renderers.mitsuba import transport_integrator  # noqa: E402,F401

_mi: Any = mi


def make_integrator(*, mode: str = "bsdf", max_depth: int = 2) -> Any:
    return _mi.load_dict({
        "type": "npg_transport",
        "mode": mode,
        "max_depth": max_depth,
        "rr_depth": 5,
        "guiding_probability": 0.5,
    })


def make_diffuse_environment_scene() -> Any:
    # A camera ray hits the convex sphere from outside. Every nonzero
    # reflected Lambertian direction leaves the sphere into the
    # constant environment, so L_o = rho * L_env = 0.5.
    return _mi.load_dict({
        "type": "scene",
        "object": {
            "type": "sphere",
            "bsdf": {
                "type": "diffuse",
                "reflectance": {"type": "rgb", "value": [0.5, 0.5, 0.5]},
            },
        },
        "environment": {
            "type": "constant",
            "radiance": {"type": "rgb", "value": [1.0, 1.0, 1.0]},
        },
    })


def camera_ray() -> Any:
    return _mi.Ray3f(
        _mi.Point3f(0.0, 0.0, 3.0),
        _mi.Vector3f(0.0, 0.0, -1.0),
    )


def mean_ray_radiance(scene: Any, integrator: Any, *, seed: int,
                      samples: int = 768) -> np.ndarray:
    sampler = _mi.load_dict({"type": "independent", "sample_count": 1})
    sampler.seed(seed, 1)
    ray = camera_ray()
    total = np.zeros(3, dtype=np.float64)
    for _ in range(samples):
        value, _, _ = integrator.sample(scene, sampler, ray)
        rgb = np.array([float(value[0]), float(value[1]), float(value[2])])
        if not np.all(np.isfinite(rgb)):
            raise AssertionError("Non-finite radiance")
        total += rgb
        sampler.schedule_state()
    return total / samples


class TestTransportPhysics(unittest.TestCase):
    def setUp(self) -> None:
        self.scene = make_diffuse_environment_scene()

    def test_max_depth_one_does_not_reflect_environment(self) -> None:
        # At max_depth=1 only directly visible emission is included;
        # the first diffuse surface may not spawn a continuation.
        result = mean_ray_radiance(
            self.scene, make_integrator(max_depth=1), seed=11, samples=32
        )
        np.testing.assert_allclose(result, np.zeros(3), atol=1e-12)

    def test_bsdf_recovers_analytic_diffuse_reflection(self) -> None:
        result = mean_ray_radiance(
            self.scene, make_integrator(mode="bsdf", max_depth=2),
            seed=21,
        )
        np.testing.assert_allclose(result, [0.5, 0.5, 0.5], rtol=0, atol=0.055)

    def test_uniform_mixture_recovers_analytic_diffuse_reflection(self) -> None:
        result = mean_ray_radiance(
            self.scene, make_integrator(mode="uniform", max_depth=2),
            seed=31,
        )
        np.testing.assert_allclose(result, [0.5, 0.5, 0.5], rtol=0, atol=0.075)

    def test_additional_depth_does_not_change_convex_sphere_solution(self) -> None:
        result = mean_ray_radiance(
            self.scene, make_integrator(mode="bsdf", max_depth=5),
            seed=41,
        )
        np.testing.assert_allclose(result, [0.5, 0.5, 0.5], rtol=0, atol=0.055)

    def test_directly_visible_area_emitter_at_depth_one(self) -> None:
        scene = _mi.load_dict({
            "type": "scene",
            "light_sphere": {
                "type": "sphere",
                "emitter": {
                    "type": "area",
                    "radiance": {"type": "rgb", "value": [0.7, 0.4, 0.2]},
                },
            },
        })
        value, valid, _ = make_integrator(max_depth=1).sample(
            scene,
            self._fresh_sampler(71),
            camera_ray(),
        )
        rgb = np.array([float(value[0]), float(value[1]), float(value[2])])
        self.assertTrue(bool(valid))
        np.testing.assert_allclose(rgb, [0.7, 0.4, 0.2], rtol=0, atol=1e-5)

    @staticmethod
    def _fresh_sampler(seed: int) -> Any:
        sampler = _mi.load_dict({"type": "independent", "sample_count": 1})
        sampler.seed(seed, 1)
        return sampler


if __name__ == "__main__":
    unittest.main()
