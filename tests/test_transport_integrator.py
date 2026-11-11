"""Basic physical and structural regression tests for the new transport loop.

These are smoke tests, NOT a proof of unbiasedness.
"""
from __future__ import annotations

import unittest
from typing import Any

import mitsuba as mi
import numpy as np

mi.set_variant("scalar_rgb")

from neural_path_guiding.renderers.mitsuba import transport_integrator  # noqa: E402,F401

_mi: Any = mi


class TestTransportIntegrator(unittest.TestCase):
    def test_invalid_mode_rejected(self):
        with self.assertRaises(Exception):
            _mi.load_dict({"type": "npg_transport", "mode": "wrong",
                           "max_depth": 3, "rr_depth": 2,
                           "guiding_probability": 0.5})

    def test_bsdf_single_ray_constant_environment(self):
        # At the front of a convex Lambertian sphere in a constant
        # environment, the reflected radiance is exactly rho * Le = 0.5.
        # The expected value remains 0.5 when direct light is estimated
        # with next-event estimation and sampled-path emission MIS.
        scene = _mi.load_dict({
            "type": "scene",
            "sphere": {"type": "sphere", "bsdf": {
                "type": "diffuse", "reflectance": {
                    "type": "rgb", "value": [0.5, 0.5, 0.5]}}},
            "light": {"type": "constant", "radiance": {
                "type": "rgb", "value": [1.0, 1.0, 1.0]}},
        })
        integrator = _mi.load_dict({
            "type": "npg_transport", "mode": "bsdf",
            "max_depth": 3, "rr_depth": 2,
            "guiding_probability": 0.5,
        })
        ray = _mi.Ray3f(_mi.Point3f(0, 0, 3), _mi.Vector3f(0, 0, -1))
        sampler = _mi.load_dict({"type": "independent", "sample_count": 1})
        sampler.seed(37, 1)
        samples = []
        for _ in range(512):
            value, _, _ = integrator.sample(scene, sampler, ray)
            samples.append([float(value[0]), float(value[1]), float(value[2])])
        rgb = np.mean(samples, axis=0)
        self.assertTrue(np.all(np.isfinite(rgb)))
        np.testing.assert_allclose(rgb, [0.5, 0.5, 0.5], atol=0.09, rtol=0)


if __name__ == "__main__":
    unittest.main()
