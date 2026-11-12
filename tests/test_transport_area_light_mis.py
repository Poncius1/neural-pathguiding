"""Physical regression: diffuse surface illuminated by a finite area light.

This is a scalar_rgb, single-ray Monte Carlo test. It compares against
both Mitsuba's native path tracer and an independent numerical quadrature
of the direct-lighting equation. It does not validate every scene/BSDF.
"""

from __future__ import annotations

import unittest
from typing import Any

import mitsuba as mi
import numpy as np

mi.set_variant("scalar_rgb")

from neural_path_guiding.renderers.mitsuba import (  # noqa: E402
    transport_integrator,
)

_mi: Any = mi

ALBEDO = 0.6
EMISSION = 4.0
LIGHT_X = 1.0
LIGHT_Z = 2.0
LIGHT_HALF_EXTENT = 0.8


def make_scene() -> Any:
    """A diffuse rectangle at z=0 and downward-facing emitter at z=2.

    Camera test ray starts at (0,0,1) and points down, so the source
    is not directly camera-visible on the diffuse-patch test.
    """
    return _mi.load_dict({
        "type": "scene",
        "floor": {
            "type": "rectangle",
            "to_world": _mi.ScalarTransform4f.scale([5, 5, 1]),
            "bsdf": {
                "type": "diffuse",
                "reflectance": {
                    "type": "rgb",
                    "value": [ALBEDO] * 3,
                },
            },
        },
        "area_source": {
            "type": "rectangle",
            "to_world": (
                _mi.ScalarTransform4f.translate([LIGHT_X, 0, LIGHT_Z])
                .rotate([0, 1, 0], 180)
                .scale([LIGHT_HALF_EXTENT, LIGHT_HALF_EXTENT, 1])
            ),
            "emitter": {
                "type": "area",
                "radiance": {
                    "type": "rgb",
                    "value": [EMISSION] * 3,
                },
            },
        },
    })


def make_integrator(kind: str, *, max_depth: int = 2) -> Any:
    if kind == "path":
        return _mi.load_dict({
            "type": "path",
            "max_depth": max_depth,
            "rr_depth": 5,
        })
    return _mi.load_dict({
        "type": "npg_transport",
        "mode": kind,
        "max_depth": max_depth,
        "rr_depth": 5,
        "guiding_probability": 0.5,
    })


def mean_ray_radiance(
    scene: Any,
    integrator: Any,
    ray: Any,
    *,
    count: int,
    seed: int,
) -> np.ndarray:
    sampler = _mi.load_dict({
        "type": "independent",
        "sample_count": 1,
    })
    sampler.seed(seed, 1)

    total = np.zeros(3, dtype=np.float64)
    for _ in range(count):
        radiance, _, _ = integrator.sample(scene, sampler, ray)
        total += np.array(
            [float(radiance[i]) for i in range(3)],
            dtype=np.float64,
        )
    return total / count


def expected_direct_radiance(grid_size: int = 160) -> float:
    """Deterministic midpoint quadrature for the area-light integral.

    The two planes are parallel, horizontal, and directly visible.
    At x=(0,0,0):
      Lo = (rho / pi) * Le * integral_A (z^2 / r^4) dA.
    This has no environment or indirect illumination.
    """
    e = LIGHT_HALF_EXTENT
    step = (2.0 * e) / grid_size
    centers = (np.arange(grid_size) + 0.5) * step - e

    xs = LIGHT_X + centers[:, None]
    ys = centers[None, :]
    r2 = xs * xs + ys * ys + LIGHT_Z * LIGHT_Z

    integral = np.sum((LIGHT_Z * LIGHT_Z) / (r2 * r2)) * (step * step)
    return float((ALBEDO / np.pi) * EMISSION * integral)


class TestTransportAreaLightMIS(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scene = make_scene()
        cls.floor_ray = _mi.Ray3f(
            _mi.Point3f(0, 0, 1),
            _mi.Vector3f(0, 0, -1),
        )
        # From below, directly toward the emitting side.
        cls.light_ray = _mi.Ray3f(
            _mi.Point3f(LIGHT_X, 0, 1),
            _mi.Vector3f(0, 0, 1),
        )

    def test_camera_visible_emitter_is_not_mis_weighted(self) -> None:
        for kind in ("path", "bsdf"):
            with self.subTest(kind=kind):
                value = mean_ray_radiance(
                    self.scene,
                    make_integrator(kind, max_depth=1),
                    self.light_ray,
                    count=1,
                    seed=11,
                )
                np.testing.assert_allclose(
                    value, [EMISSION] * 3,
                    atol=1e-5, rtol=0,
                )

    def test_max_depth_one_excludes_reflected_light(self) -> None:
        for kind in ("path", "bsdf"):
            with self.subTest(kind=kind):
                value = mean_ray_radiance(
                    self.scene,
                    make_integrator(kind, max_depth=1),
                    self.floor_ray,
                    count=32,
                    seed=12,
                )
                np.testing.assert_allclose(
                    value, [0, 0, 0], atol=1e-7, rtol=0
                )

    def test_area_light_nee_and_emitter_hit_mis(self) -> None:
        expected = expected_direct_radiance()
        self.assertGreater(expected, 0.0)

        # A moderately large area emitter keeps this regression stable.
        # This is a loose statistical test, not a proof of unbiasedness.
        for kind in ("path", "bsdf", "uniform"):
            with self.subTest(kind=kind):
                value = mean_ray_radiance(
                    self.scene,
                    make_integrator(kind, max_depth=2),
                    self.floor_ray,
                    count=2048,
                    seed=2345,
                )
                self.assertTrue(np.all(np.isfinite(value)))
                self.assertTrue(np.all(value >= 0.0))
                np.testing.assert_allclose(
                    value,
                    [expected] * 3,
                    atol=0.06,
                    rtol=0,
                    err_msg=f"Area-light direct transport mismatch ({kind})",
                )

    def test_depth_two_is_not_double_counted_at_depth_four(self) -> None:
        # The floor and light are the only surfaces. The floor cannot
        # receive indirect reflection from the one-sided light.
        expected = expected_direct_radiance()
        value = mean_ray_radiance(
            self.scene,
            make_integrator("bsdf", max_depth=4),
            self.floor_ray,
            count=2048,
            seed=4567,
        )
        np.testing.assert_allclose(
            value, [expected] * 3, atol=0.06, rtol=0
        )


if __name__ == "__main__":
    unittest.main()
