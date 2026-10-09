"""Compare emitter PDF reconstruction with two ray origins.

Diagnostic only. Does not modify the production renderer.
"""

from __future__ import annotations
from typing import Any, cast
from pathlib import Path

import mitsuba as mi
import numpy as np


def main():
    mi.set_variant("scalar_rgb")

    load_scene = cast(Any, mi.load_file)

    scene = load_scene(
        str(Path("scenes/simple_diffuse/scene.xml").resolve())
    )
    sensor = scene.sensors()[0]
    width, height = map(int, sensor.film().size())

    ray, _ = sensor.sample_ray(
        0.0,
        0.5,
        mi.Point2f(
            481.5 / width,
            440.5 / height,
        ),
        mi.Point2f(0.5, 0.5),
    )

    si = scene.ray_intersect(ray)

    if not bool(si.is_valid()):
        raise RuntimeError("No surface intersection.")

    rng = np.random.default_rng(42)

    print("Comparing ray origins...", flush=True)

    for index in range(20):
        u, v = rng.random(2)

        ds, _ = scene.sample_emitter_direction(
            si,
            mi.Point2f(float(u), float(v)),
            True,
        )

        if float(ds.pdf) <= 0.0:
            continue

        direction = mi.Vector3f(ds.d)

        # Method A: offset origin.
        ray_offset = si.spawn_ray(direction)
        hit_offset = scene.ray_intersect(ray_offset)

        reconstructed_offset = mi.DirectionSample3f(
            scene,
            si=hit_offset,
            ref=si,
        )

        pdf_offset = float(
            scene.pdf_emitter_direction(
                si,
                reconstructed_offset,
                True,
            )
        )

        # Method B: exact reference position.
        ray_exact = mi.Ray3f(
            si.p,
            direction,
            si.time,
        )

        hit_exact = scene.ray_intersect(ray_exact)

        reconstructed_exact = mi.DirectionSample3f(
            scene,
            si=hit_exact,
            ref=si,
        )

        pdf_exact = float(
            scene.pdf_emitter_direction(
                si,
                reconstructed_exact,
                True,
            )
        )

        original = float(ds.pdf)

        print(
            f"{index:02d} "
            f"original={original:.9f} "
            f"offset={pdf_offset:.9f} "
            f"exact={pdf_exact:.9f} "
            f"error_offset={abs(original-pdf_offset):.6e} "
            f"error_exact={abs(original-pdf_exact):.6e}",
            flush=True,
        )


if __name__ == "__main__":
    main()