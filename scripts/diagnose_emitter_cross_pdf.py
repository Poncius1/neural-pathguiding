"""Diagnose Mitsuba emitter PDF reconstruction.

Compares:
1. PDF returned by emitter sampling.
2. PDF evaluated using the original DirectionSample3f.
3. PDF evaluated after reconstructing the direction sample.

Target runtime: Mitsuba 3.8.0, scalar_rgb.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import mitsuba as mi
import numpy as np

from neural_path_guiding.renderers.mitsuba.adapters import (
    mitsuba_vector_to_numpy,
)
from neural_path_guiding.renderers.mitsuba.emitter_pdf import (
    evaluate_emitter_pdf,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scene",
        type=Path,
        default=Path("scenes/simple_diffuse/scene.xml"),
    )
    parser.add_argument("--pixel-x", type=int, default=481)
    parser.add_argument("--pixel-y", type=int, default=440)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def get_interaction(scene, pixel_x, pixel_y):
    sensor = scene.sensors()[0]
    size = sensor.film().size()

    width = int(size[0])
    height = int(size[1])

    if not (0 <= pixel_x < width and 0 <= pixel_y < height):
        raise ValueError("Pixel outside film.")

    uv = mi.Point2f(
        (pixel_x + 0.5) / width,
        (pixel_y + 0.5) / height,
    )

    ray, _ = sensor.sample_ray(
        0.0,
        0.5,
        uv,
        mi.Point2f(0.5, 0.5),
    )

    si = scene.ray_intersect(ray)

    if not bool(si.is_valid()):
        raise RuntimeError("Camera ray missed the scene.")

    return si


def main():
    args = parse_args()

    if args.samples < 1:
        raise ValueError("--samples must be positive.")

    mi.set_variant("scalar_rgb")

    load_scene = cast(Any, mi.load_file)
    scene = load_scene(str(args.scene.resolve()))

    si = get_interaction(
        scene,
        args.pixel_x,
        args.pixel_y,
    )

    rng = np.random.default_rng(args.seed)

    records = []
    zero_count = 0

    print("=== EMITTER PDF RECONSTRUCTION ===", flush=True)
    print("Mitsuba:", mi.__version__, flush=True)
    print("Variant:", mi.variant(), flush=True)

    for index in range(args.samples):
        u, v = rng.random(2)

        ds, _ = scene.sample_emitter_direction(
            si,
            mi.Point2f(float(u), float(v)),
            True,
        )

        if bool(ds.delta):
            raise NotImplementedError(
                "Delta emitter encountered."
            )

        sampled_pdf = float(ds.pdf)

        if not np.isfinite(sampled_pdf) or sampled_pdf < 0:
            raise ValueError("Invalid sampled PDF.")

        if sampled_pdf == 0:
            zero_count += 1
            continue

        same_record_pdf = float(
            scene.pdf_emitter_direction(
                si,
                ds,
                True,
            )
        )

        direction = mitsuba_vector_to_numpy(ds.d)

        result = evaluate_emitter_pdf(
            scene=scene,
            interaction=si,
            world_direction=direction,
        )

        reconstructed_pdf = result.pdf

        if not all(
            np.isfinite(v)
            for v in (
                same_record_pdf,
                reconstructed_pdf,
            )
        ):
            raise ValueError("Non-finite evaluated PDF.")

        if same_record_pdf < 0 or reconstructed_pdf < 0:
            raise ValueError("Negative evaluated PDF.")

        absolute_error = abs(
            same_record_pdf - reconstructed_pdf
        )

        relative_error = (
            absolute_error
            / max(abs(same_record_pdf), 1e-12)
        )

        kind = (
            "ENV"
            if result.environment
            else "AREA"
            if result.hit_emitter
            else "NONE"
        )

        records.append({
            "index": index,
            "kind": kind,
            "sampled": sampled_pdf,
            "same": same_record_pdf,
            "reconstructed": reconstructed_pdf,
            "absolute_error": absolute_error,
            "relative_error": relative_error,
        })

        print(
            f"{index:03d} [{kind}] "
            f"sampled={sampled_pdf:.9f} "
            f"same={same_record_pdf:.9f} "
            f"reconstructed={reconstructed_pdf:.9f} "
            f"abs={absolute_error:.9f} "
            f"rel={relative_error:.3e}",
            flush=True,
        )

    print()
    print("=== SUMMARY ===")
    print("Requested:", args.samples)
    print("Zero PDF:", zero_count)
    print("Compared:", len(records))

    if not records:
        raise RuntimeError("No positive-PDF samples.")

    for kind in ("AREA", "ENV", "NONE"):
        group = [
            r for r in records
            if r["kind"] == kind
        ]

        if not group:
            continue

        print()
        print(f"{kind}: {len(group)} samples")

        print(
            "Max absolute error:",
            max(r["absolute_error"] for r in group),
        )

        print(
            "Max relative error:",
            max(r["relative_error"] for r in group),
        )

        print(
            "Mean relative error:",
            np.mean([
                r["relative_error"] for r in group
            ]),
        )

    original_errors = [
        abs(r["sampled"] - r["same"])
        for r in records
    ]

    print()
    print(
        "Max original-record PDF difference:",
        max(original_errors),
    )

    print()
    print("Diagnostic completed.")
    print(
        "No correctness verdict is issued until "
        "the per-emitter differences are analyzed."
    )


if __name__ == "__main__":
    main()