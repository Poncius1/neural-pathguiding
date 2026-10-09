"""Diagnose scalar Mitsuba emitter sampling at a fixed camera pixel."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, cast

import mitsuba as mi
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scene",
        type=Path,
        default=Path("scenes/simple_diffuse/scene.xml"),
    )
    parser.add_argument("--pixel-x", type=int, default=481)
    parser.add_argument("--pixel-y", type=int, default=440)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--variant",
        type=str,
        default="scalar_rgb",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.samples < 1:
        raise ValueError("--samples must be >= 1.")

    mi.set_variant(args.variant)

    from neural_path_guiding.renderers.mitsuba.emitter_sampling import (
        evaluate_sampled_emitter_pdf,
        sample_emitter_direction,
    )
    from neural_path_guiding.renderers.mitsuba.bsdf import (
        evaluate_bsdf_pdf,
    )

    load_file = cast(Callable[[str], Any], mi.load_file)
    scene = load_file(str(args.scene.resolve()))

    sensor = scene.sensors()[0]
    film_size = sensor.film().size()

    width = int(film_size[0])
    height = int(film_size[1])

    if not (0 <= args.pixel_x < width):
        raise ValueError("Invalid pixel-x.")

    if not (0 <= args.pixel_y < height):
        raise ValueError("Invalid pixel-y.")

    uv = mi.Point2f(
        (args.pixel_x + 0.5) / width,
        (args.pixel_y + 0.5) / height,
    )

    ray, _ = sensor.sample_ray(
        0.0,
        0.5,
        uv,
        mi.Point2f(0.5, 0.5),
    )

    interaction = scene.ray_intersect(ray)

    if not interaction.is_valid():
        raise RuntimeError("Camera ray missed the scene.")

    rng = np.random.default_rng(args.seed)

    pdf_errors = []
    bsdf_pdfs = []
    emitter_pdfs = []
    contributions = []

    zero_pdf_count = 0

    print("Starting emitter diagnostic...", flush=True)
    print(f"Mitsuba: {mi.__version__}", flush=True)
    print(f"Variant: {mi.variant()}", flush=True)

    for index in range(args.samples):

        # Obtain the actual sampling record from Mitsuba.
        u, v = rng.random(2)

        ds, _ = scene.sample_emitter_direction(
            interaction,
            mi.Point2f(float(u), float(v)),
            True,
        )

        if bool(ds.delta):
            raise NotImplementedError(
                "Delta emitter encountered."
            )

        sampled_pdf = float(ds.pdf)

        if not np.isfinite(sampled_pdf):
            raise ValueError("Non-finite sampled emitter PDF.")

        if sampled_pdf < 0.0:
            raise ValueError("Negative sampled emitter PDF.")

        if sampled_pdf == 0.0:
            zero_pdf_count += 1
            continue

        # Evaluate using the SAME sampling record.
        evaluated_pdf = evaluate_sampled_emitter_pdf(
            scene=scene,
            interaction=interaction,
            direction_sample=ds,
        )

        pdf_errors.append(
            abs(sampled_pdf - evaluated_pdf)
        )

        # Independently sample another emitter direction
        # using the project's new adapter.
        result = sample_emitter_direction(
            scene=scene,
            interaction=interaction,
            rng=rng,
        )

        if result.emitter_pdf > 0.0:

            bsdf_pdf = evaluate_bsdf_pdf(
                surface_interaction=interaction,
                world_direction=result.direction_world,
            )

            if not np.isclose(
                bsdf_pdf,
                result.bsdf_pdf,
                rtol=1e-5,
                atol=1e-7,
            ):
                raise AssertionError(
                    "BSDF PDF mismatch."
                )

            emitter_pdfs.append(result.emitter_pdf)
            bsdf_pdfs.append(result.bsdf_pdf)
            contributions.append(
                result.contribution_over_pdf
            )

        if (index + 1) % 25 == 0:
            print(
                f"Processed {index + 1}/{args.samples}",
                flush=True,
            )

    if not pdf_errors:
        raise RuntimeError(
            "No positive-PDF emitter samples were obtained."
        )

    print()
    print("=== DIAGNOSTIC RESULTS ===")
    print(f"Samples requested: {args.samples}")
    print(f"Zero-PDF samples: {zero_pdf_count}")

    print(
        "Maximum emitter PDF difference:",
        max(pdf_errors),
    )

    print(
        "Mean emitter PDF difference:",
        float(np.mean(pdf_errors)),
    )

    if emitter_pdfs:
        print(
            "Mean emitter PDF:",
            float(np.mean(emitter_pdfs)),
        )
        print(
            "Mean BSDF PDF at emitter directions:",
            float(np.mean(bsdf_pdfs)),
        )
        print(
            "Mean RGB contribution:",
            np.mean(contributions, axis=0),
        )

    if max(pdf_errors) > 1e-4:
        raise AssertionError(
            "Emitter PDF consistency check failed."
        )

    print()
    print("Emitter diagnostic PASSED.")


if __name__ == "__main__":
    main()