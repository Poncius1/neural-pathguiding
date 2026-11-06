"""Smoke-test neural inference, local/world sample and PDF at Mitsuba hit."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, cast

import mitsuba as mi
import numpy as np

mi.set_variant("scalar_rgb")

from neural_path_guiding.renderers.mitsuba.neural_inference_v2 import (
    NeuralMitsubaRuntimeV2,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scene", type=Path,
                        default=Path("scenes/simple_diffuse/scene.xml"))
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("--samples must be positive")
    load_file = cast(Callable[[str], mi.Scene], mi.load_file)
    scene = load_file(str(args.scene))
    # Known front-facing sphere point in the reference scene.
    direction = np.array([0.0, -0.6, -4.5], dtype=np.float64)
    direction /= np.linalg.norm(direction)
    ray = mi.Ray3f(mi.Point3f(0, 1.15, 4.5),
                   mi.Vector3f(*direction))
    interaction = scene.ray_intersect(ray)
    if not bool(interaction.is_valid()):
        raise RuntimeError("Smoke camera ray missed scene")
    runtime = NeuralMitsubaRuntimeV2.from_checkpoint(args.checkpoint)
    dist = runtime.distribution_from_camera_ray(interaction, ray)
    rng = np.random.default_rng(args.seed)
    errors = []
    for _ in range(args.samples):
        sampled = runtime.sample_world(interaction, dist, rng)
        checked = runtime.pdf_world(interaction, dist, sampled.world)
        errors.append(abs(checked - sampled.pdf_solid_angle))
    print("=== NEURAL MITSUBA V2 (PHASE 3.3.1) ===")
    print("PMF sum:", float(np.sum(dist.probabilities)))
    print("Min bin probability:", float(np.min(dist.probabilities)))
    print("Samples:", args.samples)
    print("Max |sample PDF - evaluated PDF|:", max(errors))
    if max(errors) > 1e-8:
        raise RuntimeError("PDF mismatch")
    print("Status: PASS (not yet a Monte Carlo render)")


if __name__ == "__main__":
    main()
