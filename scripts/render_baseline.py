"""Render a baseline Mitsuba scene.

This script renders a standard Mitsuba scene without neural path guiding.

Default behavior:
- variant: scalar_rgb
- scene: built-in Cornell Box
- output: outputs/baselines/cornell_baseline.exr
"""

from __future__ import annotations

import argparse
from pathlib import Path

from neural_path_guiding.renderers.mitsuba.baseline_renderer import render_scene
from neural_path_guiding.renderers.mitsuba.scene_loader import (
    DEFAULT_VARIANT,
    configure_mitsuba,
    load_cornell_box_scene,
    validate_windows_llvm_runtime,
)


DEFAULT_OUTPUT_PATH = Path("outputs/baselines/cornell_baseline.exr")
DEFAULT_SPP = 64


def parse_arguments() -> argparse.Namespace:
    # Defines command-line options for the baseline render.
    parser = argparse.ArgumentParser(
        description="Render a baseline Cornell Box scene with Mitsuba."
    )

    parser.add_argument(
        "--variant",
        type=str,
        default=DEFAULT_VARIANT,
        help="Mitsuba variant to use.",
    )

    parser.add_argument(
        "--spp",
        type=int,
        default=DEFAULT_SPP,
        help="Samples per pixel.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output image path.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    validate_windows_llvm_runtime()
    configure_mitsuba(args.variant)

    scene = load_cornell_box_scene()

    result = render_scene(
        scene=scene,
        output_path=args.output,
        spp=args.spp,
    )

    print("Baseline render completed.")
    print(f"Variant: {args.variant}")
    print(f"Samples per pixel: {result.spp}")
    print(f"Render time: {result.elapsed_seconds:.3f} seconds")
    print(f"Output: {result.output_path}")


if __name__ == "__main__":
    main()