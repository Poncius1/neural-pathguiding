"""Render the canonical single-loop transport controls and record manifest v1.

Usage: python -m scripts.render_transport_controls --modes path bsdf uniform neural
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import mitsuba as mi
import numpy as np

from neural_path_guiding.evaluation.image_io import load_image_as_array
from neural_path_guiding.evaluation.transport_metrics import (
    file_sha256,
    summarize_transport_image,
    validate_rgb_image,
)

_mi: Any = mi


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, default=Path("scenes/simple_diffuse/scene.xml"))
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "outputs/experiments/t2_mlp_v2_seed42/best_t2_v2.pt"))
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("outputs/experiments/transport_controls"))
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--spp", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--rr-depth", type=int, default=5)
    parser.add_argument("--guiding-probability", type=float, default=0.5)
    parser.add_argument("--uniform-mix", type=float, default=0.05)
    parser.add_argument("--modes", nargs="+", choices=["path", "bsdf", "uniform", "neural"],
                        default=["path", "bsdf", "uniform", "neural"])
    return parser.parse_args()


def load_rgb(path: Path) -> np.ndarray:
    """Compatibility wrapper for callers of this script's previous API."""
    return validate_rgb_image(load_image_as_array(path), name=str(path))


def build_integrator_properties(args: argparse.Namespace, mode: str) -> dict[str, object]:
    props: dict[str, object] = {
        "type": "path" if mode == "path" else "npg_transport",
        "max_depth": args.max_depth,
        "rr_depth": args.rr_depth,
    }
    if mode != "path":
        props.update(mode=mode, guiding_probability=args.guiding_probability)
    if mode == "neural":
        props.update(checkpoint=str(args.checkpoint.resolve()), uniform_mix=args.uniform_mix)
    return props


def main() -> None:
    args = arguments()
    if min(args.width, args.height, args.spp, args.max_depth, args.rr_depth) <= 0:
        raise ValueError("Dimensions, spp and depths must be positive")
    if args.seed < 0 or not 0 < args.guiding_probability < 1:
        raise ValueError("Invalid seed or guiding probability")
    if not 0 < args.uniform_mix <= 1:
        raise ValueError("uniform_mix must be in (0, 1]")
    if not args.scene.is_file():
        raise FileNotFoundError(args.scene)
    if "neural" in args.modes and not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    _mi.set_variant("scalar_rgb")
    from neural_path_guiding.renderers.mitsuba import transport_integrator  # noqa: F401

    scene = _mi.load_file(str(args.scene.resolve()))
    parameters = _mi.traverse(scene)
    size_key = next((k for k in parameters.keys() if str(k).endswith(".film.size")), None)
    if size_key is None:
        raise ValueError("Cannot find scene film.size")
    parameters[size_key] = _mi.ScalarVector2u(args.width, args.height)
    parameters.update()

    reference = load_rgb(args.reference) if args.reference else None
    if reference is not None and reference.shape != (args.height, args.width, 3):
        raise ValueError("Reference dimensions do not match render")

    out = args.out / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S%fZ")
    out.mkdir(parents=True, exist_ok=False)
    results: dict[str, dict[str, object]] = {}

    for mode in args.modes:
        integrator = _mi.load_dict(build_integrator_properties(args, mode))
        print(f"Rendering {mode}...", flush=True)
        start = perf_counter()
        rendered = _mi.render(scene, integrator=integrator, spp=args.spp, seed=args.seed)
        elapsed = perf_counter() - start
        output_path = out / f"{mode}.exr"
        _mi.Bitmap(rendered).write(str(output_path))
        rgb = load_rgb(output_path)
        item = summarize_transport_image(rgb, seconds=elapsed, reference=reference)
        results[mode] = item
        print(f" {elapsed:.3f}s, mean L={item['mean_luminance']:.6f}, "
              f"MSE={item.get('rgb_mse', float('nan')):.7f}", flush=True)

    # Keep manifest schema unchanged so historical runs and the aggregation
    # script can be grouped together without migration.
    manifest = {
        "experiment": "single_loop_transport_controls",
        "scope": "surface scalar_rgb; first-hit diffuse-only guiding",
        "limitations": [
            "Research implementation, not GPU optimized",
            "No participating media, polarization or null transitions",
            "One continuation sample per bounce in all three custom modes",
            "A single seed is not an unbiasedness or variance validation",
        ],
        "scene": str(args.scene.resolve()),
        "scene_sha256": file_sha256(args.scene),
        "checkpoint": str(args.checkpoint.resolve()) if "neural" in args.modes else None,
        "checkpoint_sha256": file_sha256(args.checkpoint) if "neural" in args.modes else None,
        "reference": str(args.reference.resolve()) if args.reference else None,
        "reference_sha256": file_sha256(args.reference) if args.reference else None,
        "settings": {k: getattr(args, k) for k in (
            "width", "height", "spp", "seed", "max_depth", "rr_depth",
            "guiding_probability", "uniform_mix", "modes")},
        "results": results,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nOutput: {out.resolve()}")


if __name__ == "__main__":
    main()
