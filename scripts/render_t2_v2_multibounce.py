
"""Render four first-hit sampling strategies for controlled diagnostics.

Methods:
    1. baseline_path
    2. bsdf_only
    3. uniform_bsdf_mis
    4. neural_t2_multibounce

Optionally compares each image against an independent EXR reference.

Mitsuba 3.8.0, scalar_rgb.
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

from scripts.compare_direct_mis_estimators import sha256_file

# Workaround for dynamically generated Mitsuba bindings in Pylance.
_mi: Any = mi


LUMINANCE_WEIGHTS = np.array(
    [0.2126, 0.7152, 0.0722],
    dtype=np.float64,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare standard Path Tracing, BSDF-only, "
            "uniform MIS and T2 neural MIS image rendering."
        )
    )

    parser.add_argument(
        "--scene",
        type=Path,
        default=Path("scenes/simple_diffuse/scene.xml"),
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "outputs/experiments/t2_mlp_v2_seed42/"
            "best_t2_v2.pt"
        ),
    )

    parser.add_argument(
        "--width",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--height",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--spp",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--rr-depth",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--uniform-mix",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Optional independent high-spp EXR reference.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/experiments/t2_v2_image_controls"
        ),
    )

    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate experiment inputs before initializing Mitsuba."""

    if min(args.width, args.height, args.spp) <= 0:
        raise ValueError(
            "Image width, height and spp must be positive."
        )

    if args.seed < 0:
        raise ValueError("Seed must be non-negative.")

    if args.max_depth < 2:
        raise ValueError(
            "max-depth must be at least 2."
        )

    if not (0 <= args.rr_depth <= args.max_depth):
        raise ValueError(
            "rr-depth must satisfy 0 <= rr-depth <= max-depth."
        )

    if not (0.0 < args.uniform_mix <= 1.0):
        raise ValueError(
            "uniform-mix must satisfy 0 < value <= 1."
        )

    if not args.scene.is_file():
        raise FileNotFoundError(
            f"Scene not found: {args.scene}"
        )

    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}"
        )

    if args.reference is not None:
        if not args.reference.is_file():
            raise FileNotFoundError(
                f"Reference image not found: {args.reference}"
            )


def read_rgb(path: Path) -> np.ndarray:
    """Load an EXR file as a finite RGB NumPy array."""

    bitmap = _mi.Bitmap(str(path))

    image = np.asarray(
        bitmap,
        dtype=np.float64,
    )

    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(
            f"Unexpected image shape {image.shape}: {path}"
        )

    rgb = image[:, :, :3].copy()

    if not np.all(np.isfinite(rgb)):
        raise ValueError(
            f"Image contains NaN or infinity: {path}"
        )

    return rgb


def image_metrics(
    image: np.ndarray,
    reference: np.ndarray,
) -> dict[str, Any]:
    """Compute per-image error statistics against a reference."""

    if image.shape != reference.shape:
        raise ValueError(
            "Image and reference dimensions differ: "
            f"{image.shape} vs {reference.shape}"
        )

    error = image - reference

    luminance_error = error @ LUMINANCE_WEIGHTS

    rgb_mse = float(
        np.mean(np.square(error))
    )

    rgb_mae = float(
        np.mean(np.abs(error))
    )

    luminance_rmse = float(
        np.sqrt(np.mean(np.square(luminance_error)))
    )

    return {
        "rgb_mse": rgb_mse,
        "rgb_rmse": float(np.sqrt(rgb_mse)),
        "rgb_mae": rgb_mae,
        "luminance_rmse": luminance_rmse,
        "mean_rgb_error": (
            error.mean(axis=(0, 1)).tolist()
        ),
        "max_abs_error": float(
            np.max(np.abs(error))
        ),
    }


def configure_film(
    scene: Any,
    width: int,
    height: int,
) -> None:
    """Set the resolution of the scene's existing sensor."""

    parameters = _mi.traverse(scene)

    film_key = next(
        (
            key
            for key in parameters.keys()
            if str(key).endswith(".film.size")
        ),
        None,
    )

    if film_key is None:
        raise RuntimeError(
            "Could not locate the sensor film resolution."
        )

    parameters[film_key] = _mi.ScalarVector2u(
        width,
        height,
    )

    parameters.update()


def create_baseline_integrator(
    args: argparse.Namespace,
) -> Any:
    """Create Mitsuba's standard Path Tracing integrator."""

    return _mi.load_dict({
        "type": "path",
        "max_depth": args.max_depth,
        "rr_depth": args.rr_depth,
    })


def create_custom_integrator(
    mode: str,
    args: argparse.Namespace,
) -> Any:
    """Create a custom diagnostic integrator.

    Crucially, neural-only properties are passed only when
    the integrator will consume them.

    This avoids Mitsuba's 'unreferenced properties' error.
    """

    valid_modes = {
        "bsdf_only",
        "uniform_bsdf_mis",
        "neural_bsdf_mis",
    }

    if mode not in valid_modes:
        raise ValueError(
            f"Unsupported custom integrator mode: {mode}"
        )

    properties: dict[str, Any] = {
        "type": "neural_t2_multibounce",
        "mode": mode,
        "max_depth": args.max_depth,
        "rr_depth": args.rr_depth,
    }

    # Only neural inference consumes these properties.
    if mode == "neural_bsdf_mis":
        properties["checkpoint"] = str(
            args.checkpoint.resolve()
        )

        properties["uniform_mix"] = (
            args.uniform_mix
        )

    # Important: return OUTSIDE the conditional.
    return _mi.load_dict(properties)


def build_integrators(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Build all four integrators in a consistent order."""

    return {
        "baseline_path": create_baseline_integrator(
            args
        ),

        "bsdf_only": create_custom_integrator(
            "bsdf_only",
            args,
        ),

        "uniform_bsdf_mis": create_custom_integrator(
            "uniform_bsdf_mis",
            args,
        ),

        "neural_t2_multibounce": create_custom_integrator(
            "neural_bsdf_mis",
            args,
        ),
    }


def render_and_measure(
    *,
    scene: Any,
    integrator: Any,
    name: str,
    spp: int,
    seed: int,
    output_dir: Path,
    reference: np.ndarray | None,
) -> dict[str, Any]:
    """Render one integrator and collect numerical diagnostics."""

    print(
        f"Rendering {name}...",
        flush=True,
    )

    start_time = perf_counter()

    image = _mi.render(
        scene,
        integrator=integrator,
        spp=spp,
        seed=seed,
    )

    elapsed_seconds = (
        perf_counter() - start_time
    )

    output_path = output_dir / f"{name}.exr"

    _mi.Bitmap(image).write(
        str(output_path)
    )

    rgb = read_rgb(output_path)

    luminance = rgb @ LUMINANCE_WEIGHTS

    negative_channels = int(
        np.count_nonzero(rgb < 0.0)
    )

    if negative_channels > 0:
        raise ValueError(
            f"{name} produced {negative_channels} "
            "negative RGB channel values."
        )

    record: dict[str, Any] = {
        "filename": output_path.name,
        "seconds": float(elapsed_seconds),
        "mean_rgb": rgb.mean(
            axis=(0, 1)
        ).tolist(),
        "mean_luminance": float(
            luminance.mean()
        ),
        "min_luminance": float(
            luminance.min()
        ),
        "max_luminance": float(
            luminance.max()
        ),
        "luminance_percentile_99": float(
            np.percentile(luminance, 99)
        ),
        "finite": True,
        "negative_channels": negative_channels,
        "zero_pixels": int(
            np.count_nonzero(
                np.all(rgb == 0.0, axis=2)
            )
        ),
        "pixels_above_luminance_1": int(
            np.count_nonzero(luminance > 1.0)
        ),
        "pixels_above_luminance_2": int(
            np.count_nonzero(luminance > 2.0)
        ),
    }

    if reference is not None:
        record["errors"] = image_metrics(
            rgb,
            reference,
        )

    message = (
        f"  time={elapsed_seconds:.3f}s "
        f"mean_L={record['mean_luminance']:.6f} "
        f"max_L={record['max_luminance']:.6f}"
    )

    if reference is not None:
        mse = record["errors"]["rgb_mse"]

        message += (
            f" MSE={mse:.8e}"
        )

    print(
        message,
        flush=True,
    )

    return record


def main() -> None:
    args = parse_args()

    validate_arguments(args)

    _mi.set_variant("scalar_rgb")

    # Importing this module registers the custom integrator.
    from neural_path_guiding.renderers.mitsuba import (
        neural_multibounce_integrator,  # noqa: F401
    )

    # Keep the scene object alive throughout all renders.
    scene = _mi.load_file(
        str(args.scene.resolve())
    )

    configure_film(
        scene,
        args.width,
        args.height,
    )

    integrators = build_integrators(args)

    timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )

    output_dir = (
        args.output_dir
        / f"run_{timestamp}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    reference: np.ndarray | None = None

    if args.reference is not None:
        reference = read_rgb(
            args.reference
        )

        expected_shape = (
            args.height,
            args.width,
            3,
        )

        if reference.shape != expected_shape:
            raise ValueError(
                "Reference has incorrect resolution: "
                f"expected {expected_shape}, "
                f"got {reference.shape}"
            )

    print(
        "=== T2 V2 IMAGE CONTROL EXPERIMENT ===",
        flush=True,
    )

    print(
        f"Resolution: {args.width}x{args.height}",
        flush=True,
    )

    print(
        f"SPP: {args.spp}",
        flush=True,
    )

    print(
        f"Seed: {args.seed}",
        flush=True,
    )

    print(
        f"Max depth: {args.max_depth}",
        flush=True,
    )

    print(
        f"RR depth: {args.rr_depth}",
        flush=True,
    )

    results: dict[str, Any] = {}

    for name, integrator in integrators.items():
        record = render_and_measure(
            scene=scene,
            integrator=integrator,
            name=name,
            spp=args.spp,
            seed=args.seed,
            output_dir=output_dir,
            reference=reference,
        )

        results[name] = record

    manifest = {
        "experiment": "t2_v2_image_controls",
        "timestamp_utc": timestamp,

        "scene": str(
            args.scene.resolve()
        ),
        "scene_sha256": sha256_file(
            args.scene
        ),

        "checkpoint": str(
            args.checkpoint.resolve()
        ),
        "checkpoint_sha256": sha256_file(
            args.checkpoint
        ),

        "reference": (
            str(args.reference.resolve())
            if args.reference is not None
            else None
        ),

        "reference_sha256": (
            sha256_file(args.reference)
            if args.reference is not None
            else None
        ),

        "mitsuba_version": _mi.__version__,
        "mitsuba_variant": _mi.variant(),
        "numpy_version": np.__version__,

        "width": args.width,
        "height": args.height,
        "spp": args.spp,
        "seed": args.seed,
        "max_depth": args.max_depth,
        "rr_depth": args.rr_depth,
        "uniform_mix": args.uniform_mix,

        "modes": list(
            integrators.keys()
        ),

        "results": results,

        "limitations": [
            "Neural guiding only at camera-first-hit diffuse surfaces",
            "bsdf_only uses one primary directional proposal",
            "Both MIS modes use two primary directional proposals",
            "Equal spp does not imply equal computational work",
            "Python scalar_rgb research prototype",
            "Single-seed image results do not establish variance",
            "A finite high-spp reference is not ground truth",
            "Physical equivalence with Mitsuba path remains to be tested",
        ],
    }

    manifest_path = (
        output_dir / "manifest.json"
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
        )

    print()
    print(
        "=== CONTROL RESULTS ===",
        flush=True,
    )

    for name, record in results.items():
        print(
            f"{name:<24}"
            f"{record['seconds']:.3f}s",
            flush=True,
        )

        if "errors" in record:
            errors = record["errors"]

            print(
                "  RGB MSE: "
                f"{errors['rgb_mse']:.8e}",
                flush=True,
            )

            print(
                "  RGB MAE: "
                f"{errors['rgb_mae']:.8e}",
                flush=True,
            )

            print(
                "  Luminance RMSE: "
                f"{errors['luminance_rmse']:.8e}",
                flush=True,
            )

    print()
    print(
        f"Output: {output_dir.resolve()}",
        flush=True,
    )

    print(
        "All control renders completed.",
        flush=True,
    )


if __name__ == "__main__":
    main()
