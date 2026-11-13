"""Legacy fixed-point direct-light estimator experiment CLI.

Reusable estimators now live in evaluation.direct_estimators. This script
continues to re-export their names for compatibility with historical code.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import mitsuba as mi
import numpy as np

from neural_path_guiding.evaluation.direct_estimators import (
    RgbEstimate, _LUMINANCE_WEIGHTS, sha256_file,
    sample_fixed_surface, sample_uniform_hemisphere, sample_bsdf_direction,
    evaluate_direct_integrand, sample_emitter_term,
    sample_emitter_contribution, estimate_uniform, estimate_bsdf,
    estimate_neural, estimate_light, estimate_neural_bsdf_mis,
    summarize_estimates,
)
from neural_path_guiding.renderers.mitsuba.adapters import (
    mitsuba_vector_to_numpy, normalize_numpy_vector,
)
from neural_path_guiding.renderers.mitsuba.guiding_adapter import MitsubaGuidingAdapter
from neural_path_guiding.training.inference import GuidingInferenceRuntime

Estimator = Callable[[np.random.Generator], RgbEstimate]

def parse_arguments() -> argparse.Namespace:
    """Read command-line options."""

    parser = argparse.ArgumentParser(
        description=(
            "Compare direct-light estimators at one fixed Mitsuba "
            "surface interaction."
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
            "checkpoints/simple_diffuse_product_integrand_overfit_v1.pt"
        ),
    )
    parser.add_argument("--pixel-x", type=int, default=481)
    parser.add_argument("--pixel-y", type=int, default=440)
    parser.add_argument("--repetitions", type=int, default=1000)
    parser.add_argument("--reference-samples", type=int, default=10000)
    parser.add_argument("--uniform-mix", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--variant",
        default="scalar_rgb",
        help="Mitsuba variant used for this scalar experiment.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experiments"),
    )
    return parser.parse_args()


def run_estimator(
    estimator: Estimator,
    *,
    repetitions: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    """Run one estimator repeatedly with a deterministic random stream."""

    rng = np.random.default_rng(seed)
    estimates = np.empty((repetitions, 3), dtype=np.float64)
    start_time = perf_counter()

    for index in range(repetitions):
        try:
            estimates[index] = estimator(rng)
        except Exception as error:
            raise RuntimeError(
                f"Estimator failed on repetition {index + 1} "
                f"of {repetitions}."
            ) from error

    return estimates, perf_counter() - start_time


def main() -> None:
    """Load inputs, run the estimators, and save results."""

    args = parse_arguments()
    print("Starting direct-light estimator comparison.", flush=True)

    if args.repetitions < 2:
        raise ValueError("--repetitions must be at least 2.")
    if args.reference_samples < 2:
        raise ValueError("--reference-samples must be at least 2.")
    if not args.scene.is_file():
        raise FileNotFoundError(f"Scene not found: {args.scene}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}"
        )

    mi.set_variant(args.variant)
    load_scene = cast(Callable[[str], Any], getattr(mi, "load_file"))
    scene = load_scene(str(args.scene.resolve()))

    camera_ray, interaction, film_size = sample_fixed_surface(
        scene,
        args.pixel_x,
        args.pixel_y,
    )
    outgoing_direction = normalize_numpy_vector(
        -mitsuba_vector_to_numpy(camera_ray.d)
    )

    runtime = GuidingInferenceRuntime.from_checkpoint(
        args.checkpoint,
        device="cpu",
        uniform_mix=args.uniform_mix,
    )
    adapter = MitsubaGuidingAdapter(runtime)
    proposal = adapter.prepare_proposal(
        surface_interaction=interaction,
        outgoing_direction=outgoing_direction,
        bounce_depth=0,
    )

    print(
        f"Calculating reference with {args.reference_samples} samples...",
        flush=True,
    )
    reference_samples, reference_seconds = run_estimator(
        lambda rng: sample_emitter_contribution(
            scene,
            interaction,
            rng,
        ),
        repetitions=args.reference_samples,
        seed=args.seed + 1,
    )
    reference = reference_samples.mean(axis=0)
    reference_standard_error = (
        reference_samples.std(axis=0, ddof=1)
        / np.sqrt(args.reference_samples)
    )
    print("Reference calculated.", flush=True)

    estimators = build_estimators(
        scene,
        interaction,
        proposal,
    )
    raw_estimates: dict[str, np.ndarray] = {}
    results: dict[str, Any] = {}

    for index, (name, estimator) in enumerate(estimators.items()):
        print(
            f"Running {name} "
            f"({args.repetitions} repetitions)...",
            flush=True,
        )
        samples, elapsed_seconds = run_estimator(
            estimator,
            repetitions=args.repetitions,
            seed=args.seed + 100 + index,
        )
        raw_estimates[name] = samples
        results[name] = summarize_estimates(samples, reference)
        results[name]["seconds"] = elapsed_seconds
        results[name]["seconds_per_estimate"] = (
            elapsed_seconds / args.repetitions
        )
        print(f"Finished {name}.", flush=True)

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    output_path = args.output_dir / f"direct_mis_{timestamp}"
    output_path.mkdir(parents=True, exist_ok=True)

    savez_kwargs: dict[str, Any] = {
        "reference_samples": reference_samples,
        **raw_estimates,
    }
    np.savez_compressed(
        output_path / "estimates.npz",
        allow_pickle=False,
        **savez_kwargs,
    )

    position = mitsuba_vector_to_numpy(interaction.p).tolist()
    manifest = {
        "experiment": "fixed_point_direct_lighting_estimators",
        "timestamp_utc": timestamp,
        "scene": str(args.scene.resolve()),
        "scene_sha256": sha256_file(args.scene),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "mitsuba_version": mi.__version__,
        "mitsuba_variant": mi.variant(),
        "numpy_version": np.__version__,
        "pixel": [args.pixel_x, args.pixel_y],
        "film_size": list(film_size),
        "position": position,
        "normal": mitsuba_vector_to_numpy(interaction.n).tolist(),
        "target_type": runtime.target_type,
        "num_bins": runtime.num_bins,
        "uniform_mix": args.uniform_mix,
        "seed": args.seed,
        "repetitions": args.repetitions,
        "reference_method": "mitsuba_sample_emitter_direction",
        "reference_samples": args.reference_samples,
        "reference_mean_rgb": reference.tolist(),
        "reference_standard_error_rgb": (
            reference_standard_error.tolist()
        ),
        "reference_seconds": reference_seconds,
        "estimators": results,
        "interpretation": (
            "Fixed-point direct-light comparison. "
            "This is not yet image-level render variance evidence."
        ),
    }

    with (output_path / "manifest.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(manifest, file, indent=2)

    print("Direct-light estimator comparison completed.")
    print(f"Results: {output_path.resolve()}")
    print(f"Pixel: ({args.pixel_x}, {args.pixel_y})")
    print(f"Position: {position}")
    print(f"Reference RGB: {reference}")
    print(f"Reference standard error: {reference_standard_error}")
    print()
    print(
        f"{'Estimator':<22}"
        f"{'Mean luminance':>18}"
        f"{'Luminance var':>18}"
        f"{'RMSE':>14}"
    )

    for name, metrics in results.items():
        mean_luminance = float(
            np.dot(metrics["mean_rgb"], _LUMINANCE_WEIGHTS)
        )
        print(
            f"{name:<22}"
            f"{mean_luminance:>18.6f}"
            f"{metrics['luminance_variance']:>18.6e}"
            f"{metrics['rmse_rgb']:>14.6e}"
        )


if __name__ == "__main__":
    main()