
"""Phase 3.3.2b: fixed-point T2 v2 multibounce comparison.

Compare neural_2, bsdf_2 and neural_bsdf_mis_2.
Each estimate uses exactly two primary directional proposals.

The reference is an independent BSDF estimate using the same
multibounce integrand, but a larger sampling budget.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, cast

import mitsuba as mi
import numpy as np

from neural_path_guiding.evaluation.direct_estimators import (
    _LUMINANCE_WEIGHTS,
    sample_fixed_surface,
    sha256_file,
    summarize_estimates,
)
from neural_path_guiding.core.frames import build_frame_from_normal
from neural_path_guiding.renderers.mitsuba.adapters import (
    mitsuba_vector_to_numpy,
)
from neural_path_guiding.renderers.mitsuba.guiding_adapter import (
    GuidingProposal,
)
from neural_path_guiding.renderers.mitsuba.neural_inference_v2 import (
    NeuralMitsubaRuntimeV2,
)
from neural_path_guiding.renderers.mitsuba.multibounce_estimator import (
    MultibounceEvaluator,
    estimate_bsdf,
    estimate_neural,
    estimate_neural_bsdf_mis,
)


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="T2 v2 neural vs BSDF multibounce Monte Carlo"
    )

    p.add_argument(
        "--scene",
        type=Path,
        default=Path("scenes/simple_diffuse/scene.xml"),
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "outputs/experiments/t2_mlp_v2_seed42/best_t2_v2.pt"
        ),
    )
    p.add_argument("--pixel-x", type=int, default=224)
    p.add_argument("--pixel-y", type=int, default=224)
    p.add_argument("--repetitions", type=int, default=100)
    p.add_argument("--reference-samples", type=int, default=500)
    p.add_argument("--radiance-samples", type=int, default=2)
    p.add_argument("--reference-radiance-samples", type=int, default=4)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--rr-depth", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--uniform-mix", type=float, default=0.05)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experiments/t2_v2_multibounce"),
    )
    return p.parse_args()


def measure(
    estimator,
    *,
    repetitions: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    values = np.zeros((repetitions, 3), dtype=np.float64)

    start = perf_counter()

    for i in range(repetitions):
        values[i] = estimator(rng)

        if not np.all(np.isfinite(values[i])):
            raise ValueError(f"Invalid estimate at repetition {i}")

    return values, perf_counter() - start


def main() -> None:
    args = arguments()

    if args.repetitions < 2 or args.reference_samples < 2:
        raise ValueError("At least two repetitions are required.")

    if args.radiance_samples <= 0:
        raise ValueError("radiance-samples must be positive.")

    if args.reference_radiance_samples <= 0:
        raise ValueError("reference-radiance-samples must be positive.")

    if args.max_depth <= 0 or not (
        0 <= args.rr_depth <= args.max_depth
    ):
        raise ValueError("Invalid path depth or RR depth.")

    if not args.scene.is_file():
        raise FileNotFoundError(args.scene)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    mi.set_variant("scalar_rgb")

    # Scene remains alive while all interactions are used.
    load_file = cast(Callable[[str], Any], mi.load_file)
    scene = load_file(str(args.scene.resolve()))

    ray, interaction, film_size = sample_fixed_surface(
        scene, args.pixel_x, args.pixel_y
    )

    if interaction.emitter(scene) is not None:
        raise ValueError("Select a non-emissive first-hit surface.")

    runtime = NeuralMitsubaRuntimeV2.from_checkpoint(
        args.checkpoint,
        uniform_mix=args.uniform_mix,
        device="cpu",
    )

    start_inference = perf_counter()

    distribution = runtime.distribution_from_camera_ray(
        interaction, ray
    )

    proposal = GuidingProposal(
        frame=build_frame_from_normal(
            mitsuba_vector_to_numpy(interaction.n)
        ),
        distribution=distribution,
    )

    inference_seconds = perf_counter() - start_inference

    # Create separate Mitsuba random streams for each method.
    # Do not reuse the reference sampler for the tested methods.
    reference_evaluator = MultibounceEvaluator.create(
        scene,
        radiance_samples=args.reference_radiance_samples,
        max_depth=args.max_depth,
        rr_depth=args.rr_depth,
        seed=args.seed + 10,
    )

    evaluator_neural = MultibounceEvaluator.create(
        scene,
        radiance_samples=args.radiance_samples,
        max_depth=args.max_depth,
        rr_depth=args.rr_depth,
        seed=args.seed + 20,
    )

    evaluator_bsdf = MultibounceEvaluator.create(
        scene,
        radiance_samples=args.radiance_samples,
        max_depth=args.max_depth,
        rr_depth=args.rr_depth,
        seed=args.seed + 30,
    )

    evaluator_mis = MultibounceEvaluator.create(
        scene,
        radiance_samples=args.radiance_samples,
        max_depth=args.max_depth,
        rr_depth=args.rr_depth,
        seed=args.seed + 40,
    )

    print("=== T2 V2 MULTIBOUNCE FIXED-POINT ===", flush=True)
    print(f"Pixel: ({args.pixel_x}, {args.pixel_y})", flush=True)
    print(f"Max depth: {args.max_depth}", flush=True)
    print(f"Inner radiance samples: {args.radiance_samples}", flush=True)
    print(f"PMF sum: {distribution.probabilities.sum():.10f}", flush=True)
    print(f"Min PMF: {distribution.probabilities.min():.10f}", flush=True)

    print("Computing independent BSDF reference...", flush=True)

    reference, reference_time = measure(
        lambda rng: estimate_bsdf(
            interaction,
            reference_evaluator,
            rng,
            sample_count=2,
        ),
        repetitions=args.reference_samples,
        seed=args.seed + 1000,
    )

    reference_mean = reference.mean(axis=0)
    reference_se = (
        reference.std(axis=0, ddof=1)
        / np.sqrt(args.reference_samples)
    )

    print(f"Reference RGB: {reference_mean}", flush=True)
    print(f"Reference SE: {reference_se}", flush=True)

    methods = {
        "neural_2": (
            evaluator_neural,
            lambda rng: estimate_neural(
                interaction, proposal, evaluator_neural, rng, 2
            ),
        ),
        "bsdf_2": (
            evaluator_bsdf,
            lambda rng: estimate_bsdf(
                interaction, evaluator_bsdf, rng, 2
            ),
        ),
        "neural_bsdf_mis_2": (
            evaluator_mis,
            lambda rng: estimate_neural_bsdf_mis(
                interaction, proposal, evaluator_mis, rng
            ),
        ),
    }

    results = {}
    raw = {}

    for index, (name, (_evaluator, estimator)) in enumerate(
        methods.items()
    ):
        print(f"Running {name}...", flush=True)

        values, seconds = measure(
            estimator,
            repetitions=args.repetitions,
            seed=args.seed + 2000 + index,
        )

        metrics = summarize_estimates(values, reference_mean)

        metrics["seconds_sampling_and_continuation"] = seconds
        metrics["seconds_per_estimate"] = (
            seconds / args.repetitions
        )
        metrics["primary_proposals_per_estimate"] = 2
        metrics["inner_samples_per_proposal"] = args.radiance_samples

        estimate_se = np.asarray(
            metrics["standard_error_rgb"],
            dtype=np.float64,
        )

        combined_se = np.sqrt(
            estimate_se**2 + reference_se**2
        )

        difference = (
            np.asarray(metrics["mean_rgb"])
            - reference_mean
        )

        metrics["difference_in_combined_se_rgb"] = (
            np.divide(
                difference,
                combined_se,
                out=np.zeros(3),
                where=combined_se > 0,
            ).tolist()
        )

        results[name] = metrics
        raw[name] = values

        luminance = float(
            np.dot(metrics["mean_rgb"], _LUMINANCE_WEIGHTS)
        )

        print(
            f"  mean={luminance:.8f} "
            f"variance={metrics['luminance_variance']:.8e} "
            f"time={seconds:.3f}s",
            flush=True,
        )

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )

    output = args.output_dir / f"run_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)

    np.savez_compressed(
        output / "estimates.npz",
        reference_samples=reference,
        **raw,
    )

    manifest = {
        "experiment": "t2_v2_fixed_point_multibounce",
        "timestamp_utc": timestamp,
        "scene": str(args.scene.resolve()),
        "scene_sha256": sha256_file(args.scene),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "mitsuba_version": mi.__version__,
        "mitsuba_variant": mi.variant(),
        "pixel": [args.pixel_x, args.pixel_y],
        "film_size": list(film_size),
        "seed": args.seed,
        "repetitions": args.repetitions,
        "reference_samples": args.reference_samples,
        "radiance_samples": args.radiance_samples,
        "reference_radiance_samples": args.reference_radiance_samples,
        "max_depth": args.max_depth,
        "rr_depth": args.rr_depth,
        "feature_schema_version": 2,
        "target_type": "product_integrand_v1",
        "frame_convention": "core.frames.build_frame_from_normal",
        "uniform_mix": args.uniform_mix,
        "pmf_sum": float(distribution.probabilities.sum()),
        "min_bin_probability": float(
            distribution.probabilities.min()
        ),
        "proposal_preparation_seconds": inference_seconds,
        "reference_mean_rgb": reference_mean.tolist(),
        "reference_standard_error_rgb": reference_se.tolist(),
        "reference_seconds": reference_time,
        "estimators": results,
        "limitations": [
            "Fixed camera-first-hit point, not full-image render",
            "Only pure Lambertian first-hit materials",
            "Neural guiding is used only at the first continuation",
            "Mitsuba samples subsequent path directions",
            "Inner path integrator may use NEE and MIS internally",
            "Two primary directional proposals per estimate",
            "Reference uses finite independent BSDF Monte Carlo",
            "Proposal preparation and model loading excluded from method timers",
            "CPU scalar_rgb Python orchestration; not optimized rendering",
            "No explicit first-vertex emitter sampling strategy",
        ],
    }

    with (output / "manifest.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(manifest, file, indent=2)

    print("\n=== MULTIBOUNCE RESULTS ===")

    for name, metrics in results.items():
        mean_lum = float(
            np.dot(
                np.asarray(metrics["mean_rgb"]),
                _LUMINANCE_WEIGHTS,
            )
        )

        print(
            f"{name:<22} "
            f"mean={mean_lum:.8f} "
            f"var={metrics['luminance_variance']:.8e} "
            f"rmse={metrics['rmse_rgb']:.8e}"
        )

    print(f"\nOutput: {output.resolve()}")
    print("Multibounce experiment completed.", flush=True)


if __name__ == "__main__":
    main()
