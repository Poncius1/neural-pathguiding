
"""
Compares:
- neural_3
- bsdf_3
- light_3
- neural_bsdf_light_mis_3

This is a FIRST-HIT DIRECT-LIGHT experiment, not a full path tracer.

Important:
The neural distribution must use the same local frame as the T2
teacher: core.frames.build_frame_from_normal().

Target: Mitsuba 3.8.0, scalar_rgb, Python 3.11.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Callable, cast

import mitsuba as mi
import numpy as np

# Existing legacy comparator uses imports relative to scripts/.
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scripts.compare_direct_mis_estimators import (
    _LUMINANCE_WEIGHTS,
    estimate_bsdf,
    estimate_light,
    estimate_neural,
    sample_emitter_contribution,
    sample_fixed_surface,
    sha256_file,
    summarize_estimates,
)
from scripts.compare_three_technique_mis import (
    estimate_three_technique_mis,
    run_repetitions,
)
from neural_path_guiding.core.frames import build_frame_from_normal
from neural_path_guiding.renderers.mitsuba.adapters import (
    mitsuba_vector_to_numpy,
)
from neural_path_guiding.renderers.mitsuba.emitter_pdf import (
    EmitterPdfStats,
)
from neural_path_guiding.renderers.mitsuba.guiding_adapter import (
    GuidingProposal,
)
from neural_path_guiding.renderers.mitsuba.neural_inference_v2 import (
    NeuralMitsubaRuntimeV2,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate T2 v2 neural guiding in direct Monte Carlo MIS."
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
            "outputs/experiments/t2_mlp_v2_seed42/best_t2_v2.pt"
        ),
    )
    parser.add_argument("--pixel-x", type=int, default=224)
    parser.add_argument("--pixel-y", type=int, default=224)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--reference-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--uniform-mix", type=float, default=0.05)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experiments/t2_v2_fixed_point_mis"),
    )

    return parser.parse_args()


def prepare_t2_proposal(
    runtime: NeuralMitsubaRuntimeV2,
    interaction,
    camera_ray,
) -> GuidingProposal:
    """Match the directional convention used to produce T2 labels."""

    distribution = runtime.distribution_from_camera_ray(
        interaction,
        camera_ray,
    )

    normal = mitsuba_vector_to_numpy(interaction.n)

    frame = build_frame_from_normal(normal)

    return GuidingProposal(
        frame=frame,
        distribution=distribution,
    )


def validate_proposal(proposal: GuidingProposal, seed: int) -> dict:
    rng = np.random.default_rng(seed)

    probabilities = proposal.distribution.probabilities

    if not np.isclose(probabilities.sum(), 1.0, atol=1e-10):
        raise ValueError("Neural probabilities do not sum to one.")

    if not np.all(probabilities > 0):
        raise ValueError("Neural distribution lacks positive support.")

    max_pdf_error = 0.0

    for _ in range(100):
        sample = proposal.sample(rng)
        evaluated = proposal.pdf(sample.direction_world)

        max_pdf_error = max(
            max_pdf_error,
            abs(sample.pdf - evaluated),
        )

        if not np.isclose(
            sample.pdf, evaluated, rtol=1e-7, atol=1e-10
        ):
            raise ValueError("Neural sample() and pdf() disagree.")

    return {
        "pmf_sum": float(probabilities.sum()),
        "min_bin_probability": float(probabilities.min()),
        "max_pdf_consistency_error": float(max_pdf_error),
    }


def main() -> None:
    args = parse_args()

    if args.repetitions < 2:
        raise ValueError("--repetitions must be >= 2")
    if args.reference_samples < 2:
        raise ValueError("--reference-samples must be >= 2")
    if args.uniform_mix <= 0 or args.uniform_mix > 1:
        raise ValueError("--uniform-mix must be in (0, 1]")

    if not args.scene.is_file():
        raise FileNotFoundError(args.scene)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    mi.set_variant("scalar_rgb")

    # Keep scene alive for the entire experiment.
    load_file = cast(Callable[[str], mi.Scene], mi.load_file)
    scene = load_file(str(args.scene.resolve()))

    ray, interaction, film_size = sample_fixed_surface(
        scene,
        args.pixel_x,
        args.pixel_y,
    )

    if not bool(interaction.is_valid()):
        raise ValueError("Selected pixel did not hit a surface")

    runtime = NeuralMitsubaRuntimeV2.from_checkpoint(
        args.checkpoint,
        device="cpu",
        uniform_mix=args.uniform_mix,
    )

    inference_start = perf_counter()
    proposal = prepare_t2_proposal(runtime, interaction, ray)
    inference_seconds = perf_counter() - inference_start

    proposal_checks = validate_proposal(
        proposal,
        args.seed,
    )

    print("=== T2 V2 FIXED-POINT MIS ===", flush=True)
    print(f"Pixel: ({args.pixel_x}, {args.pixel_y})", flush=True)
    print(f"Checkpoint: {args.checkpoint}", flush=True)
    print(f"Feature schema: 2", flush=True)
    print(f"PMF sum: {proposal_checks['pmf_sum']:.10f}", flush=True)
    print(
        f"Proposal preparation: {inference_seconds:.6f}s",
        flush=True,
    )

    print("Computing independent light-sampling reference...", flush=True)

    reference_values, reference_seconds = run_repetitions(
        lambda rng: sample_emitter_contribution(
            scene, interaction, rng
        ),
        repetitions=args.reference_samples,
        seed=args.seed + 1,
    )

    reference_rgb = reference_values.mean(axis=0)
    reference_se = (
        reference_values.std(axis=0, ddof=1)
        / np.sqrt(args.reference_samples)
    )

    print(f"Reference RGB: {reference_rgb}", flush=True)
    print(f"Reference SE: {reference_se}", flush=True)

    pdf_stats = EmitterPdfStats()

    estimators = {
        "neural_3": lambda rng: estimate_neural(
            scene, interaction, proposal, rng, 3
        ),
        "bsdf_3": lambda rng: estimate_bsdf(
            scene, interaction, rng, 3
        ),
        "light_3": lambda rng: estimate_light(
            scene, interaction, rng, 3
        ),
        "neural_bsdf_light_mis_3": lambda rng:
            estimate_three_technique_mis(
                scene,
                interaction,
                proposal,
                rng,
                pdf_stats=pdf_stats,
            ),
    }

    raw_estimates = {}
    results = {}

    for index, (name, estimator) in enumerate(
        estimators.items()
    ):
        print(f"Running {name}...", flush=True)

        values, elapsed = run_repetitions(
            estimator,
            repetitions=args.repetitions,
            seed=args.seed + 100 + index,
        )

        raw_estimates[name] = values

        metrics = summarize_estimates(
            values,
            reference_rgb,
        )

        metrics["seconds_sampling_only"] = float(elapsed)
        metrics["seconds_per_estimate_sampling_only"] = float(
            elapsed / args.repetitions
        )
        metrics["proposal_samples_per_estimate"] = 3

        estimate_se = np.asarray(
            metrics["standard_error_rgb"],
            dtype=np.float64,
        )

        combined_se = np.sqrt(
            estimate_se**2 + reference_se**2
        )

        difference = (
            np.asarray(metrics["mean_rgb"], dtype=np.float64)
            - reference_rgb
        )

        metrics["difference_in_combined_se_rgb"] = np.divide(
            difference,
            combined_se,
            out=np.zeros(3, dtype=np.float64),
            where=combined_se > 0,
        ).tolist()

        if name == "neural_bsdf_light_mis_3":
            if pdf_stats.total_queries != 2 * args.repetitions:
                raise RuntimeError(
                    "Unexpected emitter cross-PDF query count."
                )

            metrics["emitter_pdf_diagnostics"] = pdf_stats.to_dict()

        results[name] = metrics

        print(
            f"  var={metrics['luminance_variance']:.8e} "
            f"time={elapsed:.3f}s",
            flush=True,
        )

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )

    output_path = args.output_dir / f"run_{timestamp}"
    output_path.mkdir(parents=True, exist_ok=False)

    np.savez_compressed(
        output_path / "estimates.npz",
        reference_samples=reference_values,
        **raw_estimates,
    )

    manifest = {
        "experiment": "t2_v2_fixed_point_direct_mis",
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
        "reference_mean_rgb": reference_rgb.tolist(),
        "reference_standard_error_rgb": reference_se.tolist(),
        "reference_seconds": reference_seconds,
        "feature_schema_version": 2,
        "target_type": "product_integrand_v1",
        "frame_convention": "core.frames.build_frame_from_normal",
        "uniform_mix": args.uniform_mix,
        "proposal_checks": proposal_checks,
        "proposal_preparation_seconds": inference_seconds,
        "estimators": results,
        "emitter_pdf_diagnostics": pdf_stats.to_dict(),
        "limitations": [
            "Direct illumination at one camera-first-hit point only",
            "Pure diffuse surfaces only",
            "Equal number of proposals, not equal computational work",
            "Neural inference prepared once, outside estimator timers",
            "Reference is a finite Monte Carlo estimate",
            "Emitter cross-PDF fallback requires further validation",
            "Not a multibounce renderer or a variance-per-second result",
        ],
    }

    with (output_path / "manifest.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(manifest, file, indent=2)

    print("\n=== T2 V2 RESULTS ===")

    for name, metrics in results.items():
        mean_luminance = float(
            np.dot(
                np.asarray(metrics["mean_rgb"]),
                _LUMINANCE_WEIGHTS,
            )
        )

        print(
            f"{name:<27} "
            f"mean={mean_luminance:.6f} "
            f"var={metrics['luminance_variance']:.8e} "
            f"rmse={metrics['rmse_rgb']:.8e}"
        )

    print("\n=== EMITTER PDF DIAGNOSTICS ===")
    for key, value in pdf_stats.to_dict().items():
        print(f"{key}: {value}")

    print(f"\nResults: {output_path.resolve()}")
    print("Test dataset split was not used.")
    print("Experiment completed.")


if __name__ == "__main__":
    main()
