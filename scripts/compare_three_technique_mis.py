"""Compare Neural + BSDF + Light MIS at one fixed surface point.

Mitsuba 3.8.0, scalar_rgb.
Includes emitter cross-PDF diagnostics for the MIS estimator.
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

from compare_direct_mis_estimators import (
    _LUMINANCE_WEIGHTS,
    estimate_bsdf,
    estimate_light,
    estimate_neural,
    evaluate_direct_integrand,
    sample_bsdf_direction,
    sample_emitter_contribution,
    sample_emitter_term,
    sample_fixed_surface,
    sha256_file,
    summarize_estimates,
)
from neural_path_guiding.core.mis import (
    power_heuristic_weights,
)
from neural_path_guiding.renderers.mitsuba.adapters import (
    mitsuba_vector_to_numpy,
    normalize_numpy_vector,
)
from neural_path_guiding.renderers.mitsuba.bsdf import (
    evaluate_bsdf_pdf,
)
from neural_path_guiding.renderers.mitsuba.emitter_pdf import (
    EmitterPdfStats,
    evaluate_emitter_pdf,
)
from neural_path_guiding.renderers.mitsuba.guiding_adapter import (
    MitsubaGuidingAdapter,
)
from neural_path_guiding.training.inference import (
    GuidingInferenceRuntime,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Three-technique direct MIS with PDF diagnostics."
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
            "checkpoints/"
            "simple_diffuse_product_integrand_overfit_v1.pt"
        ),
    )
    parser.add_argument("--pixel-x", type=int, default=481)
    parser.add_argument("--pixel-y", type=int, default=440)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--reference-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--uniform-mix", type=float, default=0.05)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experiments"),
    )

    return parser.parse_args()


def mis_weight(
    densities: tuple[float, float, float],
    technique_index: int,
) -> float:
    values = np.asarray(densities, dtype=np.float64)

    if not np.all(np.isfinite(values)):
        raise ValueError("Non-finite MIS PDF.")

    if np.any(values < 0.0):
        raise ValueError("Negative MIS PDF.")

    if values[technique_index] <= 0.0:
        raise ValueError("Sampled technique has zero PDF.")

    weights = power_heuristic_weights(
        tuple(float(value) for value in values),
        sample_counts=[1, 1, 1],
        exponent=2.0,
    )

    return float(weights[technique_index])


def estimate_three_technique_mis(
    scene: Any,
    interaction: Any,
    proposal: Any,
    rng: np.random.Generator,
    *,
    pdf_stats: EmitterPdfStats | None = None,
) -> np.ndarray:
    """One neural, one BSDF and one emitter sample.

    The stats object records the two cross-PDF queries:
    light PDF at the neural and BSDF sampled directions.
    """

    result = np.zeros(3, dtype=np.float64)

    # Neural sample.
    neural_sample = proposal.sample(rng)
    neural_direction = neural_sample.direction_world
    p_neural = float(neural_sample.pdf)

    p_bsdf_at_neural = evaluate_bsdf_pdf(
        surface_interaction=interaction,
        world_direction=neural_direction,
    )

    p_light_at_neural = evaluate_emitter_pdf(
        scene=scene,
        interaction=interaction,
        world_direction=neural_direction,
        stats=pdf_stats,
    ).pdf

    neural_weight = mis_weight(
        (p_neural, p_bsdf_at_neural, p_light_at_neural),
        technique_index=0,
    )

    neural_integrand = evaluate_direct_integrand(
        scene,
        interaction,
        neural_direction,
    )

    result += (
        neural_weight * neural_integrand / p_neural
    )

    # BSDF sample.
    bsdf_direction, p_bsdf = sample_bsdf_direction(
        interaction,
        rng,
    )

    p_neural_at_bsdf = float(
        proposal.pdf(bsdf_direction)
    )

    p_light_at_bsdf = evaluate_emitter_pdf(
        scene=scene,
        interaction=interaction,
        world_direction=bsdf_direction,
        stats=pdf_stats,
    ).pdf

    bsdf_weight = mis_weight(
        (p_neural_at_bsdf, p_bsdf, p_light_at_bsdf),
        technique_index=1,
    )

    bsdf_integrand = evaluate_direct_integrand(
        scene,
        interaction,
        bsdf_direction,
    )

    result += (
        bsdf_weight * bsdf_integrand / p_bsdf
    )

    # Light sample.
    (
        light_direction,
        p_light,
        light_contribution_over_pdf,
    ) = sample_emitter_term(
        scene,
        interaction,
        rng,
    )

    if p_light > 0.0:
        p_neural_at_light = float(
            proposal.pdf(light_direction)
        )

        p_bsdf_at_light = evaluate_bsdf_pdf(
            surface_interaction=interaction,
            world_direction=light_direction,
        )

        light_weight = mis_weight(
            (p_neural_at_light, p_bsdf_at_light, p_light),
            technique_index=2,
        )

        # Mitsuba's light sample is already divided by p_light.
        result += (
            light_weight * light_contribution_over_pdf
        )

    if result.shape != (3,):
        raise RuntimeError("MIS result must have RGB shape (3,).")

    if not np.all(np.isfinite(result)):
        raise ValueError("MIS produced non-finite RGB.")

    return result


def run_repetitions(
    estimator: Callable[[np.random.Generator], np.ndarray],
    *,
    repetitions: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    values = np.zeros((repetitions, 3), dtype=np.float64)

    start = perf_counter()

    for index in range(repetitions):
        try:
            values[index] = estimator(rng)
        except Exception as exc:
            raise RuntimeError(
                f"Estimator failed at repetition {index + 1}."
            ) from exc

    return values, perf_counter() - start


def main() -> None:
    args = parse_arguments()

    if args.repetitions < 2:
        raise ValueError("--repetitions must be >= 2.")

    if args.reference_samples < 2:
        raise ValueError("--reference-samples must be >= 2.")

    if not args.scene.is_file():
        raise FileNotFoundError(args.scene)

    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    mi.set_variant("scalar_rgb")

    load_scene = cast(
        Callable[[str], Any],
        getattr(mi, "load_file"),
    )

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

    print("Three-technique MIS experiment started.", flush=True)
    print(f"Mitsuba: {mi.__version__}", flush=True)
    print(f"Seed: {args.seed}", flush=True)

    # Independent reference.
    print("Calculating reference...", flush=True)

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

    # Independent counters for this experiment.
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

    results: dict[str, Any] = {}
    raw: dict[str, np.ndarray] = {}

    for index, (name, estimator) in enumerate(
        estimators.items()
    ):
        print(
            f"Running {name} ({args.repetitions})...",
            flush=True,
        )

        values, seconds = run_repetitions(
            estimator,
            repetitions=args.repetitions,
            seed=args.seed + 100 + index,
        )

        raw[name] = values

        metrics = summarize_estimates(
            values,
            reference_rgb,
        )

        metrics["seconds"] = seconds
        metrics["seconds_per_estimate"] = (
            seconds / args.repetitions
        )
        metrics["proposal_samples_per_estimate"] = 3

        estimate_se = np.asarray(
            metrics["standard_error_rgb"],
            dtype=np.float64,
        )

        combined_se = np.sqrt(
            estimate_se ** 2 + reference_se ** 2
        )

        difference = (
            np.asarray(metrics["mean_rgb"], dtype=np.float64)
            - reference_rgb
        )

        metrics["difference_in_combined_standard_errors_rgb"] = (
            np.divide(
                difference,
                combined_se,
                out=np.zeros(3, dtype=np.float64),
                where=combined_se > 0.0,
            ).tolist()
        )

        if name == "neural_bsdf_light_mis_3":
            if pdf_stats.total_queries != 2 * args.repetitions:
                raise RuntimeError(
                    "Expected exactly two emitter cross-PDF "
                    "queries per completed MIS estimate."
                )

            metrics["emitter_pdf_diagnostics"] = (
                pdf_stats.to_dict()
            )

        results[name] = metrics

        luminance = float(
            np.dot(
                metrics["mean_rgb"],
                _LUMINANCE_WEIGHTS,
            )
        )

        print(
            f"  mean={luminance:.6f} "
            f"variance={metrics['luminance_variance']:.6e} "
            f"time={seconds:.3f}s",
            flush=True,
        )

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )

    output_path = (
        args.output_dir / f"three_mis_{timestamp}"
    )
    output_path.mkdir(parents=True, exist_ok=False)

    np.savez_compressed(
        output_path / "estimates.npz",
        reference_samples=reference_values,
        neural_3=raw["neural_3"],
        bsdf_3=raw["bsdf_3"],
        light_3=raw["light_3"],
        neural_bsdf_light_mis_3=raw[
            "neural_bsdf_light_mis_3"
        ],
    )

    manifest = {
        "experiment": "three_technique_fixed_point_direct_mis",
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
        "uniform_mix": args.uniform_mix,
        "emitter_pdf_diagnostics": pdf_stats.to_dict(),
        "estimators": results,
        "limitations": [
            "Fixed-point direct-light experiment",
            "One sample per MIS technique",
            "Continuous non-delta strategies only",
            "Equal proposal samples, not equal execution time",
            "Neural checkpoint may be overfitted",
            "Reference has Monte Carlo uncertainty",
            "Visibility fallback has not been proven unbiased",
        ],
    }

    with (output_path / "manifest.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(manifest, file, indent=2)

    print()
    print("=== THREE-TECHNIQUE MIS RESULTS ===")

    for name, metrics in results.items():
        mean = float(
            np.dot(
                metrics["mean_rgb"],
                _LUMINANCE_WEIGHTS,
            )
        )

        print(
            f"{name:<27} "
            f"mean={mean:.6f} "
            f"var={metrics['luminance_variance']:.6e} "
            f"rmse={metrics['rmse_rgb']:.6e}"
        )

    print()
    print("=== EMITTER PDF DIAGNOSTICS ===")

    for key, value in pdf_stats.to_dict().items():
        print(f"{key}: {value}")

    print(f"Results: {output_path.resolve()}")
    print("Experiment completed.", flush=True)


if __name__ == "__main__":
    main()