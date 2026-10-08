"""Compare direct-light estimators at one fixed surface interaction."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import mitsuba as mi
import numpy as np

from neural_path_guiding.core.frames import local_to_world
from neural_path_guiding.core.mis import power_heuristic
from neural_path_guiding.renderers.mitsuba.adapters import (
    mitsuba_vector_to_numpy,
    normalize_numpy_vector,
)
from neural_path_guiding.renderers.mitsuba.bsdf import (
    evaluate_bsdf_pdf,
    evaluate_bsdf_times_cosine,
)
from neural_path_guiding.renderers.mitsuba.guiding_adapter import (
    MitsubaGuidingAdapter,
)
from neural_path_guiding.renderers.mitsuba.radiance import (
    mitsuba_spectrum_to_rgb,
)
from neural_path_guiding.training.inference import (
    GuidingInferenceRuntime,
)


RgbEstimate = np.ndarray
Estimator = Callable[[np.random.Generator], RgbEstimate]

_LUMINANCE_WEIGHTS = np.array(
    [0.2126, 0.7152, 0.0722],
    dtype=np.float64,
)


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


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hash of a file."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def sample_fixed_surface(
    scene: Any,
    pixel_x: int,
    pixel_y: int,
) -> tuple[Any, Any, tuple[int, int]]:
    """Get the first surface hit for the center of one camera pixel."""

    sensors = scene.sensors()
    if not sensors:
        raise ValueError("The scene does not contain a sensor.")

    sensor = sensors[0]
    film_size = sensor.film().size()
    width = int(film_size[0])
    height = int(film_size[1])

    if not (0 <= pixel_x < width and 0 <= pixel_y < height):
        raise ValueError(
            f"Pixel ({pixel_x}, {pixel_y}) is outside the film "
            f"dimensions ({width}, {height})."
        )

    image_sample = mi.Point2f(
        (pixel_x + 0.5) / width,
        (pixel_y + 0.5) / height,
    )
    ray, _ = sensor.sample_ray(
        0.0,
        0.5,
        image_sample,
        mi.Point2f(0.5, 0.5),
    )
    interaction = scene.ray_intersect(ray)

    if not interaction.is_valid():
        raise ValueError(
            f"Camera ray at pixel ({pixel_x}, {pixel_y}) missed the scene."
        )

    return ray, interaction, (width, height)


def sample_uniform_hemisphere(
    frame: Any,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a direction uniformly in solid angle on the local hemisphere."""

    u_mu, u_phi = rng.random(2)
    mu = float(u_mu)
    radius = np.sqrt(max(0.0, 1.0 - mu * mu))
    phi = 2.0 * np.pi * float(u_phi)

    direction_local = np.array(
        [
            radius * np.cos(phi),
            radius * np.sin(phi),
            mu,
        ],
        dtype=np.float64,
    )
    direction_world = local_to_world(frame, direction_local)
    return normalize_numpy_vector(direction_world)


def sample_bsdf_direction(
    interaction: Any,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Sample one BSDF direction and evaluate its solid-angle PDF."""

    bsdf = interaction.bsdf()
    if bsdf is None:
        raise ValueError("Surface interaction has no BSDF.")

    bsdf_sample, _ = bsdf.sample(
        mi.BSDFContext(),
        interaction,
        float(rng.random()),
        mi.Point2f(
            float(rng.random()),
            float(rng.random()),
        ),
        True,
    )
    direction_world = normalize_numpy_vector(
        mitsuba_vector_to_numpy(
            interaction.to_world(bsdf_sample.wo)
        )
    )
    pdf = evaluate_bsdf_pdf(
        surface_interaction=interaction,
        world_direction=direction_world,
    )

    if not np.isfinite(pdf) or pdf <= 0.0:
        raise ValueError(
            "The BSDF returned a sample with a non-positive or "
            "non-finite PDF. This experiment expects a continuous, "
            "non-delta BSDF."
        )

    return direction_world, float(pdf)


def evaluate_direct_integrand(
    scene: Any,
    interaction: Any,
    direction_world: np.ndarray,
) -> RgbEstimate:
    """Evaluate incident emission times the BSDF*cosine term."""

    direction = normalize_numpy_vector(direction_world)
    ray = interaction.spawn_ray(mi.Vector3f(*direction))
    hit = scene.ray_intersect(ray)

    if not hit.is_valid():
        environment = scene.environment()
        if environment is None:
            return np.zeros(3, dtype=np.float64)

        incident_radiance = mitsuba_spectrum_to_rgb(
            environment.eval(hit)
        )
    else:
        emitter = hit.emitter(scene)
        if emitter is None:
            return np.zeros(3, dtype=np.float64)

        incident_radiance = mitsuba_spectrum_to_rgb(
            emitter.eval(hit)
        )

    bsdf_times_cosine = evaluate_bsdf_times_cosine(
        surface_interaction=interaction,
        world_direction=direction,
    )
    return incident_radiance * bsdf_times_cosine


def sample_emitter_term(
    scene: Any,
    interaction: Any,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Return light direction, PDF, and contribution divided by that PDF."""

    direction_sample, emitter_weight = scene.sample_emitter_direction(
        interaction,
        mi.Point2f(
            float(rng.random()),
            float(rng.random()),
        ),
        True,
    )

    light_pdf = float(direction_sample.pdf)
    if not np.isfinite(light_pdf) or light_pdf < 0.0:
        raise ValueError(
            f"Mitsuba returned an invalid emitter PDF: {light_pdf}."
        )

    emitter_weight_rgb = mitsuba_spectrum_to_rgb(emitter_weight)

    if light_pdf == 0.0:
        if np.any(emitter_weight_rgb > 0.0):
            raise NotImplementedError(
                "This experiment does not support delta emitters."
            )

        return (
            np.zeros(3, dtype=np.float64),
            0.0,
            np.zeros(3, dtype=np.float64),
        )

    direction_world = normalize_numpy_vector(
        mitsuba_vector_to_numpy(direction_sample.d)
    )
    bsdf_times_cosine = evaluate_bsdf_times_cosine(
        surface_interaction=interaction,
        world_direction=direction_world,
    )

    # Mitsuba returns incident radiance divided by the emitter PDF.
    contribution_over_pdf = emitter_weight_rgb * bsdf_times_cosine
    return direction_world, light_pdf, contribution_over_pdf


def sample_emitter_contribution(
    scene: Any,
    interaction: Any,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample one emitter contribution for the light-only estimator."""

    _, _, contribution_over_pdf = sample_emitter_term(
        scene,
        interaction,
        rng,
    )
    return contribution_over_pdf


def estimate_uniform(
    scene: Any,
    interaction: Any,
    frame: Any,
    rng: np.random.Generator,
    sample_count: int,
) -> RgbEstimate:
    """Estimate direct lighting with uniform-hemisphere sampling."""

    pdf = 1.0 / (2.0 * np.pi)
    total = np.zeros(3, dtype=np.float64)

    for _ in range(sample_count):
        direction = sample_uniform_hemisphere(frame, rng)
        total += evaluate_direct_integrand(
            scene,
            interaction,
            direction,
        ) / pdf

    return total / sample_count


def estimate_bsdf(
    scene: Any,
    interaction: Any,
    rng: np.random.Generator,
    sample_count: int,
) -> RgbEstimate:
    """Estimate direct lighting with BSDF sampling."""

    total = np.zeros(3, dtype=np.float64)

    for _ in range(sample_count):
        direction, pdf = sample_bsdf_direction(interaction, rng)
        total += evaluate_direct_integrand(
            scene,
            interaction,
            direction,
        ) / pdf

    return total / sample_count


def estimate_neural(
    scene: Any,
    interaction: Any,
    proposal: Any,
    rng: np.random.Generator,
    sample_count: int,
) -> RgbEstimate:
    """Estimate direct lighting with neural directional sampling."""

    total = np.zeros(3, dtype=np.float64)

    for _ in range(sample_count):
        sample = proposal.sample(rng)
        total += evaluate_direct_integrand(
            scene,
            interaction,
            sample.direction_world,
        ) / sample.pdf

    return total / sample_count


def estimate_light(
    scene: Any,
    interaction: Any,
    rng: np.random.Generator,
    sample_count: int,
) -> RgbEstimate:
    """Estimate direct lighting with emitter sampling."""

    total = np.zeros(3, dtype=np.float64)

    for _ in range(sample_count):
        total += sample_emitter_contribution(
            scene,
            interaction,
            rng,
        )

    return total / sample_count


def estimate_neural_bsdf_mis(
    scene: Any,
    interaction: Any,
    proposal: Any,
    rng: np.random.Generator,
) -> RgbEstimate:
    """Combine one neural and one BSDF sample using the power heuristic."""

    neural_sample = proposal.sample(rng)
    bsdf_direction, bsdf_pdf = sample_bsdf_direction(
        interaction,
        rng,
    )

    neural_contribution = evaluate_direct_integrand(
        scene,
        interaction,
        neural_sample.direction_world,
    )
    bsdf_contribution = evaluate_direct_integrand(
        scene,
        interaction,
        bsdf_direction,
    )

    bsdf_pdf_at_neural = evaluate_bsdf_pdf(
        surface_interaction=interaction,
        world_direction=neural_sample.direction_world,
    )
    neural_pdf_at_bsdf = proposal.pdf(bsdf_direction)

    neural_weight = power_heuristic(
        neural_sample.pdf,
        bsdf_pdf_at_neural,
    )
    bsdf_weight = power_heuristic(
        bsdf_pdf,
        neural_pdf_at_bsdf,
    )

    return (
        neural_weight
        * neural_contribution
        / neural_sample.pdf
        + bsdf_weight
        * bsdf_contribution
        / bsdf_pdf
    )


def summarize_estimates(
    estimates: np.ndarray,
    reference: np.ndarray,
) -> dict[str, Any]:
    """Calculate summary statistics for repeated estimates."""

    mean_rgb = estimates.mean(axis=0)
    variance_rgb = estimates.var(axis=0, ddof=1)
    standard_error_rgb = (
        estimates.std(axis=0, ddof=1)
        / np.sqrt(estimates.shape[0])
    )
    luminance = estimates @ _LUMINANCE_WEIGHTS

    return {
        "mean_rgb": mean_rgb.tolist(),
        "variance_rgb": variance_rgb.tolist(),
        "standard_error_rgb": standard_error_rgb.tolist(),
        "bias_rgb": (mean_rgb - reference).tolist(),
        "rmse_rgb": float(
            np.sqrt(np.mean((estimates - reference) ** 2))
        ),
        "luminance_variance": float(luminance.var(ddof=1)),
    }


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


def build_estimators(
    scene: Any,
    interaction: Any,
    proposal: Any,
) -> dict[str, Estimator]:
    """Build the estimator variants used in this comparison."""

    return {
        "uniform_1": lambda rng: estimate_uniform(
            scene, interaction, proposal.frame, rng, 1
        ),
        "uniform_2": lambda rng: estimate_uniform(
            scene, interaction, proposal.frame, rng, 2
        ),
        "bsdf_1": lambda rng: estimate_bsdf(
            scene, interaction, rng, 1
        ),
        "bsdf_2": lambda rng: estimate_bsdf(
            scene, interaction, rng, 2
        ),
        "neural_1": lambda rng: estimate_neural(
            scene, interaction, proposal, rng, 1
        ),
        "neural_2": lambda rng: estimate_neural(
            scene, interaction, proposal, rng, 2
        ),
        "light_1": lambda rng: estimate_light(
            scene, interaction, rng, 1
        ),
        "light_2": lambda rng: estimate_light(
            scene, interaction, rng, 2
        ),
        "neural_bsdf_mis_2": lambda rng: estimate_neural_bsdf_mis(
            scene,
            interaction,
            proposal,
            rng,
        ),
        "uniform_3": lambda rng: estimate_uniform(
            scene, interaction, proposal.frame, rng, 3
        ),
        "bsdf_3": lambda rng: estimate_bsdf(
            scene, interaction, rng, 3
        ),
        "neural_3": lambda rng: estimate_neural(
            scene, interaction, proposal, rng, 3
        ),
        "light_3": lambda rng: estimate_light(
            scene, interaction, rng, 3
        ),
    }


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