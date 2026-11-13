"""Evaluate Monte Carlo stability of the T2 product-integrand teacher.

Phase 3.1.1.

Uses the same fixed camera surface interactions for all budgets.
Runs two independent replicas per budget and pixel.

Produces:
    teacher_stability_records.csv
    teacher_stability_summary.json
    teacher_targets.npz

Target:
    Mitsuba 3.8.0
    Python 3.11
    scalar_rgb
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import mitsuba as mi
import numpy as np

from neural_path_guiding.evaluation.direct_estimators import (
    sample_fixed_surface,
)

from neural_path_guiding.core.bins import HemisphereBins
from neural_path_guiding.core.targets import (
    contributions_to_target_distribution,
)
from neural_path_guiding.renderers.mitsuba.adapters import (
    mitsuba_vector_to_numpy,
    normalize_numpy_vector,
)
from neural_path_guiding.renderers.mitsuba.teachers import (
    PRODUCT_INTEGRAND_TARGET,
    PhysicalTeacherSettings,
    TeacherContext,
    create_teacher_runtime,
    estimate_bin_contributions,
)


@dataclass(frozen=True)
class Budget:
    name: str
    samples_per_bin: int
    radiance_samples: int


BUDGETS = (
    Budget("low", 4, 2),
    Budget("medium", 16, 4),
    Budget("high", 64, 8),
)

DEFAULT_PIXELS = (
    (0, 160),
    (224, 224),
    (0, 351),
    (481, 440),
)


def validate_distribution(
    values: np.ndarray,
) -> np.ndarray:
    """Validate a discrete probability distribution."""

    probabilities = np.asarray(
        values,
        dtype=np.float64,
    )

    if probabilities.ndim != 1 or probabilities.size == 0:
        raise ValueError(
            "Distribution must be a nonempty 1D array."
        )

    if (
        not np.all(np.isfinite(probabilities))
        or np.any(probabilities < 0.0)
    ):
        raise ValueError(
            "Distribution must be finite and nonnegative."
        )

    if not np.isclose(
        probabilities.sum(),
        1.0,
        atol=1e-6,
        rtol=0.0,
    ):
        raise ValueError(
            "Distribution probabilities must sum to one."
        )

    return probabilities


def total_variation_distance(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    """Calculate TVD between two normalized discrete PMFs."""

    p = validate_distribution(first)
    q = validate_distribution(second)

    if p.shape != q.shape:
        raise ValueError(
            "Distribution shapes must match."
        )

    return float(
        0.5 * np.abs(p - q).sum()
    )


def jensen_shannon_divergence(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    """Calculate Jensen-Shannon divergence in nats."""

    p = validate_distribution(first)
    q = validate_distribution(second)

    if p.shape != q.shape:
        raise ValueError(
            "Distribution shapes must match."
        )

    midpoint = 0.5 * (p + q)

    def kl_divergence(
        distribution: np.ndarray,
    ) -> float:
        positive = distribution > 0.0

        return float(
            np.sum(
                distribution[positive]
                * np.log(
                    distribution[positive]
                    / midpoint[positive]
                )
            )
        )

    return float(
        0.5 * kl_divergence(p)
        + 0.5 * kl_divergence(q)
    )


def compare_distributions(
    first: np.ndarray,
    second: np.ndarray,
) -> dict[str, float]:
    return {
        "tvd": total_variation_distance(
            first,
            second,
        ),
        "jsd": jensen_shannon_divergence(
            first,
            second,
        ),
    }


def parse_pixel(
    value: str,
) -> tuple[int, int]:
    """Parse one pixel in x,y format."""

    parts = value.split(",")

    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            "Pixel must have format x,y."
        )

    try:
        x = int(parts[0])
        y = int(parts[1])
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Pixel coordinates must be integers."
        ) from error

    if x < 0 or y < 0:
        raise argparse.ArgumentTypeError(
            "Pixel coordinates cannot be negative."
        )

    return x, y


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare independent T2 teacher estimates "
            "at fixed Mitsuba shading points."
        )
    )

    parser.add_argument(
        "--scene",
        type=Path,
        default=Path(
            "scenes/simple_diffuse/scene.xml"
        ),
    )

    parser.add_argument(
        "--pixels",
        nargs="+",
        type=parse_pixel,
        default=list(DEFAULT_PIXELS),
    )

    parser.add_argument(
        "--budgets",
        nargs="+",
        choices=[budget.name for budget in BUDGETS],
        default=[
            "low",
            "medium",
            "high",
        ],
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--n-mu",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--n-phi",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--rr-depth",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--smoothing",
        type=float,
        default=1e-6,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/experiments/t2_teacher_stability"
        ),
    )

    return parser.parse_args()


def create_teacher(
    budget: Budget,
    *,
    smoothing: float,
    max_depth: int,
    rr_depth: int,
) -> PhysicalTeacherSettings:
    return PhysicalTeacherSettings(
        target_type=PRODUCT_INTEGRAND_TARGET,
        samples_per_bin=budget.samples_per_bin,
        radiance_samples=budget.radiance_samples,
        smoothing=smoothing,
        max_depth=max_depth,
        rr_depth=rr_depth,
    )


def evaluate_target(
    *,
    context: TeacherContext,
    bins: HemisphereBins,
    budget: Budget,
    seed: int,
    smoothing: float,
    max_depth: int,
    rr_depth: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the teacher using independent local RNG state."""

    teacher = create_teacher(
        budget,
        smoothing=smoothing,
        max_depth=max_depth,
        rr_depth=rr_depth,
    )

    rng = np.random.default_rng(seed)

    runtime = create_teacher_runtime(
        teacher=teacher,
        seed=seed,
    )

    contributions = estimate_bin_contributions(
        context=context,
        bins=bins,
        teacher=teacher,
        rng=rng,
        runtime=runtime,
    )

    target = contributions_to_target_distribution(
        contributions=contributions,
        smoothing=smoothing,
    )

    return (
        np.asarray(target, dtype=np.float64),
        np.asarray(contributions, dtype=np.float64),
    )


def make_independent_seeds(
    base_seed: int,
    count: int,
) -> list[int]:
    """Produce reproducible, separate seeds for experiments."""

    if base_seed < 0:
        raise ValueError(
            "base_seed must be nonnegative."
        )

    sequence = np.random.SeedSequence(base_seed)
    children = sequence.spawn(count)

    return [
        int(
            child.generate_state(
                1,
                dtype=np.uint32,
            )[0]
        )
        for child in children
    ]


def add_comparison(
    records: list[dict[str, Any]],
    *,
    pixel: tuple[int, int],
    comparison: str,
    first: np.ndarray,
    second: np.ndarray,
) -> None:
    metrics = compare_distributions(
        first,
        second,
    )

    records.append({
        "pixel_x": pixel[0],
        "pixel_y": pixel[1],
        "comparison": comparison,
        **metrics,
    })


def write_records(
    output_path: Path,
    records: list[dict[str, Any]],
) -> None:
    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "pixel_x",
                "pixel_y",
                "comparison",
                "tvd",
                "jsd",
            ],
        )

        writer.writeheader()
        writer.writerows(records)


def summarize_comparisons(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Aggregate metrics for each comparison type."""

    names = sorted({
        record["comparison"]
        for record in records
    })

    result: dict[str, dict[str, float]] = {}

    for name in names:
        group = [
            record
            for record in records
            if record["comparison"] == name
        ]

        tvds = np.array(
            [record["tvd"] for record in group],
            dtype=np.float64,
        )

        jsds = np.array(
            [record["jsd"] for record in group],
            dtype=np.float64,
        )

        result[name] = {
            "count": len(group),
            "mean_tvd": float(np.mean(tvds)),
            "max_tvd": float(np.max(tvds)),
            "mean_jsd": float(np.mean(jsds)),
            "max_jsd": float(np.max(jsds)),
        }

    return result


def main() -> None:
    args = parse_arguments()

    if not args.scene.is_file():
        raise FileNotFoundError(args.scene)

    if args.seed < 0:
        raise ValueError(
            "Seed must be nonnegative."
        )

    if not args.pixels:
        raise ValueError(
            "At least one pixel is required."
        )

    if len(set(args.pixels)) != len(args.pixels):
        raise ValueError(
            "Pixel coordinates must not be repeated."
        )

    if args.rr_depth > args.max_depth:
        raise ValueError(
            "rr_depth cannot exceed max_depth."
        )

    mi.set_variant("scalar_rgb")
    scene = cast(Any, mi.load_file)(
        str(args.scene.resolve())
    )

    bins = HemisphereBins(
        n_mu=args.n_mu,
        n_phi=args.n_phi,
    )

    selected_budgets = [
        budget
        for budget in BUDGETS
        if budget.name in args.budgets
    ]

    # Two independent estimates for each selected budget.
    experiment_count = (
        len(args.pixels)
        * len(selected_budgets)
        * 2
    )

    seeds = iter(
        make_independent_seeds(
            args.seed,
            experiment_count,
        )
    )

    records: list[dict[str, Any]] = []

    stored_targets: dict[str, np.ndarray] = {}
    stored_contributions: dict[str, np.ndarray] = {}

    print("T2 teacher stability validation started.")
    print(f"Scene: {args.scene}")
    print(f"Pixels: {args.pixels}")
    print(f"Bins: {bins.n_bins}")
    print(
        "Budgets: "
        + ", ".join(
            budget.name
            for budget in selected_budgets
        )
    )

    for pixel in args.pixels:
        print()
        print(f"=== PIXEL {pixel} ===")

        camera_ray, interaction, _ = sample_fixed_surface(
            scene,
            pixel[0],
            pixel[1],
        )

        normal = normalize_numpy_vector(
            mitsuba_vector_to_numpy(
                interaction.n
            )
        )

        outgoing = normalize_numpy_vector(
            -mitsuba_vector_to_numpy(
                camera_ray.d
            )
        )

        context = TeacherContext(
            scene=scene,
            surface_interaction=interaction,
            normal=normal,
            outgoing_direction=outgoing,
        )

        pixel_targets: dict[
            str,
            tuple[np.ndarray, np.ndarray]
        ] = {}

        for budget in selected_budgets:
            replicas: list[np.ndarray] = []

            for replica_index in range(2):
                seed = next(seeds)

                print(
                    f"  {budget.name} "
                    f"replica={replica_index + 1} "
                    f"seed={seed}"
                )

                target, contributions = evaluate_target(
                    context=context,
                    bins=bins,
                    budget=budget,
                    seed=seed,
                    smoothing=args.smoothing,
                    max_depth=args.max_depth,
                    rr_depth=args.rr_depth,
                )

                validate_distribution(target)

                key = (
                    f"pixel_{pixel[0]}_{pixel[1]}"
                    f"_{budget.name}"
                    f"_replica_{replica_index + 1}"
                )

                stored_targets[key] = target
                stored_contributions[key] = contributions

                replicas.append(target)

            pixel_targets[budget.name] = (
                replicas[0],
                replicas[1],
            )

            add_comparison(
                records,
                pixel=pixel,
                comparison=(
                    f"{budget.name}_replica_difference"
                ),
                first=replicas[0],
                second=replicas[1],
            )

        # Compare the same replica number across budgets.
        # Replicas are independent across budgets by design.
        for first_index, first_budget in enumerate(
            selected_budgets
        ):
            for second_budget in selected_budgets[
                first_index + 1:
            ]:
                first_replicas = pixel_targets[
                    first_budget.name
                ]

                second_replicas = pixel_targets[
                    second_budget.name
                ]

                for replica_index in range(2):
                    add_comparison(
                        records,
                        pixel=pixel,
                        comparison=(
                            f"{first_budget.name}_vs_"
                            f"{second_budget.name}"
                        ),
                        first=first_replicas[
                            replica_index
                        ],
                        second=second_replicas[
                            replica_index
                        ],
                    )

    summary = summarize_comparisons(records)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_records(
        args.output_dir
        / "teacher_stability_records.csv",
        records,
    )

    np.savez_compressed(
        args.output_dir / "teacher_targets.npz",
        **cast(
            Any,
            {
                **{
                    f"target_{key}": value
                    for key, value in stored_targets.items()
                },
                **{
                    f"contributions_{key}": value
                    for key, value in stored_contributions.items()
                },
            },
        ),
    )

    manifest = {
        "experiment": "t2_teacher_stability_v1",
        "scene": str(args.scene.resolve()),
        "mitsuba_version": mi.__version__,
        "seed": args.seed,
        "pixels": [
            list(pixel)
            for pixel in args.pixels
        ],
        "n_mu": args.n_mu,
        "n_phi": args.n_phi,
        "smoothing": args.smoothing,
        "max_depth": args.max_depth,
        "rr_depth": args.rr_depth,
        "budgets": [
            {
                "name": budget.name,
                "samples_per_bin": budget.samples_per_bin,
                "radiance_samples": budget.radiance_samples,
            }
            for budget in selected_budgets
        ],
        "replicas_per_budget": 2,
        "comparisons": summary,
        "notes": [
            "All budgets use the same fixed shading points.",
            "Replicas and budgets use independent random seeds.",
            "TVD and JSD compare estimated target PMFs.",
            "High budget is not an exact ground truth.",
            "These experiments do not establish unbiasedness.",
        ],
    }

    with (
        args.output_dir
        / "teacher_stability_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
        )

    print()
    print("=== T2 STABILITY SUMMARY ===")

    for name, values in summary.items():
        print(
            f"{name:32s} "
            f"mean_TVD={values['mean_tvd']:.6f} "
            f"max_TVD={values['max_tvd']:.6f} "
            f"mean_JSD={values['mean_jsd']:.6f}"
        )

    print()
    print(f"Output: {args.output_dir.resolve()}")
    print("T2 teacher stability validation completed.")


if __name__ == "__main__":
    main()