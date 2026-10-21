"""Validate fixed-point MIS convergence across budgets and pixels.

Mitsuba 3.8.0, scalar_rgb.

Runs the existing compare_three_technique_mis.py script without
changing its estimators, reads numerical samples from estimates.npz,
and produces CSV/JSON reports.

Example:
    python -u scripts/validate_mis_convergence.py \
        --pixels 481,440 \
        --budgets 100 1000 5000 \
        --seeds 42 123 2026 \
        --reference-samples 50000
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


ESTIMATORS = (
    "neural_3",
    "bsdf_3",
    "light_3",
    "neural_bsdf_light_mis_3",
)

LUMINANCE_WEIGHTS = np.array(
    [0.2126, 0.7152, 0.0722],
    dtype=np.float64,
)


def parse_pixel(value: str) -> tuple[int, int]:
    """Parse a pixel argument written as x,y."""

    try:
        parts = value.split(",")

        if len(parts) != 2:
            raise ValueError

        x, y = (int(part.strip()) for part in parts)

    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid pixel {value!r}. Expected x,y."
        ) from exc

    if x < 0 or y < 0:
        raise argparse.ArgumentTypeError(
            "Pixel coordinates must be non-negative."
        )

    return x, y


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate fixed-point MIS convergence."
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

    parser.add_argument(
        "--pixels",
        nargs="+",
        type=parse_pixel,
        default=[(481, 440)],
        help="Camera pixels as x,y, e.g. 481,440 450,420.",
    )

    parser.add_argument(
        "--budgets",
        nargs="+",
        type=int,
        default=[100, 1000, 5000],
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 2026],
    )

    parser.add_argument(
        "--reference-samples",
        type=int,
        default=50000,
    )

    parser.add_argument(
        "--uniform-mix",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experiments"),
    )

    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    if any(value < 2 for value in args.budgets):
        raise ValueError("Every budget must be >= 2.")

    if args.reference_samples < 2:
        raise ValueError("--reference-samples must be >= 2.")

    if any(seed < 0 for seed in args.seeds):
        raise ValueError("Seeds must be non-negative.")

    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must be unique.")

    if len(set(args.budgets)) != len(args.budgets):
        raise ValueError("Budgets must be unique.")

    if len(set(args.pixels)) != len(args.pixels):
        raise ValueError("Pixels must be unique.")

    if not 0.0 <= args.uniform_mix <= 1.0:
        raise ValueError("--uniform-mix must be in [0, 1].")

    if not args.scene.is_file():
        raise FileNotFoundError(args.scene)

    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)


def luminance_samples(rgb_samples: np.ndarray) -> np.ndarray:
    """Convert an Nx3 RGB sample array to luminance samples."""

    samples = np.asarray(rgb_samples, dtype=np.float64)

    if (
        samples.ndim != 2
        or samples.shape[1] != 3
        or samples.shape[0] < 2
    ):
        raise ValueError(
            "Expected at least two RGB samples with shape (N, 3)."
        )

    if not np.all(np.isfinite(samples)):
        raise ValueError("Non-finite RGB samples.")

    return samples @ LUMINANCE_WEIGHTS


def sample_statistics(values: np.ndarray) -> dict[str, float]:
    """Compute mean, sample variance and mean standard error."""

    values = np.asarray(values, dtype=np.float64)

    if values.ndim != 1 or len(values) < 2:
        raise ValueError("Expected at least two scalar samples.")

    if not np.all(np.isfinite(values)):
        raise ValueError("Non-finite scalar samples.")

    variance = float(np.var(values, ddof=1))

    return {
        "mean": float(np.mean(values)),
        "variance": variance,
        "standard_error": float(
            np.sqrt(variance / len(values))
        ),
    }


def compare_with_reference(
    estimator_values: np.ndarray,
    reference_values: np.ndarray,
) -> dict[str, float | bool | None]:
    """Compare independent estimator and reference sample sets.

    The z-like score uses estimated standard errors and should be
    interpreted diagnostically, not as proof of unbiasedness.
    """

    estimate = sample_statistics(
        luminance_samples(estimator_values)
    )

    reference = sample_statistics(
        luminance_samples(reference_values)
    )

    difference = estimate["mean"] - reference["mean"]

    combined_se = float(
        np.hypot(
            estimate["standard_error"],
            reference["standard_error"],
        )
    )

    z_score = (
        float(difference / combined_se)
        if combined_se > 0.0
        else None
    )

    return {
        "mean_luminance": estimate["mean"],
        "variance_luminance": estimate["variance"],
        "standard_error_luminance": estimate["standard_error"],
        "reference_luminance": reference["mean"],
        "reference_standard_error_luminance": (
            reference["standard_error"]
        ),
        "difference_luminance": float(difference),
        "absolute_difference_luminance": float(abs(difference)),
        "combined_standard_error": combined_se,
        "z_score": z_score,
        "within_1_96_combined_se": (
            bool(abs(z_score) <= 1.96)
            if z_score is not None
            else bool(difference == 0.0)
        ),
    }


def execute_experiment(
    *,
    script_path: Path,
    scene: Path,
    checkpoint: Path,
    pixel: tuple[int, int],
    budget: int,
    seed: int,
    reference_samples: int,
    uniform_mix: float,
    output_dir: Path,
) -> Path:
    """Run one experiment and return its manifest path."""

    output_dir.mkdir(parents=True, exist_ok=False)

    command = [
        sys.executable,
        "-u",
        str(script_path),
        "--scene",
        str(scene),
        "--checkpoint",
        str(checkpoint),
        "--pixel-x",
        str(pixel[0]),
        "--pixel-y",
        str(pixel[1]),
        "--repetitions",
        str(budget),
        "--reference-samples",
        str(reference_samples),
        "--uniform-mix",
        str(uniform_mix),
        "--seed",
        str(seed),
        "--output-dir",
        str(output_dir),
    ]

    print(
        f"\n=== PIXEL {pixel} | "
        f"BUDGET {budget} | SEED {seed} ===",
        flush=True,
    )

    completed = subprocess.run(
        command,
        check=False,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            f"Experiment failed at pixel {pixel}, "
            f"budget {budget}, seed {seed}. "
            f"Exit code: {completed.returncode}."
        )

    manifests = list(
        output_dir.glob("three_mis_*/manifest.json")
    )

    if len(manifests) != 1:
        raise RuntimeError(
            f"Expected exactly one manifest, found "
            f"{len(manifests)} in {output_dir}."
        )

    return manifests[0]


def extract_rows(
    manifest_path: Path,
) -> list[dict[str, Any]]:
    """Read manifest and raw samples for one completed run."""

    with manifest_path.open(
        "r", encoding="utf-8"
    ) as file:
        manifest = json.load(file)

    npz_path = manifest_path.parent / "estimates.npz"

    rows: list[dict[str, Any]] = []

    with np.load(npz_path, allow_pickle=False) as arrays:
        reference = arrays["reference_samples"]

        for name in ESTIMATORS:
            samples = arrays[name]

            if len(samples) != manifest["repetitions"]:
                raise RuntimeError(
                    f"Unexpected sample count for {name}."
                )

            comparison = compare_with_reference(
                samples,
                reference,
            )

            metrics = manifest["estimators"][name]

            diagnostics = metrics.get(
                "emitter_pdf_diagnostics",
                {},
            )

            rows.append({
                "pixel_x": int(manifest["pixel"][0]),
                "pixel_y": int(manifest["pixel"][1]),
                "budget": int(manifest["repetitions"]),
                "seed": int(manifest["seed"]),
                "estimator": name,
                **comparison,
                "seconds_per_estimate": float(
                    metrics["seconds_per_estimate"]
                ),
                "variance_times_seconds": float(
                    comparison["variance_luminance"]
                    * metrics["seconds_per_estimate"]
                ),
                "fallback_queries": diagnostics.get(
                    "fallback_queries"
                ),
                "total_pdf_queries": diagnostics.get(
                    "total_queries"
                ),
                "manifest_path": str(
                    manifest_path.resolve()
                ),
            })

    return rows


def summarize_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate results across seeds for each pixel and budget."""

    keys = sorted({
        (
            row["pixel_x"],
            row["pixel_y"],
            row["budget"],
            row["estimator"],
        )
        for row in rows
    })

    summary: list[dict[str, Any]] = []

    for pixel_x, pixel_y, budget, name in keys:
        selected = [
            row for row in rows
            if (
                row["pixel_x"] == pixel_x
                and row["pixel_y"] == pixel_y
                and row["budget"] == budget
                and row["estimator"] == name
            )
        ]

        def average(field: str) -> float:
            return float(
                np.mean([
                    float(row[field])
                    for row in selected
                ])
            )

        valid_z = [
            abs(float(row["z_score"]))
            for row in selected
            if row["z_score"] is not None
        ]

        fallbacks = [
            int(row["fallback_queries"])
            for row in selected
            if row["fallback_queries"] is not None
        ]

        summary.append({
            "pixel_x": pixel_x,
            "pixel_y": pixel_y,
            "budget": budget,
            "estimator": name,
            "seed_count": len(selected),
            "mean_variance": average(
                "variance_luminance"
            ),
            "mean_standard_error": average(
                "standard_error_luminance"
            ),
            "mean_absolute_difference": average(
                "absolute_difference_luminance"
            ),
            "mean_signed_difference": average(
                "difference_luminance"
            ),
            "mean_seconds_per_estimate": average(
                "seconds_per_estimate"
            ),
            "mean_variance_times_seconds": average(
                "variance_times_seconds"
            ),
            "max_abs_z_score": (
                max(valid_z) if valid_z else None
            ),
            "fraction_within_1_96_se": float(
                np.mean([
                    row["within_1_96_combined_se"]
                    for row in selected
                ])
            ),
            "fallback_queries_total": (
                sum(fallbacks) if fallbacks else None
            ),
        })

    return summary


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        raise ValueError("Cannot write an empty report.")

    with path.open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_arguments()
    validate_arguments(args)

    script_path = (
        Path(__file__).resolve().parent
        / "compare_three_technique_mis.py"
    )

    if not script_path.is_file():
        raise FileNotFoundError(script_path)

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )

    run_dir = (
        args.output_dir
        / f"mis_validation_{timestamp}"
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    rows: list[dict[str, Any]] = []

    expected_runs = (
        len(args.pixels)
        * len(args.budgets)
        * len(args.seeds)
    )

    print("MIS convergence validation started.")
    print(f"Expected experiments: {expected_runs}")

    for pixel in args.pixels:
        for budget in sorted(args.budgets):
            for seed in args.seeds:
                experiment_dir = (
                    run_dir
                    / f"pixel_{pixel[0]}_{pixel[1]}"
                    / f"budget_{budget}"
                    / f"seed_{seed}"
                )

                manifest_path = execute_experiment(
                    script_path=script_path,
                    scene=args.scene,
                    checkpoint=args.checkpoint,
                    pixel=pixel,
                    budget=budget,
                    seed=seed,
                    reference_samples=args.reference_samples,
                    uniform_mix=args.uniform_mix,
                    output_dir=experiment_dir,
                )

                rows.extend(
                    extract_rows(manifest_path)
                )

    summary = summarize_rows(rows)

    raw_csv = run_dir / "convergence_detailed.csv"
    summary_csv = run_dir / "convergence_summary.csv"
    summary_json = run_dir / "convergence_summary.json"

    write_csv(raw_csv, rows)
    write_csv(summary_csv, summary)

    report = {
        "experiment": "mis_fixed_point_budget_validation",
        "timestamp_utc": timestamp,
        "scene": str(args.scene.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "pixels": [list(pixel) for pixel in args.pixels],
        "budgets": sorted(args.budgets),
        "seeds": args.seeds,
        "reference_samples_per_run": (
            args.reference_samples
        ),
        "uniform_mix": args.uniform_mix,
        "completed_runs": expected_runs,
        "summary": summary,
        "limitations": [
            "Fixed-point direct-light experiment only.",
            "One sample per MIS technique.",
            "Reference is estimated independently per run.",
            "Repeated budgets with the same seed are correlated.",
            "Z scores use estimated Monte Carlo uncertainty.",
            "Compatibility with reference does not prove unbiasedness.",
            "Equal sample budgets do not imply equal render time.",
            "Some pixels may not correspond to valid surfaces.",
            "Neural checkpoint may not generalize to new pixels.",
        ],
    }

    with summary_json.open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=2)

    print("\n=== CONVERGENCE VALIDATION SUMMARY ===")

    print(
        f"{'Budget':>7} "
        f"{'Estimator':<27} "
        f"{'Mean variance':>14} "
        f"{'Mean abs diff':>14} "
        f"{'Mean SE':>12} "
        f"{'Max |z|':>9}"
    )

    for record in summary:
        z_value = record["max_abs_z_score"]

        z_text = (
            f"{z_value:.3f}"
            if z_value is not None
            else "N/A"
        )

        print(
            f"{record['budget']:>7} "
            f"{record['estimator']:<27} "
            f"{record['mean_variance']:>14.6e} "
            f"{record['mean_absolute_difference']:>14.6e} "
            f"{record['mean_standard_error']:>12.6e} "
            f"{z_text:>9}"
        )

    print()
    print(f"Detailed CSV: {raw_csv.resolve()}")
    print(f"Summary CSV: {summary_csv.resolve()}")
    print(f"Summary JSON: {summary_json.resolve()}")
    print("MIS convergence validation completed.", flush=True)


if __name__ == "__main__":
    main()