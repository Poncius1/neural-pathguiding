"""Run three-technique MIS experiments across independent seeds.

Produces:
    Individual experiment directories with NPZ and JSON.
    convergence_summary.csv
    convergence_summary.json

Example:
    python -u scripts/run_mis_convergence.py 
        --seeds 42 123 456 789 2026 
        --repetitions 1000 
        --reference-samples 10000
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

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


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 456, 789, 2026],
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--reference-samples",
        type=int,
        default=10000,
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experiments"),
    )

    return parser.parse_args()


def load_manifest(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def execute_seed(
    *,
    args,
    seed: int,
    output_dir: Path,
):
    script_path = (
        Path(__file__).resolve().parent
        / "compare_three_technique_mis.py"
    )

    # Each seed has its own output directory.
    seed_output = output_dir / f"seed_{seed}"
    seed_output.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-u",
        str(script_path),
        "--scene", str(args.scene),
        "--checkpoint", str(args.checkpoint),
        "--pixel-x", str(args.pixel_x),
        "--pixel-y", str(args.pixel_y),
        "--repetitions", str(args.repetitions),
        "--reference-samples", str(args.reference_samples),
        "--seed", str(seed),
        "--output-dir", str(seed_output),
    ]

    print()
    print(f"=== RUNNING SEED {seed} ===", flush=True)

    completed = subprocess.run(
        command,
        check=False,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            f"Seed {seed} failed with return code "
            f"{completed.returncode}."
        )

    manifests = sorted(
        seed_output.glob("three_mis_*/manifest.json")
    )

    if len(manifests) != 1:
        raise RuntimeError(
            f"Expected one manifest for seed {seed}, "
            f"found {len(manifests)}."
        )

    return load_manifest(manifests[0])


def manifest_to_rows(manifest):
    rows = []
    seed = manifest["seed"]
    reference = np.asarray(
        manifest["reference_mean_rgb"],
        dtype=np.float64,
    )
    reference_se = np.asarray(
        manifest["reference_standard_error_rgb"],
        dtype=np.float64,
    )

    reference_luminance = float(
        reference @ LUMINANCE_WEIGHTS
    )

    # The reference SE assumes independent RGB samples.
    # Its luminance SE cannot be recovered from component
    # SEs alone because RGB channels can be correlated.
    for name in ESTIMATORS:
        metrics = manifest["estimators"][name]

        mean_rgb = np.asarray(
            metrics["mean_rgb"],
            dtype=np.float64,
        )
        estimate_se_rgb = np.asarray(
            metrics["standard_error_rgb"],
            dtype=np.float64,
        )

        mean_luminance = float(
            mean_rgb @ LUMINANCE_WEIGHTS
        )

        rows.append({
            "seed": seed,
            "estimator": name,
            "repetitions": manifest["repetitions"],
            "mean_luminance": mean_luminance,
            "reference_luminance": reference_luminance,
            "difference_luminance": (
                mean_luminance - reference_luminance
            ),
            "luminance_variance": float(
                metrics["luminance_variance"]
            ),
            "rgb_mean_se_max": float(
                np.max(estimate_se_rgb)
            ),
            "reference_rgb_se_max": float(
                np.max(reference_se)
            ),
            "seconds": float(metrics["seconds"]),
            "seconds_per_estimate": float(
                metrics["seconds_per_estimate"]
            ),
            "rmse_rgb": float(metrics["rmse_rgb"]),
        })

    return rows


def summarize_across_seeds(rows):
    summary = {}

    for estimator in ESTIMATORS:
        selected = [
            row for row in rows
            if row["estimator"] == estimator
        ]

        variances = np.asarray(
            [row["luminance_variance"] for row in selected],
            dtype=np.float64,
        )
        differences = np.asarray(
            [row["difference_luminance"] for row in selected],
            dtype=np.float64,
        )
        times = np.asarray(
            [row["seconds_per_estimate"] for row in selected],
            dtype=np.float64,
        )

        summary[estimator] = {
            "seeds": len(selected),
            "mean_luminance_variance": float(
                variances.mean()
            ),
            "std_luminance_variance_across_seeds": (
                float(variances.std(ddof=1))
                if len(variances) > 1
                else None
            ),
            "mean_difference_from_reference": float(
                differences.mean()
            ),
            "mean_absolute_difference_from_reference": float(
                np.abs(differences).mean()
            ),
            "mean_seconds_per_estimate": float(
                times.mean()
            ),
            "mean_variance_times_seconds": float(
                np.mean(variances * times)
            ),
        }

    return summary


def main():
    args = parse_arguments()

    if args.repetitions < 2:
        raise ValueError("--repetitions must be >= 2.")

    if args.reference_samples < 2:
        raise ValueError("--reference-samples must be >= 2.")

    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must be unique.")

    if any(seed < 0 for seed in args.seeds):
        raise ValueError("Seeds must be non-negative.")

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )

    run_dir = (
        args.output_dir
        / f"mis_convergence_{timestamp}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    all_rows = []
    completed_seeds = []

    for seed in args.seeds:
        manifest = execute_seed(
            args=args,
            seed=seed,
            output_dir=run_dir,
        )

        all_rows.extend(
            manifest_to_rows(manifest)
        )
        completed_seeds.append(seed)

    csv_path = run_dir / "convergence_summary.csv"

    with csv_path.open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(all_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(all_rows)

    summary = summarize_across_seeds(all_rows)

    report = {
        "experiment": "three_technique_mis_convergence",
        "timestamp_utc": timestamp,
        "seeds": completed_seeds,
        "repetitions_per_seed": args.repetitions,
        "reference_samples_per_seed": (
            args.reference_samples
        ),
        "pixel": [args.pixel_x, args.pixel_y],
        "summary": summary,
        "notes": [
            "Each seed has an independently estimated reference.",
            "Reported reference differences are not proof of bias.",
            "Equal proposal sample counts do not imply equal time.",
            "Mean variance-times-time is an exploratory efficiency metric.",
            "This is a fixed-point direct-light experiment.",
            "No image-level or multibounce validation is implied.",
        ],
    }

    json_path = run_dir / "convergence_summary.json"

    with json_path.open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=2)

    print()
    print("=== CONVERGENCE SUMMARY ===")

    print(
        f"{'Estimator':<28}"
        f"{'Mean variance':>17}"
        f"{'Mean abs diff':>17}"
        f"{'Seconds/estimate':>19}"
    )

    for name, values in summary.items():
        print(
            f"{name:<28}"
            f"{values['mean_luminance_variance']:>17.6e}"
            f"{values['mean_absolute_difference_from_reference']:>17.6e}"
            f"{values['mean_seconds_per_estimate']:>19.6e}"
        )

    print()
    print(f"CSV: {csv_path.resolve()}")
    print(f"JSON: {json_path.resolve()}")
    print("Convergence campaign completed.")


if __name__ == "__main__":
    main()