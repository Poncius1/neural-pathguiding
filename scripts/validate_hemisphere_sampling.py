"""Validate hemisphere sampling against analytical integrals."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from neural_path_guiding.core.bins import HemisphereBins
from neural_path_guiding.core.directional_distribution import (
    DiscreteDirectionalDistribution,
)


ANALYTICAL_INTEGRALS = np.array([
    2.0 * math.pi,
    math.pi,
    2.0 * math.pi / 3.0,
])

INTEGRAND_NAMES = (
    "constant",
    "cosine",
    "cosine_squared",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--samples",
        type=int,
        nargs="+",
        default=[128, 512, 2048, 8192],
    )

    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/experiments/hemisphere_validation"),
    )

    return parser.parse_args()


def sample_discrete(
    distribution: DiscreteDirectionalDistribution,
    count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:

    directions = np.empty((count, 3))
    pdfs = np.empty(count)

    for index in range(count):
        sample = distribution.sample(rng)
        directions[index] = sample.direction_local
        pdfs[index] = sample.pdf

    return directions, pdfs


def sample_analytical(
    method: str,
    count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:

    u = rng.random(count)
    v = rng.random(count)

    phi = 2.0 * math.pi * v

    if method == "cosine_weighted":
        mu = np.sqrt(u)
        pdfs = mu / math.pi

    elif method == "naive_angular":
        theta = (math.pi / 2.0) * u
        mu = np.cos(theta)

        # Actual PDF with respect to solid angle.
        # This proposal is deliberately not uniform.
        sin_theta = np.sin(theta)
        sin_theta = np.maximum(
            sin_theta,
            np.finfo(np.float64).tiny,
        )

        pdfs = 1.0 / (
            math.pi**2 * sin_theta
        )

    else:
        raise ValueError(f"Unknown method: {method}")

    radius = np.sqrt(
        np.maximum(0.0, 1.0 - mu**2)
    )

    directions = np.column_stack((
        radius * np.cos(phi),
        radius * np.sin(phi),
        mu,
    ))

    return directions, pdfs


def evaluate_integrals(
    directions: np.ndarray,
    pdfs: np.ndarray,
) -> np.ndarray:

    mu = directions[:, 2]

    values = np.column_stack((
        np.ones_like(mu),
        mu,
        mu**2,
    ))

    if not np.all(np.isfinite(pdfs)) or np.any(pdfs <= 0):
        raise ValueError("Invalid sampling PDF")

    return np.mean(
        values / pdfs[:, None],
        axis=0,
    )


def main() -> None:
    args = parse_arguments()

    if min(args.samples) <= 0:
        raise ValueError("Sample counts must be positive")

    if args.repetitions < 2:
        raise ValueError("At least two repetitions required")

    bins = HemisphereBins(n_mu=4, n_phi=8)

    uniform = DiscreteDirectionalDistribution(
        bins=bins,
        probabilities=np.ones(bins.n_bins),
    )

    nonuniform = DiscreteDirectionalDistribution(
        bins=bins,
        probabilities=np.linspace(
            0.5,
            2.0,
            bins.n_bins,
        ),
        uniform_mix=0.05,
    )

    methods = {
        "uniform_bins": lambda n, rng: sample_discrete(
            uniform, n, rng
        ),
        "nonuniform_bins": lambda n, rng: sample_discrete(
            nonuniform, n, rng
        ),
        "cosine_weighted": lambda n, rng: sample_analytical(
            "cosine_weighted", n, rng
        ),
        "naive_angular": lambda n, rng: sample_analytical(
            "naive_angular", n, rng
        ),
    }

    args.output.mkdir(parents=True, exist_ok=True)

    rows = []
    histogram_data = {}

    for method_index, (name, sampler) in enumerate(
        methods.items()
    ):
        for count in sorted(set(args.samples)):

            estimates = np.empty(
                (args.repetitions, 3),
                dtype=np.float64,
            )

            for repetition in range(args.repetitions):

                seed = np.random.SeedSequence([
                    args.seed,
                    method_index,
                    count,
                    repetition,
                ])

                rng = np.random.default_rng(seed)

                directions, pdfs = sampler(count, rng)

                estimates[repetition] = evaluate_integrals(
                    directions,
                    pdfs,
                )

            mean = estimates.mean(axis=0)
            variance = estimates.var(axis=0, ddof=1)

            standard_error = np.sqrt(
                variance / args.repetitions
            )

            mse = np.mean(
                (estimates - ANALYTICAL_INTEGRALS) ** 2,
                axis=0,
            )

            for index, integrand in enumerate(
                INTEGRAND_NAMES
            ):
                rows.append({
                    "method": name,
                    "samples": count,
                    "repetitions": args.repetitions,
                    "integrand": integrand,
                    "reference": float(
                        ANALYTICAL_INTEGRALS[index]
                    ),
                    "mean": float(mean[index]),
                    "variance": float(variance[index]),
                    "standard_error": float(
                        standard_error[index]
                    ),
                    "mse": float(mse[index]),
                })

            print(
                f"{name:20s} N={count:6d} "
                f"cosine MSE={mse[1]:.6e}",
                flush=True,
            )

        rng = np.random.default_rng(
            np.random.SeedSequence([
                args.seed,
                method_index,
                999,
            ])
        )

        directions, _ = sampler(8192, rng)

        counts, edges = np.histogram(
            directions[:, 2],
            bins=20,
            range=(0.0, 1.0),
        )

        histogram_data[name] = {
            "directions": directions,
            "mu_counts": counts,
            "mu_edges": edges,
        }

    with (args.output / "results.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    np.savez_compressed(
        args.output / "sampling_distributions.npz",
        **{
            f"{method}_{key}": array
            for method, data in histogram_data.items()
            for key, array in data.items()
        },
    )

    manifest = {
        "experiment": "analytical_hemisphere_validation",
        "seed": args.seed,
        "samples": sorted(set(args.samples)),
        "repetitions": args.repetitions,
        "bins": {"n_mu": 4, "n_phi": 8},
        "methods": list(methods),
        "analytical_integrals": dict(zip(
            INTEGRAND_NAMES,
            ANALYTICAL_INTEGRALS.tolist(),
        )),
        "results": rows,
    }

    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(f"\nResults: {args.output.resolve()}")


if __name__ == "__main__":
    main()