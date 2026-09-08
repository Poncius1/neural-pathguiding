"""Compare discrete uniform and neural importance sampling variance."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from neural_path_guiding.core.directional_distribution import (
    DiscreteDirectionalDistribution,
)
from neural_path_guiding.data.dataset import load_dataset
from neural_path_guiding.data.schema import NeuralGuidingDataset
from neural_path_guiding.evaluation.discrete_variance import (
    DiscreteEstimatorStatistics,
    calculate_discrete_estimator_statistics,
    simulate_discrete_estimates,
)
from neural_path_guiding.training.inference import (
    GuidingInferenceRuntime,
)


@dataclass(frozen=True)
class MethodResult:
    """Analytical and empirical results for one proposal."""

    name: str
    statistics: DiscreteEstimatorStatistics
    empirical_mean: float
    empirical_variance: float
    empirical_rmse: float


def parse_arguments() -> argparse.Namespace:
    """Read experiment settings from the command line."""

    parser = argparse.ArgumentParser(
        description=(
            "Compare uniform and neural proposals "
            "on one discretized T2 target."
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to the T2 NPZ dataset.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the trained guiding checkpoint.",
    )
    parser.add_argument(
        "--row",
        type=int,
        default=0,
        help="Dataset row used as the discrete integrand.",
    )
    parser.add_argument(
        "--samples-per-estimate",
        type=int,
        default=8,
        help="Number of samples in each Monte Carlo estimate.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=10000,
        help="Number of independent estimates.",
    )
    parser.add_argument(
        "--uniform-mix",
        type=float,
        default=0.05,
        help="Uniform support mixed with the neural proposal.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for the Monte Carlo simulations.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Device used for MLP inference.",
    )

    return parser.parse_args()


def validate_experiment(
    arguments: argparse.Namespace,
    dataset: NeuralGuidingDataset,
    runtime: GuidingInferenceRuntime,
) -> None:
    """Validate experiment settings and data compatibility."""

    if arguments.row < 0 or arguments.row >= dataset.num_samples:
        raise ValueError(
            "row must be between 0 and "
            f"{dataset.num_samples - 1}."
        )

    if arguments.samples_per_estimate <= 0:
        raise ValueError(
            "samples-per-estimate must be positive."
        )

    if arguments.repetitions < 2:
        raise ValueError(
            "repetitions must be at least 2."
        )

    if arguments.seed < 0:
        raise ValueError(
            "seed must be non-negative."
        )

    if (
        arguments.device == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA was requested, but it is not available."
        )

    if dataset.features.shape[1] != runtime.feature_dimension:
        raise ValueError(
            "Dataset feature dimension does not match "
            "the checkpoint."
        )

    if dataset.targets.shape[1] != runtime.num_bins:
        raise ValueError(
            "Dataset bin count does not match "
            "the checkpoint."
        )

    dataset_n_mu = int(
        dataset.metadata["n_mu"]
    )
    dataset_n_phi = int(
        dataset.metadata["n_phi"]
    )

    if (
        dataset_n_mu != runtime.bins.n_mu
        or dataset_n_phi != runtime.bins.n_phi
    ):
        raise ValueError(
            "Dataset bin geometry does not match "
            "the checkpoint."
        )

    dataset_target_type = str(
        dataset.metadata["target_type"]
    )

    if dataset_target_type != runtime.target_type:
        raise ValueError(
            "Dataset target type does not match "
            "the checkpoint."
        )


def evaluate_method(
    *,
    name: str,
    contributions: np.ndarray,
    probabilities: np.ndarray,
    bin_solid_angle: float,
    samples_per_estimate: int,
    repetitions: int,
    rng: np.random.Generator,
) -> MethodResult:
    """Evaluate one discrete sampling proposal."""

    statistics = calculate_discrete_estimator_statistics(
        contributions,
        probabilities,
        bin_solid_angle=bin_solid_angle,
        sample_count=samples_per_estimate,
    )

    estimates = simulate_discrete_estimates(
        contributions,
        probabilities,
        bin_solid_angle=bin_solid_angle,
        samples_per_estimate=samples_per_estimate,
        repetitions=repetitions,
        rng=rng,
    )

    empirical_mean = float(
        np.mean(estimates)
    )
    empirical_variance = float(
        np.var(
            estimates,
            ddof=1,
        )
    )
    empirical_rmse = float(
        np.sqrt(
            np.mean(
                (
                    estimates
                    - statistics.exact_integral
                )
                ** 2
            )
        )
    )

    return MethodResult(
        name=name,
        statistics=statistics,
        empirical_mean=empirical_mean,
        empirical_variance=empirical_variance,
        empirical_rmse=empirical_rmse,
    )


def print_results(
    results: list[MethodResult],
) -> None:
    """Print the variance comparison as a compact table."""

    baseline_variance = (
        results[0].statistics.estimator_variance
    )

    print("\nResults")
    print(
        f"{'Method':<22}"
        f"{'Mean':>13}"
        f"{'Theory var':>15}"
        f"{'Observed var':>16}"
        f"{'RMSE':>13}"
        f"{'Reduction':>13}"
    )

    for result in results:
        method_variance = (
            result.statistics.estimator_variance
        )

        if method_variance > 0.0:
            reduction = (
                baseline_variance
                / method_variance
            )
            reduction_text = f"{reduction:.2f}x"
        else:
            reduction_text = "infinite"

        print(
            f"{result.name:<22}"
            f"{result.empirical_mean:>13.6f}"
            f"{method_variance:>15.6e}"
            f"{result.empirical_variance:>16.6e}"
            f"{result.empirical_rmse:>13.6e}"
            f"{reduction_text:>13}"
        )


def main() -> None:
    """Run the controlled discrete variance experiment."""

    arguments = parse_arguments()

    dataset = load_dataset(
        arguments.dataset
    )

    runtime = GuidingInferenceRuntime.from_checkpoint(
        arguments.checkpoint,
        device=arguments.device,
        uniform_mix=0.0,
    )

    validate_experiment(
        arguments,
        dataset,
        runtime,
    )

    row = arguments.row

    features = np.asarray(
        dataset.features[row],
        dtype=np.float64,
    )
    contributions = np.asarray(
        dataset.mean_contributions[row],
        dtype=np.float64,
    )

    neural_distribution = (
        runtime.predict_distribution(
            features
        )
    )

    mixed_distribution = (
        DiscreteDirectionalDistribution(
            bins=runtime.bins,
            probabilities=(
                neural_distribution.probabilities
            ),
            uniform_mix=arguments.uniform_mix,
        )
    )

    uniform_probabilities = np.full(
        runtime.num_bins,
        1.0 / runtime.num_bins,
        dtype=np.float64,
    )

    proposals = {
        "Uniform": uniform_probabilities,
        "Neural": neural_distribution.probabilities,
        "Neural + uniform": (
            mixed_distribution.probabilities
        ),
    }

    seed_sequence = np.random.SeedSequence(
        arguments.seed
    )
    method_seeds = seed_sequence.spawn(
        len(proposals)
    )

    results = []

    for (
        name,
        probabilities,
    ), method_seed in zip(
        proposals.items(),
        method_seeds,
        strict=True,
    ):
        results.append(
            evaluate_method(
                name=name,
                contributions=contributions,
                probabilities=probabilities,
                bin_solid_angle=(
                    runtime.bins.bin_solid_angle
                ),
                samples_per_estimate=(
                    arguments.samples_per_estimate
                ),
                repetitions=arguments.repetitions,
                rng=np.random.default_rng(
                    method_seed
                ),
            )
        )

    reference = (
        results[0].statistics.exact_integral
    )

    print(
        "Discrete sampling comparison completed."
    )

    print("\nExperiment")
    print(f"  Dataset: {arguments.dataset.resolve()}")
    print(f"  Checkpoint: {arguments.checkpoint.resolve()}")
    print(f"  Row: {row}")
    print(f"  Pixel: {dataset.pixels[row]}")
    print(f"  Target type: {runtime.target_type}")
    print(f"  Bins: {runtime.num_bins}")
    print(
        "  Samples per estimate: "
        f"{arguments.samples_per_estimate}"
    )
    print(
        f"  Repetitions: {arguments.repetitions}"
    )
    print(
        f"  Uniform mix: {arguments.uniform_mix}"
    )
    print(f"  Exact discrete integral: {reference:.8f}")

    print_results(results)

    print(
        "\nImportant: this compares proposals on a "
        "single discretized training target. It is not "
        "yet evidence of render-level variance reduction."
    )


if __name__ == "__main__":
    main()