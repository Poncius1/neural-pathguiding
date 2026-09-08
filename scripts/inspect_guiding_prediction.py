"""Inspect one guiding-model prediction against its dataset target."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from neural_path_guiding.data.dataset import load_dataset
from neural_path_guiding.data.schema import NeuralGuidingDataset
from neural_path_guiding.training.inference import (
    GuidingInferenceRuntime,
)


def parse_arguments() -> argparse.Namespace:
    """Read prediction-inspection options."""

    parser = argparse.ArgumentParser(
        description=(
            "Compare one neural guiding prediction "
            "against its dataset target."
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to the NPZ dataset.",
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
        help="Dataset row to inspect.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used to sample the predicted distribution.",
    )
    parser.add_argument(
        "--uniform-mix",
        type=float,
        default=0.0,
        help="Uniform support mixed with the neural PMF.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Device used for MLP inference.",
    )

    return parser.parse_args()


def validate_arguments(
    arguments: argparse.Namespace,
    dataset: NeuralGuidingDataset,
) -> None:
    """Validate CLI values that depend on the loaded dataset."""

    if arguments.row < 0 or arguments.row >= dataset.num_samples:
        raise ValueError(
            "row must be between 0 and "
            f"{dataset.num_samples - 1}."
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


def validate_compatibility(
    dataset: NeuralGuidingDataset,
    runtime: GuidingInferenceRuntime,
) -> None:
    """Ensure the dataset and checkpoint describe the same problem."""

    if dataset.features.shape[1] != runtime.feature_dimension:
        raise ValueError(
            "Dataset feature dimension does not match "
            "the checkpoint."
        )

    if dataset.targets.shape[1] != runtime.num_bins:
        raise ValueError(
            "Dataset target dimension does not match "
            "the checkpoint bin count."
        )

    dataset_n_mu = require_metadata_integer(
        dataset.metadata,
        "n_mu",
    )
    dataset_n_phi = require_metadata_integer(
        dataset.metadata,
        "n_phi",
    )

    if (
        dataset_n_mu != runtime.bins.n_mu
        or dataset_n_phi != runtime.bins.n_phi
    ):
        raise ValueError(
            "Dataset hemisphere bins do not match "
            "the checkpoint."
        )

    dataset_target_type = require_metadata_string(
        dataset.metadata,
        "target_type",
    )

    if dataset_target_type != runtime.target_type:
        raise ValueError(
            "Dataset target_type does not match "
            "the checkpoint."
        )

    dataset_schema_version = require_metadata_integer(
        dataset.metadata,
        "feature_schema_version",
    )
    checkpoint_schema_version = require_metadata_integer(
        runtime.checkpoint.metadata,
        "feature_schema_version",
    )

    if dataset_schema_version != checkpoint_schema_version:
        raise ValueError(
            "Dataset feature schema does not match "
            "the checkpoint."
        )


def require_metadata_integer(
    metadata: dict[str, Any],
    key: str,
) -> int:
    """Read an integer from metadata without accepting booleans."""

    value = metadata.get(key)

    if (
        isinstance(value, bool)
        or not isinstance(
            value,
            (int, np.integer),
        )
    ):
        raise ValueError(
            f"Metadata '{key}' must be an integer."
        )

    return int(value)


def require_metadata_string(
    metadata: dict[str, Any],
    key: str,
) -> str:
    """Read a non-empty string from metadata."""

    value = metadata.get(key)

    if (
        not isinstance(value, str)
        or value.strip() == ""
    ):
        raise ValueError(
            f"Metadata '{key}' must be a non-empty string."
        )

    return value


def calculate_kl_divergence(
    target: np.ndarray,
    prediction: np.ndarray,
) -> float:
    """Calculate KL(target || prediction)."""

    positive_target = target > 0.0
    safe_prediction = np.clip(
        prediction,
        np.finfo(np.float64).tiny,
        None,
    )

    terms = target[positive_target] * (
        np.log(target[positive_target])
        - np.log(safe_prediction[positive_target])
    )

    return float(np.sum(terms))


def format_vector(values: np.ndarray) -> str:
    """Format a compact numeric vector for console output."""

    return np.array2string(
        values,
        precision=5,
        suppress_small=True,
        max_line_width=120,
    )


def print_top_bins(
    target: np.ndarray,
    prediction: np.ndarray,
    count: int = 5,
) -> None:
    """Print the bins with the largest target probabilities."""

    top_indices = np.argsort(target)[::-1][:count]

    print("\nTop target bins")

    for bin_index in top_indices:
        print(
            f"  Bin {int(bin_index):2d}: "
            f"target={target[bin_index]:.8f}, "
            f"prediction={prediction[bin_index]:.8f}"
        )


def main() -> None:
    """Load one dataset row and inspect its guiding prediction."""

    arguments = parse_arguments()

    dataset = load_dataset(
        arguments.dataset
    )

    validate_arguments(
        arguments,
        dataset,
    )

    runtime = GuidingInferenceRuntime.from_checkpoint(
        arguments.checkpoint,
        device=arguments.device,
        uniform_mix=arguments.uniform_mix,
    )

    validate_compatibility(
        dataset,
        runtime,
    )

    row = arguments.row

    features = np.asarray(
        dataset.features[row],
        dtype=np.float64,
    )
    target = np.asarray(
        dataset.targets[row],
        dtype=np.float64,
    )

    distribution = runtime.predict_distribution(
        features
    )
    prediction = distribution.probabilities

    target_dominant_bin = int(
        np.argmax(target)
    )
    predicted_dominant_bin = int(
        np.argmax(prediction)
    )

    mean_absolute_error = float(
        np.mean(
            np.abs(target - prediction)
        )
    )
    kl_divergence = calculate_kl_divergence(
        target,
        prediction,
    )

    rng = np.random.default_rng(
        arguments.seed
    )
    sample = distribution.sample(rng)
    evaluated_pdf = distribution.pdf(
        sample.direction_local
    )

    print("Guiding prediction inspection completed.")

    print("\nInputs")
    print(f"  Dataset: {arguments.dataset.resolve()}")
    print(f"  Checkpoint: {arguments.checkpoint.resolve()}")
    print(f"  Row: {row}")
    print(f"  Pixel: {dataset.pixels[row]}")
    print(f"  Position: {format_vector(dataset.positions[row])}")
    print(f"  Normal: {format_vector(dataset.normals[row])}")
    print(f"  Target type: {runtime.target_type}")
    print(f"  Device: {runtime.device}")
    print(f"  Uniform mix: {runtime.uniform_mix:.6f}")

    print("\nPrediction metrics")
    print(f"  Target sum: {np.sum(target):.12f}")
    print(f"  Prediction sum: {np.sum(prediction):.12f}")
    print(f"  Target dominant bin: {target_dominant_bin}")
    print(f"  Predicted dominant bin: {predicted_dominant_bin}")
    print(f"  Mean absolute error: {mean_absolute_error:.12f}")
    print(f"  KL divergence: {kl_divergence:.12f}")

    print("\nDistributions")
    print(f"  Target: {format_vector(target)}")
    print(f"  Prediction: {format_vector(prediction)}")

    print_top_bins(
        target,
        prediction,
    )

    print("\nSample")
    print(f"  Bin: {sample.bin_index}")
    print(
        "  Direction local: "
        f"{format_vector(sample.direction_local)}"
    )
    print(f"  Sample PDF: {sample.pdf:.12f}")
    print(f"  Evaluated PDF: {evaluated_pdf:.12f}")
    print(
        "  Absolute PDF difference: "
        f"{abs(sample.pdf - evaluated_pdf):.12e}"
    )


if __name__ == "__main__":
    main()