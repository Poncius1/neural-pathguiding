"""Train a minimal guiding MLP and save its checkpoint."""

import argparse
from pathlib import Path
from time import perf_counter

import torch

from neural_path_guiding.core.features import FEATURE_SCHEMA_VERSION
from neural_path_guiding.training.checkpoint import (
    MetadataValue,
    save_checkpoint,
)
from neural_path_guiding.training.dataset import load_training_subset
from neural_path_guiding.training.model import GuidingMLP
from neural_path_guiding.training.trainer import train_full_batch


def parse_arguments() -> argparse.Namespace:
    """Read training options from the command line."""

    parser = argparse.ArgumentParser(
        description="Train a minimal neural path guiding MLP."
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to the NPZ training dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path where the checkpoint will be saved.",
    )
    parser.add_argument(
        "--experiment",
        default="minimal_mlp_overfit",
        help="Name stored in the checkpoint metadata.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=32,
        help="Number of dataset samples used for training.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=2000,
        help="Number of full-batch optimization steps.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Adam learning rate.",
    )
    parser.add_argument(
        "--hidden-dimension",
        type=int,
        default=64,
        help="Width of the two hidden layers.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for subset selection and initialization.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Device used during training.",
    )

    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    """Validate and construct the requested PyTorch device."""

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but it is not available in PyTorch"
        )

    return torch.device(device_name)


def calculate_metrics(
    model: GuidingMLP,
    features: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[float, float, float]:
    """Calculate entropy, KL divergence, and dominant-bin accuracy."""

    model.eval()

    with torch.inference_mode():
        logits = model(features)
        log_probabilities = torch.log_softmax(logits, dim=1)
        probabilities = log_probabilities.exp()

        smallest_value = torch.finfo(targets.dtype).tiny
        safe_targets = targets.clamp_min(smallest_value)
        target_log_probabilities = safe_targets.log()

        target_entropy = -(
            targets * target_log_probabilities
        ).sum(dim=1).mean()

        kl_divergence = (
            targets
            * (target_log_probabilities - log_probabilities)
        ).sum(dim=1).mean()

        dominant_bin_accuracy = (
            probabilities.argmax(dim=1)
            == targets.argmax(dim=1)
        ).float().mean()

    return (
        target_entropy.item(),
        kl_divergence.item(),
        dominant_bin_accuracy.item(),
    )


def build_checkpoint_metadata(
    *,
    experiment: str,
    dataset_path: Path,
    sample_count: int,
    device: torch.device,
    final_loss: float,
    n_mu: int,
    n_phi: int,
    target_type: str,
    dataset_format_version: int,
) -> dict[str, MetadataValue]:
    """Build the reproducibility metadata stored with the model."""

    return {
        "experiment": experiment,
        "dataset": dataset_path.name,
        "sample_count": sample_count,
        "device": str(device),
        "final_loss": final_loss,
        "torch_version": str(torch.__version__),
        "n_mu": n_mu,
        "n_phi": n_phi,
        "target_type": target_type,
        "dataset_format_version": dataset_format_version,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
    }


def main() -> None:
    """Train the model, report metrics, and save its checkpoint."""

    arguments = parse_arguments()
    device = resolve_device(arguments.device)

    torch.manual_seed(arguments.seed)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(arguments.seed)

    subset = load_training_subset(
        arguments.dataset,
        sample_count=arguments.sample_count,
        seed=arguments.seed,
    )

    features = subset.features.to(device)
    targets = subset.targets.to(device)

    model = GuidingMLP(
        feature_dimension=subset.feature_dimension,
        hidden_dimension=arguments.hidden_dimension,
        num_bins=subset.num_bins,
    ).to(device)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = perf_counter()

    result = train_full_batch(
        model,
        features,
        targets,
        num_steps=arguments.steps,
        learning_rate=arguments.learning_rate,
    )

    if device.type == "cuda":
        torch.cuda.synchronize()

    training_time = perf_counter() - start_time

    target_entropy, kl_divergence, dominant_bin_accuracy = (
        calculate_metrics(
            model,
            features,
            targets,
        )
    )

    metadata = build_checkpoint_metadata(
        experiment=arguments.experiment,
        dataset_path=arguments.dataset,
        sample_count=subset.num_samples,
        device=device,
        final_loss=result.final_loss,
        n_mu=subset.n_mu,
        n_phi=subset.n_phi,
        target_type=subset.target_type,
        dataset_format_version=subset.dataset_format_version,
    )

    checkpoint_path = save_checkpoint(
        arguments.output,
        model,
        subset.feature_mean,
        subset.feature_scale,
        seed=arguments.seed,
        num_steps=arguments.steps,
        learning_rate=arguments.learning_rate,
        metadata=metadata,
    )

    print("MLP training completed.")
    print(f"Experiment: {arguments.experiment}")
    print(f"Dataset: {arguments.dataset.resolve()}")
    print(f"Checkpoint: {checkpoint_path.resolve()}")
    print(f"Samples: {subset.num_samples}")
    print(f"Feature dimension: {subset.feature_dimension}")
    print(f"Bins: {subset.num_bins}")
    print(f"Device: {device}")
    print(f"Initial loss: {result.initial_loss:.8f}")
    print(f"Final loss: {result.final_loss:.8f}")
    print(f"Target entropy: {target_entropy:.8f}")
    print(f"KL divergence: {kl_divergence:.8f}")
    print(
        "Dominant-bin accuracy: "
        f"{dominant_bin_accuracy:.4f}"
    )
    print(f"Training time: {training_time:.3f} seconds")


if __name__ == "__main__":
    main()