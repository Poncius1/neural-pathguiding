"""Utilities for saving and loading trained guiding models."""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import TypeAlias

import torch

from neural_path_guiding.training.model import GuidingMLP


MetadataValue: TypeAlias = str | int | float | bool


@dataclass(frozen=True)
class LoadedCheckpoint:
    """Model and preprocessing information recovered from a checkpoint."""

    model: GuidingMLP
    feature_mean: torch.Tensor
    feature_scale: torch.Tensor
    seed: int
    num_steps: int
    learning_rate: float
    metadata: dict[str, MetadataValue]


def _validate_normalization(
    feature_mean: torch.Tensor,
    feature_scale: torch.Tensor,
    feature_dimension: int,
) -> None:
    """Validate the tensors required to normalize model inputs."""

    for name, tensor in (
        ("feature_mean", feature_mean),
        ("feature_scale", feature_scale),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")

        if tensor.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")

        if tensor.numel() != feature_dimension:
            raise ValueError(
                f"{name} must contain {feature_dimension} values"
            )

        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} must contain only finite values")

    if torch.any(feature_scale <= 0):
        raise ValueError("feature_scale values must be positive")


def _validate_metadata(
    metadata: dict[str, MetadataValue],
) -> None:
    """Ensure metadata can be safely stored in the checkpoint."""

    supported_types = (str, int, float, bool)

    for key, value in metadata.items():
        if not isinstance(key, str):
            raise TypeError("metadata keys must be strings")

        if not isinstance(value, supported_types):
            raise TypeError(
                "metadata values must be strings, integers, floats, or booleans"
            )


def save_checkpoint(
    checkpoint_path: str | Path,
    model: GuidingMLP,
    feature_mean: torch.Tensor,
    feature_scale: torch.Tensor,
    *,
    seed: int,
    num_steps: int,
    learning_rate: float,
    metadata: dict[str, MetadataValue] | None = None,
) -> Path:
    """Save a guiding model and the information needed for inference."""

    if not isinstance(model, GuidingMLP):
        raise TypeError("model must be a GuidingMLP")

    if seed < 0:
        raise ValueError("seed must be non-negative")

    if num_steps <= 0:
        raise ValueError("num_steps must be positive")

    if not isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")

    _validate_normalization(
        feature_mean,
        feature_scale,
        model.feature_dimension,
    )

    checkpoint_metadata = dict(metadata or {})
    _validate_metadata(checkpoint_metadata)

    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state_dict = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
    }

    payload = {
        "format_version": 1,
        "model": {
            "feature_dimension": model.feature_dimension,
            "hidden_dimension": model.hidden_dimension,
            "num_bins": model.num_bins,
            "state_dict": state_dict,
        },
        "normalization": {
            "feature_mean": feature_mean.detach().cpu(),
            "feature_scale": feature_scale.detach().cpu(),
        },
        "training": {
            "seed": seed,
            "num_steps": num_steps,
            "learning_rate": learning_rate,
        },
        "metadata": checkpoint_metadata,
    }

    torch.save(payload, path)
    return path


def load_checkpoint(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> LoadedCheckpoint:
    """Load a guiding model and its input-normalization information."""

    path = Path(checkpoint_path)

    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    target_device = torch.device(device)

    payload = torch.load(
        path,
        map_location=target_device,
        weights_only=True,
    )

    if payload.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format version")

    model_data = payload["model"]
    normalization = payload["normalization"]
    training = payload["training"]

    model = GuidingMLP(
        feature_dimension=model_data["feature_dimension"],
        hidden_dimension=model_data["hidden_dimension"],
        num_bins=model_data["num_bins"],
    )

    model.load_state_dict(model_data["state_dict"])
    model.to(target_device)
    model.eval()

    feature_mean = normalization["feature_mean"].to(target_device)
    feature_scale = normalization["feature_scale"].to(target_device)

    _validate_normalization(
        feature_mean,
        feature_scale,
        model.feature_dimension,
    )

    return LoadedCheckpoint(
        model=model,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        seed=training["seed"],
        num_steps=training["num_steps"],
        learning_rate=training["learning_rate"],
        metadata=dict(payload["metadata"]),
    )