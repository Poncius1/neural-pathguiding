"""PyTorch dataset preparation for neural path guiding training.

This module loads an already validated dataset and prepares a deterministic
subset for the initial overfitting experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import torch

from neural_path_guiding.data.dataset import load_dataset


IndexArray = NDArray[np.int64]

_NORMALIZATION_EPSILON = 1e-8


@dataclass(frozen=True)
class TrainingSubset:
    """Tensors and normalization statistics for a small training subset."""

    features: torch.Tensor
    targets: torch.Tensor
    feature_mean: torch.Tensor
    feature_scale: torch.Tensor
    indices: IndexArray
    n_mu: int
    n_phi: int
    target_type: str
    dataset_format_version: int

    @property
    def num_samples(self) -> int:
     return int(self.features.shape[0])


    @property
    def feature_dimension(self) -> int:
        return int(self.features.shape[1])


    @property
    def num_bins(self) -> int:
        return int(self.targets.shape[1])


def load_training_subset(
    dataset_path: Path,
    sample_count: int,
    seed: int,
) -> TrainingSubset:
    """Load and normalize a deterministic subset without replacement."""

    _validate_subset_settings(
        sample_count=sample_count,
        seed=seed,
    )

    dataset = load_dataset(dataset_path)

    if sample_count > dataset.num_samples:
        raise ValueError(
            f"sample_count must not exceed the dataset size "
            f"({dataset.num_samples}), got {sample_count}."
        )

    rng = np.random.default_rng(seed)
    indices = np.asarray(
        rng.choice(
            dataset.num_samples,
            size=sample_count,
            replace=False,
        ),
        dtype=np.int64,
    )
    indices.sort()

    features = torch.from_numpy(
        np.asarray(
            dataset.features[indices],
            dtype=np.float32,
        ).copy()
    )
    targets = torch.from_numpy(
        np.asarray(
            dataset.targets[indices],
            dtype=np.float32,
        ).copy()
    )

    feature_mean = features.mean(dim=0)
    feature_standard_deviation = features.std(
        dim=0,
        unbiased=False,
    )

    feature_scale = torch.where(
        feature_standard_deviation > _NORMALIZATION_EPSILON,
        feature_standard_deviation,
        torch.ones_like(feature_standard_deviation),
    )

    normalized_features = (
        features - feature_mean
    ) / feature_scale

    return TrainingSubset(
    features=normalized_features,
    targets=targets,
    feature_mean=feature_mean,
    feature_scale=feature_scale,
    indices=indices,
    n_mu=int(dataset.metadata["n_mu"]),
    n_phi=int(dataset.metadata["n_phi"]),
    target_type=str(dataset.metadata["target_type"]),
    dataset_format_version=int(
        dataset.metadata["dataset_format_version"]
    ),
)


def _validate_subset_settings(
    sample_count: int,
    seed: int,
) -> None:
    if isinstance(sample_count, bool) or not isinstance(
        sample_count,
        (int, np.integer),
    ):
        raise TypeError("sample_count must be an integer.")

    if sample_count <= 0:
        raise ValueError("sample_count must be greater than zero.")

    if isinstance(seed, bool) or not isinstance(
        seed,
        (int, np.integer),
    ):
        raise TypeError("seed must be an integer.")

    if seed < 0:
        raise ValueError("seed must be non-negative.")