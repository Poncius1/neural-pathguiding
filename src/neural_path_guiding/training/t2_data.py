"""Leakage-checked T2 dataset loader for feature schema v2.

The train/validation/test indices come from the stratified generator.
Only training rows are used to compute input normalization statistics.
No model is trained here, and legacy overfit loaders remain unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from neural_path_guiding.data.dataset import load_dataset

SPLIT_NAMES = ("train", "validation", "test")
EXPECTED_FEATURE_SCHEMA = 2
EXPECTED_TARGET_TYPE = "product_integrand_v1"
EXPECTED_MATERIAL_MODE = "lambertian_reflectance_v1"
NORMALIZATION_EPSILON = 1e-8


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_split_indices(path: str | Path) -> dict[str, np.ndarray]:
    """Read split indices without enabling NumPy pickle loading."""
    with np.load(Path(path), allow_pickle=False) as archive:
        if set(archive.files) != set(SPLIT_NAMES):
            raise ValueError(
                f"Split archive must contain exactly {SPLIT_NAMES}; "
                f"got {sorted(archive.files)}"
            )
        return {name: np.array(archive[name], copy=True) for name in SPLIT_NAMES}


def validate_split_indices(
    indices: dict[str, np.ndarray],
    pixels: np.ndarray,
) -> dict[str, np.ndarray]:
    """Require complete, exclusive row coverage and disjoint camera pixels."""
    pixel_array = np.asarray(pixels)
    if pixel_array.ndim != 2 or pixel_array.shape[1] != 2:
        raise ValueError("pixels must have shape (N, 2)")
    if not np.issubdtype(pixel_array.dtype, np.integer):
        raise ValueError("pixels must contain integer coordinates")
    if set(indices) != set(SPLIT_NAMES):
        raise ValueError("Split names must be train, validation, test")

    n = len(pixel_array)
    result: dict[str, np.ndarray] = {}
    groups: dict[str, set[tuple[int, int]]] = {}

    for name in SPLIT_NAMES:
        values = np.asarray(indices[name])
        if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"{name} indices must be a 1D integer array")
        if values.size == 0:
            raise ValueError(f"{name} split must not be empty")
        if np.any(values < 0) or np.any(values >= n):
            raise ValueError(f"{name} contains out-of-range row indices")
        clean = values.astype(np.int64, copy=True)
        if np.unique(clean).size != clean.size:
            raise ValueError(f"{name} contains duplicated row indices")
        result[name] = clean
        groups[name] = {tuple(map(int, p)) for p in pixel_array[clean]}

    joined = np.concatenate([result[name] for name in SPLIT_NAMES])
    if joined.size != n or np.unique(joined).size != n:
        raise ValueError("Split indices must partition all dataset rows exactly once")

    for i, name in enumerate(SPLIT_NAMES):
        for other in SPLIT_NAMES[i + 1:]:
            if groups[name].intersection(groups[other]):
                raise ValueError(f"Camera pixel leakage between {name} and {other}")

    return result


@dataclass(frozen=True)
class PreparedSplit:
    features: torch.Tensor
    targets: torch.Tensor
    indices: np.ndarray

    @property
    def num_samples(self) -> int:
        return int(self.features.shape[0])


@dataclass(frozen=True)
class PreparedT2Data:
    train: PreparedSplit
    validation: PreparedSplit
    test: PreparedSplit
    feature_mean: torch.Tensor
    feature_scale: torch.Tensor
    metadata: dict[str, Any]
    dataset_sha256: str
    splits_sha256: str

    @property
    def feature_dimension(self) -> int:
        return int(self.train.features.shape[1])

    @property
    def num_bins(self) -> int:
        return int(self.train.targets.shape[1])

    def make_dataloaders(
        self,
        *,
        batch_size: int = 64,
        seed: int = 42,
    ) -> dict[str, DataLoader]:
        """Create reproducible training order; never shuffle validation/test."""
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        generator = torch.Generator().manual_seed(seed)
        return {
            "train": DataLoader(
                TensorDataset(self.train.features, self.train.targets),
                batch_size=batch_size,
                shuffle=True,
                generator=generator,
                num_workers=0,
            ),
            "validation": DataLoader(
                TensorDataset(self.validation.features, self.validation.targets),
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
            ),
            "test": DataLoader(
                TensorDataset(self.test.features, self.test.targets),
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
            ),
        }


def prepare_t2_data(
    dataset_path: str | Path,
    splits_path: str | Path,
) -> PreparedT2Data:
    """Read validated data, enforce v2 semantics and normalize using train only."""
    dataset_path = Path(dataset_path)
    splits_path = Path(splits_path)
    dataset = load_dataset(dataset_path)
    metadata = dataset.metadata
    if metadata.get("feature_schema_version") != EXPECTED_FEATURE_SCHEMA:
        raise ValueError("Expected feature_schema_version=2")
    if metadata.get("material_feature_mode") != EXPECTED_MATERIAL_MODE:
        raise ValueError(f"Expected material_feature_mode={EXPECTED_MATERIAL_MODE}")
    if metadata.get("target_type") != EXPECTED_TARGET_TYPE:
        raise ValueError(f"Expected target_type={EXPECTED_TARGET_TYPE}")

    indices = validate_split_indices(load_split_indices(splits_path), dataset.pixels)
    raw_features = np.asarray(dataset.features, dtype=np.float64)
    raw_targets = np.asarray(dataset.targets, dtype=np.float64)

    # All normalization statistics must come from the training subset alone.
    train_features = raw_features[indices["train"]]
    mean = train_features.mean(axis=0)
    standard_deviation = train_features.std(axis=0, ddof=0)
    scale = np.where(standard_deviation > NORMALIZATION_EPSILON, standard_deviation, 1.0)

    prepared: dict[str, PreparedSplit] = {}
    for name in SPLIT_NAMES:
        selected = indices[name]
        normalized = (raw_features[selected] - mean) / scale
        features_f32 = np.asarray(normalized, dtype=np.float32)
        targets_f32 = np.asarray(raw_targets[selected], dtype=np.float32)
        if not np.all(np.isfinite(features_f32)):
            raise ValueError(f"Non-finite normalized features in {name}")
        prepared[name] = PreparedSplit(
            features=torch.from_numpy(features_f32.copy()),
            targets=torch.from_numpy(targets_f32.copy()),
            indices=selected.copy(),
        )

    return PreparedT2Data(
        train=prepared["train"],
        validation=prepared["validation"],
        test=prepared["test"],
        feature_mean=torch.from_numpy(np.asarray(mean, dtype=np.float32).copy()),
        feature_scale=torch.from_numpy(np.asarray(scale, dtype=np.float32).copy()),
        metadata=dict(metadata),
        dataset_sha256=sha256_file(dataset_path),
        splits_sha256=sha256_file(splits_path),
    )
