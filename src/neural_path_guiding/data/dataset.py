"""Load and save validated neural path guiding NPZ datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from neural_path_guiding.data.schema import (
    DATASET_ARRAY_KEYS,
    FEATURES_KEY,
    MEAN_CONTRIBUTIONS_KEY,
    METADATA_KEY,
    NORMALS_KEY,
    PIXELS_KEY,
    POSITIONS_KEY,
    TARGETS_KEY,
    NeuralGuidingDataset,
)
from neural_path_guiding.data.validation import validate_dataset


def load_dataset(dataset_path: Path) -> NeuralGuidingDataset:
    """Load a complete dataset and reject invalid or incompatible files."""
    dataset_path = Path(dataset_path)
    _validate_dataset_path(dataset_path, must_exist=True)

    with np.load(dataset_path, allow_pickle=False) as stored:
        arrays = {
            name: _read_required_array(stored, name)
            for name in DATASET_ARRAY_KEYS
        }
        metadata = _read_metadata(stored)

    dataset = NeuralGuidingDataset(
        features=arrays[FEATURES_KEY],
        targets=arrays[TARGETS_KEY],
        mean_contributions=arrays[MEAN_CONTRIBUTIONS_KEY],
        positions=arrays[POSITIONS_KEY],
        normals=arrays[NORMALS_KEY],
        pixels=arrays[PIXELS_KEY],
        metadata=metadata,
    )
    validate_dataset(dataset)

    return dataset


def save_dataset(dataset_path: Path, dataset: NeuralGuidingDataset) -> None:
    """Validate and write a complete dataset using the canonical schema."""
    dataset_path = Path(dataset_path)
    _validate_dataset_path(dataset_path, must_exist=False)
    validate_dataset(dataset)

    try:
        metadata_json = json.dumps(dataset.metadata, indent=2)
    except (TypeError, ValueError) as error:
        raise ValueError("dataset metadata must be JSON serializable.") from error

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dataset_path,
        features=dataset.features,
        targets=dataset.targets,
        mean_contributions=dataset.mean_contributions,
        positions=dataset.positions,
        normals=dataset.normals,
        pixels=dataset.pixels,
        metadata_json=metadata_json,
    )


def _validate_dataset_path(dataset_path: Path, must_exist: bool) -> None:
    if dataset_path.suffix.lower() != ".npz":
        raise ValueError("dataset path must use the .npz extension.")

    if must_exist and not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")


def _read_required_array(stored: Any, name: str) -> np.ndarray:
    if name not in stored.files:
        raise ValueError(f"Dataset is missing required array: {name}")

    return np.array(stored[name], copy=True)


def _read_metadata(stored: Any) -> dict[str, Any]:
    if METADATA_KEY not in stored.files:
        raise ValueError(f"Dataset is missing required array: {METADATA_KEY}")

    try:
        raw_metadata = stored[METADATA_KEY].item()
        metadata = json.loads(str(raw_metadata))
    except (ValueError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("metadata_json must contain a valid JSON object.") from error

    if not isinstance(metadata, dict):
        raise ValueError("metadata_json must contain a JSON object.")

    return metadata
