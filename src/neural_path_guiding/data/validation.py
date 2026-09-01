"""Central validation for in-memory neural path guiding datasets."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from neural_path_guiding.core.features import FEATURE_DIMENSION
from neural_path_guiding.data.schema import (
    DATASET_FORMAT_VERSION,
    NeuralGuidingDataset,
)


TARGET_SUM_TOLERANCE = 1e-4
NORMAL_LENGTH_TOLERANCE = 1e-4


def validate_training_arrays(features: Any, targets: Any) -> None:
    """Validate the two arrays required by training and inspection."""
    features = np.asarray(features)
    targets = np.asarray(targets)

    if features.ndim != 2:
        raise ValueError(f"features must be 2D, got {features.shape}.")

    if targets.ndim != 2:
        raise ValueError(f"targets must be 2D, got {targets.shape}.")

    if features.shape[0] <= 0:
        raise ValueError("dataset must contain at least one sample.")

    if features.shape[0] != targets.shape[0]:
        raise ValueError("features and targets sample counts differ.")

    if features.shape[1] != FEATURE_DIMENSION:
        raise ValueError(f"features must have dimension {FEATURE_DIMENSION}.")

    if targets.shape[1] <= 0:
        raise ValueError("targets must contain at least one bin.")

    _validate_floating_array(features, "features")
    _validate_floating_array(targets, "targets")

    if bool(np.any(targets < 0.0)):
        raise ValueError("targets must be non-negative.")

    target_sums = targets.sum(axis=1, dtype=np.float64)
    if not bool(
        np.allclose(
            target_sums,
            1.0,
            atol=TARGET_SUM_TOLERANCE,
            rtol=0.0,
        )
    ):
        raise ValueError("every target row must sum to 1.")


def validate_dataset(dataset: NeuralGuidingDataset) -> None:
    """Validate the complete version-1 dataset contract."""
    validate_training_arrays(dataset.features, dataset.targets)

    num_samples = dataset.num_samples
    num_bins = dataset.num_bins
    mean_contributions = np.asarray(dataset.mean_contributions)
    positions = np.asarray(dataset.positions)
    normals = np.asarray(dataset.normals)
    pixels = np.asarray(dataset.pixels)

    _validate_matrix(
        mean_contributions,
        "mean_contributions",
        expected_shape=(num_samples, num_bins),
        floating=True,
    )
    _validate_matrix(
        positions,
        "positions",
        expected_shape=(num_samples, 3),
        floating=True,
    )
    _validate_matrix(
        normals,
        "normals",
        expected_shape=(num_samples, 3),
        floating=True,
    )
    _validate_matrix(
        pixels,
        "pixels",
        expected_shape=(num_samples, 2),
        floating=False,
    )

    if bool(np.any(mean_contributions < 0.0)):
        raise ValueError("mean_contributions must be non-negative.")

    normal_lengths = np.linalg.norm(normals.astype(np.float64), axis=1)
    if not bool(
        np.allclose(
            normal_lengths,
            1.0,
            atol=NORMAL_LENGTH_TOLERANCE,
            rtol=0.0,
        )
    ):
        raise ValueError("normals must have unit length.")

    if not np.issubdtype(pixels.dtype, np.integer):
        raise ValueError("pixels must use an integer dtype.")

    if bool(np.any(pixels < 0)):
        raise ValueError("pixels must be non-negative.")

    validate_metadata(
        metadata=dataset.metadata,
        num_samples=num_samples,
        feature_dimension=dataset.feature_dimension,
        num_bins=num_bins,
    )


def validate_metadata(
    metadata: Mapping[str, Any],
    num_samples: int,
    feature_dimension: int,
    num_bins: int,
) -> None:
    """Validate metadata fields that must agree with stored arrays."""
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a mapping.")

    _require_metadata_equal(
        metadata,
        "dataset_format_version",
        DATASET_FORMAT_VERSION,
    )
    _require_metadata_equal(metadata, "feature_dimension", feature_dimension)
    _require_metadata_equal(metadata, "num_bins", num_bins)
    _require_metadata_equal(metadata, "num_shading_points", num_samples)

    experiment_name = metadata.get("experiment_name")
    if not isinstance(experiment_name, str) or experiment_name.strip() == "":
        raise ValueError("metadata.experiment_name must be a non-empty string.")

    target_type = metadata.get("target_type")
    if not isinstance(target_type, str) or target_type.strip() == "":
        raise ValueError("metadata.target_type must be a non-empty string.")

    n_mu = metadata.get("n_mu")
    n_phi = metadata.get("n_phi")
    if not _is_positive_integer(n_mu) or not _is_positive_integer(n_phi):
        raise ValueError("metadata n_mu and n_phi must be positive integers.")

    if int(n_mu) * int(n_phi) != num_bins:
        raise ValueError("metadata n_mu * n_phi must equal num_bins.")


def _validate_matrix(
    values: Any,
    name: str,
    expected_shape: tuple[int, int],
    floating: bool,
) -> None:
    values = np.asarray(values)

    if values.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {values.shape}.")

    if floating:
        _validate_floating_array(values, name)
    elif not bool(np.all(np.isfinite(values))):
        raise ValueError(f"{name} must contain only finite values.")


def _validate_floating_array(values: Any, name: str) -> None:
    values = np.asarray(values)

    if not np.issubdtype(values.dtype, np.floating):
        raise ValueError(f"{name} must use a floating-point dtype.")

    if not bool(np.all(np.isfinite(values))):
        raise ValueError(f"{name} must contain only finite values.")


def _require_metadata_equal(
    metadata: Mapping[str, Any],
    key: str,
    expected: int,
) -> None:
    value = metadata.get(key)

    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"metadata.{key} must be an integer.")

    if int(value) != expected:
        raise ValueError(
            f"metadata.{key} must equal {expected}, got {value}."
        )


def _is_positive_integer(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, np.integer))
        and int(value) > 0
    )
