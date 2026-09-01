"""Typed in-memory schema for neural path guiding datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from numpy.typing import NDArray


Array = NDArray[Any]
Metadata = dict[str, Any]

DATASET_FORMAT_VERSION = 1

FEATURES_KEY = "features"
TARGETS_KEY = "targets"
MEAN_CONTRIBUTIONS_KEY = "mean_contributions"
POSITIONS_KEY = "positions"
NORMALS_KEY = "normals"
PIXELS_KEY = "pixels"
METADATA_KEY = "metadata_json"

DATASET_ARRAY_KEYS = (
    FEATURES_KEY,
    TARGETS_KEY,
    MEAN_CONTRIBUTIONS_KEY,
    POSITIONS_KEY,
    NORMALS_KEY,
    PIXELS_KEY,
)


@dataclass(frozen=True)
class NeuralGuidingDataset:
    """All arrays and metadata stored in a version-1 dataset."""

    features: Array
    targets: Array
    mean_contributions: Array
    positions: Array
    normals: Array
    pixels: Array
    metadata: Metadata

    @property
    def num_samples(self) -> int:
        return int(self.features.shape[0]) if self.features.ndim > 0 else 0

    @property
    def feature_dimension(self) -> int:
        return int(self.features.shape[1]) if self.features.ndim == 2 else 0

    @property
    def num_bins(self) -> int:
        return int(self.targets.shape[1]) if self.targets.ndim == 2 else 0
