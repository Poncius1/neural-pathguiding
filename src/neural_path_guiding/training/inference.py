"""Inference bridge between trained MLP checkpoints and directional sampling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from neural_path_guiding.core.bins import (
    FloatArray,
    HemisphereBins,
)
from neural_path_guiding.core.directional_distribution import (
    DiscreteDirectionalDistribution,
)
from neural_path_guiding.core.features import FEATURE_SCHEMA_VERSION
from neural_path_guiding.training.checkpoint import (
    LoadedCheckpoint,
    MetadataValue,
    load_checkpoint,
)


@dataclass(frozen=True)
class GuidingInferenceRuntime:
    """Loaded guiding model ready to predict directional distributions."""

    checkpoint: LoadedCheckpoint
    bins: HemisphereBins
    uniform_mix: float = 0.0

    def __post_init__(self) -> None:
        alpha = _validate_uniform_mix(self.uniform_mix)

        if self.bins.n_bins != self.checkpoint.model.num_bins:
            raise ValueError(
                "Checkpoint output dimension does not match "
                "the hemisphere bin count."
            )

        object.__setattr__(
            self,
            "uniform_mix",
            alpha,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device = "cpu",
        uniform_mix: float = 0.0,
    ) -> GuidingInferenceRuntime:
        """Load and validate a guiding checkpoint for inference."""

        checkpoint = load_checkpoint(
            checkpoint_path,
            device=device,
        )

        n_mu = _require_positive_metadata_integer(
            checkpoint.metadata,
            "n_mu",
        )
        n_phi = _require_positive_metadata_integer(
            checkpoint.metadata,
            "n_phi",
        )

        feature_schema_version = (
            _require_positive_metadata_integer(
                checkpoint.metadata,
                "feature_schema_version",
            )
        )

        if feature_schema_version != FEATURE_SCHEMA_VERSION:
            raise ValueError(
                "Checkpoint feature schema version "
                f"{feature_schema_version} does not match "
                f"the current version {FEATURE_SCHEMA_VERSION}."
            )

        _require_target_type(checkpoint.metadata)

        bins = HemisphereBins(
            n_mu=n_mu,
            n_phi=n_phi,
        )

        return cls(
            checkpoint=checkpoint,
            bins=bins,
            uniform_mix=uniform_mix,
        )

    @property
    def feature_dimension(self) -> int:
        """Return the number of features expected by the MLP."""

        return self.checkpoint.model.feature_dimension

    @property
    def num_bins(self) -> int:
        """Return the number of directional bins predicted by the MLP."""

        return self.checkpoint.model.num_bins

    @property
    def target_type(self) -> str:
        """Return the teacher target used to train the model."""

        value = self.checkpoint.metadata["target_type"]

        if not isinstance(value, str):
            raise TypeError(
                "Checkpoint target_type must be a string."
            )

        return value

    @property
    def device(self) -> torch.device:
        """Return the device used for neural inference."""

        return self.checkpoint.feature_mean.device

    def predict_logits(
        self,
        features: FloatArray,
    ) -> FloatArray:
        """Predict directional logits from one raw feature vector."""

        normalized_features = self._normalize_features(
            features
        )

        with torch.inference_mode():
            logits = self.checkpoint.model(
                normalized_features.unsqueeze(0)
            )

        expected_shape = (
            1,
            self.num_bins,
        )

        if tuple(logits.shape) != expected_shape:
            raise RuntimeError(
                "The guiding model returned shape "
                f"{tuple(logits.shape)}, expected {expected_shape}."
            )

        if not bool(
            torch.isfinite(logits).all().item()
        ):
            raise ValueError(
                "The guiding model produced non-finite logits."
            )

        return np.asarray(
            logits[0].detach().cpu().numpy(),
            dtype=np.float64,
        )

    def predict_distribution(
        self,
        features: FloatArray,
    ) -> DiscreteDirectionalDistribution:
        """Predict the directional distribution for one shading point."""

        logits = self.predict_logits(features)

        return DiscreteDirectionalDistribution.from_logits(
            bins=self.bins,
            logits=logits,
            uniform_mix=self.uniform_mix,
        )

    def _normalize_features(
        self,
        features: FloatArray,
    ) -> torch.Tensor:
        """Validate and normalize one raw feature vector."""

        values = _convert_feature_vector(
            features,
            expected_dimension=self.feature_dimension,
        )

        feature_tensor = torch.as_tensor(
            values,
            dtype=self.checkpoint.feature_mean.dtype,
            device=self.device,
        )

        normalized_features = (
            feature_tensor
            - self.checkpoint.feature_mean
        ) / self.checkpoint.feature_scale

        if not bool(
            torch.isfinite(
                normalized_features
            ).all().item()
        ):
            raise ValueError(
                "Feature normalization produced non-finite values."
            )

        return normalized_features


def _convert_feature_vector(
    features: Any,
    *,
    expected_dimension: int,
) -> FloatArray:
    """Convert one numeric feature vector into validated NumPy data."""

    try:
        values = np.asarray(
            features,
            dtype=np.float64,
        )
    except (TypeError, ValueError) as error:
        raise TypeError(
            "features must contain numeric values."
        ) from error

    expected_shape = (
        expected_dimension,
    )

    if values.shape != expected_shape:
        raise ValueError(
            "features must have shape "
            f"{expected_shape}, got {values.shape}."
        )

    if not bool(
        np.all(np.isfinite(values))
    ):
        raise ValueError(
            "features must contain only finite values."
        )

    return values


def _require_positive_metadata_integer(
    metadata: dict[str, MetadataValue],
    key: str,
) -> int:
    """Read one required positive integer from checkpoint metadata."""

    value = metadata.get(key)

    if (
        isinstance(value, bool)
        or not isinstance(
            value,
            (int, np.integer),
        )
    ):
        raise TypeError(
            f"Checkpoint metadata '{key}' must be an integer."
        )

    integer_value = int(value)

    if integer_value <= 0:
        raise ValueError(
            f"Checkpoint metadata '{key}' must be positive."
        )

    return integer_value


def _require_target_type(
    metadata: dict[str, MetadataValue],
) -> str:
    """Read the teacher target identifier from checkpoint metadata."""

    value = metadata.get("target_type")

    if (
        not isinstance(value, str)
        or value.strip() == ""
    ):
        raise ValueError(
            "Checkpoint metadata 'target_type' "
            "must be a non-empty string."
        )

    return value


def _validate_uniform_mix(
    uniform_mix: float,
) -> float:
    """Validate the probability assigned to uniform support."""

    if (
        isinstance(uniform_mix, bool)
        or not isinstance(
            uniform_mix,
            (int, float, np.integer, np.floating),
        )
    ):
        raise TypeError(
            "uniform_mix must be a real number."
        )

    alpha = float(uniform_mix)

    if (
        not np.isfinite(alpha)
        or alpha < 0.0
        or alpha > 1.0
    ):
        raise ValueError(
            "uniform_mix must be finite and between 0 and 1."
        )

    return alpha