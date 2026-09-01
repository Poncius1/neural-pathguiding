"""Minimal MLP for discrete neural path guiding.

The model maps shading features to one logit per directional bin.
It does not apply softmax because training losses should operate directly
on logits for numerical stability.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class GuidingMLP(nn.Module):
    """Small MLP that predicts directional-bin logits."""

    def __init__(
        self,
        feature_dimension: int,
        num_bins: int,
        hidden_dimension: int = 64,
    ) -> None:
        super().__init__()

        _validate_positive_integer(
            feature_dimension,
            "feature_dimension",
        )
        _validate_positive_integer(
            num_bins,
            "num_bins",
        )
        _validate_positive_integer(
            hidden_dimension,
            "hidden_dimension",
        )

        self.feature_dimension = int(feature_dimension)
        self.num_bins = int(num_bins)
        self.hidden_dimension = int(hidden_dimension)

        self.network = nn.Sequential(
            nn.Linear(
                self.feature_dimension,
                self.hidden_dimension,
            ),
            nn.ReLU(),
            nn.Linear(
                self.hidden_dimension,
                self.hidden_dimension,
            ),
            nn.ReLU(),
            nn.Linear(
                self.hidden_dimension,
                self.num_bins,
            ),
        )

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Return one unnormalized logit per directional bin."""

        if features.ndim != 2:
            raise ValueError(
                f"features must be 2D, got shape {tuple(features.shape)}."
            )

        if features.shape[1] != self.feature_dimension:
            raise ValueError(
                f"features must have dimension "
                f"{self.feature_dimension}, "
                f"got {features.shape[1]}."
            )

        return self.network(features)


def _validate_positive_integer(
    value: int,
    name: str,
) -> None:
    if isinstance(value, bool) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise TypeError(f"{name} must be an integer.")

    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")