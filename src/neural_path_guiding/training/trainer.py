"""Minimal full-batch trainer for the initial overfitting experiment."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from neural_path_guiding.training.losses import (
    soft_target_cross_entropy,
)
from neural_path_guiding.training.model import GuidingMLP


@dataclass(frozen=True)
class TrainingResult:
    """Loss history produced by a completed training run."""

    losses: tuple[float, ...]
    final_loss: float

    @property
    def initial_loss(self) -> float:
        return self.losses[0]

    @property
    def num_steps(self) -> int:
        return len(self.losses)


def train_full_batch(
    model: GuidingMLP,
    features: torch.Tensor,
    targets: torch.Tensor,
    num_steps: int,
    learning_rate: float,
) -> TrainingResult:
    """Train the model using the complete subset on every step."""

    _validate_training_settings(
        num_steps=num_steps,
        learning_rate=learning_rate,
    )
    _validate_devices(
        model=model,
        features=features,
        targets=targets,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    loss_values: list[float] = []

    model.train()

    for step in range(num_steps):
        optimizer.zero_grad(set_to_none=True)

        logits = model(features)

        loss = soft_target_cross_entropy(
            logits=logits,
            targets=targets,
        )

        if not bool(torch.isfinite(loss)):
            raise RuntimeError(
                f"Training produced a non-finite loss "
                f"at step {step}."
            )

        loss.backward()
        optimizer.step()

        loss_values.append(float(loss.item()))

    model.eval()

    with torch.no_grad():
        final_logits = model(features)
        final_loss = soft_target_cross_entropy(
            logits=final_logits,
            targets=targets,
        )

    return TrainingResult(
        losses=tuple(loss_values),
        final_loss=float(final_loss.item()),
    )


def _validate_training_settings(
    num_steps: int,
    learning_rate: float,
) -> None:
    if isinstance(num_steps, bool) or not isinstance(
        num_steps,
        int,
    ):
        raise TypeError("num_steps must be an integer.")

    if num_steps <= 0:
        raise ValueError("num_steps must be greater than zero.")

    if isinstance(learning_rate, bool) or not isinstance(
        learning_rate,
        (int, float),
    ):
        raise TypeError("learning_rate must be numeric.")

    if not math.isfinite(float(learning_rate)):
        raise ValueError("learning_rate must be finite.")

    if learning_rate <= 0.0:
        raise ValueError(
            "learning_rate must be greater than zero."
        )


def _validate_devices(
    model: GuidingMLP,
    features: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    model_device = next(model.parameters()).device

    if features.device != model_device:
        raise ValueError(
            f"features are on {features.device}, "
            f"but the model is on {model_device}."
        )

    if targets.device != model_device:
        raise ValueError(
            f"targets are on {targets.device}, "
            f"but the model is on {model_device}."
        )