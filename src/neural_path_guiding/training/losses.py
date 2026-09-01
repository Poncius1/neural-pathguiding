"""Loss functions for neural path guiding training."""

from __future__ import annotations

import torch
from torch.nn import functional as F


_TARGET_SUM_TOLERANCE = 1e-4


def soft_target_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Compute cross entropy between logits and target PMFs.

    Args:
        logits:
            Unnormalized model outputs with shape
            ``(batch_size, num_bins)``.

        targets:
            Target probability distributions with the same shape.
            Every row must be non-negative and sum to one.

    Returns:
        A scalar containing the mean loss over the batch.
    """

    _validate_loss_inputs(
        logits=logits,
        targets=targets,
    )

    log_probabilities = F.log_softmax(
        logits,
        dim=1,
    )

    losses_per_sample = -torch.sum(
        targets * log_probabilities,
        dim=1,
    )

    return losses_per_sample.mean()


def _validate_loss_inputs(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    if logits.ndim != 2:
        raise ValueError(
            f"logits must be 2D, got shape {tuple(logits.shape)}."
        )

    if targets.ndim != 2:
        raise ValueError(
            f"targets must be 2D, got shape {tuple(targets.shape)}."
        )

    if logits.shape != targets.shape:
        raise ValueError(
            f"logits and targets must have the same shape, "
            f"got {tuple(logits.shape)} and {tuple(targets.shape)}."
        )

    if logits.shape[0] == 0:
        raise ValueError("loss inputs must contain at least one sample.")

    if logits.shape[1] == 0:
        raise ValueError("loss inputs must contain at least one bin.")

    if not torch.is_floating_point(logits):
        raise TypeError("logits must use a floating-point dtype.")

    if not torch.is_floating_point(targets):
        raise TypeError("targets must use a floating-point dtype.")

    if logits.device != targets.device:
        raise ValueError(
            "logits and targets must be on the same device."
        )

    if not bool(torch.all(torch.isfinite(logits))):
        raise ValueError("logits must contain only finite values.")

    if not bool(torch.all(torch.isfinite(targets))):
        raise ValueError("targets must contain only finite values.")

    if bool(torch.any(targets < 0.0)):
        raise ValueError("targets must be non-negative.")

    target_sums = targets.sum(dim=1)

    if not bool(
        torch.allclose(
            target_sums,
            torch.ones_like(target_sums),
            atol=_TARGET_SUM_TOLERANCE,
            rtol=0.0,
        )
    ):
        raise ValueError("every target row must sum to one.")