"""Discrete directional distributions over hemisphere bins.

This module combines:

- probability normalization
- optional uniform-support mixing
- stable conversion from logits
- directional sampling
- solid-angle PDF evaluation

The same effective PMF is always used by sample() and pdf().
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neural_path_guiding.core.bins import (
    FloatArray,
    HemisphereBins,
)
from neural_path_guiding.core.pdf import (
    evaluate_pdf_from_probabilities,
    mix_with_uniform,
)
from neural_path_guiding.core.sampling import (
    DirectionalSample,
    sample_direction,
)


@dataclass(frozen=True)
class DiscreteDirectionalDistribution:
    """Directional PMF with consistent sampling and PDF evaluation."""

    bins: HemisphereBins
    probabilities: FloatArray
    uniform_mix: float = 0.0

    def __post_init__(self) -> None:
        effective_probabilities = mix_with_uniform(
            probabilities=self.probabilities,
            alpha=self.uniform_mix,
            expected_size=self.bins.n_bins,
        )

        stored_probabilities = np.array(
            effective_probabilities,
            dtype=np.float64,
            copy=True,
        )
        stored_probabilities.setflags(
            write=False
        )

        object.__setattr__(
            self,
            "probabilities",
            stored_probabilities,
        )
        object.__setattr__(
            self,
            "uniform_mix",
            float(self.uniform_mix),
        )

    @classmethod
    def from_logits(
        cls,
        *,
        bins: HemisphereBins,
        logits: FloatArray,
        uniform_mix: float = 0.0,
    ) -> DiscreteDirectionalDistribution:
        """Construct a distribution from unnormalized neural logits."""
        probabilities = softmax_logits(
            logits=logits,
            expected_size=bins.n_bins,
        )

        return cls(
            bins=bins,
            probabilities=probabilities,
            uniform_mix=uniform_mix,
        )

    def sample(
        self,
        rng: np.random.Generator,
    ) -> DirectionalSample:
        """Sample a local direction from the effective PMF."""
        return sample_direction(
            bins=self.bins,
            probabilities=self.probabilities,
            rng=rng,
            alpha=0.0,
        )

    def pdf(
        self,
        direction_local: FloatArray,
    ) -> float:
        """Evaluate the solid-angle PDF of a local direction."""
        return evaluate_pdf_from_probabilities(
            bins=self.bins,
            probabilities=self.probabilities,
            omega_local=direction_local,
        )

    def probability_for_bin(
        self,
        bin_index: int,
    ) -> float:
        """Return the effective probability assigned to one bin."""
        if (
            isinstance(bin_index, bool)
            or not isinstance(
                bin_index,
                (int, np.integer),
            )
        ):
            raise TypeError(
                "bin_index must be an integer."
            )

        if (
            bin_index < 0
            or bin_index >= self.bins.n_bins
        ):
            raise ValueError(
                "bin_index is outside the distribution."
            )

        return float(
            self.probabilities[bin_index]
        )


def softmax_logits(
    *,
    logits: FloatArray,
    expected_size: int,
) -> FloatArray:
    """Convert one finite logit vector into a stable PMF."""
    values = np.asarray(
        logits,
        dtype=np.float64,
    )

    if values.shape != (expected_size,):
        raise ValueError(
            "logits must have shape "
            f"({expected_size},), got {values.shape}."
        )

    if not bool(np.all(np.isfinite(values))):
        raise ValueError(
            "logits must contain only finite values."
        )

    shifted_values = (
        values - float(np.max(values))
    )

    exponential_values = np.exp(
        shifted_values
    )

    total = float(
        np.sum(exponential_values)
    )

    if (
        not np.isfinite(total)
        or total <= 0.0
    ):
        raise ValueError(
            "logits could not be converted into probabilities."
        )

    probabilities = (
        exponential_values / total
    )

    return np.asarray(
        probabilities,
        dtype=np.float64,
    )