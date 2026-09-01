"""Directional sampling from discrete hemisphere-bin probabilities.

The sampler uses one effective PMF for both bin selection and PDF evaluation:

    p(omega) = P(bin(omega)) / solid_angle(bin)

Directions are expressed in the local frame where +Z is the surface normal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neural_path_guiding.core.bins import FloatArray, HemisphereBins
from neural_path_guiding.core.pdf import mix_with_uniform


_DIRECTION_TOLERANCE = 1e-10


@dataclass(frozen=True)
class DirectionalSample:
    """One local direction sampled from a discrete guiding distribution."""

    direction_local: FloatArray
    bin_index: int
    pdf: float

    def __post_init__(self) -> None:
        direction = np.asarray(self.direction_local, dtype=np.float64)

        if direction.shape != (3,):
            raise ValueError(
                f"direction_local must have shape (3,), got {direction.shape}."
            )

        if not bool(np.all(np.isfinite(direction))):
            raise ValueError("direction_local must contain only finite values.")

        if not np.isclose(
            np.linalg.norm(direction),
            1.0,
            atol=_DIRECTION_TOLERANCE,
            rtol=0.0,
        ):
            raise ValueError("direction_local must have unit length.")

        if float(direction[2]) < -_DIRECTION_TOLERANCE:
            raise ValueError("direction_local must lie in the upper hemisphere.")

        if isinstance(self.bin_index, bool) or not isinstance(
            self.bin_index,
            (int, np.integer),
        ):
            raise TypeError("bin_index must be an integer.")

        if self.bin_index < 0:
            raise ValueError("bin_index must be non-negative.")

        if not np.isfinite(self.pdf) or self.pdf <= 0.0:
            raise ValueError("pdf must be finite and greater than zero.")

        stored_direction = np.array(direction, dtype=np.float64, copy=True)
        stored_direction.setflags(write=False)
        object.__setattr__(self, "direction_local", stored_direction)
        object.__setattr__(self, "bin_index", int(self.bin_index))
        object.__setattr__(self, "pdf", float(self.pdf))


def sample_direction(
    bins: HemisphereBins,
    probabilities: FloatArray,
    rng: np.random.Generator,
    alpha: float = 0.0,
) -> DirectionalSample:
    """Sample a bin and a uniform-solid-angle direction inside it.

    ``alpha`` mixes the input probabilities with the uniform bin PMF. The same
    mixed PMF determines both the selected bin and the returned solid-angle PDF.
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be an instance of numpy.random.Generator.")

    effective_probabilities = mix_with_uniform(
        probabilities=probabilities,
        alpha=alpha,
        expected_size=bins.n_bins,
    )

    bin_index = int(rng.choice(bins.n_bins, p=effective_probabilities))
    random_values = rng.random(2)
    direction_local = bins.sample_direction_in_bin(
        bin_index=bin_index,
        u_mu=float(random_values[0]),
        u_phi=float(random_values[1]),
    )
    pdf = float(effective_probabilities[bin_index] / bins.bin_solid_angle)

    return DirectionalSample(
        direction_local=direction_local,
        bin_index=bin_index,
        pdf=pdf,
    )
