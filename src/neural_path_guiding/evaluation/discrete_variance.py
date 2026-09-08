"""Variance analysis for discrete importance-sampling estimators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class DiscreteEstimatorStatistics:
    """Exact statistics of a discrete Monte Carlo estimator."""

    exact_integral: float
    single_sample_variance: float
    estimator_variance: float
    standard_error: float
    relative_standard_error: float


def calculate_discrete_estimator_statistics(
    contributions: FloatArray,
    probabilities: FloatArray,
    *,
    bin_solid_angle: float,
    sample_count: int,
) -> DiscreteEstimatorStatistics:
    """Calculate the exact variance of an importance-sampling estimator."""

    values, proposal = _prepare_distribution(
        contributions,
        probabilities,
    )
    solid_angle = _validate_solid_angle(
        bin_solid_angle
    )
    count = _validate_positive_integer(
        sample_count,
        "sample_count",
    )

    exact_integral = float(
        solid_angle * np.sum(values)
    )

    positive_contributions = values > 0.0

    second_moment = float(
        solid_angle**2
        * np.sum(
            values[positive_contributions] ** 2
            / proposal[positive_contributions]
        )
    )

    single_sample_variance = max(
        0.0,
        second_moment - exact_integral**2,
    )
    estimator_variance = (
        single_sample_variance / count
    )
    standard_error = float(
        np.sqrt(estimator_variance)
    )

    if exact_integral > 0.0:
        relative_standard_error = (
            standard_error / exact_integral
        )
    else:
        relative_standard_error = 0.0

    return DiscreteEstimatorStatistics(
        exact_integral=exact_integral,
        single_sample_variance=single_sample_variance,
        estimator_variance=estimator_variance,
        standard_error=standard_error,
        relative_standard_error=relative_standard_error,
    )


def simulate_discrete_estimates(
    contributions: FloatArray,
    probabilities: FloatArray,
    *,
    bin_solid_angle: float,
    samples_per_estimate: int,
    repetitions: int,
    rng: np.random.Generator,
) -> FloatArray:
    """Generate repeated Monte Carlo estimates of a discrete integral."""

    if not isinstance(rng, np.random.Generator):
        raise TypeError(
            "rng must be a numpy.random.Generator."
        )

    values, proposal = _prepare_distribution(
        contributions,
        probabilities,
    )
    solid_angle = _validate_solid_angle(
        bin_solid_angle
    )
    sample_count = _validate_positive_integer(
        samples_per_estimate,
        "samples_per_estimate",
    )
    repetition_count = _validate_positive_integer(
        repetitions,
        "repetitions",
    )

    estimates = np.empty(
        repetition_count,
        dtype=np.float64,
    )

    for repetition in range(repetition_count):
        selected_bins = rng.choice(
            proposal.size,
            size=sample_count,
            replace=True,
            p=proposal,
        )

        sample_values = (
            solid_angle
            * values[selected_bins]
            / proposal[selected_bins]
        )

        estimates[repetition] = float(
            np.mean(sample_values)
        )

    return estimates


def _prepare_distribution(
    contributions: FloatArray,
    probabilities: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Validate contributions and normalize the proposal PMF."""

    values = np.asarray(
        contributions,
        dtype=np.float64,
    )
    proposal = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    if values.ndim != 1:
        raise ValueError(
            "contributions must be one-dimensional."
        )

    if proposal.shape != values.shape:
        raise ValueError(
            "probabilities must have the same shape "
            "as contributions."
        )

    if values.size == 0:
        raise ValueError(
            "contributions must not be empty."
        )

    if not bool(
        np.all(np.isfinite(values))
    ):
        raise ValueError(
            "contributions must contain only finite values."
        )

    if not bool(
        np.all(np.isfinite(proposal))
    ):
        raise ValueError(
            "probabilities must contain only finite values."
        )

    if bool(np.any(values < 0.0)):
        raise ValueError(
            "contributions must be non-negative."
        )

    if bool(np.any(proposal < 0.0)):
        raise ValueError(
            "probabilities must be non-negative."
        )

    probability_sum = float(
        np.sum(proposal)
    )

    if probability_sum <= 0.0:
        raise ValueError(
            "probabilities must have a positive sum."
        )

    normalized_proposal = (
        proposal / probability_sum
    )

    unsupported_contributions = (
        (values > 0.0)
        & (normalized_proposal <= 0.0)
    )

    if bool(
        np.any(unsupported_contributions)
    ):
        raise ValueError(
            "Every positive contribution must have "
            "positive sampling probability."
        )

    return (
        np.array(
            values,
            dtype=np.float64,
            copy=True,
        ),
        np.array(
            normalized_proposal,
            dtype=np.float64,
            copy=True,
        ),
    )


def _validate_solid_angle(
    bin_solid_angle: float,
) -> float:
    """Validate the equal-bin solid angle."""

    if (
        isinstance(bin_solid_angle, bool)
        or not isinstance(
            bin_solid_angle,
            (int, float, np.integer, np.floating),
        )
    ):
        raise TypeError(
            "bin_solid_angle must be a real number."
        )

    value = float(bin_solid_angle)

    if (
        not np.isfinite(value)
        or value <= 0.0
    ):
        raise ValueError(
            "bin_solid_angle must be finite and positive."
        )

    return value


def _validate_positive_integer(
    value: int,
    name: str,
) -> int:
    """Validate a positive integer setting."""

    if (
        isinstance(value, bool)
        or not isinstance(
            value,
            (int, np.integer),
        )
    ):
        raise TypeError(
            f"{name} must be an integer."
        )

    integer_value = int(value)

    if integer_value <= 0:
        raise ValueError(
            f"{name} must be positive."
        )

    return integer_value