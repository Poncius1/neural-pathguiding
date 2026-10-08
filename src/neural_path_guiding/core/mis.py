"""Multiple importance sampling heuristics for directional PDFs."""

from __future__ import annotations

from math import exp, isfinite, log
from numbers import Integral, Real
from collections.abc import Sequence
import numpy as np


def balance_heuristic(
    pdf_a: float,
    pdf_b: float,
    *,
    samples_a: int = 1,
    samples_b: int = 1,
) -> float:
    """Return technique A's balance-heuristic weight.

    The weight is:

        n_a * p_a / (n_a * p_a + n_b * p_b)

    Both PDFs must be expressed with respect to the same measure, such as
    solid angle.
    """

    return _heuristic_weight(
        pdf_a,
        pdf_b,
        samples_a=samples_a,
        samples_b=samples_b,
        exponent=1.0,
    )


def power_heuristic(
    pdf_a: float,
    pdf_b: float,
    *,
    samples_a: int = 1,
    samples_b: int = 1,
    exponent: float = 2.0,
) -> float:
    """Return technique A's power-heuristic weight.

    The weight is:

        (n_a * p_a)^exponent
        ---------------------------------------------
        (n_a * p_a)^exponent + (n_b * p_b)^exponent

    Both PDFs must use the same measure. The default exponent is two.
    """

    power = _validate_exponent(exponent)

    return _heuristic_weight(
        pdf_a,
        pdf_b,
        samples_a=samples_a,
        samples_b=samples_b,
        exponent=power,
    )


def _heuristic_weight(
    pdf_a: float,
    pdf_b: float,
    *,
    samples_a: int,
    samples_b: int,
    exponent: float,
) -> float:
    density_a = _validate_pdf(pdf_a, "pdf_a")
    density_b = _validate_pdf(pdf_b, "pdf_b")
    count_a = _validate_sample_count(samples_a, "samples_a")
    count_b = _validate_sample_count(samples_b, "samples_b")

    if count_a + count_b == 0:
        raise ValueError(
            "At least one technique must have a positive sample count."
        )

    log_weight_a = _log_scaled_density(count_a, density_a)
    log_weight_b = _log_scaled_density(count_b, density_b)

    if log_weight_a == -np.inf and log_weight_b == -np.inf:
        raise ValueError(
            "Both techniques have zero weighted density."
        )

    maximum_log_weight = max(
        log_weight_a,
        log_weight_b,
    )

    scaled_a = _scaled_power(
        log_weight_a,
        maximum_log_weight,
        exponent,
    )
    scaled_b = _scaled_power(
        log_weight_b,
        maximum_log_weight,
        exponent,
    )

    return float(scaled_a / (scaled_a + scaled_b))


def _log_scaled_density(
    sample_count: int,
    pdf: float,
) -> float:
    if sample_count == 0 or pdf == 0.0:
        return -np.inf

    return log(float(sample_count)) + log(pdf)


def _scaled_power(
    log_value: float,
    maximum_log_value: float,
    exponent: float,
) -> float:
    if log_value == -np.inf:
        return 0.0

    return exp(
        exponent * (log_value - maximum_log_value)
    )


def _validate_pdf(
    value: float,
    name: str,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")

    pdf = float(value)

    if not isfinite(pdf) or pdf < 0.0:
        raise ValueError(
            f"{name} must be finite and non-negative."
        )

    return pdf


def _validate_sample_count(
    value: int,
    name: str,
) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")

    count = int(value)

    if count < 0:
        raise ValueError(
            f"{name} must be non-negative."
        )

    return count


def _validate_exponent(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("exponent must be a real number.")

    exponent = float(value)

    if not isfinite(exponent) or exponent <= 0.0:
        raise ValueError(
            "exponent must be finite and greater than zero."
        )

    return exponent


def combine_two_technique_samples(
    *,
    contribution_a: np.ndarray,
    pdf_a_at_a: float,
    pdf_b_at_a: float,
    contribution_b: np.ndarray,
    pdf_a_at_b: float,
    pdf_b_at_b: float,
) -> np.ndarray:
    """Combine one sample from each technique using the power heuristic.

    Each contribution is the integrand F(omega), before division by its
    sampling PDF. All PDFs must use the same measure, such as solid angle.
    """

    values_a = _validate_rgb_contribution(
        contribution_a,
        "contribution_a",
    )
    values_b = _validate_rgb_contribution(
        contribution_b,
        "contribution_b",
    )

    if values_a.shape != values_b.shape:
        raise ValueError(
            "Both contributions must have the same shape."
        )

    density_a_at_a = _validate_pdf(
        pdf_a_at_a,
        "pdf_a_at_a",
    )
    density_b_at_b = _validate_pdf(
        pdf_b_at_b,
        "pdf_b_at_b",
    )

    if density_a_at_a <= 0.0:
        raise ValueError(
            "pdf_a_at_a must be greater than zero."
        )

    if density_b_at_b <= 0.0:
        raise ValueError(
            "pdf_b_at_b must be greater than zero."
        )

    weight_a = power_heuristic(
        density_a_at_a,
        pdf_b_at_a,
    )
    weight_b = power_heuristic(
        density_b_at_b,
        pdf_a_at_b,
    )

    return (
        weight_a * values_a / density_a_at_a
        + weight_b * values_b / density_b_at_b
    )


def _validate_rgb_contribution(
    value: np.ndarray,
    name: str,
) -> np.ndarray:
    """Return one finite RGB contribution."""

    values = np.asarray(value, dtype=np.float64)

    if values.shape != (3,):
        raise ValueError(
            f"{name} must have shape (3,), got {values.shape}."
        )

    if not bool(np.all(np.isfinite(values))):
        raise ValueError(
            f"{name} must contain only finite values."
        )

    return values

def power_heuristic_weights(
    pdfs: Sequence[float],
    *,
    sample_counts: Sequence[int] | None = None,
    exponent: float = 2.0,
) -> np.ndarray:
    """Return normalized power-heuristic weights for multiple techniques.

    ``pdfs[i]`` is the density of technique ``i`` at the same sample
    direction. All densities must use the same measure, such as solid angle.
    """

    densities = np.asarray(pdfs, dtype=np.float64)

    if densities.ndim != 1 or densities.size < 2:
        raise ValueError("pdfs must contain at least two densities.")

    if not bool(np.all(np.isfinite(densities))):
        raise ValueError("pdfs must contain only finite values.")

    if bool(np.any(densities < 0.0)):
        raise ValueError("pdfs must be non-negative.")

    power = _validate_exponent(exponent)

    if sample_counts is None:
        counts = np.ones(densities.size, dtype=np.float64)
    else:
        if len(sample_counts) != densities.size:
            raise ValueError(
                "sample_counts must have the same length as pdfs."
            )

        counts = np.array(
            [
                _validate_sample_count(count, f"sample_counts[{index}]")
                for index, count in enumerate(sample_counts)
            ],
            dtype=np.float64,
        )

    if not bool(np.any(counts > 0.0)):
        raise ValueError("At least one technique must have samples.")

    active = (counts > 0.0) & (densities > 0.0)
    if not bool(np.any(active)):
        raise ValueError(
            "At least one technique must have a positive weighted density."
        )

    log_weights = np.full(densities.shape, -np.inf, dtype=np.float64)
    log_weights[active] = (
        power
        * (
            np.log(counts[active])
            + np.log(densities[active])
        )
    )

    maximum = float(np.max(log_weights[active]))
    scaled_weights = np.zeros_like(densities)
    scaled_weights[active] = np.exp(
        log_weights[active] - maximum
    )

    total = float(np.sum(scaled_weights))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Could not normalize MIS weights.")

    return scaled_weights / total