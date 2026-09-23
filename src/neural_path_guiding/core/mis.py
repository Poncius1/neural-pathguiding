"""Multiple importance sampling heuristics for directional PDFs."""

from __future__ import annotations

from math import exp, isfinite, log
from numbers import Integral, Real

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