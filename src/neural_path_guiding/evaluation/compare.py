"""Image comparison utilities.

This module compares two rendered images and writes metric results.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

from neural_path_guiding.evaluation.image_io import load_image_as_array
from neural_path_guiding.evaluation.metrics import (
    automatic_peak_value,
    mean_absolute_error,
    mean_squared_error,
    peak_signal_to_noise_ratio,
    root_mean_squared_error,
)


# Result produced by an image comparison.
@dataclass(frozen=True)
class ImageComparisonResult:
    reference_path: Path
    candidate_path: Path
    mse: float
    rmse: float
    mae: float
    psnr: float
    peak_value: float


def compare_image_files(
    reference_path: Path,
    candidate_path: Path,
    peak_value: float | None = None,
) -> ImageComparisonResult:
    # Loads two images and computes quality metrics.
    reference = load_image_as_array(reference_path)
    candidate = load_image_as_array(candidate_path)

    resolved_peak_value = (
        automatic_peak_value(reference) if peak_value is None else peak_value
    )

    mse = mean_squared_error(reference, candidate)
    rmse = root_mean_squared_error(reference, candidate)
    mae = mean_absolute_error(reference, candidate)
    psnr = peak_signal_to_noise_ratio(mse, resolved_peak_value)

    return ImageComparisonResult(
        reference_path=reference_path,
        candidate_path=candidate_path,
        mse=mse,
        rmse=rmse,
        mae=mae,
        psnr=psnr,
        peak_value=resolved_peak_value,
    )


def write_comparison_result(
    result: ImageComparisonResult,
    output_path: Path,
) -> None:
    # Writes comparison metrics to JSON.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = asdict(result)
    data["reference_path"] = str(result.reference_path)
    data["candidate_path"] = str(result.candidate_path)
    data["psnr"] = _json_safe_number(result.psnr)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def _json_safe_number(value: float) -> float | str:
    # JSON has no portable infinity value.
    if math.isinf(value):
        return "Infinity"

    return value