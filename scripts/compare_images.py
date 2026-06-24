"""Compare two rendered images using a YAML config.

This script is intentionally thin.
Metric logic lives in src/neural_path_guiding/evaluation/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from neural_path_guiding.config import (
    load_yaml_config,
    require_mapping,
    require_string,
    resolve_project_path,
)
from neural_path_guiding.evaluation.compare import (
    compare_image_files,
    write_comparison_result,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs/simple_diffuse_compare.yaml"


def parse_arguments() -> argparse.Namespace:
    # Defines command-line options for image comparison.
    parser = argparse.ArgumentParser(
        description="Compare two rendered images and write metrics to JSON."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the comparison YAML config file.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    config_path = resolve_project_path(args.config, PROJECT_ROOT)

    config = load_yaml_config(config_path)
    comparison_config = require_mapping(config, "comparison")

    reference_path = resolve_project_path(
        require_string(comparison_config, "reference_path"),
        PROJECT_ROOT,
    )

    candidate_path = resolve_project_path(
        require_string(comparison_config, "candidate_path"),
        PROJECT_ROOT,
    )

    output_path = resolve_project_path(
        require_string(comparison_config, "output_path"),
        PROJECT_ROOT,
    )

    peak_value = parse_peak_value(comparison_config.get("peak_value", "auto"))

    result = compare_image_files(
        reference_path=reference_path,
        candidate_path=candidate_path,
        peak_value=peak_value,
    )

    write_comparison_result(result, output_path)

    print("Image comparison completed.")
    print(f"Reference: {result.reference_path}")
    print(f"Candidate: {result.candidate_path}")
    print(f"MSE: {result.mse:.8f}")
    print(f"RMSE: {result.rmse:.8f}")
    print(f"MAE: {result.mae:.8f}")
    print(f"PSNR: {result.psnr:.3f} dB")
    print(f"Metrics: {output_path}")


def parse_peak_value(value: object) -> float | None:
    # Supports either "auto" or a numeric peak value.
    if value == "auto":
        return None

    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)

    raise ValueError("peak_value must be 'auto' or a number.")


if __name__ == "__main__":
    main()