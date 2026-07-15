"""Inspect a generated dataset.

This script reads a YAML config and prints useful dataset statistics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from neural_path_guiding.config import (
    load_yaml_config,
    optional_int,
    require_mapping,
    require_string,
    resolve_project_path,
)
from neural_path_guiding.evaluation.dataset_inspection import (
    format_inspection_report,
    inspect_dataset_file,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs/simple_diffuse_dataset_inspect.yaml"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a dataset NPZ file.")

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the dataset inspection config.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    config_path = resolve_project_path(args.config, PROJECT_ROOT)

    config = load_yaml_config(config_path)
    dataset_config = require_mapping(config, "dataset")
    inspection_config = config.get("inspection", {})

    if not isinstance(inspection_config, dict):
        raise ValueError("inspection must be a config section.")

    dataset_path = resolve_project_path(
        require_string(dataset_config, "path"),
        PROJECT_ROOT,
    )

    report = inspect_dataset_file(
        dataset_path=dataset_path,
        top_k_bins=optional_int(inspection_config, "top_k_bins", default=8),
    )

    print(format_inspection_report(report))


if __name__ == "__main__":
    main()