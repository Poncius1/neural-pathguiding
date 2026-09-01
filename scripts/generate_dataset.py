"""Generate a neural path guiding dataset from a YAML config.

This script is intentionally thin.
Dataset generation logic lives in src/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from neural_path_guiding.config import (
    load_yaml_config,
    resolve_project_path,
)
from neural_path_guiding.renderers.mitsuba.dataset_config import (
    build_settings_from_config,
)
from neural_path_guiding.renderers.mitsuba.dataset_generator import (
    generate_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs/simple_diffuse_dataset.yaml"


def parse_arguments() -> argparse.Namespace:
    # Defines command-line options for dataset generation.
    parser = argparse.ArgumentParser(
        description="Generate a dataset for neural path guiding."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the dataset YAML config.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    config_path = resolve_project_path(args.config, PROJECT_ROOT)

    config = load_yaml_config(config_path)
    settings = build_settings_from_config(
        config=config,
        config_path=config_path,
        project_root=PROJECT_ROOT,
    )

    result = generate_dataset(settings)

    print("Dataset generation completed.")
    print(f"Experiment: {settings.experiment_name}")
    print(f"Output: {result.output_path}")
    print(f"Samples: {result.num_samples}")
    print(f"Feature dimension: {result.feature_dimension}")
    print(f"Bins: {result.num_bins}")


if __name__ == "__main__":
    main()
