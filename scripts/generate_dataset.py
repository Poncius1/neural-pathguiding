"""Generate a neural path guiding dataset from a YAML config.

This script is intentionally thin.
Dataset generation logic lives in src/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from neural_path_guiding.config import (
    load_yaml_config,
    optional_float,
    require_float,
    require_float_sequence,
    require_int,
    require_mapping,
    require_string,
    resolve_project_path,
)
from neural_path_guiding.core.bins import HemisphereBins
from neural_path_guiding.renderers.mitsuba.dataset_generator import (
    DatasetCamera,
    DatasetGenerationSettings,
    VisibilityCosineTeacherSettings,
    generate_dataset,
)
from neural_path_guiding.renderers.mitsuba.scene_loader import (
    DEFAULT_VARIANT,
    configure_mitsuba,
    load_scene_from_config,
    validate_windows_llvm_runtime,
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
    settings = build_settings_from_config(config, config_path)

    result = generate_dataset(settings)

    print("Dataset generation completed.")
    print(f"Experiment: {settings.experiment_name}")
    print(f"Output: {result.output_path}")
    print(f"Samples: {result.num_samples}")
    print(f"Feature dimension: {result.feature_dimension}")
    print(f"Bins: {result.num_bins}")


def build_settings_from_config(
    config: dict,
    config_path: Path,
) -> DatasetGenerationSettings:
    # Converts YAML data into dataset generation settings.
    mitsuba_config = require_mapping(config, "mitsuba")
    scene_config = require_mapping(config, "scene")
    camera_config = require_mapping(config, "camera")
    bins_config = require_mapping(config, "bins")
    dataset_config = require_mapping(config, "dataset")
    teacher_config = require_mapping(config, "teacher")

    variant = str(mitsuba_config.get("variant", DEFAULT_VARIANT))

    validate_windows_llvm_runtime()
    configure_mitsuba(variant)

    scene = load_scene_from_config(
        scene_config=scene_config,
        project_root=PROJECT_ROOT,
    )

    camera = DatasetCamera(
        origin=np.array(require_float_sequence(camera_config, "origin", 3)),
        target=np.array(require_float_sequence(camera_config, "target", 3)),
        up=np.array(require_float_sequence(camera_config, "up", 3)),
        fov_degrees=require_float(camera_config, "fov_degrees"),
        image_width=require_int(camera_config, "image_width"),
        image_height=require_int(camera_config, "image_height"),
    )

    bins = HemisphereBins(
        n_mu=require_int(bins_config, "n_mu"),
        n_phi=require_int(bins_config, "n_phi"),
    )

    teacher = VisibilityCosineTeacherSettings(
        samples_per_bin=require_int(teacher_config, "samples_per_bin"),
        smoothing=optional_float(teacher_config, "smoothing", default=1e-6),
        ray_epsilon=optional_float(teacher_config, "ray_epsilon", default=1e-4),
        max_distance=optional_float(teacher_config, "max_distance", default=1000.0),
        environment_weight=optional_float(
            teacher_config,
            "environment_weight",
            default=1.0,
        ),
        emitter_weight=optional_float(
            teacher_config,
            "emitter_weight",
            default=8.0,
        ),
        occluded_weight=optional_float(
            teacher_config,
            "occluded_weight",
            default=0.02,
        ),
    )

    output_path = resolve_project_path(
        require_string(dataset_config, "output_path"),
        PROJECT_ROOT,
    )

    return DatasetGenerationSettings(
        scene=scene,
        camera=camera,
        bins=bins,
        teacher=teacher,
        num_shading_points=require_int(dataset_config, "num_shading_points"),
        max_sampling_attempts=require_int(dataset_config, "max_sampling_attempts"),
        seed=require_int(dataset_config, "seed"),
        output_path=output_path,
        experiment_name=str(config.get("experiment_name", config_path.stem)),
    )


if __name__ == "__main__":
    main()