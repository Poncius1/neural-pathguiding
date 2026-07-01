"""Inspect Mitsuba scene intersections and extracted features.

This script verifies that we can:
- load a Mitsuba scene from YAML
- shoot simple rays into the scene
- read surface position and normal
- convert intersections into feature vectors

It is a bridge toward dataset generation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import mitsuba as mi
import numpy as np

from neural_path_guiding.config import (
    load_yaml_config,
    require_mapping,
    resolve_project_path,
)
from neural_path_guiding.core.features import FEATURE_NAMES
from neural_path_guiding.renderers.mitsuba.adapters import (
    normalize_numpy_vector,
    surface_interaction_to_features,
)
from neural_path_guiding.renderers.mitsuba.scene_loader import (
    DEFAULT_VARIANT,
    configure_mitsuba,
    load_scene_from_config,
    validate_windows_llvm_runtime,
)


_mi: Any = mi

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs/simple_diffuse_baseline.yaml"


def parse_arguments() -> argparse.Namespace:
    # Defines command-line options for this script.
    parser = argparse.ArgumentParser(
        description="Inspect Mitsuba intersections and extracted feature vectors."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to a render YAML config.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    config_path = resolve_project_path(args.config, PROJECT_ROOT)

    config = load_yaml_config(config_path)
    mitsuba_config = require_mapping(config, "mitsuba")
    scene_config = require_mapping(config, "scene")

    variant = str(mitsuba_config.get("variant", DEFAULT_VARIANT))

    validate_windows_llvm_runtime()
    configure_mitsuba(variant)

    scene = load_scene_from_config(
        scene_config=scene_config,
        project_root=PROJECT_ROOT,
    )

    inspect_scene(scene)


def inspect_scene(scene: Any) -> None:
    # Shoots a few deterministic rays into the scene.
    ray_specs = [
        ("center_sphere", np.array([0.0, 1.15, 4.5]), np.array([0.0, 0.55, 0.0])),
        ("floor", np.array([0.0, 1.15, 4.5]), np.array([0.0, 0.0, 0.0])),
        ("back_wall", np.array([0.0, 1.15, 4.5]), np.array([0.0, 1.0, -2.5])),
    ]

    print("Inspecting scene intersections.")

    for name, origin, target in ray_specs:
        direction = normalize_numpy_vector(target - origin)
        ray = _mi.Ray3f(origin, direction)

        surface_interaction = scene.ray_intersect(ray)

        print(f"\nRay target: {name}")

        if not bool(surface_interaction.is_valid()):
            print("No intersection.")
            continue

        outgoing_direction = -direction

        features = surface_interaction_to_features(
            surface_interaction=surface_interaction,
            outgoing_direction=outgoing_direction,
            bounce_depth=0,
        )

        feature_vector = features.to_array()

        print(f"Position: {features.position}")
        print(f"Normal:   {features.normal}")
        print(f"Outgoing: {features.outgoing_direction}")
        print("Feature vector:")

        for feature_name, value in zip(FEATURE_NAMES, feature_vector, strict=True):
            print(f"  {feature_name}: {value:.6f}")


if __name__ == "__main__":
    main()