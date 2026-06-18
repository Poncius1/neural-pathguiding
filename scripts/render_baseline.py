"""Render a baseline Mitsuba scene from a YAML config.

This script is intentionally thin.
Experiment parameters live in configs/.
Rendering logic lives in src/.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neural_path_guiding.config import (
    load_yaml_config,
    optional_int,
    require_int,
    require_mapping,
    require_string,
    resolve_project_path,
)
from neural_path_guiding.renderers.mitsuba.baseline_renderer import render_scene
from neural_path_guiding.renderers.mitsuba.scene_loader import (
    DEFAULT_VARIANT,
    configure_mitsuba,
    load_scene_from_config,
    validate_windows_llvm_runtime,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs/cornell_baseline.yaml"


# Parsed baseline experiment settings.
@dataclass(frozen=True)
class BaselineRenderSettings:
    experiment_name: str
    variant: str
    scene_config: dict[str, Any]
    spp: int
    seed: int
    output_path: Path


def parse_arguments() -> argparse.Namespace:
    # Defines command-line options for this script.
    parser = argparse.ArgumentParser(
        description="Render a Mitsuba baseline from a YAML config."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the YAML config file.",
    )

    return parser.parse_args()


def load_baseline_settings(config_path: Path) -> BaselineRenderSettings:
    # Converts a YAML config into typed baseline settings.
    config = load_yaml_config(config_path)

    mitsuba_config = require_mapping(config, "mitsuba")
    scene_config = require_mapping(config, "scene")
    render_config = require_mapping(config, "render")

    output_path = resolve_project_path(
        path=require_string(render_config, "output_path"),
        project_root=PROJECT_ROOT,
    )

    return BaselineRenderSettings(
        experiment_name=str(config.get("experiment_name", "baseline_render")),
        variant=str(mitsuba_config.get("variant", DEFAULT_VARIANT)),
        scene_config=scene_config,
        spp=require_int(render_config, "spp"),
        seed=optional_int(render_config, "seed", default=0),
        output_path=output_path,
    )


def run_baseline_render(settings: BaselineRenderSettings, config_path: Path) -> None:
    # Executes the baseline render pipeline.
    validate_windows_llvm_runtime()
    configure_mitsuba(settings.variant)

    scene = load_scene_from_config(
        scene_config=settings.scene_config,
        project_root=PROJECT_ROOT,
    )

    result = render_scene(
        scene=scene,
        output_path=settings.output_path,
        spp=settings.spp,
        seed=settings.seed,
    )

    print("Baseline render completed.")
    print(f"Experiment: {settings.experiment_name}")
    print(f"Config: {config_path}")
    print(f"Variant: {settings.variant}")
    print(f"Samples per pixel: {result.spp}")
    print(f"Seed: {result.seed}")
    print(f"Render time: {result.elapsed_seconds:.3f} seconds")
    print(f"Output: {result.output_path}")


def main() -> None:
    args = parse_arguments()
    config_path = resolve_project_path(args.config, PROJECT_ROOT)

    settings = load_baseline_settings(config_path)
    run_baseline_render(settings, config_path)


if __name__ == "__main__":
    main()