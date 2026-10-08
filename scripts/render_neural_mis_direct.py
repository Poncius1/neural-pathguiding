"""Render a low-resolution single-bounce neural MIS smoke test."""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any

import mitsuba as mi

from neural_path_guiding.config import (
    load_yaml_config,
    require_mapping,
    require_string,
    resolve_project_path,
)
from neural_path_guiding.renderers.mitsuba.scene_loader import (
    DEFAULT_VARIANT,
    configure_mitsuba,
    validate_windows_llvm_runtime,
)


_mi: Any = mi

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/simple_diffuse_baseline.yaml"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a low-resolution neural MIS smoke test."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Baseline YAML config used to locate the scene.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "checkpoints/simple_diffuse_product_integrand_overfit_v1.pt"
        ),
        help="Trained guiding checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/neural_mis_direct_smoke.exr"),
        help="Output EXR path.",
    )
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--spp", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--uniform-mix", type=float, default=0.05)

    return parser.parse_args()


def _require_positive_integer(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")


def _read_variant(mitsuba_config: dict[str, Any]) -> str:
    variant = mitsuba_config.get("variant", DEFAULT_VARIANT)

    if not isinstance(variant, str) or not variant.strip():
        raise ValueError("mitsuba.variant must be a non-empty string.")

    return variant


def _set_film_resolution(
    scene: Any,
    width: int,
    height: int,
) -> None:
    scene_parameters = _mi.traverse(scene)
    parameter_names = list(scene_parameters.keys())

    film_size_key = next(
        (
            name
            for name in parameter_names
            if str(name).endswith(".film.size")
        ),
        None,
    )

    if film_size_key is None:
        available_parameters = ", ".join(
            str(name) for name in parameter_names
        )
        raise RuntimeError(
            "Could not find the film-size parameter. "
            f"Scene parameters: {available_parameters}"
        )

    scene_parameters[film_size_key] = _mi.ScalarVector2u(
        width,
        height,
    )
    scene_parameters.update()


def main() -> None:
    arguments = parse_arguments()

    _require_positive_integer(arguments.width, "width")
    _require_positive_integer(arguments.height, "height")
    _require_positive_integer(arguments.spp, "spp")

    config_path = resolve_project_path(
        arguments.config,
        PROJECT_ROOT,
    )
    checkpoint_path = resolve_project_path(
        arguments.checkpoint,
        PROJECT_ROOT,
    )
    output_path = resolve_project_path(
        arguments.output,
        PROJECT_ROOT,
    )

    config = load_yaml_config(config_path)
    mitsuba_config = require_mapping(config, "mitsuba")
    scene_config = require_mapping(config, "scene")

    if require_string(scene_config, "type") != "xml":
        raise ValueError(
            "This smoke-test script currently requires an XML scene."
        )

    scene_path = resolve_project_path(
        require_string(scene_config, "path"),
        PROJECT_ROOT,
    )

    validate_windows_llvm_runtime()
    configure_mitsuba(_read_variant(mitsuba_config))

    # Import after selecting the Mitsuba variant; this registers the plugin.
    from neural_path_guiding.renderers.mitsuba import neural_integrator  # noqa: F401

    scene = _mi.load_file(
        str(scene_path),
        parallel=False,
        optimize=False,
    )
    _set_film_resolution(
        scene,
        width=arguments.width,
        height=arguments.height,
    )

    integrator = _mi.load_dict(
        {
            "type": "neural_mis_direct",
            "checkpoint": str(checkpoint_path),
            "uniform_mix": arguments.uniform_mix,
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    image = _mi.render(
        scene,
        integrator=integrator,
        spp=arguments.spp,
        seed=arguments.seed,
    )
    elapsed_seconds = time.perf_counter() - start_time

    _mi.Bitmap(image).write(str(output_path))

    print("Neural MIS smoke render completed.")
    print(f"Scene: {scene_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Resolution: {arguments.width} x {arguments.height}")
    print(f"SPP: {arguments.spp}")
    print(f"Seed: {arguments.seed}")
    print(f"Uniform mix: {arguments.uniform_mix}")
    print(f"Render time: {elapsed_seconds:.3f} seconds")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()