"""Mitsuba scene loading utilities.

This module centralizes Mitsuba setup and scene loading.

It supports:
- variant configuration
- Windows LLVM runtime validation
- built-in Mitsuba scenes
- XML scene files
"""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
import platform
from typing import Any

import mitsuba as mi


_mi: Any = mi

DEFAULT_VARIANT = "scalar_rgb"


def configure_mitsuba(variant: str = DEFAULT_VARIANT) -> None:
    # Selects the Mitsuba rendering variant.
    _mi.set_variant(variant)


def validate_windows_llvm_runtime() -> None:
    # On Windows, Dr.Jit needs the full path to LLVM-C.dll.
    if platform.system() != "Windows":
        return

    llvm_path = os.environ.get("DRJIT_LIBLLVM_PATH")

    if llvm_path is None:
        raise RuntimeError(
            "DRJIT_LIBLLVM_PATH is not set. "
            "Set it to the full path of LLVM-C.dll."
        )

    if not Path(llvm_path).is_file():
        raise RuntimeError(
            "DRJIT_LIBLLVM_PATH does not point to a valid file: "
            f"{llvm_path}"
        )


def load_scene_from_config(
    scene_config: Mapping[str, Any],
    project_root: Path,
    parallel: bool = False,
    optimize: bool = False,
) -> Any:
    # Loads a scene using the scene section from a YAML config.
    scene_type = _read_required_string(scene_config, "type")

    if scene_type == "builtin":
        return _load_builtin_scene(
            scene_config=scene_config,
            parallel=parallel,
            optimize=optimize,
        )

    if scene_type == "xml":
        scene_path = _resolve_project_path(
            path_value=_read_required_string(scene_config, "path"),
            project_root=project_root,
        )

        return load_scene_from_file(
            scene_path=scene_path,
            parallel=parallel,
            optimize=optimize,
        )

    raise ValueError(f"Unsupported scene type: {scene_type}")


def load_scene_from_file(
    scene_path: Path,
    parallel: bool = False,
    optimize: bool = False,
) -> Any:
    # Loads a Mitsuba scene from an XML file.
    if not scene_path.is_file():
        raise FileNotFoundError(f"Mitsuba scene file not found: {scene_path}")

    return _mi.load_file(
        str(scene_path),
        parallel=parallel,
        optimize=optimize,
    )


def load_scene_from_dict(
    scene_description: Mapping[str, Any],
    parallel: bool = False,
    optimize: bool = False,
) -> Any:
    # Loads a Mitsuba scene from a Python dictionary.
    return _mi.load_dict(
        dict(scene_description),
        parallel=parallel,
        optimize=optimize,
    )


def load_cornell_box_scene(
    parallel: bool = False,
    optimize: bool = False,
) -> Any:
    # Loads Mitsuba's built-in Cornell Box scene.
    return load_scene_from_dict(
        scene_description=_mi.cornell_box(),
        parallel=parallel,
        optimize=optimize,
    )


def _load_builtin_scene(
    scene_config: Mapping[str, Any],
    parallel: bool,
    optimize: bool,
) -> Any:
    # Loads built-in scenes supported by this project.
    scene_name = _read_required_string(scene_config, "name")

    if scene_name == "cornell_box":
        return load_cornell_box_scene(
            parallel=parallel,
            optimize=optimize,
        )

    raise ValueError(f"Unsupported built-in scene: {scene_name}")


def _resolve_project_path(path_value: str, project_root: Path) -> Path:
    # Resolves relative scene paths from the project root.
    path = Path(path_value)

    if path.is_absolute():
        return path

    return project_root / path


def _read_required_string(config: Mapping[str, Any], key: str) -> str:
    # Reads a required string value from a small config section.
    value = config.get(key)

    if not isinstance(value, str) or value == "":
        raise ValueError(f"Missing or invalid string value: {key}")

    return value