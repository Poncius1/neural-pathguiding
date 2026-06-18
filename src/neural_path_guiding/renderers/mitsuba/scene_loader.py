"""Mitsuba scene loading utilities.

This module centralizes Mitsuba setup and scene loading.

It provides:
- Mitsuba variant configuration
- Windows LLVM runtime validation
- stable scene loading from dictionaries
- a built-in Cornell Box loader for first baselines
"""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
import platform
from typing import Any

import mitsuba as mi


# Mitsuba exposes dynamic Python bindings. Using Any avoids false type errors.
_mi: Any = mi

DEFAULT_VARIANT = "scalar_rgb"


def configure_mitsuba(variant: str = DEFAULT_VARIANT) -> None:
    # Selects the Mitsuba rendering variant.
    _mi.set_variant(variant)


def validate_windows_llvm_runtime() -> None:
    # On Windows, Dr.Jit needs this path to render correctly.
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