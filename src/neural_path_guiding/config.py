"""YAML configuration helpers.

Configs define experiments.
Code defines behavior.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from pathlib import Path
from typing import Any

import yaml


ConfigData = dict[str, Any]


def load_yaml_config(config_path: Path) -> ConfigData:
    # Loads a YAML config file.
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")

    return dict(data)


def require_mapping(config: Mapping[str, Any], key: str) -> dict[str, Any]:
    # Reads a required nested section.
    value = config.get(key)

    if not isinstance(value, dict):
        raise ValueError(f"Missing or invalid config section: {key}")

    return dict(value)


def require_string(config: Mapping[str, Any], key: str) -> str:
    # Reads a required string value.
    value = config.get(key)

    if not isinstance(value, str) or value.strip() == "":
        raise ValueError(f"Missing or invalid string value: {key}")

    return value


def require_int(config: Mapping[str, Any], key: str) -> int:
    # Reads a required integer value.
    value = config.get(key)

    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"Missing or invalid integer value: {key}")

    return value


def optional_int(config: Mapping[str, Any], key: str, default: int) -> int:
    # Reads an optional integer value.
    value = config.get(key, default)

    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"Invalid integer value: {key}")

    return value


def require_float(config: Mapping[str, Any], key: str) -> float:
    # Reads a required numeric value.
    value = config.get(key)

    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"Missing or invalid numeric value: {key}")

    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError(f"Numeric value must be finite: {key}")

    return numeric_value


def optional_float(config: Mapping[str, Any], key: str, default: float) -> float:
    # Reads an optional numeric value.
    value = config.get(key, default)

    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"Invalid numeric value: {key}")

    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError(f"Numeric value must be finite: {key}")

    return numeric_value


def require_float_sequence(
    config: Mapping[str, Any],
    key: str,
    length: int,
) -> list[float]:
    # Reads a required numeric list.
    value = config.get(key)

    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ValueError(f"Missing or invalid numeric sequence: {key}")

    if len(value) != length:
        raise ValueError(f"{key} must have length {length}, got {len(value)}.")

    if any(isinstance(item, bool) for item in value):
        raise ValueError(f"Invalid numeric sequence: {key}")

    try:
        numeric_values = [float(item) for item in value]
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid numeric sequence: {key}") from error

    if not all(math.isfinite(item) for item in numeric_values):
        raise ValueError(f"Numeric sequence must contain only finite values: {key}")

    return numeric_values


def resolve_project_path(path: Path | str, project_root: Path) -> Path:
    # Resolves relative paths from the project root.
    resolved_path = Path(path)

    if resolved_path.is_absolute():
        return resolved_path

    return project_root / resolved_path