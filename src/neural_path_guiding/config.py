"""YAML configuration helpers.

Configs define experiments.
Code defines behavior.
"""

from __future__ import annotations

from collections.abc import Mapping
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

    if not isinstance(value, str) or value == "":
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


def resolve_project_path(path: Path | str, project_root: Path) -> Path:
    # Resolves relative paths from the project root.
    resolved_path = Path(path)

    if resolved_path.is_absolute():
        return resolved_path

    return project_root / resolved_path