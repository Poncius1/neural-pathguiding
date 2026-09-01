"""Dataset inspection utilities.

This module checks generated NPZ datasets before training.

It verifies shapes, numeric validity and target distributions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from neural_path_guiding.data.dataset import load_dataset
from neural_path_guiding.data.validation import validate_training_arrays


Array = NDArray[Any]
Report = dict[str, Any]


def inspect_dataset_file(dataset_path: Path, top_k_bins: int = 8) -> Report:
    # Loads a dataset from disk and builds an inspection report.
    dataset = load_dataset(dataset_path)

    return inspect_dataset_arrays(
        features=dataset.features,
        targets=dataset.targets,
        metadata=dataset.metadata,
        dataset_path=dataset_path,
        top_k_bins=top_k_bins,
    )


def inspect_dataset_arrays(
    features: Array,
    targets: Array,
    metadata: dict[str, Any] | None = None,
    dataset_path: Path | None = None,
    top_k_bins: int = 8,
) -> Report:
    # Builds a compact report from dataset arrays.
    features = np.asarray(features)
    targets = np.asarray(targets)

    validate_training_arrays(features, targets)

    target_sums = targets.sum(axis=1, dtype=np.float64)
    target_entropy = compute_target_entropy(targets)
    normalized_entropy = target_entropy / np.log(float(targets.shape[1]))
    max_target_probability = targets.max(axis=1)

    average_target = targets.mean(axis=0, dtype=np.float64)
    top_bins = _top_average_bins(average_target, top_k_bins)

    return {
        "dataset_path": str(dataset_path) if dataset_path else None,
        "num_samples": int(features.shape[0]),
        "feature_dimension": int(features.shape[1]),
        "num_bins": int(targets.shape[1]),
        "features_dtype": str(features.dtype),
        "targets_dtype": str(targets.dtype),
        "feature_stats": _stats(features),
        "target_stats": _stats(targets),
        "target_sum_stats": _stats(target_sums),
        "target_entropy_stats": _stats(target_entropy),
        "normalized_entropy_stats": _stats(normalized_entropy),
        "max_probability_stats": _stats(max_target_probability),
        "top_average_bins": top_bins,
        "metadata": metadata or {},
        "warnings": [],
    }


def compute_target_entropy(targets: Array) -> NDArray[np.float64]:
    # Entropy measures how uniform or concentrated each target is.
    targets = np.asarray(targets, dtype=np.float64)
    safe_targets = np.clip(targets, 1e-12, 1.0)

    return -np.sum(targets * np.log(safe_targets), axis=1)


def format_inspection_report(report: Report) -> str:
    # Converts a report dict into readable console text.
    lines = [
        "Dataset inspection completed.",
        "",
        f"Dataset: {report['dataset_path']}",
        "",
        "Shapes",
        f"  Samples: {report['num_samples']}",
        f"  Feature dimension: {report['feature_dimension']}",
        f"  Bins: {report['num_bins']}",
        f"  Features dtype: {report['features_dtype']}",
        f"  Targets dtype: {report['targets_dtype']}",
        "",
        "Feature stats",
        *_format_stats(report["feature_stats"]),
        "",
        "Target stats",
        *_format_stats(report["target_stats"]),
        "",
        "Target row sums",
        *_format_stats(report["target_sum_stats"]),
        "",
        "Normalized target entropy",
        *_format_stats(report["normalized_entropy_stats"]),
        "",
        "Max probability per target",
        *_format_stats(report["max_probability_stats"]),
        "",
        "Top average bins",
        *_format_top_bins(report["top_average_bins"]),
    ]

    if report["warnings"]:
        lines.extend(["", "Warnings", *_format_warnings(report["warnings"])])

    if report["metadata"]:
        lines.extend(["", "Metadata", json.dumps(report["metadata"], indent=2)])

    return "\n".join(lines)


def _stats(values: Array) -> dict[str, float]:
    # Basic numeric statistics.
    values = np.asarray(values, dtype=np.float64)

    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
    }


def _top_average_bins(
    average_target: Array,
    top_k_bins: int,
) -> list[dict[str, float | int]]:
    # Finds the bins with the highest average probability.
    average_target = np.asarray(average_target, dtype=np.float64)
    top_k = min(top_k_bins, average_target.size)

    indices = np.argsort(average_target)[::-1][:top_k]

    return [
        {
            "bin_index": int(index),
            "probability": float(average_target[index]),
        }
        for index in indices
    ]


def _format_stats(stats: dict[str, float]) -> list[str]:
    return [
        f"  min:  {stats['min']:.6f}",
        f"  max:  {stats['max']:.6f}",
        f"  mean: {stats['mean']:.6f}",
        f"  std:  {stats['std']:.6f}",
    ]


def _format_top_bins(top_bins: list[dict[str, float | int]]) -> list[str]:
    return [
        f"  bin {item['bin_index']}: {item['probability']:.6f}"
        for item in top_bins
    ]


def _format_warnings(warnings: list[str]) -> list[str]:
    return [f"  - {warning}" for warning in warnings]
