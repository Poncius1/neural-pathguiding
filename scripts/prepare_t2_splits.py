"""Inspect and split a T2 neural path guiding dataset.

The original NPZ dataset is never modified.

Splits are grouped by camera pixel so repeated samples
from one pixel cannot cross train/validation/test boundaries.

This does not guarantee spatial independence between
neighboring pixels or between scenes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from neural_path_guiding.data.dataset import load_dataset


IndexArray = NDArray[np.int64]

EXPECTED_TARGET_TYPE = "product_integrand_v1"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect and split a T2 dataset."
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "data/simple_diffuse_t2_pilot_v1.npz"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/experiments/t2_pilot_analysis"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
    )

    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
    )

    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def grouped_pixel_split(
    pixels: np.ndarray,
    *,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> dict[str, IndexArray]:
    """Split row indices without sharing any pixel across splits."""

    pixels = np.asarray(pixels)

    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError(
            "pixels must have shape (N, 2)."
        )

    if pixels.shape[0] < 3:
        raise ValueError(
            "At least three samples are required."
        )

    if not np.issubdtype(pixels.dtype, np.integer):
        raise TypeError(
            "Pixel coordinates must be integers."
        )

    if np.any(pixels < 0):
        raise ValueError(
            "Pixel coordinates must be non-negative."
        )

    fractions = (train_fraction, val_fraction)

    if not all(np.isfinite(value) for value in fractions):
        raise ValueError(
            "Split fractions must be finite."
        )

    if (
        train_fraction <= 0.0
        or val_fraction <= 0.0
        or train_fraction + val_fraction >= 1.0
    ):
        raise ValueError(
            "Expected positive train and validation fractions "
            "with a positive remaining test fraction."
        )

    if isinstance(seed, bool) or not isinstance(
        seed, (int, np.integer)
    ) or seed < 0:
        raise ValueError(
            "seed must be a non-negative integer."
        )

    unique_pixels, group_ids = np.unique(
        pixels,
        axis=0,
        return_inverse=True,
    )

    group_count = len(unique_pixels)

    if group_count < 3:
        raise ValueError(
            "At least three distinct pixels are required "
            "for grouped splitting."
        )

    rng = np.random.default_rng(int(seed))
    shuffled_groups = rng.permutation(group_count)

    # Allocate at least one group to each split.
    n_train = max(
        1,
        int(np.floor(group_count * train_fraction)),
    )

    n_val = max(
        1,
        int(np.floor(group_count * val_fraction)),
    )

    if n_train + n_val >= group_count:
        n_train = min(
            n_train,
            group_count - 2,
        )
        n_val = min(
            n_val,
            group_count - n_train - 1,
        )

    train_groups = shuffled_groups[:n_train]

    val_groups = shuffled_groups[
        n_train:n_train + n_val
    ]

    test_groups = shuffled_groups[
        n_train + n_val:
    ]

    splits = {
        "train": np.flatnonzero(
            np.isin(group_ids, train_groups)
        ).astype(np.int64),
        "validation": np.flatnonzero(
            np.isin(group_ids, val_groups)
        ).astype(np.int64),
        "test": np.flatnonzero(
            np.isin(group_ids, test_groups)
        ).astype(np.int64),
    }

    validate_splits(
        pixels=pixels,
        splits=splits,
    )

    return splits


def validate_splits(
    *,
    pixels: np.ndarray,
    splits: dict[str, IndexArray],
) -> None:
    """Ensure complete coverage and no pixel leakage."""

    expected_names = {
        "train",
        "validation",
        "test",
    }

    if set(splits) != expected_names:
        raise ValueError(
            "Unexpected split names."
        )

    n_rows = len(pixels)
    pixel_sets: dict[str, set[tuple[int, int]]] = {}
    index_sets: dict[str, set[int]] = {}

    for name, indices in splits.items():
        values = np.asarray(indices)

        if values.ndim != 1:
            raise ValueError(
                f"{name} indices must be one-dimensional."
            )

        if not np.issubdtype(values.dtype, np.integer):
            raise TypeError(
                f"{name} indices must be integers."
            )

        if len(values) == 0:
            raise ValueError(
                f"{name} split cannot be empty."
            )

        if np.any(values < 0) or np.any(values >= n_rows):
            raise ValueError(
                f"{name} has out-of-range indices."
            )

        index_sets[name] = {
            int(index) for index in values
        }

        if len(index_sets[name]) != len(values):
            raise ValueError(
                f"{name} contains duplicate row indices."
            )

        pixel_sets[name] = {
            (int(x), int(y))
            for x, y in pixels[values]
        }

    names = ("train", "validation", "test")

    for i, first in enumerate(names):
        for second in names[i + 1:]:
            if index_sets[first] & index_sets[second]:
                raise ValueError(
                    "Row indices overlap across splits."
                )

            if pixel_sets[first] & pixel_sets[second]:
                raise ValueError(
                    "Camera pixel leakage across splits."
                )

    all_indices = set().union(*index_sets.values())

    if all_indices != set(range(n_rows)):
        raise ValueError(
            "Splits do not cover the complete dataset."
        )


def analyze_targets(
    targets: np.ndarray,
    contributions: np.ndarray,
) -> dict[str, Any]:
    """Summarize normalized targets and raw teacher signals."""

    probabilities = np.asarray(
        targets,
        dtype=np.float64,
    )

    values = np.asarray(
        contributions,
        dtype=np.float64,
    )

    if (
        probabilities.ndim != 2
        or values.shape != probabilities.shape
        or probabilities.shape[0] == 0
        or probabilities.shape[1] == 0
    ):
        raise ValueError(
            "Targets and contributions require matching "
            "nonempty 2D shapes."
        )

    if (
        not np.all(np.isfinite(probabilities))
        or not np.all(np.isfinite(values))
        or np.any(probabilities < 0.0)
        or np.any(values < 0.0)
    ):
        raise ValueError(
            "Targets and contributions must be "
            "finite and non-negative."
        )

    if not np.allclose(
        probabilities.sum(axis=1),
        1.0,
        atol=1e-4,
        rtol=0.0,
    ):
        raise ValueError(
            "Each target distribution must sum to one."
        )

    n_bins = probabilities.shape[1]
    uniform_entropy = np.log(n_bins)

    safe_log = np.zeros_like(probabilities)

    np.log(
        probabilities,
        out=safe_log,
        where=probabilities > 0.0,
    )

    entropy = -np.sum(
        probabilities * safe_log,
        axis=1,
    )

    if uniform_entropy > 0.0:
        normalized_entropy = entropy / uniform_entropy
    else:
        normalized_entropy = np.zeros_like(entropy)

    raw_sums = values.sum(axis=1)

    return {
        "samples": int(probabilities.shape[0]),
        "bins": int(n_bins),
        "non_finite_values": 0,
        "target_sum_min": float(
            np.min(probabilities.sum(axis=1))
        ),
        "target_sum_max": float(
            np.max(probabilities.sum(axis=1))
        ),
        "mean_normalized_entropy": float(
            np.mean(normalized_entropy)
        ),
        "mean_max_bin_probability": float(
            np.mean(np.max(probabilities, axis=1))
        ),
        "zero_raw_contribution_rows": int(
            np.count_nonzero(raw_sums == 0.0)
        ),
        "mean_nonzero_bins_per_row": float(
            np.mean(np.count_nonzero(values > 0.0, axis=1))
        ),
        "raw_contribution_mean": float(
            np.mean(values)
        ),
        "raw_contribution_max": float(
            np.max(values)
        ),
    }


def main() -> None:
    args = parse_arguments()

    dataset = load_dataset(args.dataset)

    target_type = dataset.metadata["target_type"]

    if target_type != EXPECTED_TARGET_TYPE:
        raise ValueError(
            f"Expected target_type='{EXPECTED_TARGET_TYPE}', "
            f"received '{target_type}'."
        )

    splits = grouped_pixel_split(
        dataset.pixels,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    quality = analyze_targets(
        dataset.targets,
        dataset.mean_contributions,
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    split_path = args.output_dir / "split_indices.npz"

    np.savez_compressed(
        split_path,
        train=splits["train"],
        validation=splits["validation"],
        test=splits["test"],
    )

    split_summary = {
        name: {
            "samples": int(len(indices)),
            "unique_pixels": int(
                len(np.unique(
                    dataset.pixels[indices],
                    axis=0,
                ))
            ),
        }
        for name, indices in splits.items()
    }

    report = {
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256_file(args.dataset),
        "target_type": target_type,
        "experiment_name": dataset.metadata["experiment_name"],
        "feature_schema_version": dataset.metadata.get(
            "feature_schema_version"
        ),
        "dataset_format_version": dataset.metadata[
            "dataset_format_version"
        ],
        "seed": args.seed,
        "requested_train_fraction": args.train_fraction,
        "requested_validation_fraction": args.val_fraction,
        "split_strategy": "grouped_by_exact_camera_pixel",
        "total_samples": dataset.num_samples,
        "unique_pixels": int(
            len(np.unique(dataset.pixels, axis=0))
        ),
        "splits": split_summary,
        "target_quality": quality,
        "limitations": [
            "Neighboring pixels may still be strongly correlated.",
            "All samples belong to one scene.",
            "Samples are first-hit camera intersections.",
            "Current features may use default material properties.",
            "Raw T2 targets contain Monte Carlo estimation noise.",
        ],
    }

    report_path = args.output_dir / "t2_dataset_report.json"

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
        )

    print()
    print("=== T2 DATASET QUALITY ===")

    for key, value in quality.items():
        print(f"{key}: {value}")

    print()
    print("=== GROUPED SPLITS ===")

    for name, details in split_summary.items():
        print(
            f"{name}: "
            f"{details['samples']} samples, "
            f"{details['unique_pixels']} unique pixels"
        )

    print()
    print(f"Split indices: {split_path.resolve()}")
    print(f"Report: {report_path.resolve()}")
    print("T2 preparation completed.")


if __name__ == "__main__":
    main()