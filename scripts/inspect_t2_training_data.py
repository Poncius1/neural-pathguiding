"""Phase 3.2.1: validate stratified T2 data before MLP training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from neural_path_guiding.training.t2_data import prepare_t2_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=Path,
        default=Path("data/simple_diffuse_t2_stratified_medium_v2.npz"),
    )
    parser.add_argument(
        "--splits", type=Path,
        default=Path("data/simple_diffuse_t2_stratified_medium_v2_splits.npz"),
    )
    parser.add_argument(
        "--report", type=Path,
        default=Path("outputs/experiments/t2_training_data_v2/report.json"),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = prepare_t2_data(args.dataset, args.splits)
    loaders = data.make_dataloaders(batch_size=args.batch_size, seed=args.seed)
    feature_mean = data.feature_mean.numpy()
    feature_scale = data.feature_scale.numpy()
    train_normalized = data.train.features.numpy()

    report = {
        "phase": "3.2.1",
        "dataset": str(args.dataset.resolve()),
        "splits": str(args.splits.resolve()),
        "dataset_sha256": data.dataset_sha256,
        "splits_sha256": data.splits_sha256,
        "feature_schema_version": data.metadata["feature_schema_version"],
        "target_type": data.metadata["target_type"],
        "feature_dimension": data.feature_dimension,
        "num_bins": data.num_bins,
        "counts": {
            "train": data.train.num_samples,
            "validation": data.validation.num_samples,
            "test": data.test.num_samples,
        },
        "batch_counts": {name: len(loader) for name, loader in loaders.items()},
        "feature_mean_train_only": feature_mean.tolist(),
        "feature_scale_train_only": feature_scale.tolist(),
        "train_normalized_max_abs_mean": float(
            np.abs(train_normalized.mean(axis=0)).max()
        ),
        "checks": {
            "finite_features": all(
                bool(np.isfinite(split.features.numpy()).all())
                for split in (data.train, data.validation, data.test)
            ),
            "finite_targets": all(
                bool(np.isfinite(split.targets.numpy()).all())
                for split in (data.train, data.validation, data.test)
            ),
            "positive_scales": bool(np.all(feature_scale > 0)),
            "test_not_used_for_normalization": True,
        },
        "limitations": [
            "Pixel-disjoint splits do not guarantee spatial independence.",
            "All samples are camera-first-hit interactions in one scene.",
            "No model has been trained or evaluated in phase 3.2.1.",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("=== T2 TRAINING DATA (PHASE 3.2.1) ===")
    for name, count in report["counts"].items():
        print(f"{name}: {count} samples, {report['batch_counts'][name]} batches")
    print(f"Features: {data.feature_dimension}, bins: {data.num_bins}")
    print(f"Feature schema: {report['feature_schema_version']}")
    print(f"Max train normalized mean magnitude: {report['train_normalized_max_abs_mean']:.3e}")
    print(f"Checks: {report['checks']}")
    print(f"Report: {args.report.resolve()}")


if __name__ == "__main__":
    main()
