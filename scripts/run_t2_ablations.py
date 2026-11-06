
"""Phase 3.2.4b: train feature ablations without touching held-out test.

Run from repo root: python -m scripts.run_t2_ablations --dataset ... --splits ...
Uses the existing validated v2 loader and the phase 3.2.2 trainer.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from neural_path_guiding.training.checkpoint import (
    load_checkpoint,
    save_checkpoint,
)
from neural_path_guiding.training.t2_data import (
    PreparedT2Data,
    PreparedSplit,
    prepare_t2_data,
)
from scripts.train_t2_mlp import TrainSettings, train_t2


# Feature schema v2:
# position:  [0, 1, 2]
# normal:    [3, 4, 5]
# outgoing:  [6, 7, 8]
# roughness: [9]
# albedo:    [10, 11, 12]
# depth:     [13]

FEATURE_GROUPS: dict[str, tuple[int, ...]] = {
    "full": (),
    "no_albedo": (10, 11, 12),
    "no_position": (0, 1, 2),
    "no_outgoing": (6, 7, 8),
}


def mask_features(
    features: torch.Tensor,
    ablation: str,
) -> torch.Tensor:
    """Mask selected features in normalized space.

    Zero corresponds to the training-set mean after normalization.
    Returns a copy to avoid modifying the original tensors.
    """
    if ablation not in FEATURE_GROUPS:
        raise ValueError(
            f"Unknown ablation: {ablation!r}"
        )

    if features.ndim != 2 or features.shape[1] != 14:
        raise ValueError(
            "Expected normalized [N, 14] features"
        )

    result = features.clone()
    columns = FEATURE_GROUPS[ablation]

    if columns:
        result[:, list(columns)] = 0.0

    return result


def make_ablated_data(
    data: PreparedT2Data,
    ablation: str,
) -> PreparedT2Data:
    """Apply the same mask to all splits.

    The original PreparedT2Data remains unchanged.
    """

    def masked(part: PreparedSplit) -> PreparedSplit:
        return replace(
            part,
            features=mask_features(
                part.features,
                ablation,
            ),
        )

    return replace(
        data,
        train=masked(data.train),
        validation=masked(data.validation),
        test=masked(data.test),
    )


def annotate_checkpoint(
    checkpoint_path: Path,
    ablation: str,
) -> None:
    """Save ablation metadata alongside model parameters.

    Keeps the existing checkpoint format and normalizer.
    """
    ckpt = load_checkpoint(
        checkpoint_path,
        device="cpu",
    )

    metadata = dict(ckpt.metadata)

    metadata["feature_ablation"] = ablation
    metadata["ablated_feature_indices"] = ",".join(
        map(str, FEATURE_GROUPS[ablation])
    )
    metadata["feature_mask_space"] = (
        "train_normalized_zero_is_train_mean"
    )

    save_checkpoint(
        checkpoint_path,
        ckpt.model,
        ckpt.feature_mean,
        ckpt.feature_scale,
        seed=ckpt.seed,
        num_steps=ckpt.num_steps,
        learning_rate=ckpt.learning_rate,
        metadata=metadata,
    )


def run_ablations(
    dataset: Path,
    splits: Path,
    output_dir: Path,
    *,
    seeds: tuple[int, ...],
    ablations: tuple[str, ...],
    batch_size: int,
    max_epochs: int,
    patience: int,
    hidden_dimension: int,
    learning_rate: float,
) -> list[dict]:
    """Train every ablation from scratch for every seed."""

    if (
        not seeds
        or len(seeds) != len(set(seeds))
        or any(x < 0 for x in seeds)
    ):
        raise ValueError(
            "Seeds must be distinct nonnegative integers"
        )

    if (
        not ablations
        or len(ablations) != len(set(ablations))
    ):
        raise ValueError(
            "Choose distinct ablations"
        )

    if any(a not in FEATURE_GROUPS for a in ablations):
        raise ValueError(
            "Unknown ablation name"
        )

    data = prepare_t2_data(
        dataset,
        splits,
    )

    if data.feature_dimension != 14 or data.num_bins != 32:
        raise ValueError(
            "Expected v2 14-feature / 32-bin model"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results: list[dict] = []

    for ablation in ablations:
        ablated = make_ablated_data(
            data,
            ablation,
        )

        for seed in seeds:
            run_dir = (
                output_dir
                / ablation
                / f"seed{seed}"
            )

            print(
                f"\n=== ABLATION {ablation} seed={seed} ===",
                flush=True,
            )

            settings = TrainSettings(
                batch_size=batch_size,
                hidden_dimension=hidden_dimension,
                learning_rate=learning_rate,
                max_epochs=max_epochs,
                patience=patience,
                seed=seed,
            )

            report = train_t2(
                ablated,
                settings,
                run_dir,
            )

            annotate_checkpoint(
                run_dir / "best_t2_v2.pt",
                ablation,
            )

            # Identify the ablation in the training report.
            report["phase"] = "3.2.4b"
            report["feature_ablation"] = ablation
            report["ablated_feature_indices"] = list(
                FEATURE_GROUPS[ablation]
            )
            report["test_evaluated"] = False

            (run_dir / "report.json").write_text(
                json.dumps(report, indent=2),
                encoding="utf-8",
            )

            metrics = report[
                "selected_model_metrics"
            ]["validation"]

            result = {
                "ablation": ablation,
                "seed": seed,
                "best_epoch": report["best_epoch"],
                "ce": metrics["cross_entropy"],
                "kl": metrics["kl"],
                "tvd": metrics["tvd"],
                "jsd": metrics["jsd"],
                "dataset_sha256": data.dataset_sha256,
                "splits_sha256": data.splits_sha256,
            }

            results.append(result)

            print(
                "VALIDATION",
                {
                    key: round(result[key], 6)
                    for key in (
                        "ce",
                        "kl",
                        "tvd",
                        "jsd",
                    )
                },
                flush=True,
            )

            _write_summary(
                output_dir,
                results,
            )

    return results


def _write_summary(
    output_dir: Path,
    results: list[dict],
) -> None:
    """Write individual runs and per-ablation statistics."""

    with (
        output_dir / "ablation_runs.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0]),
        )
        writer.writeheader()
        writer.writerows(results)

    summary: dict[str, dict] = {}

    for ablation in sorted(
        {r["ablation"] for r in results}
    ):
        group = [
            r
            for r in results
            if r["ablation"] == ablation
        ]

        summary[ablation] = {
            "completed_seeds": [
                r["seed"]
                for r in group
            ],
            "n": len(group),
            **{
                metric: {
                    "mean": float(
                        np.mean([
                            r[metric]
                            for r in group
                        ])
                    ),
                    "std_sample": (
                        float(
                            np.std(
                                [
                                    r[metric]
                                    for r in group
                                ],
                                ddof=1,
                            )
                        )
                        if len(group) > 1
                        else None
                    ),
                }
                for metric in (
                    "ce",
                    "kl",
                    "tvd",
                    "jsd",
                )
            },
        }

    (output_dir / "ablation_summary.json").write_text(
        json.dumps(
            {
                "phase": "3.2.4b",
                "split_evaluated": "validation",
                "test_evaluated": False,
                "completed_runs": len(results),
                "summary": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "data/simple_diffuse_t2_stratified_medium_v2.npz"
        ),
    )

    parser.add_argument(
        "--splits",
        type=Path,
        default=Path(
            "data/simple_diffuse_t2_stratified_medium_v2_splits.npz"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/experiments/t2_ablations_v2"
        ),
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[7, 21, 42, 84, 123],
    )

    parser.add_argument(
        "--ablations",
        nargs="+",
        choices=list(FEATURE_GROUPS),
        default=[
            "full",
            "no_albedo",
            "no_position",
            "no_outgoing",
        ],
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--hidden-dimension",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--max-epochs",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=30,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    results = run_ablations(
        args.dataset,
        args.splits,
        args.output_dir,
        seeds=tuple(args.seeds),
        ablations=tuple(args.ablations),
        batch_size=args.batch_size,
        hidden_dimension=args.hidden_dimension,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        patience=args.patience,
    )

    print("\n=== T2 FEATURE ABLATIONS ===")
    print(f"Runs completed: {len(results)}")

    for ablation in args.ablations:
        rows = [
            r
            for r in results
            if r["ablation"] == ablation
        ]

        print(
            f"{ablation:>13}: "
            f"N={len(rows)} "
            f"CE={np.mean([r['ce'] for r in rows]):.6f} "
            f"TVD={np.mean([r['tvd'] for r in rows]):.6f}"
        )

    print(
        "Summary:",
        (
            args.output_dir / "ablation_summary.json"
        ).resolve(),
    )

    print(
        "Test split intentionally not evaluated."
    )


if __name__ == "__main__":
    main()
