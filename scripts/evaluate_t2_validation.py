"""Phase 3.2.3: validation-only diagnostics for a T2 feature-schema-v2 checkpoint.

Does not evaluate or report predictions for reserved test rows.
Run from the repository root: python -m scripts.evaluate_t2_validation ...
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from neural_path_guiding.data.dataset import load_dataset
from neural_path_guiding.training.checkpoint import load_checkpoint
from neural_path_guiding.training.t2_data import prepare_t2_data


# Known Lambertian reflectances from scenes/simple_diffuse/scene.xml.
# These labels identify reflectance groups, NOT individual mesh objects.
MATERIAL_ALBEDOS = {
    "blue": (0.25, 0.38, 0.85),
    "dark_gray": (0.45, 0.45, 0.45),
    "light_gray": (0.72, 0.72, 0.72),
}
METRICS = ("cross_entropy", "kl", "tvd", "jsd")


def identify_material(albedo: np.ndarray, *, tolerance: float = 1e-4) -> str:
    """Identify a material by feature-schema-v2 albedo, or fail explicitly."""
    a = np.asarray(albedo, dtype=np.float64)
    if a.shape != (3,) or not np.isfinite(a).all():
        raise ValueError("Material albedo must be a finite RGB triplet")
    matches = [name for name, rgb in MATERIAL_ALBEDOS.items()
               if np.allclose(a, rgb, rtol=0, atol=tolerance)]
    if len(matches) != 1:
        raise ValueError(f"Unknown or ambiguous material reflectance: {a.tolist()}")
    return matches[0]


def row_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, np.ndarray]:
    """Return per-row CE, KL(p||q), TVD and JSD in nats.

    Zero teacher probabilities are handled by 0*log(0)=0. The model and
    mean-PMF baseline must have strictly positive probabilities.
    """
    p = np.asarray(targets, dtype=np.float64)
    q = np.asarray(predictions, dtype=np.float64)
    if p.ndim != 2 or p.shape != q.shape or min(p.shape) < 1:
        raise ValueError("Expected matching nonempty [samples,bins] matrices")
    if not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("Nonfinite target or prediction")
    if np.any(p < 0) or np.any(q <= 0):
        raise ValueError("Targets must be nonnegative and predictions strictly positive")
    if not np.allclose(p.sum(axis=1), 1, rtol=0, atol=1e-4):
        raise ValueError("Target rows must sum to one")
    if not np.allclose(q.sum(axis=1), 1, rtol=0, atol=1e-4):
        raise ValueError("Prediction rows must sum to one")

    logp = np.zeros_like(p)
    np.log(p, out=logp, where=p > 0)
    logq = np.log(q)
    ce = -np.sum(p * logq, axis=1)
    ent = -np.sum(p * logp, axis=1)
    m = (p + q) * 0.5
    jsd = 0.5 * (np.sum(p * (logp - np.log(m)), axis=1)
                 + np.sum(q * (logq - np.log(m)), axis=1))
    return {
        "cross_entropy": ce,
        "kl": np.maximum(ce - ent, 0.0),
        "tvd": 0.5 * np.abs(p - q).sum(axis=1),
        "jsd": np.maximum(jsd, 0.0),
    }


def summarize(metrics: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, float]:
    if np.count_nonzero(mask) == 0:
        raise ValueError("Cannot summarize an empty group")
    return {key: float(metrics[key][mask].mean()) for key in METRICS}


def validate_checkpoint(ckpt: Any, data: Any) -> None:
    """Avoid mixing checkpoint, schema, data, partitions or normalizers."""
    meta = ckpt.metadata
    for key, expected in (
        ("feature_schema_version", 2),
        ("material_feature_mode", "lambertian_reflectance_v1"),
        ("target_type", "product_integrand_v1"),
        ("n_mu", int(data.metadata["n_mu"])),
        ("n_phi", int(data.metadata["n_phi"])),
        ("dataset_sha256", data.dataset_sha256),
        ("splits_sha256", data.splits_sha256),
        ("selection_metric", "validation_cross_entropy"),
    ):
        if meta.get(key) != expected:
            raise ValueError(f"Checkpoint incompatible: {key} (got {meta.get(key)!r})")
    if ckpt.model.feature_dimension != data.feature_dimension or ckpt.model.num_bins != data.num_bins:
        raise ValueError("Checkpoint model dimensions do not match dataset")
    for name, actual, expected in (
        ("feature_mean", ckpt.feature_mean, data.feature_mean),
        ("feature_scale", ckpt.feature_scale, data.feature_scale),
    ):
        if not torch.allclose(actual.detach().cpu(), expected, rtol=1e-5, atol=1e-6):
            raise ValueError(f"Checkpoint {name} does not match train-only normalization")


def evaluate_validation(dataset_path: Path, splits_path: Path,
                        checkpoint_path: Path, output_dir: Path,
                        *, top_k: int = 10) -> dict[str, Any]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    prepared = prepare_t2_data(dataset_path, splits_path)
    checkpoint = load_checkpoint(checkpoint_path, device="cpu")
    validate_checkpoint(checkpoint, prepared)
    # Only validation tensors are passed through the model; test is never evaluated.
    validation = prepared.validation
    with torch.inference_mode():
        logits = checkpoint.model(validation.features)
        predicted = torch.softmax(logits.double(), dim=1).cpu().numpy()
    targets = validation.targets.detach().cpu().numpy().astype(np.float64)
    train_mean = prepared.train.targets.double().mean(dim=0).cpu().numpy()
    train_mean = np.maximum(train_mean, 1e-12)
    train_mean /= train_mean.sum()
    n, bins = targets.shape
    distributions = {
        "mlp": predicted,
        "uniform": np.full((n, bins), 1.0 / bins, dtype=np.float64),
        "train_mean": np.broadcast_to(train_mean, (n, bins)),
    }
    metrics = {name: row_metrics(targets, q) for name, q in distributions.items()}

    # Pixel/material metadata is aligned using original row indices, not row order guesses.
    dataset = load_dataset(dataset_path)
    original_indices = validation.indices
    pixels = np.asarray(dataset.pixels)[original_indices]
    materials = [identify_material(a) for a in np.asarray(dataset.features)[original_indices, 10:13]]
    if len({tuple(map(int, p)) for p in pixels}) != n:
        raise ValueError("Validation pixels must be unique")

    all_mask = np.ones(n, dtype=bool)
    report: dict[str, Any] = {
        "phase": "3.2.3", "split_evaluated": "validation", "test_evaluated": False,
        "validation_count": int(n), "test_count_reserved": int(prepared.test.num_samples),
        "feature_schema_version": 2, "checkpoint_selected_epoch": checkpoint.metadata.get("selected_epoch"),
        "dataset_sha256": prepared.dataset_sha256, "splits_sha256": prepared.splits_sha256,
        "metrics_units": {"cross_entropy": "nats", "kl": "nats", "jsd": "nats", "tvd": "unitless"},
        "overall": {name: summarize(m, all_mask) for name, m in metrics.items()},
        "by_material": {},
    }
    material_values = np.asarray(materials)
    for material in sorted(set(materials)):
        mask = material_values == material
        report["by_material"][material] = {
            "count": int(mask.sum()),
            "methods": {name: summarize(m, mask) for name, m in metrics.items()},
        }
    report["mlp_ce_win_rate_vs_train_mean"] = float(np.mean(
        metrics["mlp"]["cross_entropy"] < metrics["train_mean"]["cross_entropy"]
    ))

    rows: list[dict[str, Any]] = []
    for i in range(n):
        row: dict[str, Any] = {
            "dataset_row": int(original_indices[i]), "pixel_x": int(pixels[i, 0]),
            "pixel_y": int(pixels[i, 1]), "material": materials[i],
            "target_max_probability": float(targets[i].max()),
            "target_entropy": float(-np.sum(np.where(targets[i] > 0, targets[i], 0)
                                                    * np.log(np.maximum(targets[i], 1e-300)))),
        }
        for method, result in metrics.items():
            for metric in METRICS:
                row[f"{method}_{metric}"] = float(result[metric][i])
        row["mlp_ce_gain_vs_train_mean"] = row["train_mean_cross_entropy"] - row["mlp_cross_entropy"]
        rows.append(row)

    report["worst_mlp_tvd_points"] = [
        {"dataset_row": r["dataset_row"], "pixel": [r["pixel_x"], r["pixel_y"]],
         "material": r["material"], "mlp_tvd": r["mlp_tvd"], "mlp_kl": r["mlp_kl"]}
        for r in sorted(rows, key=lambda r: r["mlp_tvd"], reverse=True)[:top_k]
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "validation_per_pixel.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report_path = output_dir / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=Path("data/simple_diffuse_t2_stratified_medium_v2.npz"))
    p.add_argument("--splits", type=Path, default=Path("data/simple_diffuse_t2_stratified_medium_v2_splits.npz"))
    p.add_argument("--checkpoint", type=Path, default=Path("outputs/experiments/t2_mlp_v2_seed42/best_t2_v2.pt"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/experiments/t2_validation_v2_seed42"))
    p.add_argument("--top-k", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate_validation(args.dataset, args.splits, args.checkpoint, args.output_dir, top_k=args.top_k)
    print("\n=== T2 VALIDATION DIAGNOSTICS (PHASE 3.2.3) ===")
    print(f"Validation: {report['validation_count']} samples; test reserved: {report['test_count_reserved']}")
    for name, m in report["overall"].items():
        print(f"{name:>10}: CE={m['cross_entropy']:.6f} KL={m['kl']:.6f} "
              f"TVD={m['tvd']:.6f} JSD={m['jsd']:.6f}")
    print("By material (MLP):")
    for name, result in report["by_material"].items():
        m = result["methods"]["mlp"]
        print(f"  {name}: N={result['count']} CE={m['cross_entropy']:.6f} TVD={m['tvd']:.6f}")
    print(f"MLP CE win rate vs train mean: {report['mlp_ce_win_rate_vs_train_mean']:.1%}")
    print(f"Report: {(args.output_dir / 'validation_report.json').resolve()}")
    print(f"CSV: {(args.output_dir / 'validation_per_pixel.csv').resolve()}")
    print("Test split intentionally not evaluated.")


if __name__ == "__main__":
    main()
