"""
Only train and validation data are used. The test split remains untouched.
Run from repository root with: python -m scripts.train_t2_mlp ...
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from neural_path_guiding.training.checkpoint import save_checkpoint
from neural_path_guiding.training.losses import soft_target_cross_entropy
from neural_path_guiding.training.model import GuidingMLP
from neural_path_guiding.training.t2_data import PreparedT2Data, prepare_t2_data


@dataclass(frozen=True)
class TrainSettings:
    batch_size: int = 64
    hidden_dimension: int = 64
    learning_rate: float = 1e-3
    max_epochs: int = 300
    patience: int = 30
    seed: int = 42
    min_delta: float = 0.0


def validate_settings(settings: TrainSettings) -> None:
    for name in ("batch_size", "hidden_dimension", "max_epochs", "patience"):
        value = getattr(settings, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(settings.seed, bool) or not isinstance(settings.seed, int) or settings.seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    for name in ("learning_rate", "min_delta"):
        value = getattr(settings, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite and numeric")
    if settings.learning_rate <= 0 or settings.min_delta < 0:
        raise ValueError("learning_rate must be positive and min_delta nonnegative")


def distribution_metrics(targets: torch.Tensor, probabilities: torch.Tensor) -> dict[str, float]:
    """Macro-averaged CE, KL(p||q), TVD and JS; tensors shape [N, B]."""
    if targets.shape != probabilities.shape or targets.ndim != 2 or targets.numel() == 0:
        raise ValueError("targets and probabilities must have equal nonempty [N,B] shapes")
    p = targets.double()
    q = probabilities.double()
    if not torch.isfinite(p).all() or not torch.isfinite(q).all():
        raise ValueError("metrics require finite distributions")
    if (p < 0).any() or (q <= 0).any():
        raise ValueError("targets must be nonnegative; predictions must be positive")
    if not torch.allclose(p.sum(1), torch.ones_like(p[:, 0]), atol=1e-4, rtol=0):
        raise ValueError("targets must sum to one")
    if not torch.allclose(q.sum(1), torch.ones_like(q[:, 0]), atol=1e-4, rtol=0):
        raise ValueError("predictions must sum to one")
    log_p = torch.where(p > 0, torch.log(p.clamp_min(1e-300)), torch.zeros_like(p))
    log_q = torch.log(q)
    ce = -(p * log_q).sum(1)
    entropy = -(p * log_p).sum(1)
    kl = ce - entropy
    tvd = 0.5 * (p - q).abs().sum(1)
    m = 0.5 * (p + q)
    jsd = 0.5 * (
        (p * (log_p - torch.log(m))).sum(1)
        + (q * (log_q - torch.log(m))).sum(1)
    )
    return {
        "cross_entropy": float(ce.mean()),
        "target_entropy": float(entropy.mean()),
        "kl": float(kl.clamp_min(0).mean()),
        "tvd": float(tvd.mean()),
        "jsd": float(jsd.clamp_min(0).mean()),
    }


def baseline_probabilities(data: PreparedT2Data) -> dict[str, torch.Tensor]:
    """Baseline PMFs derived from the training partition only."""
    n_bins = data.num_bins
    uniform = torch.full((n_bins,), 1.0 / n_bins, dtype=torch.float64)
    mean_target = data.train.targets.double().mean(dim=0)
    mean_target = mean_target / mean_target.sum()
    # Use floor for numerical safety; renormalize afterwards.
    mean_target = mean_target.clamp_min(1e-12)
    mean_target /= mean_target.sum()
    return {"uniform": uniform, "train_mean": mean_target}


def evaluate_model(model: GuidingMLP, features: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    model.eval()
    with torch.inference_mode():
        logits = model(features)
        probs = torch.softmax(logits.double(), dim=1)
        result = distribution_metrics(targets, probs)
    return result


def evaluate_baselines(data: PreparedT2Data, split: str) -> dict[str, dict[str, float]]:
    if split not in ("train", "validation"):
        raise ValueError("Phase 3.2.2 only evaluates train and validation")
    chosen = getattr(data, split)
    return {
        name: distribution_metrics(
            chosen.targets, pmf.unsqueeze(0).expand(chosen.num_samples, -1)
        )
        for name, pmf in baseline_probabilities(data).items()
    }


def train_t2(data: PreparedT2Data, settings: TrainSettings, output_dir: Path) -> dict[str, Any]:
    """Train on train, select by validation CE, save best weights and metrics.

    NEVER reads data.test tensors or evaluates them in this phase.
    """
    validate_settings(settings)
    if data.metadata.get("feature_schema_version") != 2:
        raise ValueError("Only feature schema v2 is supported")
    if data.metadata.get("target_type") != "product_integrand_v1":
        raise ValueError("T2 target_type is required")

    random.seed(settings.seed)
    np.random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    torch.set_num_threads(1)  # predictable CPU behavior for a small MLP

    model = GuidingMLP(
        feature_dimension=data.feature_dimension,
        num_bins=data.num_bins,
        hidden_dimension=settings.hidden_dimension,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    loaders = data.make_dataloaders(batch_size=settings.batch_size, seed=settings.seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_t2_v2.pt"
    history: list[dict[str, float | int]] = []
    best_val = float("inf")
    best_epoch = 0
    bad_epochs = 0
    steps = 0

    for epoch in range(1, settings.max_epochs + 1):
        model.train()
        for features, targets in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            loss = soft_target_cross_entropy(model(features), targets)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite training loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            steps += 1

        # Evaluate full partitions using the same exact loss as checkpoint selection.
        train_metrics = evaluate_model(model, data.train.features, data.train.targets)
        val_metrics = evaluate_model(model, data.validation.features, data.validation.targets)
        train_ce = train_metrics["cross_entropy"]
        val_ce = val_metrics["cross_entropy"]
        history.append({"epoch": epoch, "train_ce": train_ce, "val_ce": val_ce,
                        "val_kl": val_metrics["kl"], "val_tvd": val_metrics["tvd"],
                        "val_jsd": val_metrics["jsd"]})

        if val_ce < best_val - settings.min_delta:
            best_val = val_ce
            best_epoch = epoch
            bad_epochs = 0
            save_checkpoint(
                checkpoint_path, model, data.feature_mean, data.feature_scale,
                seed=settings.seed, num_steps=steps, learning_rate=settings.learning_rate,
                metadata={
                    "feature_schema_version": 2,
                    "material_feature_mode": "lambertian_reflectance_v1",
                    "target_type": "product_integrand_v1",
                    "n_mu": int(data.metadata["n_mu"]),
                    "n_phi": int(data.metadata["n_phi"]),
                    "dataset_sha256": data.dataset_sha256,
                    "splits_sha256": data.splits_sha256,
                    "selected_epoch": best_epoch,
                    "selection_metric": "validation_cross_entropy",
                },
            )
        else:
            bad_epochs += 1
        if epoch == 1 or epoch % 10 == 0 or bad_epochs >= settings.patience:
            print(f"epoch={epoch:03d} train_ce={train_ce:.6f} val_ce={val_ce:.6f} "
                  f"best={best_val:.6f} (epoch {best_epoch})", flush=True)
        if bad_epochs >= settings.patience:
            break

    # Reload selected model, do not accidentally report the last epoch.
    from neural_path_guiding.training.checkpoint import load_checkpoint
    selected = load_checkpoint(checkpoint_path)
    selected_train = evaluate_model(selected.model, data.train.features, data.train.targets)
    selected_val = evaluate_model(selected.model, data.validation.features, data.validation.targets)
    baselines = {
        "train": evaluate_baselines(data, "train"),
        "validation": evaluate_baselines(data, "validation"),
    }
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    report: dict[str, Any] = {
        "phase": "3.2.2",
        "feature_schema_version": 2,
        "test_evaluated": False,
        "train_count": data.train.num_samples,
        "validation_count": data.validation.num_samples,
        "test_count_reserved": data.test.num_samples,
        "model": {"features": data.feature_dimension, "hidden": settings.hidden_dimension,
                  "bins": data.num_bins},
        "settings": vars(settings),
        "dataset_sha256": data.dataset_sha256,
        "splits_sha256": data.splits_sha256,
        "best_epoch": best_epoch,
        "best_validation_cross_entropy": best_val,
        "epochs_completed": len(history),
        "optimizer_steps": steps,
        "selected_model_metrics": {"train": selected_train, "validation": selected_val},
        "baselines": baselines,
        "checkpoint": str(checkpoint_path.resolve()),
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/simple_diffuse_t2_stratified_medium_v2.npz"))
    parser.add_argument("--splits", type=Path, default=Path("data/simple_diffuse_t2_stratified_medium_v2_splits.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiments/t2_mlp_v2_seed42"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dimension", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = TrainSettings(
        batch_size=args.batch_size,
        hidden_dimension=args.hidden_dimension,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        patience=args.patience,
        seed=args.seed,
    )
    data = prepare_t2_data(args.dataset, args.splits)
    report = train_t2(data, settings, args.output_dir)
    print("\n=== T2 MLP TRAINING ===")
    print(f"Selected epoch: {report['best_epoch']}")
    print(f"Validation CE: {report['best_validation_cross_entropy']:.6f}")
    for name, stats in report["baselines"]["validation"].items():
        print(f"Validation baseline {name}: CE={stats['cross_entropy']:.6f}")
    print(f"Report: {(args.output_dir / 'report.json').resolve()}")
    print(f"Checkpoint: {report['checkpoint']}")
    print("Test split intentionally not evaluated.")


if __name__ == "__main__":
    main()
