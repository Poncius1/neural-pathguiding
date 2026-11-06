"""Generate a material-balanced, first-bounce T2 dataset (feature schema v2).

Quotas describe effective diffuse reflectance, not geometric object IDs.
Candidate selection and teacher evaluation use independent random streams.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from neural_path_guiding.config import load_yaml_config, resolve_project_path
from neural_path_guiding.core.targets import contributions_to_target_distribution
from neural_path_guiding.data.dataset import save_dataset
from neural_path_guiding.data.schema import NeuralGuidingDataset
from neural_path_guiding.renderers.mitsuba.adapters import (
    mitsuba_vector_to_numpy,
    normalize_numpy_vector,
)
from neural_path_guiding.renderers.mitsuba.dataset_config import build_settings_from_config
from neural_path_guiding.renderers.mitsuba.dataset_generator import (
    build_metadata,
    sample_camera_ray,
    validate_dataset_settings,
)
from neural_path_guiding.renderers.mitsuba.material_features import (
    MATERIAL_FEATURE_MODE,
    MATERIAL_FEATURE_SCHEMA_VERSION,
    is_supported_diffuse_bsdf,
    surface_to_material_features,
)
from neural_path_guiding.renderers.mitsuba.teachers import (
    PRODUCT_INTEGRAND_TARGET,
    TeacherContext,
    create_teacher_runtime,
    estimate_bin_contributions,
)

ROOT = Path(__file__).resolve().parents[1]


def read_material_quotas(value: Any) -> dict[str, tuple[np.ndarray, int]]:
    """Parse exact RGB group definitions and positive quotas."""
    if not isinstance(value, dict) or not value:
        raise ValueError("dataset.material_quotas must be a nonempty mapping")
    result: dict[str, tuple[np.ndarray, int]] = {}
    colors: list[np.ndarray] = []
    for name, entry in value.items():
        if not isinstance(name, str) or not name or not isinstance(entry, dict):
            raise ValueError("Each material group needs a name and mapping")
        rgb = np.asarray(entry.get("albedo"), dtype=np.float64)
        count = entry.get("count")
        if rgb.shape != (3,) or not np.all(np.isfinite(rgb)) or np.any((rgb < 0) | (rgb > 1)):
            raise ValueError(f"Invalid albedo for {name}")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"Invalid positive integer count for {name}")
        if any(np.allclose(rgb, other, atol=1e-4, rtol=0) for other in colors):
            raise ValueError("Material quota colors must be distinct")
        colors.append(rgb)
        result[name] = (rgb, count)
    return result


def classify_material(albedo: np.ndarray, quotas: dict[str, tuple[np.ndarray, int]]) -> str | None:
    """Return the matching diffuse material label, or None."""
    matches = [
        name for name, (color, _) in quotas.items()
        if np.allclose(albedo, color, atol=1e-4, rtol=0)
    ]
    if len(matches) > 1:
        raise ValueError("Ambiguous material reflectance")
    return matches[0] if matches else None


def stratified_indices(
    labels: list[str] | np.ndarray,
    pixels: np.ndarray,
    *,
    seed: int,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
) -> dict[str, np.ndarray]:
    """Group by pixel inside each material to balance all three splits."""
    labels = np.asarray(labels)
    pixels = np.asarray(pixels)
    if labels.ndim != 1 or pixels.shape != (len(labels), 2):
        raise ValueError("labels and pixels have incompatible shapes")
    if not np.issubdtype(pixels.dtype, np.integer):
        raise ValueError("pixels must contain integer coordinates")
    if len(set(map(tuple, pixels.tolist()))) != len(pixels):
        raise ValueError("Duplicate camera pixels are not permitted")
    if seed < 0 or not 0 < train_fraction < 1 or not 0 < val_fraction < 1 or train_fraction + val_fraction >= 1:
        raise ValueError("Invalid split seed or fractions")
    rng = np.random.default_rng(seed)
    splits: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
    for material in sorted(set(labels.tolist())):
        indices = np.flatnonzero(labels == material)
        if len(indices) < 3:
            raise ValueError(f"At least 3 samples required for material {material}")
        indices = rng.permutation(indices)
        n_train = max(1, int(np.floor(len(indices) * train_fraction)))
        n_val = max(1, int(np.floor(len(indices) * val_fraction)))
        if n_train + n_val >= len(indices):
            n_train = len(indices) - 2
            n_val = 1
        splits["train"].extend(indices[:n_train].tolist())
        splits["validation"].extend(indices[n_train:n_train + n_val].tolist())
        splits["test"].extend(indices[n_train + n_val:].tolist())
    result = {name: np.sort(np.asarray(ids, dtype=np.int64)) for name, ids in splits.items()}
    all_ids = np.concatenate(list(result.values()))
    if len(all_ids) != len(labels) or len(np.unique(all_ids)) != len(labels):
        raise AssertionError("Split coverage or disjointness failure")
    return result


def select_candidates(settings: Any, quotas: dict[str, tuple[np.ndarray, int]], rng: np.random.Generator):
    """Collect unique camera pixels before spending any teacher samples."""
    selected: list[tuple[int, int, Any, Any, Any, str]] = []
    counts: Counter[str] = Counter()
    rejects: Counter[str] = Counter()
    occupied: set[tuple[int, int]] = set()
    attempts = 0
    required = sum(count for _, count in quotas.values())
    while len(selected) < required and attempts < settings.max_sampling_attempts:
        attempts += 1
        x, y, ray = sample_camera_ray(settings.camera, rng)
        pixel = (x, y)
        if pixel in occupied:
            rejects["repeated_pixel"] += 1
            continue
        si = settings.scene.ray_intersect(ray)
        if not bool(si.is_valid()):
            rejects["miss"] += 1
            continue
        if not is_supported_diffuse_bsdf(si.bsdf()):
            rejects["unsupported_bsdf"] += 1
            continue
        if float(si.wi.z) <= 0:
            rejects["backface"] += 1
            continue
        outgoing = normalize_numpy_vector(-mitsuba_vector_to_numpy(ray.d))
        features = surface_to_material_features(
            surface_interaction=si, outgoing_direction=outgoing, bounce_depth=0
        )
        group = classify_material(features.albedo, quotas)
        if group is None:
            rejects["unlisted_material"] += 1
            continue
        if counts[group] >= quotas[group][1]:
            rejects["full_quota"] += 1
            continue
        occupied.add(pixel)
        counts[group] += 1
        selected.append((x, y, si, outgoing, features, group))
    if len(selected) != required:
        raise RuntimeError(
            f"Could not meet quotas within {settings.max_sampling_attempts} attempts: "
            f"found={dict(counts)} required={ {name: q[1] for name, q in quotas.items()} } "
            f"rejects={dict(rejects)}"
        )
    return selected, attempts, dict(rejects)


def generate(settings: Any, quotas: dict[str, tuple[np.ndarray, int]], *, split_seed: int) -> None:
    validate_dataset_settings(settings)
    if settings.teacher.target_type != PRODUCT_INTEGRAND_TARGET:
        raise ValueError("A T2 product-integrand teacher is required")
    required = sum(q[1] for q in quotas.values())
    if settings.num_shading_points != required:
        raise ValueError("num_shading_points must equal the sum of material quotas")
    if any(q[1] < 3 for q in quotas.values()):
        raise ValueError("Each material requires at least three samples for splitting")

    selection_rng = np.random.default_rng(settings.seed)
    selected, attempts, rejected = select_candidates(settings, quotas, selection_rng)
    print(f"Selected {len(selected)} unique pixels in {attempts} attempts", flush=True)

    # Independent of candidate collection/rejection and split assignment.
    sequence = np.random.SeedSequence(settings.seed)
    teacher_seeds = sequence.spawn(len(selected))
    feature_rows, target_rows, contribution_rows, position_rows, normal_rows, pixel_rows, labels = (
        [], [], [], [], [], [], []
    )
    for i, (x, y, si, outgoing, features, label) in enumerate(selected):
        seed = int(teacher_seeds[i].generate_state(1, dtype=np.uint32)[0])
        rng = np.random.default_rng(seed)
        runtime = create_teacher_runtime(settings.teacher, seed=seed)
        contributions = estimate_bin_contributions(
            context=TeacherContext(settings.scene, si, features.normal, outgoing),
            bins=settings.bins, teacher=settings.teacher, rng=rng, runtime=runtime,
        )
        target = contributions_to_target_distribution(contributions, settings.teacher.smoothing)
        feature_rows.append(features.to_array())
        target_rows.append(target)
        contribution_rows.append(contributions)
        position_rows.append(features.position)
        normal_rows.append(features.normal)
        pixel_rows.append((x, y))
        labels.append(label)
        if (i + 1) % 32 == 0 or i + 1 == len(selected):
            print(f"Teacher progress: {i + 1}/{len(selected)}", flush=True)

    metadata = build_metadata(settings)
    metadata.update({
        "feature_schema_version": MATERIAL_FEATURE_SCHEMA_VERSION,
        "material_feature_mode": MATERIAL_FEATURE_MODE,
        "roughness_semantics": "1.0 convention for Lambertian reflection",
        "material_support": "diffuse_reflection_only",
        "sampling_depth": 0,
        "sampling_strategy": "material_reflectance_quota_unique_camera_pixel",
        "material_quotas": {name: {"albedo": rgb.tolist(), "count": count} for name, (rgb, count) in quotas.items()},
        "accepted_per_material": dict(Counter(labels)),
        "candidate_attempts": attempts,
        "rejected_interactions": rejected,
        "teacher_randomization": "independent_seed_sequence_per_selected_point",
        "split_seed": split_seed,
    })
    dataset = NeuralGuidingDataset(
        features=np.asarray(feature_rows, dtype=np.float32),
        targets=np.asarray(target_rows, dtype=np.float32),
        mean_contributions=np.asarray(contribution_rows, dtype=np.float32),
        positions=np.asarray(position_rows, dtype=np.float32),
        normals=np.asarray(normal_rows, dtype=np.float32),
        pixels=np.asarray(pixel_rows, dtype=np.int32),
        metadata=metadata,
    )
    splits = stratified_indices(labels, dataset.pixels, seed=split_seed)
    save_dataset(settings.output_path, dataset)
    split_path = settings.output_path.with_name(settings.output_path.stem + "_splits.npz")
    np.savez_compressed(split_path, **splits)  # pyright: ignore[reportArgumentType]
    report = {
        "dataset": str(settings.output_path),
        "splits": {name: {"count": len(ids), "material_counts": dict(Counter(labels[i] for i in ids))} for name, ids in splits.items()},
        "candidate_attempts": attempts,
        "rejected_interactions": rejected,
        "unique_pixels": len(set(pixel_rows)),
        "target_sum_min": float(dataset.targets.sum(axis=1).min()),
        "target_sum_max": float(dataset.targets.sum(axis=1).max()),
        "zero_raw_contribution_rows": int(np.sum(dataset.mean_contributions.sum(axis=1) == 0)),
    }
    report_path = settings.output_path.with_name(settings.output_path.stem + "_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Dataset: {settings.output_path.resolve()}")
    print(f"Splits:  {split_path.resolve()}")
    print(f"Report:  {report_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = resolve_project_path(args.config, ROOT)
    config = load_yaml_config(config_path)
    settings = build_settings_from_config(config, config_path, ROOT)
    quotas = read_material_quotas(config["dataset"].get("material_quotas"))
    split_seed = config["dataset"].get("split_seed", 2026)
    if isinstance(split_seed, bool) or not isinstance(split_seed, int) or split_seed < 0:
        raise ValueError("dataset.split_seed must be a nonnegative integer")
    generate(settings, quotas, split_seed=split_seed)


if __name__ == "__main__":
    main()
