"""Regression tests for leakage-free T2 preparation (schema v2)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neural_path_guiding.data.dataset import save_dataset
from neural_path_guiding.data.schema import NeuralGuidingDataset
from neural_path_guiding.training.t2_data import (
    load_split_indices,
    prepare_t2_data,
    validate_split_indices,
)


def make_dataset(tmp_path: Path, *, version: int = 2):
    # 12 rows, exactly 4 per split; distinct pixels.
    # Extreme validation/test values expose normalization leakage.
    features = np.zeros((12, 14), dtype=np.float32)
    features[:4, 0] = [0., 2., 4., 6.]
    features[4:8, 0] = 1000.
    features[8:, 0] = -1000.
    features[:, 3] = 1.
    features[:, 9] = 1.
    features[:, 10:13] = [0.25, 0.38, 0.85]
    targets = np.full((12, 4), 0.25, dtype=np.float32)
    pixels = np.stack([np.arange(12), np.zeros(12, dtype=np.int32)], axis=1).astype(np.int32)
    metadata = {
        "dataset_format_version": 1,
        "feature_schema_version": version,
        "material_feature_mode": "lambertian_reflectance_v1",
        "target_type": "product_integrand_v1",
        "experiment_name": "unit_test",
        "feature_dimension": 14,
        "num_bins": 4,
        "n_mu": 2,
        "n_phi": 2,
        "num_shading_points": 12,
    }
    dataset = NeuralGuidingDataset(
        features=features,
        targets=targets,
        mean_contributions=np.ones((12, 4), dtype=np.float32),
        positions=np.zeros((12, 3), dtype=np.float32),
        normals=np.tile(np.array([[0, 0, 1]], dtype=np.float32), (12, 1)),
        pixels=pixels,
        metadata=metadata,
    )
    dataset_path = tmp_path / "sample.npz"
    splits_path = tmp_path / "sample_splits.npz"
    save_dataset(dataset_path, dataset)
    np.savez_compressed(
        splits_path,
        train=np.arange(4, dtype=np.int64),
        validation=np.arange(4, 8, dtype=np.int64),
        test=np.arange(8, 12, dtype=np.int64),
    )
    return dataset_path, splits_path, pixels


def test_train_only_normalization_and_constant_columns(tmp_path):
    dataset_path, splits_path, _ = make_dataset(tmp_path)
    prepared = prepare_t2_data(dataset_path, splits_path)
    assert prepared.train.num_samples == 4
    assert prepared.validation.num_samples == 4
    assert prepared.test.num_samples == 4
    assert prepared.feature_mean[0].item() == pytest.approx(3.0)
    assert prepared.feature_scale[0].item() == pytest.approx(np.sqrt(5.0))
    assert prepared.feature_scale[1].item() == pytest.approx(1.0)
    assert abs(prepared.validation.features[:, 0].mean().item()) > 100
    np.testing.assert_allclose(prepared.train.features[:, 0].numpy().mean(), 0, atol=1e-7)


def test_batches_and_targets(tmp_path):
    dataset_path, splits_path, _ = make_dataset(tmp_path)
    prepared = prepare_t2_data(dataset_path, splits_path)
    loaders = prepared.make_dataloaders(batch_size=3, seed=7)
    assert len(loaders["train"]) == 2
    assert len(loaders["validation"]) == 2
    x, y = next(iter(loaders["train"]))
    assert x.shape == (3, 14)
    assert y.shape == (3, 4)
    assert y.dtype.is_floating_point


def test_repeatability(tmp_path):
    dataset_path, splits_path, _ = make_dataset(tmp_path)
    first = prepare_t2_data(dataset_path, splits_path)
    second = prepare_t2_data(dataset_path, splits_path)
    assert first.dataset_sha256 == second.dataset_sha256
    assert first.splits_sha256 == second.splits_sha256
    assert np.array_equal(first.feature_mean.numpy(), second.feature_mean.numpy())
    def order(loader):
        return [float(value) for x, _ in loader for value in x[:, 0]]
    assert order(first.make_dataloaders(batch_size=2, seed=12)["train"]) == order(
        second.make_dataloaders(batch_size=2, seed=12)["train"]
    )


def test_rejects_duplicate_indices(tmp_path):
    _, _, pixels = make_dataset(tmp_path)
    with pytest.raises(ValueError, match="duplicated"):
        validate_split_indices({
            "train": np.array([0, 0]),
            "validation": np.array([1]),
            "test": np.array([2]),
        }, pixels)


def test_rejects_overlap_and_missing_rows(tmp_path):
    _, _, pixels = make_dataset(tmp_path)
    with pytest.raises(ValueError, match="partition"):
        validate_split_indices({
            "train": np.arange(5),
            "validation": np.arange(4, 9),
            "test": np.arange(9, 12),
        }, pixels)


def test_rejects_pixel_leakage(tmp_path):
    _, _, pixels = make_dataset(tmp_path)
    pixels[5] = pixels[0]
    with pytest.raises(ValueError, match="pixel leakage"):
        validate_split_indices({
            "train": np.arange(4),
            "validation": np.arange(4, 8),
            "test": np.arange(8, 12),
        }, pixels)


def test_rejects_out_of_range(tmp_path):
    _, _, pixels = make_dataset(tmp_path)
    with pytest.raises(ValueError, match="out-of-range"):
        validate_split_indices({
            "train": np.array([0, 12]),
            "validation": np.arange(1, 7),
            "test": np.arange(7, 12),
        }, pixels)


def test_rejects_wrong_schema(tmp_path):
    dataset_path, splits_path, _ = make_dataset(tmp_path, version=1)
    with pytest.raises(ValueError, match="schema_version=2"):
        prepare_t2_data(dataset_path, splits_path)


def test_rejects_noninteger_indices(tmp_path):
    _, _, pixels = make_dataset(tmp_path)
    with pytest.raises(ValueError, match="integer"):
        validate_split_indices({
            "train": np.array([0., 1.]),
            "validation": np.arange(2, 8),
            "test": np.arange(8, 12),
        }, pixels)


def test_rejects_unexpected_split_keys(tmp_path):
    _, splits_path, _ = make_dataset(tmp_path)
    np.savez_compressed(splits_path, train=np.arange(4), validation=np.arange(4, 8), bogus=np.arange(8, 12))
    with pytest.raises(ValueError, match="exactly"):
        load_split_indices(splits_path)
