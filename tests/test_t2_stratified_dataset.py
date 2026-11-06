"""Unit tests for material quotas and reproducible leakage-free splits."""
from __future__ import annotations

import unittest

import numpy as np

from scripts.generate_t2_stratified_dataset import (
    classify_material,
    read_material_quotas,
    stratified_indices,
)


class TestQuotas(unittest.TestCase):
    def setUp(self):
        self.quotas = read_material_quotas({
            "blue": {"albedo": [0.25, 0.38, 0.85], "count": 12},
            "gray": {"albedo": [0.45, 0.45, 0.45], "count": 12},
        })

    def test_classifies_with_float_tolerance(self):
        self.assertEqual(classify_material(np.array([0.25, 0.38, 0.85000002]), self.quotas), "blue")

    def test_unknown_color_is_not_assigned(self):
        self.assertIsNone(classify_material(np.array([0.8, 0.8, 0.8]), self.quotas))

    def test_rejects_duplicate_color_group(self):
        with self.assertRaises(ValueError):
            read_material_quotas({
                "a": {"albedo": [0.5, 0.5, 0.5], "count": 10},
                "b": {"albedo": [0.5, 0.5, 0.5], "count": 10},
            })

    def test_rejects_invalid_quota(self):
        with self.assertRaises(ValueError):
            read_material_quotas({"blue": {"albedo": [0.2, 0.3, 0.8], "count": 0}})


class TestSplits(unittest.TestCase):
    def setUp(self):
        self.labels = np.array(["blue"] * 12 + ["gray"] * 12 + ["light"] * 12)
        self.pixels = np.array([(i, i + 1) for i in range(36)], dtype=np.int32)

    def test_complete_nonoverlapping(self):
        splits = stratified_indices(self.labels, self.pixels, seed=42)
        all_ids = np.concatenate(list(splits.values()))
        np.testing.assert_array_equal(np.sort(all_ids), np.arange(36))
        for label in set(self.labels):
            self.assertGreater(sum(self.labels[splits["train"]] == label), 0)
            self.assertGreater(sum(self.labels[splits["validation"]] == label), 0)
            self.assertGreater(sum(self.labels[splits["test"]] == label), 0)

    def test_reproducible(self):
        first = stratified_indices(self.labels, self.pixels, seed=100)
        second = stratified_indices(self.labels, self.pixels, seed=100)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])

    def test_rejects_duplicate_pixel(self):
        pixels = self.pixels.copy()
        pixels[1] = pixels[0]
        with self.assertRaises(ValueError):
            stratified_indices(self.labels, pixels, seed=42)

    def test_rejects_insufficient_group(self):
        with self.assertRaises(ValueError):
            stratified_indices(["a", "b", "a"], np.array([[0,0],[1,0],[2,0]]), seed=2)

    def test_rejects_invalid_fraction(self):
        with self.assertRaises(ValueError):
            stratified_indices(self.labels, self.pixels, seed=0, train_fraction=0.95, val_fraction=0.1)


if __name__ == "__main__":
    unittest.main()
