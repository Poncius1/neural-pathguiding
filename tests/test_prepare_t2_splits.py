"""Tests for Phase 3.1 T2 dataset preparation."""

from __future__ import annotations

import unittest

import numpy as np

from scripts.prepare_t2_splits import (
    analyze_targets,
    grouped_pixel_split,
    validate_splits,
)


class TestGroupedPixelSplit(unittest.TestCase):

    def setUp(self) -> None:
        self.pixels = np.array(
            [
                [0, 0],
                [0, 0],
                [1, 0],
                [1, 0],
                [2, 0],
                [3, 0],
                [4, 0],
                [5, 0],
                [6, 0],
                [7, 0],
                [8, 0],
                [9, 0],
            ],
            dtype=np.int32,
        )

    def test_no_pixel_leakage(self) -> None:
        splits = grouped_pixel_split(
            self.pixels,
            seed=42,
        )

        validate_splits(
            pixels=self.pixels,
            splits=splits,
        )

        train_pixels = {
            tuple(pixel)
            for pixel in self.pixels[splits["train"]]
        }

        validation_pixels = {
            tuple(pixel)
            for pixel in self.pixels[splits["validation"]]
        }

        test_pixels = {
            tuple(pixel)
            for pixel in self.pixels[splits["test"]]
        }

        self.assertTrue(
            train_pixels.isdisjoint(validation_pixels)
        )

        self.assertTrue(
            train_pixels.isdisjoint(test_pixels)
        )

        self.assertTrue(
            validation_pixels.isdisjoint(test_pixels)
        )

    def test_all_rows_are_preserved(self) -> None:
        splits = grouped_pixel_split(
            self.pixels,
            seed=42,
        )

        all_indices = np.concatenate(
            list(splits.values())
        )

        np.testing.assert_array_equal(
            np.sort(all_indices),
            np.arange(len(self.pixels)),
        )

    def test_deterministic_with_same_seed(self) -> None:
        first = grouped_pixel_split(
            self.pixels,
            seed=123,
        )

        second = grouped_pixel_split(
            self.pixels,
            seed=123,
        )

        for name in first:
            np.testing.assert_array_equal(
                first[name],
                second[name],
            )

    def test_rejects_too_few_unique_pixels(self) -> None:
        pixels = np.array(
            [
                [0, 0],
                [0, 0],
                [1, 0],
                [1, 0],
            ],
            dtype=np.int32,
        )

        with self.assertRaises(ValueError):
            grouped_pixel_split(pixels)

    def test_three_groups_still_produce_three_splits(
        self,
    ) -> None:
        pixels = np.array(
            [
                [0, 0],
                [1, 0],
                [2, 0],
            ],
            dtype=np.int32,
        )

        splits = grouped_pixel_split(pixels)

        self.assertEqual(
            {len(indices) for indices in splits.values()},
            {1},
        )

    def test_rejects_invalid_fractions(self) -> None:
        with self.assertRaises(ValueError):
            grouped_pixel_split(
                self.pixels,
                train_fraction=0.9,
                val_fraction=0.2,
            )


class TestTargetQuality(unittest.TestCase):

    def test_valid_targets(self) -> None:
        targets = np.array(
            [
                [0.5, 0.5],
                [0.9, 0.1],
            ],
            dtype=np.float64,
        )

        contributions = np.array(
            [
                [1.0, 1.0],
                [9.0, 1.0],
            ],
            dtype=np.float64,
        )

        result = analyze_targets(
            targets,
            contributions,
        )

        self.assertEqual(
            result["samples"],
            2,
        )

        self.assertEqual(
            result["bins"],
            2,
        )

        self.assertEqual(
            result["zero_raw_contribution_rows"],
            0,
        )

        self.assertGreater(
            result["mean_normalized_entropy"],
            0.0,
        )

    def test_detects_zero_teacher_signal(self) -> None:
        targets = np.array(
            [
                [0.5, 0.5],
            ],
            dtype=np.float64,
        )

        contributions = np.zeros(
            (1, 2),
            dtype=np.float64,
        )

        result = analyze_targets(
            targets,
            contributions,
        )

        self.assertEqual(
            result["zero_raw_contribution_rows"],
            1,
        )

    def test_rejects_unnormalized_targets(self) -> None:
        targets = np.array(
            [
                [0.3, 0.3],
            ],
            dtype=np.float64,
        )

        contributions = np.array(
            [
                [1.0, 1.0],
            ],
            dtype=np.float64,
        )

        with self.assertRaises(ValueError):
            analyze_targets(
                targets,
                contributions,
            )

    def test_rejects_negative_contributions(self) -> None:
        targets = np.array(
            [[0.5, 0.5]],
            dtype=np.float64,
        )

        contributions = np.array(
            [[1.0, -1.0]],
            dtype=np.float64,
        )

        with self.assertRaises(ValueError):
            analyze_targets(
                targets,
                contributions,
            )


if __name__ == "__main__":
    unittest.main()