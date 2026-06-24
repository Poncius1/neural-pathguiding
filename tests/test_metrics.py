"""Tests for image quality metrics."""

from __future__ import annotations

import math
import unittest

import numpy as np

from neural_path_guiding.evaluation.metrics import (
    automatic_peak_value,
    mean_absolute_error,
    mean_squared_error,
    peak_signal_to_noise_ratio,
    root_mean_squared_error,
)


class TestImageMetrics(unittest.TestCase):
    def test_mse_zero_for_equal_images(self) -> None:
        image = np.ones((2, 2, 3), dtype=np.float64)

        mse = mean_squared_error(image, image)

        self.assertEqual(mse, 0.0)

    def test_rmse_matches_mse_square_root(self) -> None:
        reference = np.zeros((1, 1, 1), dtype=np.float64)
        candidate = np.array([[[2.0]]], dtype=np.float64)

        mse = mean_squared_error(reference, candidate)
        rmse = root_mean_squared_error(reference, candidate)

        self.assertEqual(mse, 4.0)
        self.assertEqual(rmse, 2.0)

    def test_mae(self) -> None:
        reference = np.array([[[0.0], [1.0]]], dtype=np.float64)
        candidate = np.array([[[1.0], [3.0]]], dtype=np.float64)

        mae = mean_absolute_error(reference, candidate)

        self.assertEqual(mae, 1.5)

    def test_psnr_is_infinite_when_mse_is_zero(self) -> None:
        psnr = peak_signal_to_noise_ratio(mse=0.0, peak_value=1.0)

        self.assertTrue(math.isinf(psnr))

    def test_automatic_peak_value_uses_at_least_one(self) -> None:
        image = np.full((2, 2, 3), 0.25, dtype=np.float64)

        peak = automatic_peak_value(image)

        self.assertEqual(peak, 1.0)


if __name__ == "__main__":
    unittest.main()