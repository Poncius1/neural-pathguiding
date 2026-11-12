"""Regression tests for transport manifest metrics (Mitsuba not required)."""

from __future__ import annotations

import hashlib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import numpy as np

from neural_path_guiding.evaluation.transport_metrics import (
    file_sha256, summarize_transport_image, validate_rgb_image,
)


class TestTransportMetrics(unittest.TestCase):
    def test_manifest_keys_and_numeric_values(self):
        rgb = np.array([[[0., 0., 0.], [1., 2., 3.]]])
        ref = np.zeros_like(rgb)
        result = summarize_transport_image(rgb, seconds=2.5, reference=ref)
        self.assertEqual(set(result), {
            'seconds', 'mean_luminance', 'max_luminance', 'zero_pixels',
            'rgb_mse', 'rgb_mae', 'mean_rgb_error',
        })
        self.assertEqual(result['zero_pixels'], 1)
        self.assertAlmostEqual(cast(float, result['rgb_mse']), (1+4+9)/6)
        self.assertAlmostEqual(cast(float, result['rgb_mae']), 1.0)
        np.testing.assert_allclose(
            cast(np.ndarray, result['mean_rgb_error']), [.5, 1., 1.5]
        )

    def test_no_reference_omits_error_fields(self):
        result = summarize_transport_image(np.ones((1, 1, 3)), seconds=0.0)
        self.assertEqual(set(result), {
            'seconds', 'mean_luminance', 'max_luminance', 'zero_pixels',
        })

    def test_invalid_shape_and_nonfinite_rejected(self):
        for image in (np.empty((0, 1, 3)), np.ones((2, 2, 4)),
                      np.array([[[np.nan, 0., 0.]]])):
            with self.assertRaises(ValueError):
                validate_rgb_image(image)

    def test_mismatched_reference_rejected(self):
        with self.assertRaises(ValueError):
            summarize_transport_image(np.zeros((1, 1, 3)), seconds=1.,
                                      reference=np.zeros((2, 1, 3)))

    def test_invalid_seconds_rejected(self):
        for seconds in (-1., np.nan, np.inf):
            with self.assertRaises(ValueError):
                summarize_transport_image(np.zeros((1, 1, 3)), seconds=seconds)

    def test_file_hash(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'small.bin'
            path.write_bytes(b't2\x00test')
            self.assertEqual(file_sha256(path), hashlib.sha256(b't2\x00test').hexdigest())


if __name__ == '__main__':
    unittest.main()
