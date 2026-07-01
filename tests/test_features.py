"""Tests for shading feature representation."""

from __future__ import annotations

import unittest

import numpy as np

from neural_path_guiding.core.features import (
    FEATURE_DIMENSION,
    FEATURE_NAMES,
    ShadingFeatures,
    normalize_vector3,
    validate_feature_vector,
)


class TestShadingFeatures(unittest.TestCase):
    def test_feature_names_match_dimension(self) -> None:
        self.assertEqual(len(FEATURE_NAMES), FEATURE_DIMENSION)

    def test_features_convert_to_expected_array_shape(self) -> None:
        features = ShadingFeatures(
            position=np.array([1.0, 2.0, 3.0]),
            normal=np.array([0.0, 1.0, 0.0]),
            outgoing_direction=np.array([0.0, 0.0, 1.0]),
            roughness=1.0,
            albedo=np.array([0.7, 0.7, 0.7]),
            bounce_depth=0,
        )

        feature_vector = features.to_array()

        self.assertEqual(feature_vector.shape, (FEATURE_DIMENSION,))
        self.assertTrue(np.all(np.isfinite(feature_vector)))

    def test_direction_vectors_are_normalized(self) -> None:
        vector = normalize_vector3(np.array([0.0, 2.0, 0.0]), "test_vector")

        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0)

    def test_validate_feature_vector_rejects_wrong_shape(self) -> None:
        with self.assertRaises(ValueError):
            validate_feature_vector(np.zeros(3))


if __name__ == "__main__":
    unittest.main()