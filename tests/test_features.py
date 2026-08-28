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
    def make_features(self, **overrides: object) -> ShadingFeatures:
        values: dict[str, object] = {
            "position": np.array([1.0, 2.0, 3.0]),
            "normal": np.array([0.0, 1.0, 0.0]),
            "outgoing_direction": np.array([0.0, 0.0, 1.0]),
            "roughness": 1.0,
            "albedo": np.array([0.7, 0.7, 0.7]),
            "bounce_depth": 0,
        }
        values.update(overrides)
        return ShadingFeatures(**values)  # type: ignore[arg-type]

    def test_feature_names_match_dimension(self) -> None:
        self.assertEqual(len(FEATURE_NAMES), FEATURE_DIMENSION)

    def test_features_convert_to_expected_array_shape(self) -> None:
        features = self.make_features()

        feature_vector = features.to_array()

        self.assertEqual(feature_vector.shape, (FEATURE_DIMENSION,))
        self.assertTrue(np.all(np.isfinite(feature_vector)))

    def test_direction_vectors_are_normalized(self) -> None:
        vector = normalize_vector3(np.array([0.0, 2.0, 0.0]), "test_vector")

        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0)

    def test_validate_feature_vector_rejects_wrong_shape(self) -> None:
        with self.assertRaises(ValueError):
            validate_feature_vector(np.zeros(3))

    def test_non_finite_roughness_is_rejected(self) -> None:
        for roughness in (np.nan, np.inf, -np.inf):
            with self.subTest(roughness=roughness):
                with self.assertRaises(ValueError):
                    self.make_features(roughness=roughness).to_array()

    def test_roughness_outside_unit_interval_is_rejected(self) -> None:
        for roughness in (-0.1, 1.1):
            with self.subTest(roughness=roughness):
                with self.assertRaises(ValueError):
                    self.make_features(roughness=roughness).to_array()

    def test_albedo_outside_unit_interval_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.make_features(albedo=np.array([1.1, 0.5, 0.5])).to_array()

    def test_non_integer_bounce_depth_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            self.make_features(bounce_depth=0.5).to_array()


if __name__ == "__main__":
    unittest.main()