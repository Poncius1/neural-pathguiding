"""Validate material-aware features on the simple diffuse scene."""

from __future__ import annotations

from pathlib import Path
import unittest

import mitsuba as mi
import numpy as np

from neural_path_guiding.evaluation.direct_estimators import (
    sample_fixed_surface,
)
from neural_path_guiding.renderers.mitsuba.adapters import (
    mitsuba_vector_to_numpy,
    normalize_numpy_vector,
)
from neural_path_guiding.renderers.mitsuba.material_features import (
    MATERIAL_FEATURE_SCHEMA_VERSION,
    extract_diffuse_reflectance,
    is_supported_diffuse_bsdf,
    surface_to_material_features,
    validate_diffuse_reflectance,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestMaterialFeatures(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        mi.set_variant("scalar_rgb")

        cls.scene = getattr(mi, "load_file")(
            str(
                PROJECT_ROOT
                / "scenes/simple_diffuse/scene.xml"
            )
        )

    def sample(self, x: int, y: int):
        return sample_fixed_surface(
            self.scene,
            x,
            y,
        )

    def test_sphere_reflectance(self) -> None:
        _, interaction, _ = self.sample(224, 224)

        self.assertTrue(
            is_supported_diffuse_bsdf(
                interaction.bsdf()
            )
        )

        albedo = extract_diffuse_reflectance(
            interaction
        )

        np.testing.assert_allclose(
            albedo,
            [0.25, 0.38, 0.85],
            atol=1e-4,
            rtol=0.0,
        )

    def test_back_wall_reflectance(self) -> None:
        _, interaction, _ = self.sample(0, 160)

        albedo = extract_diffuse_reflectance(
            interaction
        )

        np.testing.assert_allclose(
            albedo,
            [0.45, 0.45, 0.45],
            atol=1e-4,
            rtol=0.0,
        )

    def test_floor_reflectance(self) -> None:
        _, interaction, _ = self.sample(0, 351)

        albedo = extract_diffuse_reflectance(
            interaction
        )

        np.testing.assert_allclose(
            albedo,
            [0.72, 0.72, 0.72],
            atol=1e-4,
            rtol=0.0,
        )

    def test_materials_have_distinct_albedos(
        self,
    ) -> None:
        _, sphere, _ = self.sample(224, 224)
        _, wall, _ = self.sample(0, 160)

        sphere_albedo = extract_diffuse_reflectance(
            sphere
        )

        wall_albedo = extract_diffuse_reflectance(
            wall
        )

        self.assertFalse(
            np.allclose(
                sphere_albedo,
                wall_albedo,
            )
        )

    def test_feature_vector_keeps_dimension(
        self,
    ) -> None:
        ray, interaction, _ = self.sample(224, 224)

        outgoing = normalize_numpy_vector(
            -mitsuba_vector_to_numpy(ray.d)
        )

        features = surface_to_material_features(
            surface_interaction=interaction,
            outgoing_direction=outgoing,
            bounce_depth=0,
        )

        vector = features.to_array()

        self.assertEqual(
            vector.shape,
            (14,),
        )

        np.testing.assert_allclose(
            vector[10:13],
            [0.25, 0.38, 0.85],
            atol=1e-4,
            rtol=0.0,
        )

        self.assertEqual(
            MATERIAL_FEATURE_SCHEMA_VERSION,
            2,
        )

    def test_rejects_invalid_reflectance(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            validate_diffuse_reflectance(
                np.array([1.5, 0.2, 0.3])
            )

    def test_accepts_valid_reflectance(
        self,
    ) -> None:
        result = validate_diffuse_reflectance(
            np.array([0.2, 0.4, 0.8])
        )

        np.testing.assert_allclose(
            result,
            [0.2, 0.4, 0.8],
        )


if __name__ == "__main__":
    unittest.main()