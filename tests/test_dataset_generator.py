"""Tests for dataset generator settings, metadata and serialization.

The tests use a minimal Mitsuba module stub because they exercise only the
NumPy/configuration side of the generator, not rendering.
"""

from __future__ import annotations

from dataclasses import replace
import importlib
from pathlib import Path
import sys
import tempfile
import types
import unittest

import numpy as np


try:
    import mitsuba  # noqa: F401
except ModuleNotFoundError:
    mitsuba_stub = types.ModuleType("mitsuba")
    sys.modules["mitsuba"] = mitsuba_stub
    try:
        importlib.import_module(
            "neural_path_guiding.renderers.mitsuba.dataset_generator"
        )
    finally:
        sys.modules.pop("mitsuba", None)


from neural_path_guiding.core.bins import HemisphereBins
from neural_path_guiding.data.schema import DATASET_FORMAT_VERSION
from neural_path_guiding.renderers.mitsuba.dataset_generator import (
    DatasetCamera,
    DatasetGenerationSettings,
    DatasetProvenance,
    build_metadata,
    validate_dataset_settings,
)
from neural_path_guiding.renderers.mitsuba.teachers import (
    VISIBILITY_COSINE_TARGET,
    VisibilityCosineTeacherSettings,
    estimate_visibility_cosine_contribution,
    resolve_teacher_estimator,
    surface_hit_has_emitter,
    validate_teacher_target_type,
)


class TestDatasetGenerator(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.project_root = Path(self.temporary_directory.name)
        self.config_path = self.project_root / "dataset.yaml"
        self.scene_path = self.project_root / "scene.xml"
        self.config_path.write_text("experiment_name: test\n", encoding="utf-8")
        self.scene_path.write_text("<scene version='3.0.0' />\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def make_settings(self) -> DatasetGenerationSettings:
        return DatasetGenerationSettings(
            scene=object(),
            camera=DatasetCamera(
                origin=np.array([0.0, 0.0, 1.0]),
                target=np.array([0.0, 0.0, 0.0]),
                up=np.array([0.0, 1.0, 0.0]),
                fov_degrees=45.0,
                image_width=64,
                image_height=64,
            ),
            bins=HemisphereBins(n_mu=2, n_phi=4),
            teacher=VisibilityCosineTeacherSettings(
                target_type=VISIBILITY_COSINE_TARGET,
                samples_per_bin=4,
                smoothing=1e-6,
                max_distance=1000.0,
                environment_weight=1.0,
                emitter_weight=8.0,
                occluded_weight=0.02,
            ),
            num_shading_points=16,
            max_sampling_attempts=100,
            seed=42,
            output_path=self.project_root / "dataset.npz",
            experiment_name="test_dataset",
            provenance=DatasetProvenance(
                project_root=self.project_root,
                config_path=self.config_path,
                config_snapshot={"experiment_name": "test_dataset"},
                scene_config={"type": "xml", "path": "scene.xml"},
                mitsuba_variant="scalar_rgb",
            ),
        )

    def test_valid_settings_are_accepted(self) -> None:
        validate_dataset_settings(self.make_settings())

    def test_unsupported_teacher_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_teacher_target_type("not_implemented")

        settings = self.make_settings()
        invalid_teacher = replace(settings.teacher, target_type="not_implemented")

        with self.assertRaises(ValueError):
            validate_dataset_settings(replace(settings, teacher=invalid_teacher))

    def test_supported_teacher_resolves_to_implementation(self) -> None:
        estimator = resolve_teacher_estimator(VISIBILITY_COSINE_TARGET)

        self.assertIs(estimator, estimate_visibility_cosine_contribution)

    def test_non_finite_teacher_weight_is_rejected(self) -> None:
        settings = self.make_settings()
        invalid_teacher = replace(settings.teacher, emitter_weight=np.nan)

        with self.assertRaises(ValueError):
            validate_dataset_settings(replace(settings, teacher=invalid_teacher))

    def test_invalid_camera_fov_is_rejected(self) -> None:
        settings = self.make_settings()

        for fov in (0.0, 180.0, np.nan):
            with self.subTest(fov=fov):
                invalid_camera = replace(settings.camera, fov_degrees=fov)
                with self.assertRaises(ValueError):
                    validate_dataset_settings(replace(settings, camera=invalid_camera))

    def test_parallel_camera_up_vector_is_rejected(self) -> None:
        settings = self.make_settings()
        invalid_camera = replace(settings.camera, up=np.array([0.0, 0.0, 2.0]))

        with self.assertRaises(ValueError):
            validate_dataset_settings(replace(settings, camera=invalid_camera))

    def test_output_must_use_npz_extension(self) -> None:
        settings = replace(
            self.make_settings(),
            output_path=self.project_root / "dataset.npy",
        )

        with self.assertRaises(ValueError):
            validate_dataset_settings(settings)

    def test_metadata_contains_reproducibility_information(self) -> None:
        settings = self.make_settings()

        metadata = build_metadata(settings)

        self.assertEqual(metadata["dataset_format_version"], DATASET_FORMAT_VERSION)
        self.assertEqual(metadata["target_type"], VISIBILITY_COSINE_TARGET)
        self.assertEqual(metadata["source"]["config_path"], "dataset.yaml")
        self.assertEqual(metadata["scene"]["path"], "scene.xml")
        self.assertEqual(len(metadata["source"]["config_sha256"]), 64)
        self.assertEqual(len(metadata["scene"]["sha256"]), 64)
        self.assertIn("mean_contributions", metadata["array_schema"])
        self.assertNotIn("ray_epsilon", metadata["teacher"])

    def test_emitter_api_errors_are_not_hidden(self) -> None:
        class BrokenSurfaceInteraction:
            def emitter(self, scene: object) -> object:
                raise RuntimeError("Mitsuba API failure")

        with self.assertRaisesRegex(RuntimeError, "Mitsuba API failure"):
            surface_hit_has_emitter(BrokenSurfaceInteraction(), object())


if __name__ == "__main__":
    unittest.main()
