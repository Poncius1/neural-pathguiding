"""Tests for dataset generator settings and metadata."""

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
    setattr(mitsuba_stub, "__version__", "test-stub")
    sys.modules["mitsuba"] = mitsuba_stub

    try:
        importlib.import_module(
            "neural_path_guiding.renderers.mitsuba.dataset_generator"
        )
    finally:
        sys.modules.pop("mitsuba", None)


from neural_path_guiding.core.bins import HemisphereBins
from neural_path_guiding.data.schema import (
    DATASET_FORMAT_VERSION,
)
from neural_path_guiding.renderers.mitsuba.dataset_generator import (
    DatasetCamera,
    DatasetGenerationSettings,
    DatasetProvenance,
    build_metadata,
    validate_dataset_settings,
)
from neural_path_guiding.renderers.mitsuba.teachers import (
    INCIDENT_RADIANCE_TARGET,
    PRODUCT_INTEGRAND_TARGET,
    VISIBILITY_COSINE_TARGET,
    PhysicalTeacherSettings,
    VisibilityCosineTeacherSettings,
    estimate_incident_radiance_contribution,
    estimate_product_integrand_contribution,
    estimate_visibility_cosine_contribution,
    resolve_teacher_estimator,
    surface_hit_has_emitter,
    validate_teacher_target_type,
)


class TestDatasetGenerator(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = (
            tempfile.TemporaryDirectory()
        )
        self.project_root = Path(
            self.temporary_directory.name
        )
        self.config_path = (
            self.project_root / "dataset.yaml"
        )
        self.scene_path = (
            self.project_root / "scene.xml"
        )

        self.config_path.write_text(
            "experiment_name: test\n",
            encoding="utf-8",
        )
        self.scene_path.write_text(
            "<scene version='3.0.0' />\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def make_settings(
        self,
    ) -> DatasetGenerationSettings:
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
            bins=HemisphereBins(
                n_mu=2,
                n_phi=4,
            ),
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
            output_path=(
                self.project_root / "dataset.npz"
            ),
            experiment_name="test_dataset",
            provenance=DatasetProvenance(
                project_root=self.project_root,
                config_path=self.config_path,
                config_snapshot={
                    "experiment_name": "test_dataset",
                },
                scene_config={
                    "type": "xml",
                    "path": "scene.xml",
                },
                mitsuba_variant="scalar_rgb",
            ),
        )

    def make_physical_settings(
        self,
    ) -> DatasetGenerationSettings:
        physical_teacher = PhysicalTeacherSettings(
            target_type=INCIDENT_RADIANCE_TARGET,
            samples_per_bin=2,
            radiance_samples=2,
            smoothing=1e-6,
            max_depth=6,
            rr_depth=3,
        )

        return replace(
            self.make_settings(),
            teacher=physical_teacher,
        )

    def test_valid_settings_are_accepted(self) -> None:
        validate_dataset_settings(
            self.make_settings()
        )

    def test_physical_settings_are_accepted(self) -> None:
        validate_dataset_settings(
            self.make_physical_settings()
        )

    def test_unsupported_teacher_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_teacher_target_type(
                "not_implemented"
            )

        settings = self.make_settings()

        invalid_teacher = replace(
            settings.teacher,
            target_type="not_implemented",
        )

        with self.assertRaises(ValueError):
            validate_dataset_settings(
                replace(
                    settings,
                    teacher=invalid_teacher,
                )
            )

    def test_supported_teachers_resolve(self) -> None:
        expected_estimators = {
            VISIBILITY_COSINE_TARGET: (
                estimate_visibility_cosine_contribution
            ),
            INCIDENT_RADIANCE_TARGET: (
                estimate_incident_radiance_contribution
            ),
            PRODUCT_INTEGRAND_TARGET: (
                estimate_product_integrand_contribution
            ),
        }

        for target_type, expected in expected_estimators.items():
            with self.subTest(target_type=target_type):
                estimator = resolve_teacher_estimator(
                    target_type
                )

                self.assertIs(
                    estimator,
                    expected,
                )

    def test_non_finite_teacher_weight_is_rejected(
        self,
    ) -> None:
        settings = self.make_settings()

        invalid_teacher = replace(
            settings.teacher,
            emitter_weight=np.nan,
        )

        with self.assertRaises(ValueError):
            validate_dataset_settings(
                replace(
                    settings,
                    teacher=invalid_teacher,
                )
            )

    def test_invalid_camera_fov_is_rejected(self) -> None:
        settings = self.make_settings()

        for fov in (0.0, 180.0, np.nan):
            with self.subTest(fov=fov):
                invalid_camera = replace(
                    settings.camera,
                    fov_degrees=fov,
                )

                with self.assertRaises(ValueError):
                    validate_dataset_settings(
                        replace(
                            settings,
                            camera=invalid_camera,
                        )
                    )

    def test_parallel_camera_up_vector_is_rejected(
        self,
    ) -> None:
        settings = self.make_settings()

        invalid_camera = replace(
            settings.camera,
            up=np.array([0.0, 0.0, 2.0]),
        )

        with self.assertRaises(ValueError):
            validate_dataset_settings(
                replace(
                    settings,
                    camera=invalid_camera,
                )
            )

    def test_output_must_use_npz_extension(self) -> None:
        settings = replace(
            self.make_settings(),
            output_path=(
                self.project_root / "dataset.npy"
            ),
        )

        with self.assertRaises(ValueError):
            validate_dataset_settings(settings)

    def test_visibility_metadata_is_reproducible(
        self,
    ) -> None:
        metadata = build_metadata(
            self.make_settings()
        )

        self.assertEqual(
            metadata["dataset_format_version"],
            DATASET_FORMAT_VERSION,
        )
        self.assertEqual(
            metadata["target_type"],
            VISIBILITY_COSINE_TARGET,
        )
        self.assertEqual(
            metadata["source"]["config_path"],
            "dataset.yaml",
        )
        self.assertEqual(
            metadata["scene"]["path"],
            "scene.xml",
        )
        self.assertEqual(
            len(metadata["source"]["config_sha256"]),
            64,
        )
        self.assertEqual(
            len(metadata["scene"]["sha256"]),
            64,
        )
        self.assertIn(
            "mean_contributions",
            metadata["array_schema"],
        )
        self.assertNotIn(
            "ray_epsilon",
            metadata["teacher"],
        )

    def test_physical_metadata_is_reproducible(
        self,
    ) -> None:
        metadata = build_metadata(
            self.make_physical_settings()
        )

        self.assertEqual(
            metadata["target_type"],
            INCIDENT_RADIANCE_TARGET,
        )
        self.assertEqual(
            metadata["teacher"]["radiance_samples"],
            2,
        )
        self.assertEqual(
            metadata["teacher"]["max_depth"],
            6,
        )
        self.assertEqual(
            metadata["teacher"]["rr_depth"],
            3,
        )
        self.assertNotIn(
            "emitter_weight",
            metadata["teacher"],
        )

    def test_emitter_api_errors_are_not_hidden(
        self,
    ) -> None:
        class BrokenSurfaceInteraction:
            def emitter(
                self,
                scene: object,
            ) -> object:
                del scene
                raise RuntimeError(
                    "Mitsuba API failure"
                )

        with self.assertRaisesRegex(
            RuntimeError,
            "Mitsuba API failure",
        ):
            surface_hit_has_emitter(
                BrokenSurfaceInteraction(),
                object(),
            )


if __name__ == "__main__":
    unittest.main()