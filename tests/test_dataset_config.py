"""Tests for the Mitsuba dataset configuration boundary."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


try:
    import mitsuba  # noqa: F401
except ModuleNotFoundError:
    mitsuba_stub = types.ModuleType("mitsuba")
    setattr(mitsuba_stub, "__version__", "test-stub")

    sys.modules["mitsuba"] = mitsuba_stub

    try:
        dataset_config = importlib.import_module(
            "neural_path_guiding.renderers.mitsuba.dataset_config"
        )
    finally:
        sys.modules.pop("mitsuba", None)
else:
    dataset_config = importlib.import_module(
        "neural_path_guiding.renderers.mitsuba.dataset_config"
    )


from neural_path_guiding.renderers.mitsuba.teachers import (
    INCIDENT_RADIANCE_TARGET,
    PRODUCT_INTEGRAND_TARGET,
    VISIBILITY_COSINE_TARGET,
    PhysicalTeacherSettings,
    VisibilityCosineTeacherSettings,
)


class TestDatasetConfig(unittest.TestCase):
    def make_config(self) -> dict:
        return {
            "experiment_name": "test_dataset",
            "mitsuba": {
                "variant": "scalar_rgb",
            },
            "scene": {
                "type": "builtin",
                "name": "cornell_box",
            },
            "camera": {
                "origin": [0.0, 1.0, 4.0],
                "target": [0.0, 1.0, 0.0],
                "up": [0.0, 1.0, 0.0],
                "fov_degrees": 45.0,
                "image_width": 64,
                "image_height": 64,
            },
            "bins": {
                "n_mu": 2,
                "n_phi": 4,
            },
            "dataset": {
                "num_shading_points": 16,
                "max_sampling_attempts": 100,
                "seed": 42,
                "output_path": "data/test_dataset.npz",
            },
            "teacher": {
                "target_type": VISIBILITY_COSINE_TARGET,
                "samples_per_bin": 4,
                "smoothing": 1e-6,
                "max_distance": 1000.0,
                "environment_weight": 1.0,
                "emitter_weight": 8.0,
                "occluded_weight": 0.02,
            },
        }

    def test_builds_visibility_teacher(self) -> None:
        teacher_config = self.make_config()["teacher"]

        teacher = dataset_config.build_teacher_settings(
            teacher_config=teacher_config,
            target_type=VISIBILITY_COSINE_TARGET,
        )

        self.assertIsInstance(
            teacher,
            VisibilityCosineTeacherSettings,
        )
        self.assertEqual(
            teacher.target_type,
            VISIBILITY_COSINE_TARGET,
        )
        self.assertEqual(
            teacher.samples_per_bin,
            4,
        )
        self.assertEqual(
            teacher.emitter_weight,
            8.0,
        )

    def test_builds_physical_teacher_settings(self) -> None:
        for target_type in (
            INCIDENT_RADIANCE_TARGET,
            PRODUCT_INTEGRAND_TARGET,
        ):
            with self.subTest(target_type=target_type):
                teacher_config = {
                    "target_type": target_type,
                    "samples_per_bin": 8,
                    "radiance_samples": 4,
                    "smoothing": 1e-5,
                    "max_depth": 7,
                    "rr_depth": 4,
                }

                teacher = dataset_config.build_teacher_settings(
                    teacher_config=teacher_config,
                    target_type=target_type,
                )

                self.assertIsInstance(
                    teacher,
                    PhysicalTeacherSettings,
                )
                self.assertEqual(
                    teacher.target_type,
                    target_type,
                )
                self.assertEqual(
                    teacher.samples_per_bin,
                    8,
                )
                self.assertEqual(
                    teacher.radiance_samples,
                    4,
                )
                self.assertEqual(
                    teacher.max_depth,
                    7,
                )
                self.assertEqual(
                    teacher.rr_depth,
                    4,
                )

    def test_physical_teacher_uses_defaults(self) -> None:
        teacher_config = {
            "target_type": INCIDENT_RADIANCE_TARGET,
            "samples_per_bin": 4,
        }

        teacher = dataset_config.build_teacher_settings(
            teacher_config=teacher_config,
            target_type=INCIDENT_RADIANCE_TARGET,
        )

        self.assertIsInstance(
            teacher,
            PhysicalTeacherSettings,
        )
        self.assertEqual(
            teacher.radiance_samples,
            1,
        )
        self.assertEqual(
            teacher.smoothing,
            1e-6,
        )
        self.assertEqual(
            teacher.max_depth,
            6,
        )
        self.assertEqual(
            teacher.rr_depth,
            3,
        )

    def test_rejects_unknown_teacher_builder(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Unknown teacher target_type",
        ):
            dataset_config.build_teacher_settings(
                teacher_config={
                    "target_type": "unknown",
                    "samples_per_bin": 4,
                },
                target_type="unknown",
            )

    def test_unknown_target_is_rejected_before_scene_loading(
        self,
    ) -> None:
        config = self.make_config()
        config["teacher"]["target_type"] = "unknown"

        with (
            patch.object(
                dataset_config,
                "configure_mitsuba",
            ) as configure,
            patch.object(
                dataset_config,
                "load_scene_from_config",
            ) as loader,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "Unknown teacher target_type",
            ):
                dataset_config.build_settings_from_config(
                    config=config,
                    config_path=Path("dataset.yaml"),
                    project_root=Path.cwd(),
                )

        configure.assert_not_called()
        loader.assert_not_called()

    def test_product_target_loads_scene(
        self,
    ) -> None:
        config = self.make_config()
        fake_scene = object()

        config["teacher"] = {
            "target_type": PRODUCT_INTEGRAND_TARGET,
            "samples_per_bin": 2,
            "radiance_samples": 2,
            "smoothing": 1e-6,
            "max_depth": 6,
            "rr_depth": 3,
        }

        with (
            patch.object(
                dataset_config,
                "validate_windows_llvm_runtime",
            ),
            patch.object(
                dataset_config,
                "configure_mitsuba",
            ),
            patch.object(
                dataset_config,
                "load_scene_from_config",
                return_value=fake_scene,
            ),
        ):
            settings = (
                dataset_config
                .build_settings_from_config(
                    config=config,
                    config_path=Path("dataset.yaml"),
                    project_root=Path.cwd(),
                )
            )

        self.assertIs(
            settings.scene,
            fake_scene,
        )
        self.assertIsInstance(
            settings.teacher,
            PhysicalTeacherSettings,
        )
        self.assertEqual(
            settings.teacher.target_type,
            PRODUCT_INTEGRAND_TARGET,
        )   
        
        
    def test_incident_radiance_target_loads_scene(
        self,
    ) -> None:
        config = self.make_config()
        fake_scene = object()

        config["teacher"] = {
            "target_type": INCIDENT_RADIANCE_TARGET,
            "samples_per_bin": 2,
            "radiance_samples": 2,
            "smoothing": 1e-6,
            "max_depth": 6,
            "rr_depth": 3,
        }

        with (
            patch.object(
                dataset_config,
                "validate_windows_llvm_runtime",
            ),
            patch.object(
                dataset_config,
                "configure_mitsuba",
            ),
            patch.object(
                dataset_config,
                "load_scene_from_config",
                return_value=fake_scene,
            ),
        ):
            settings = (
                dataset_config
                .build_settings_from_config(
                    config=config,
                    config_path=Path("dataset.yaml"),
                    project_root=Path.cwd(),
                )
            )

        self.assertIs(
            settings.scene,
            fake_scene,
        )
        self.assertIsInstance(
            settings.teacher,
            PhysicalTeacherSettings,
        )
        self.assertEqual(
            settings.teacher.target_type,
            INCIDENT_RADIANCE_TARGET,
        )
            
    def test_removed_ray_epsilon_is_rejected_before_scene_loading(
        self,
    ) -> None:
        config = self.make_config()
        config["teacher"]["ray_epsilon"] = 1e-4

        with patch.object(
            dataset_config,
            "load_scene_from_config",
        ) as loader:
            with self.assertRaisesRegex(
                ValueError,
                "ray_epsilon was removed",
            ):
                dataset_config.build_settings_from_config(
                    config=config,
                    config_path=Path("dataset.yaml"),
                    project_root=Path.cwd(),
                )

        loader.assert_not_called()

    def test_valid_config_builds_settings_with_provenance(
        self,
    ) -> None:
        config = self.make_config()
        fake_scene = object()

        with (
            patch.object(
                dataset_config,
                "validate_windows_llvm_runtime",
            ),
            patch.object(
                dataset_config,
                "configure_mitsuba",
            ) as configure,
            patch.object(
                dataset_config,
                "load_scene_from_config",
                return_value=fake_scene,
            ) as loader,
        ):
            settings = dataset_config.build_settings_from_config(
                config=config,
                config_path=Path("dataset.yaml"),
                project_root=Path.cwd(),
            )

        self.assertIs(
            settings.scene,
            fake_scene,
        )
        self.assertIsInstance(
            settings.teacher,
            VisibilityCosineTeacherSettings,
        )
        self.assertEqual(
            settings.teacher.target_type,
            VISIBILITY_COSINE_TARGET,
        )
        self.assertEqual(
            settings.provenance.config_snapshot,
            config,
        )
        self.assertEqual(
            settings.provenance.mitsuba_variant,
            "scalar_rgb",
        )

        configure.assert_called_once_with("scalar_rgb")
        loader.assert_called_once()


if __name__ == "__main__":
    unittest.main()