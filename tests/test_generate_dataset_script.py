"""Tests for the Mitsuba dataset configuration boundary."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


try:
    import mitsuba  # noqa: F401
except ModuleNotFoundError:
    mitsuba_stub = types.ModuleType("mitsuba")
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


class TestDatasetConfig(unittest.TestCase):
    def make_config(self) -> dict:
        return {
            "experiment_name": "test_dataset",
            "mitsuba": {"variant": "scalar_rgb"},
            "scene": {"type": "builtin", "name": "cornell_box"},
            "camera": {
                "origin": [0.0, 1.0, 4.0],
                "target": [0.0, 1.0, 0.0],
                "up": [0.0, 1.0, 0.0],
                "fov_degrees": 45.0,
                "image_width": 64,
                "image_height": 64,
            },
            "bins": {"n_mu": 2, "n_phi": 4},
            "dataset": {
                "num_shading_points": 16,
                "max_sampling_attempts": 100,
                "seed": 42,
                "output_path": "data/test_dataset.npz",
            },
            "teacher": {
                "target_type": "visibility_cosine_v0",
                "samples_per_bin": 4,
                "smoothing": 1e-6,
                "max_distance": 1000.0,
                "environment_weight": 1.0,
                "emitter_weight": 8.0,
                "occluded_weight": 0.02,
            },
        }

    def test_target_type_rejects_unknown_value_before_scene_loading(self) -> None:
        config = self.make_config()
        config["teacher"]["target_type"] = "not_implemented"

        with patch.object(dataset_config, "load_scene_from_config") as loader:
            with self.assertRaises(ValueError):
                dataset_config.build_settings_from_config(
                    config=config,
                    config_path=Path("dataset.yaml"),
                    project_root=Path.cwd(),
                )

        loader.assert_not_called()

    def test_removed_ray_epsilon_is_rejected_before_scene_loading(self) -> None:
        config = self.make_config()
        config["teacher"]["ray_epsilon"] = 1e-4

        with patch.object(dataset_config, "load_scene_from_config") as loader:
            with self.assertRaisesRegex(ValueError, "ray_epsilon was removed"):
                dataset_config.build_settings_from_config(
                    config=config,
                    config_path=Path("dataset.yaml"),
                    project_root=Path.cwd(),
                )

        loader.assert_not_called()

    def test_valid_config_builds_settings_with_provenance(self) -> None:
        config = self.make_config()

        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "dataset.yaml"
            config_path.touch()

            with (
                patch.object(dataset_config, "validate_windows_llvm_runtime"),
                patch.object(dataset_config, "configure_mitsuba"),
                patch.object(
                    dataset_config,
                    "load_scene_from_config",
                    return_value=object(),
                ),
            ):
                settings = dataset_config.build_settings_from_config(
                    config=config,
                    config_path=config_path,
                    project_root=Path(temporary_directory),
                )

        self.assertEqual(settings.teacher.target_type, "visibility_cosine_v0")
        self.assertEqual(settings.provenance.config_snapshot, config)
        self.assertEqual(settings.provenance.mitsuba_variant, "scalar_rgb")


if __name__ == "__main__":
    unittest.main()
