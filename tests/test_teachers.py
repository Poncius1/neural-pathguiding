"""Tests for Mitsuba directional teacher estimators."""

from dataclasses import replace
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from neural_path_guiding.renderers.mitsuba.teachers import (
    INCIDENT_RADIANCE_TARGET,
    PRODUCT_INTEGRAND_TARGET,
    PhysicalTeacherSettings,
    VISIBILITY_COSINE_TARGET,
    VisibilityCosineTeacherSettings,
    build_teacher_metadata,
    validate_teacher_settings,
)

try:
    import mitsuba  # noqa: F401
except ModuleNotFoundError:
    mitsuba_stub = types.ModuleType("mitsuba")
    sys.modules["mitsuba"] = mitsuba_stub

    try:
        from neural_path_guiding.renderers.mitsuba import teachers
    finally:
        sys.modules.pop("mitsuba", None)
else:
    from neural_path_guiding.renderers.mitsuba import teachers


class FakeRay:
    """Minimal ray used by the teacher tests."""

    def __init__(self) -> None:
        self.maxt: float | None = None


class FakeSurfaceInteraction:
    """Minimal source surface interaction."""

    def __init__(self) -> None:
        self.spawned_ray: FakeRay | None = None

    def spawn_ray(
        self,
        direction: object,
    ) -> FakeRay:
        del direction

        self.spawned_ray = FakeRay()
        return self.spawned_ray


class FakeHit:
    """Minimal result returned by ray_intersect."""

    def __init__(
        self,
        *,
        valid: bool,
        has_emitter: bool = False,
    ) -> None:
        self.valid = valid
        self.has_emitter = has_emitter

    def is_valid(self) -> bool:
        return self.valid

    def emitter(
        self,
        scene: object,
    ) -> object | None:
        del scene

        if self.has_emitter:
            return object()

        return None


class FakeScene:
    """Scene that always returns a configured intersection."""

    def __init__(
        self,
        hit: FakeHit,
    ) -> None:
        self.hit = hit
        self.intersection_count = 0

    def ray_intersect(
        self,
        ray: FakeRay,
    ) -> FakeHit:
        del ray

        self.intersection_count += 1
        return self.hit


class TestTeachers(unittest.TestCase):
    """Validate teacher selection and numerical behavior."""

    def setUp(self) -> None:
        self.teacher = (
            teachers.VisibilityCosineTeacherSettings(
                target_type=(
                    teachers.VISIBILITY_COSINE_TARGET
                ),
                samples_per_bin=4,
                smoothing=1e-6,
                max_distance=25.0,
                environment_weight=1.0,
                emitter_weight=8.0,
                occluded_weight=0.02,
            )
        )

        self.physical_teacher = (
            teachers.PhysicalTeacherSettings(
                target_type=(
                    teachers.INCIDENT_RADIANCE_TARGET
                ),
                samples_per_bin=4,
                radiance_samples=2,
                smoothing=1e-6,
                max_depth=6,
                rr_depth=5,
            )
        )

        self.normal = np.array(
            [0.0, 0.0, 1.0],
            dtype=np.float64,
        )
        self.direction = np.array(
            [0.0, 0.0, 1.0],
            dtype=np.float64,
        )
        self.outgoing_direction = np.array(
            [0.0, 0.0, 1.0],
            dtype=np.float64,
        )

    def evaluate(
        self,
        hit: FakeHit,
    ) -> tuple[float, FakeSurfaceInteraction]:
        scene = FakeScene(hit)
        surface_interaction = FakeSurfaceInteraction()

        context = teachers.TeacherContext(
            scene=scene,
            surface_interaction=surface_interaction,
            normal=self.normal,
            outgoing_direction=self.outgoing_direction,
        )

        fake_mitsuba = SimpleNamespace(
            Vector3f=lambda *values: values
        )

        with patch.object(
            teachers,
            "_mi",
            fake_mitsuba,
        ):
            contribution = (
                teachers
                .estimate_visibility_cosine_contribution(
                    context=context,
                    world_direction=self.direction,
                    teacher=self.teacher,
                )
            )

        return contribution, surface_interaction

    def test_defines_t0_t1_t2_targets(self) -> None:
        expected_targets = {
            teachers.VISIBILITY_COSINE_TARGET: "T0",
            teachers.INCIDENT_RADIANCE_TARGET: "T1",
            teachers.PRODUCT_INTEGRAND_TARGET: "T2",
        }

        for target_type, label in expected_targets.items():
            with self.subTest(target_type=target_type):
                self.assertIn(
                    target_type,
                    teachers.DEFINED_TARGET_TYPES,
                )
                self.assertIn(
                    label,
                    teachers.TEACHER_TARGET_DESCRIPTIONS[
                        target_type
                    ],
                )

        self.assertEqual(
            teachers.SUPPORTED_TARGET_TYPES,
            frozenset(
                {
                    teachers.VISIBILITY_COSINE_TARGET,
                }
            ),
        )

    def test_resolves_visibility_cosine_teacher(
        self,
    ) -> None:
        estimator = teachers.resolve_teacher_estimator(
            teachers.VISIBILITY_COSINE_TARGET
        )

        self.assertIs(
            estimator,
            teachers.estimate_visibility_cosine_contribution,
        )

    def test_rejects_unknown_teacher(self) -> None:
        with self.assertRaises(ValueError):
            teachers.resolve_teacher_estimator(
                "unknown_teacher"
            )

    def test_rejects_unimplemented_physical_teachers(
        self,
    ) -> None:
        unimplemented_targets = (
            teachers.INCIDENT_RADIANCE_TARGET,
            teachers.PRODUCT_INTEGRAND_TARGET,
        )

        for target_type in unimplemented_targets:
            with self.subTest(target_type=target_type):
                with self.assertRaisesRegex(
                    ValueError,
                    "defined but not implemented",
                ):
                    teachers.resolve_teacher_estimator(
                        target_type
                    )

    def test_visibility_teacher_selects_expected_weight(
        self,
    ) -> None:
        cases = (
            (
                FakeHit(valid=False),
                self.teacher.environment_weight,
            ),
            (
                FakeHit(
                    valid=True,
                    has_emitter=True,
                ),
                self.teacher.emitter_weight,
            ),
            (
                FakeHit(
                    valid=True,
                    has_emitter=False,
                ),
                self.teacher.occluded_weight,
            ),
        )

        for hit, expected in cases:
            with self.subTest(expected=expected):
                contribution, surface_interaction = (
                    self.evaluate(hit)
                )

                self.assertAlmostEqual(
                    contribution,
                    expected,
                )

                spawned_ray = (
                    surface_interaction.spawned_ray
                )

                self.assertIsNotNone(spawned_ray)
                assert spawned_ray is not None

                self.assertEqual(
                    spawned_ray.maxt,
                    self.teacher.max_distance,
                )

    def test_visibility_teacher_rejects_tangent_direction(
        self,
    ) -> None:
        scene = FakeScene(
            FakeHit(valid=False)
        )
        surface_interaction = (
            FakeSurfaceInteraction()
        )

        context = teachers.TeacherContext(
            scene=scene,
            surface_interaction=surface_interaction,
            normal=self.normal,
            outgoing_direction=self.outgoing_direction,
        )

        tangent_direction = np.array(
            [1.0, 0.0, 0.0],
            dtype=np.float64,
        )

        contribution = (
            teachers
            .estimate_visibility_cosine_contribution(
                context=context,
                world_direction=tangent_direction,
                teacher=self.teacher,
            )
        )

        self.assertEqual(
            contribution,
            0.0,
        )
        self.assertEqual(
            scene.intersection_count,
            0,
        )
        self.assertIsNone(
            surface_interaction.spawned_ray
        )

    def test_accepts_t1_and_t2_physical_settings(
        self,
    ) -> None:
        target_types = (
            teachers.INCIDENT_RADIANCE_TARGET,
            teachers.PRODUCT_INTEGRAND_TARGET,
        )

        for target_type in target_types:
            with self.subTest(target_type=target_type):
                teacher = replace(
                    self.physical_teacher,
                    target_type=target_type,
                )

                teachers.validate_physical_teacher_settings(
                    teacher
                )
                

    def test_physical_settings_reject_wrong_target(
        self,
    ) -> None:
        teacher = replace(
            self.physical_teacher,
            target_type=(
                teachers.VISIBILITY_COSINE_TARGET
            ),
        )

        with self.assertRaises(ValueError):
            teachers.validate_physical_teacher_settings(
                teacher
            )

    def test_physical_settings_reject_invalid_counts(
        self,
    ) -> None:
        invalid_settings = (
            replace(
                self.physical_teacher,
                samples_per_bin=0,
            ),
            replace(
                self.physical_teacher,
                radiance_samples=0,
            ),
            replace(
                self.physical_teacher,
                max_depth=0,
            ),
            replace(
                self.physical_teacher,
                rr_depth=-1,
            ),
        )

        for teacher in invalid_settings:
            with self.subTest(teacher=teacher):
                with self.assertRaises(ValueError):
                    teachers.validate_physical_teacher_settings(
                        teacher
                    )

    def test_physical_settings_reject_late_rr_depth(
        self,
    ) -> None:
        teacher = replace(
            self.physical_teacher,
            max_depth=4,
            rr_depth=5,
        )

        with self.assertRaises(ValueError):
            teachers.validate_physical_teacher_settings(
                teacher
            )

    def test_physical_settings_reject_invalid_smoothing(
        self,
    ) -> None:
        invalid_values = (
            -1e-6,
            float("nan"),
            float("inf"),
        )

        for smoothing in invalid_values:
            with self.subTest(smoothing=smoothing):
                teacher = replace(
                    self.physical_teacher,
                    smoothing=smoothing,
                )

                with self.assertRaises(ValueError):
                    teachers.validate_physical_teacher_settings(
                        teacher
                    )
    def test_generic_validation_accepts_visibility_teacher(self) -> None:
        teacher = VisibilityCosineTeacherSettings(
            target_type=VISIBILITY_COSINE_TARGET,
            samples_per_bin=4,
            smoothing=1e-6,
            max_distance=1000.0,
            environment_weight=1.0,
            emitter_weight=8.0,
            occluded_weight=0.02,
        )

        validate_teacher_settings(teacher)


    def test_generic_validation_accepts_physical_teacher(self) -> None:
        teacher = PhysicalTeacherSettings(
            target_type=PRODUCT_INTEGRAND_TARGET,
            samples_per_bin=4,
            radiance_samples=2,
            smoothing=1e-6,
            max_depth=6,
            rr_depth=3,
        )

        validate_teacher_settings(teacher)


    def test_visibility_teacher_metadata_is_complete(self) -> None:
        teacher = VisibilityCosineTeacherSettings(
            target_type=VISIBILITY_COSINE_TARGET,
            samples_per_bin=4,
            smoothing=1e-6,
            max_distance=1000.0,
            environment_weight=1.0,
            emitter_weight=8.0,
            occluded_weight=0.02,
        )

        metadata = build_teacher_metadata(teacher)

        self.assertEqual(metadata["target_type"], VISIBILITY_COSINE_TARGET)
        self.assertEqual(metadata["samples_per_bin"], 4)
        self.assertEqual(metadata["emitter_weight"], 8.0)
        self.assertEqual(metadata["max_distance"], 1000.0)


    def test_physical_teacher_metadata_is_complete(self) -> None:
        teacher = PhysicalTeacherSettings(
            target_type=INCIDENT_RADIANCE_TARGET,
            samples_per_bin=8,
            radiance_samples=4,
            smoothing=1e-6,
            max_depth=6,
            rr_depth=3,
        )

        metadata = build_teacher_metadata(teacher)

        self.assertEqual(metadata["target_type"], INCIDENT_RADIANCE_TARGET)
        self.assertEqual(metadata["samples_per_bin"], 8)
        self.assertEqual(metadata["radiance_samples"], 4)
        self.assertEqual(metadata["max_depth"], 6)
        self.assertEqual(metadata["rr_depth"], 3)                    


if __name__ == "__main__":
    unittest.main()