"""Tests for Mitsuba directional teacher estimators."""

from __future__ import annotations

from dataclasses import replace
import importlib
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


try:
    import mitsuba  # noqa: F401
except ModuleNotFoundError:
    mitsuba_stub = types.ModuleType("mitsuba")
    setattr(mitsuba_stub, "__version__", "test-stub")
    sys.modules["mitsuba"] = mitsuba_stub

    try:
        teachers = importlib.import_module(
            "neural_path_guiding.renderers.mitsuba.teachers"
        )
        radiance_module = importlib.import_module(
            "neural_path_guiding.renderers.mitsuba.radiance"
        )
    finally:
        sys.modules.pop("mitsuba", None)
else:
    teachers = importlib.import_module(
        "neural_path_guiding.renderers.mitsuba.teachers"
    )
    radiance_module = importlib.import_module(
        "neural_path_guiding.renderers.mitsuba.radiance"
    )


IncidentRadianceRuntime = (
    radiance_module.IncidentRadianceRuntime
)


class FakeRay:
    def __init__(self) -> None:
        self.maxt: float | None = None


class FakeSurfaceInteraction:
    def __init__(self) -> None:
        self.spawned_ray: FakeRay | None = None
        self.spawned_direction: object | None = None

    def spawn_ray(
        self,
        direction: object,
    ) -> FakeRay:
        self.spawned_direction = direction
        self.spawned_ray = FakeRay()

        return self.spawned_ray


class FakeHit:
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

    def make_context(
        self,
        scene: object,
        surface_interaction: FakeSurfaceInteraction,
    ) -> object:
        return teachers.TeacherContext(
            scene=scene,
            surface_interaction=surface_interaction,
            normal=self.normal,
            outgoing_direction=self.outgoing_direction,
        )

    def evaluate_visibility(
        self,
        hit: FakeHit,
    ) -> tuple[float, FakeSurfaceInteraction]:
        scene = FakeScene(hit)
        surface_interaction = FakeSurfaceInteraction()

        context = self.make_context(
            scene,
            surface_interaction,
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

    def make_runtime(self) -> object:
        return IncidentRadianceRuntime(
            integrator=object(),
            sampler=object(),
            radiance_samples=2,
        )

    def make_product_teacher(self) -> object:
        return replace(
            self.physical_teacher,
            target_type=(
                teachers.PRODUCT_INTEGRAND_TARGET
            ),
        )

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
                    teachers.INCIDENT_RADIANCE_TARGET,
                    teachers.PRODUCT_INTEGRAND_TARGET,
                }
            ),
        )

    def test_resolves_visibility_teacher(self) -> None:
        estimator = teachers.resolve_teacher_estimator(
            teachers.VISIBILITY_COSINE_TARGET
        )

        self.assertIs(
            estimator,
            teachers.estimate_visibility_cosine_contribution,
        )

    def test_resolves_incident_radiance_teacher(
        self,
    ) -> None:
        estimator = teachers.resolve_teacher_estimator(
            teachers.INCIDENT_RADIANCE_TARGET
        )

        self.assertIs(
            estimator,
            teachers.estimate_incident_radiance_contribution,
        )

    def test_rejects_unknown_teacher(self) -> None:
        with self.assertRaises(ValueError):
            teachers.resolve_teacher_estimator(
                "unknown_teacher"
            )

    def test_resolves_product_integrand_teacher(
        self,
    ) -> None:
        estimator = teachers.resolve_teacher_estimator(
            teachers.PRODUCT_INTEGRAND_TARGET
        )

        self.assertIs(
            estimator,
            teachers.estimate_product_integrand_contribution,
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
                contribution, interaction = (
                    self.evaluate_visibility(hit)
                )

                self.assertAlmostEqual(
                    contribution,
                    expected,
                )

                self.assertIsNotNone(
                    interaction.spawned_ray
                )
                assert interaction.spawned_ray is not None

                self.assertEqual(
                    interaction.spawned_ray.maxt,
                    self.teacher.max_distance,
                )

    def test_visibility_teacher_rejects_tangent_direction(
        self,
    ) -> None:
        scene = FakeScene(
            FakeHit(valid=False)
        )
        interaction = FakeSurfaceInteraction()

        context = self.make_context(
            scene,
            interaction,
        )

        contribution = (
            teachers
            .estimate_visibility_cosine_contribution(
                context=context,
                world_direction=np.array(
                    [1.0, 0.0, 0.0]
                ),
                teacher=self.teacher,
            )
        )

        self.assertEqual(contribution, 0.0)
        self.assertEqual(
            scene.intersection_count,
            0,
        )
        self.assertIsNone(
            interaction.spawned_ray
        )

    def test_incident_teacher_returns_luminance(
        self,
    ) -> None:
        scene = object()
        interaction = FakeSurfaceInteraction()
        context = self.make_context(
            scene,
            interaction,
        )
        runtime = self.make_runtime()

        incident_rgb = np.array(
            [2.0, 4.0, 6.0]
        )

        fake_mitsuba = SimpleNamespace(
            Vector3f=lambda *values: np.array(values)
        )

        with (
            patch.object(
                teachers,
                "_mi",
                fake_mitsuba,
            ),
            patch.object(
                teachers,
                "estimate_incident_radiance",
                return_value=incident_rgb,
            ) as estimate_radiance,
        ):
            contribution = (
                teachers
                .estimate_incident_radiance_contribution(
                    context=context,
                    world_direction=self.direction,
                    teacher=self.physical_teacher,
                    runtime=runtime,
                )
            )

        expected = (
            0.2126 * 2.0
            + 0.7152 * 4.0
            + 0.0722 * 6.0
        )

        self.assertAlmostEqual(
            contribution,
            expected,
        )

        estimate_radiance.assert_called_once_with(
            runtime=runtime,
            scene=scene,
            ray=interaction.spawned_ray,
        )

    def test_incident_teacher_rejects_tangent_direction(
        self,
    ) -> None:
        interaction = FakeSurfaceInteraction()
        context = self.make_context(
            object(),
            interaction,
        )

        with patch.object(
            teachers,
            "estimate_incident_radiance",
        ) as estimate_radiance:
            contribution = (
                teachers
                .estimate_incident_radiance_contribution(
                    context=context,
                    world_direction=np.array(
                        [1.0, 0.0, 0.0]
                    ),
                    teacher=self.physical_teacher,
                    runtime=self.make_runtime(),
                )
            )

        self.assertEqual(contribution, 0.0)
        self.assertIsNone(
            interaction.spawned_ray
        )
        estimate_radiance.assert_not_called()

    def test_incident_teacher_requires_runtime(
        self,
    ) -> None:
        context = self.make_context(
            object(),
            FakeSurfaceInteraction(),
        )

        with self.assertRaisesRegex(
            ValueError,
            "requires an IncidentRadianceRuntime",
        ):
            teachers.estimate_incident_radiance_contribution(
                context=context,
                world_direction=self.direction,
                teacher=self.physical_teacher,
                runtime=None,
            )

    def test_incident_teacher_rejects_wrong_target(
        self,
    ) -> None:
        context = self.make_context(
            object(),
            FakeSurfaceInteraction(),
        )

        with self.assertRaisesRegex(
            ValueError,
            "incident-radiance estimator requires",
        ):
            teachers.estimate_incident_radiance_contribution(
                context=context,
                world_direction=self.direction,
                teacher=self.make_product_teacher(),
                runtime=self.make_runtime(),
            )

    def test_product_teacher_multiplies_li_and_bsdf_cosine(
        self,
    ) -> None:
        scene = object()
        interaction = FakeSurfaceInteraction()

        context = self.make_context(
            scene,
            interaction,
        )

        runtime = self.make_runtime()
        product_teacher = self.make_product_teacher()

        incident_radiance = np.array(
            [2.0, 4.0, 6.0]
        )
        bsdf_times_cosine = np.array(
            [0.1, 0.2, 0.3]
        )

        fake_mitsuba = SimpleNamespace(
            Vector3f=lambda *values: np.array(values)
        )

        with (
            patch.object(
                teachers,
                "_mi",
                fake_mitsuba,
            ),
            patch.object(
                teachers,
                "estimate_incident_radiance",
                return_value=incident_radiance,
            ) as estimate_radiance,
            patch.object(
                teachers,
                "evaluate_bsdf_times_cosine",
                return_value=bsdf_times_cosine,
            ) as evaluate_bsdf,
        ):
            contribution = (
                teachers
                .estimate_product_integrand_contribution(
                    context=context,
                    world_direction=self.direction,
                    teacher=product_teacher,
                    runtime=runtime,
                )
            )

        product = (
            incident_radiance
            * bsdf_times_cosine
        )

        expected = (
            0.2126 * product[0]
            + 0.7152 * product[1]
            + 0.0722 * product[2]
        )

        self.assertAlmostEqual(
            contribution,
            expected,
        )

        estimate_radiance.assert_called_once_with(
            runtime=runtime,
            scene=scene,
            ray=interaction.spawned_ray,
        )

        evaluate_bsdf.assert_called_once()

        arguments = evaluate_bsdf.call_args.kwargs

        self.assertIs(
            arguments["surface_interaction"],
            interaction,
        )

        np.testing.assert_allclose(
            arguments["world_direction"],
            self.direction,
        )

    def test_product_teacher_rejects_tangent_direction(
        self,
    ) -> None:
        interaction = FakeSurfaceInteraction()

        context = self.make_context(
            object(),
            interaction,
        )

        with (
            patch.object(
                teachers,
                "estimate_incident_radiance",
            ) as estimate_radiance,
            patch.object(
                teachers,
                "evaluate_bsdf_times_cosine",
            ) as evaluate_bsdf,
        ):
            contribution = (
                teachers
                .estimate_product_integrand_contribution(
                    context=context,
                    world_direction=np.array(
                        [1.0, 0.0, 0.0]
                    ),
                    teacher=self.make_product_teacher(),
                    runtime=self.make_runtime(),
                )
            )

        self.assertEqual(contribution, 0.0)
        self.assertIsNone(
            interaction.spawned_ray
        )
        estimate_radiance.assert_not_called()
        evaluate_bsdf.assert_not_called()

    def test_product_teacher_requires_runtime(
        self,
    ) -> None:
        context = self.make_context(
            object(),
            FakeSurfaceInteraction(),
        )

        with self.assertRaisesRegex(
            ValueError,
            "requires an IncidentRadianceRuntime",
        ):
            teachers.estimate_product_integrand_contribution(
                context=context,
                world_direction=self.direction,
                teacher=self.make_product_teacher(),
                runtime=None,
            )

    def test_product_teacher_rejects_wrong_target(
        self,
    ) -> None:
        context = self.make_context(
            object(),
            FakeSurfaceInteraction(),
        )

        with self.assertRaisesRegex(
            ValueError,
            "product-integrand estimator requires",
        ):
            teachers.estimate_product_integrand_contribution(
                context=context,
                world_direction=self.direction,
                teacher=self.physical_teacher,
                runtime=self.make_runtime(),
            )

    def test_visibility_teacher_does_not_need_runtime(
        self,
    ) -> None:
        runtime = teachers.create_teacher_runtime(
            teacher=self.teacher,
            seed=42,
        )

        self.assertIsNone(runtime)

    def test_incident_teacher_creates_runtime(
        self,
    ) -> None:
        expected_runtime = self.make_runtime()

        with patch.object(
            teachers,
            "create_incident_radiance_runtime",
            return_value=expected_runtime,
        ) as factory:
            runtime = teachers.create_teacher_runtime(
                teacher=self.physical_teacher,
                seed=42,
            )

        self.assertIs(
            runtime,
            expected_runtime,
        )

        factory.assert_called_once_with(
            radiance_samples=2,
            max_depth=6,
            rr_depth=5,
            seed=42,
        )

    def test_accepts_t1_and_t2_physical_settings(
        self,
    ) -> None:
        for target_type in (
            teachers.INCIDENT_RADIANCE_TARGET,
            teachers.PRODUCT_INTEGRAND_TARGET,
        ):
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

    def test_generic_validation_accepts_visibility_teacher(
        self,
    ) -> None:
        teachers.validate_teacher_settings(
            self.teacher
        )

    def test_generic_validation_accepts_physical_teacher(
        self,
    ) -> None:
        teachers.validate_teacher_settings(
            self.physical_teacher
        )

    def test_visibility_metadata_is_complete(
        self,
    ) -> None:
        metadata = teachers.build_teacher_metadata(
            self.teacher
        )

        self.assertEqual(
            metadata["target_type"],
            teachers.VISIBILITY_COSINE_TARGET,
        )
        self.assertEqual(
            metadata["samples_per_bin"],
            4,
        )
        self.assertEqual(
            metadata["emitter_weight"],
            8.0,
        )
        self.assertEqual(
            metadata["max_distance"],
            25.0,
        )

    def test_physical_metadata_is_complete(
        self,
    ) -> None:
        metadata = teachers.build_teacher_metadata(
            self.physical_teacher
        )

        self.assertEqual(
            metadata["target_type"],
            teachers.INCIDENT_RADIANCE_TARGET,
        )
        self.assertEqual(
            metadata["samples_per_bin"],
            4,
        )
        self.assertEqual(
            metadata["radiance_samples"],
            2,
        )
        self.assertEqual(
            metadata["max_depth"],
            6,
        )
        self.assertEqual(
            metadata["rr_depth"],
            5,
        )


if __name__ == "__main__":
    unittest.main()