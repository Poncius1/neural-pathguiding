"""Single-bounce neural and BSDF MIS integrator for Mitsuba."""

from __future__ import annotations

from typing import Any

import mitsuba as mi
import numpy as np

from neural_path_guiding.core.mis import (
    combine_two_technique_samples,
)
from neural_path_guiding.renderers.mitsuba.adapters import (
    mitsuba_vector_to_numpy,
    normalize_numpy_vector,
)
from neural_path_guiding.renderers.mitsuba.bsdf import (
    evaluate_bsdf_pdf,
    evaluate_bsdf_times_cosine,
)
from neural_path_guiding.renderers.mitsuba.guiding_adapter import (
    MitsubaGuidingAdapter,
)
from neural_path_guiding.renderers.mitsuba.radiance import (
    mitsuba_spectrum_to_rgb,
)
from neural_path_guiding.training.inference import (
    GuidingInferenceRuntime,
)


_mi: Any = mi


class NeuralMISDirectIntegrator(mi.SamplingIntegrator):
    """Estimate direct illumination at the first visible surface."""

    def __init__(self, props: Any) -> None:
        super().__init__(props)

        checkpoint_path = str(props["checkpoint"])
        uniform_mix = float(props["uniform_mix"])

        self.runtime = GuidingInferenceRuntime.from_checkpoint(
            checkpoint_path,
            device="cpu",
            uniform_mix=uniform_mix,
        )
        self.guiding_adapter = MitsubaGuidingAdapter(self.runtime)

    def sample(
        self,
        scene: Any,
        sampler: Any,
        ray: Any,
        medium: Any = None,
        active: Any = True,
    ) -> tuple[Any, Any, list[Any]]:
        """Return direct illumination for one camera ray."""

        if medium is not None:
            raise NotImplementedError(
                "This integrator does not support participating media."
            )

        surface_interaction = scene.ray_intersect(ray, active)

        if not surface_interaction.is_valid():
            environment = scene.environment()

            if environment is None:
                return _mi.Color3f(0.0), _mi.Bool(False), []

            return (
                environment.eval(surface_interaction),
                _mi.Bool(False),
                [],
            )

        emitter = surface_interaction.emitter(scene)

        if emitter is not None:
            return (
                emitter.eval(surface_interaction),
                _mi.Bool(True),
                [],
            )

        proposal = self._prepare_guiding_proposal(
            surface_interaction=surface_interaction,
            ray=ray,
        )

        rng = self._make_numpy_rng(sampler)
        neural_sample = proposal.sample(rng)

        bsdf = surface_interaction.bsdf()
        bsdf_context = _mi.BSDFContext()

        bsdf_sample, _ = bsdf.sample(
            bsdf_context,
            surface_interaction,
            sampler.next_1d(),
            sampler.next_2d(),
            active,
        )

        bsdf_direction = normalize_numpy_vector(
            mitsuba_vector_to_numpy(
                surface_interaction.to_world(bsdf_sample.wo)
            )
        )

        bsdf_pdf = evaluate_bsdf_pdf(
            surface_interaction=surface_interaction,
            world_direction=bsdf_direction,
        )

        if bsdf_pdf <= 0.0:
            raise RuntimeError(
                "The smoke integrator requires a non-delta BSDF sample."
            )

        neural_contribution = self._evaluate_direct_integrand(
            scene=scene,
            surface_interaction=surface_interaction,
            direction_world=neural_sample.direction_world,
        )
        bsdf_contribution = self._evaluate_direct_integrand(
            scene=scene,
            surface_interaction=surface_interaction,
            direction_world=bsdf_direction,
        )

        bsdf_pdf_at_neural_sample = evaluate_bsdf_pdf(
            surface_interaction=surface_interaction,
            world_direction=neural_sample.direction_world,
        )
        neural_pdf_at_bsdf_sample = proposal.pdf(bsdf_direction)

        estimate = combine_two_technique_samples(
            contribution_a=neural_contribution,
            pdf_a_at_a=neural_sample.pdf,
            pdf_b_at_a=bsdf_pdf_at_neural_sample,
            contribution_b=bsdf_contribution,
            pdf_a_at_b=neural_pdf_at_bsdf_sample,
            pdf_b_at_b=bsdf_pdf,
        )

        return _mi.Color3f(*estimate), _mi.Bool(True), []

    def _prepare_guiding_proposal(
        self,
        *,
        surface_interaction: Any,
        ray: Any,
    ) -> Any:
        outgoing_direction = normalize_numpy_vector(
            -mitsuba_vector_to_numpy(ray.d)
        )

        return self.guiding_adapter.prepare_proposal(
            surface_interaction=surface_interaction,
            outgoing_direction=outgoing_direction,
            bounce_depth=0,
        )

    def _make_numpy_rng(
        self,
        sampler: Any,
    ) -> np.random.Generator:
        seed = int(float(sampler.next_1d()) * (2**32 - 1))
        return np.random.default_rng(seed)

    def _evaluate_direct_integrand(
        self,
        *,
        scene: Any,
        surface_interaction: Any,
        direction_world: np.ndarray,
    ) -> np.ndarray:
        incident_radiance = self._trace_emission(
            scene=scene,
            surface_interaction=surface_interaction,
            direction_world=direction_world,
        )

        bsdf_value = evaluate_bsdf_times_cosine(
            surface_interaction=surface_interaction,
            world_direction=direction_world,
        )

        return incident_radiance * bsdf_value

    def _trace_emission(
        self,
        *,
        scene: Any,
        surface_interaction: Any,
        direction_world: np.ndarray,
    ) -> np.ndarray:
        ray = surface_interaction.spawn_ray(
            _mi.Vector3f(*direction_world)
        )
        hit = scene.ray_intersect(ray)

        if not hit.is_valid():
            environment = scene.environment()

            if environment is None:
                return np.zeros(3, dtype=np.float64)

            return mitsuba_spectrum_to_rgb(
                environment.eval(hit)
            )

        emitter = hit.emitter(scene)

        if emitter is None:
            return np.zeros(3, dtype=np.float64)

        return mitsuba_spectrum_to_rgb(emitter.eval(hit))


mi.register_integrator(
    "neural_mis_direct",
    lambda props: NeuralMISDirectIntegrator(props),
)