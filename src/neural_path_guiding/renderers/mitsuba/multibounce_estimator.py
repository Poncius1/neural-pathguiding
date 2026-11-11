
"""Fixed-point neural importance sampling with multibounce radiance.

The first direction is sampled by neural or BSDF sampling.
Mitsuba's path integrator estimates incident radiance after that
direction, using independent internal random samples.

Only camera-first-hit, pure Lambertian surfaces are intended
for the current v2 checkpoint.

This module is not a full neural path tracer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mitsuba as mi
import numpy as np

from neural_path_guiding.core.mis import (
    combine_two_technique_samples,
)
from neural_path_guiding.renderers.mitsuba.adapters import (
    normalize_numpy_vector,
)
from neural_path_guiding.renderers.mitsuba.bsdf import (
    evaluate_bsdf_pdf,
    evaluate_bsdf_times_cosine,
)
from neural_path_guiding.renderers.mitsuba.radiance import (
    IncidentRadianceRuntime,
    create_incident_radiance_runtime,
    estimate_incident_radiance,
)


@dataclass
class MultibounceEvaluator:
    """Reusable Mitsuba continuation runtime for one experiment stream."""

    scene: Any
    runtime: IncidentRadianceRuntime

    @classmethod
    def create(
        cls,
        scene: Any,
        *,
        radiance_samples: int,
        max_depth: int,
        rr_depth: int,
        seed: int,
    ) -> "MultibounceEvaluator":
        runtime = create_incident_radiance_runtime(
            radiance_samples=radiance_samples,
            max_depth=max_depth,
            rr_depth=rr_depth,
            seed=seed,
        )
        return cls(scene=scene, runtime=runtime)

    def evaluate_integrand(
        self,
        interaction: Any,
        direction_world: np.ndarray,
    ) -> np.ndarray:
        """Return an unbiased stochastic estimate of Li * BSDF*cos.

        The inner path integrator does not apply the primary
        neural/BSDF sampling PDF. The outer estimator does that.
        """

        direction = normalize_numpy_vector(direction_world)

        bsdf_cos = evaluate_bsdf_times_cosine(
            surface_interaction=interaction,
            world_direction=direction,
        )

        if not np.any(bsdf_cos > 0.0):
            return np.zeros(3, dtype=np.float64)

        ray = interaction.spawn_ray(mi.Vector3f(*direction))

        incident = estimate_incident_radiance(
            runtime=self.runtime,
            scene=self.scene,
            ray=ray,
        )

        result = incident * bsdf_cos

        if result.shape != (3,) or not np.all(np.isfinite(result)):
            raise ValueError("Non-finite multibounce contribution.")

        if np.any(result < 0.0):
            raise ValueError("Negative multibounce contribution.")

        return result


def sample_bsdf(
    interaction: Any,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Sample a continuous BSDF and evaluate its solid-angle PDF."""

    bsdf_sample, _ = interaction.bsdf().sample(
        mi.BSDFContext(),
        interaction,
        float(rng.random()),
        mi.Point2f(
            float(rng.random()),
            float(rng.random()),
        ),
        True,
    )

    from neural_path_guiding.renderers.mitsuba.adapters import (
        mitsuba_vector_to_numpy,
    )

    direction = normalize_numpy_vector(
        mitsuba_vector_to_numpy(
            interaction.to_world(bsdf_sample.wo)
        )
    )

    pdf = evaluate_bsdf_pdf(
        surface_interaction=interaction,
        world_direction=direction,
    )

    if not np.isfinite(pdf) or pdf <= 0.0:
        raise ValueError(
            "Expected a continuous BSDF sample with positive PDF."
        )

    return direction, float(pdf)


def estimate_neural(
    interaction: Any,
    proposal: Any,
    evaluator: MultibounceEvaluator,
    rng: np.random.Generator,
    sample_count: int,
) -> np.ndarray:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")

    total = np.zeros(3, dtype=np.float64)

    for _ in range(sample_count):
        sample = proposal.sample(rng)
        total += (
            evaluator.evaluate_integrand(
                interaction,
                sample.direction_world,
            )
            / sample.pdf
        )

    return total / sample_count


def estimate_bsdf(
    interaction: Any,
    evaluator: MultibounceEvaluator,
    rng: np.random.Generator,
    sample_count: int,
) -> np.ndarray:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")

    total = np.zeros(3, dtype=np.float64)

    for _ in range(sample_count):
        direction, pdf = sample_bsdf(interaction, rng)

        total += (
            evaluator.evaluate_integrand(interaction, direction)
            / pdf
        )

    return total / sample_count


def estimate_neural_bsdf_mis(
    interaction: Any,
    proposal: Any,
    evaluator: MultibounceEvaluator,
    rng: np.random.Generator,
) -> np.ndarray:
    """One neural + one BSDF proposal, with power heuristic.

    Both directions estimate the same multibounce integrand.
    All PDFs are solid-angle densities at the first hit.
    """

    neural_sample = proposal.sample(rng)
    bsdf_direction, bsdf_pdf = sample_bsdf(interaction, rng)

    neural_integrand = evaluator.evaluate_integrand(
        interaction,
        neural_sample.direction_world,
    )

    bsdf_integrand = evaluator.evaluate_integrand(
        interaction,
        bsdf_direction,
    )

    bsdf_at_neural = evaluate_bsdf_pdf(
        surface_interaction=interaction,
        world_direction=neural_sample.direction_world,
    )

    neural_at_bsdf = proposal.pdf(bsdf_direction)

    return combine_two_technique_samples(
        contribution_a=neural_integrand,
        pdf_a_at_a=neural_sample.pdf,
        pdf_b_at_a=bsdf_at_neural,
        contribution_b=bsdf_integrand,
        pdf_a_at_b=neural_at_bsdf,
        pdf_b_at_b=bsdf_pdf,
    )
