
"""Diagnostic first-hit multibounce integrator for Mitsuba 3.8.

Modes:
    bsdf_only        - One BSDF continuation
    uniform_bsdf_mis - One uniform + one BSDF continuation
    neural_bsdf_mis  - One T2 neural + one BSDF continuation

Only camera-first-hit, front-facing, pure diffuse surfaces are guided.
Other surfaces use the conventional Mitsuba path integrator.

The first direction is chosen here. Subsequent path interactions
are evaluated by Mitsuba's path integrator.

Research prototype: scalar_rgb, not a production renderer.
"""

from __future__ import annotations

from typing import Any

import mitsuba as mi
import numpy as np

from neural_path_guiding.core.bins import HemisphereBins
from neural_path_guiding.core.directional_distribution import (
    DiscreteDirectionalDistribution,
)
from neural_path_guiding.core.frames import (
    build_frame_from_normal,
)
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
    GuidingProposal,
)
from neural_path_guiding.renderers.mitsuba.material_features import (
    is_supported_diffuse_bsdf,
)
from neural_path_guiding.renderers.mitsuba.neural_inference_v2 import (
    NeuralMitsubaRuntimeV2,
)
from neural_path_guiding.renderers.mitsuba.radiance import (
    mitsuba_spectrum_to_rgb,
)

# Workaround for dynamically generated Mitsuba bindings in Pylance.
_mi: Any = mi


VALID_MODES = frozenset({
    "bsdf_only",
    "uniform_bsdf_mis",
    "neural_bsdf_mis",
})


class NeuralT2MultibounceIntegrator(mi.SamplingIntegrator):
    """Diagnostic first-hit directional importance-sampling integrator."""

    def __init__(self, props: Any) -> None:
        super().__init__(props)

        self.mode = str(props["mode"])
        self.max_depth = int(props["max_depth"])
        self.rr_depth = int(props["rr_depth"])

        if self.mode not in VALID_MODES:
            raise ValueError(
                f"Unknown integrator mode: {self.mode}"
            )

        if self.max_depth < 2:
            raise ValueError(
                "max_depth must be >= 2"
            )

        if not (0 <= self.rr_depth <= self.max_depth):
            raise ValueError(
                "Invalid rr_depth"
            )

        self.runtime: NeuralMitsubaRuntimeV2 | None = None
        self.uniform_distribution = None

        # Load neural inference only when explicitly requested.
        if self.mode == "neural_bsdf_mis":
            self.runtime = NeuralMitsubaRuntimeV2.from_checkpoint(
                str(props["checkpoint"]),
                uniform_mix=float(props["uniform_mix"]),
                device="cpu",
            )

        # Build a constant uniform directional distribution.
        if self.mode == "uniform_bsdf_mis":
            bins = HemisphereBins(4, 8)

            self.uniform_distribution = (
                DiscreteDirectionalDistribution(
                    bins=bins,
                    probabilities=np.full(
                        bins.n_bins,
                        1.0 / bins.n_bins,
                        dtype=np.float64,
                    ),
                    uniform_mix=0.05,
                )
            )

        # Conventional fallback for unsupported first-hit surfaces.
        self.fallback_path = _mi.load_dict({
            "type": "path",
            "max_depth": self.max_depth,
            "rr_depth": self.rr_depth,
        })

        # The first surface interaction has already been evaluated.
        continuation_depth = self.max_depth - 1

        self.continuation_path = _mi.load_dict({
            "type": "path",
            "max_depth": continuation_depth,
            "rr_depth": min(
                self.rr_depth,
                continuation_depth,
            ),
        })

    
    def _prepare_proposal(
        self,
        si: Any,
        ray: Any,
    ) -> GuidingProposal:
        """Build a directional proposal in the canonical T2 frame."""

        if self.mode == "uniform_bsdf_mis":
            if self.uniform_distribution is None:
                raise RuntimeError(
                    "Uniform distribution is not initialized"
                )

            return NeuralMitsubaRuntimeV2.proposal_from_distribution(
                si,
                self.uniform_distribution,
            )

        if self.mode == "neural_bsdf_mis":
            if self.runtime is None:
                raise RuntimeError(
                    "Neural runtime is not initialized"
                )

            outgoing = normalize_numpy_vector(
                -mitsuba_vector_to_numpy(ray.d)
            )

            return self.runtime.proposal_at(
                si,
                outgoing_world=outgoing,
                bounce_depth=0,
            )

        raise RuntimeError(
            f"Mode '{self.mode}' does not use a guiding proposal"
        )

    def _sample_bsdf(
        self,
        si: Any,
        sampler: Any,
        active: Any,
    ) -> tuple[np.ndarray, float]:
        """Sample one continuous BSDF direction."""

        bsdf = si.bsdf()

        bsdf_sample, _ = bsdf.sample(
            _mi.BSDFContext(),
            si,
            sampler.next_1d(),
            sampler.next_2d(),
            active,
        )

        # Mitsuba 3.8: BSDFSample3f has sampled_type,
        # not a .delta attribute.
        sampled_type = int(
            bsdf_sample.sampled_type
        )

        delta_flag = int(
            _mi.BSDFFlags.Delta
        )

        if sampled_type & delta_flag:
            raise ValueError(
                "Unexpected delta BSDF event "
                "at a pure diffuse surface"
            )

        direction = normalize_numpy_vector(
            mitsuba_vector_to_numpy(
                si.to_world(bsdf_sample.wo)
            )
        )

        pdf = evaluate_bsdf_pdf(
            surface_interaction=si,
            world_direction=direction,
        )

        if not np.isfinite(pdf) or pdf <= 0.0:
            raise ValueError(
                "Invalid BSDF sampling PDF"
            )

        return direction, float(pdf)

    def _integrand(
        self,
        scene: Any,
        sampler: Any,
        si: Any,
        direction: np.ndarray,
    ) -> np.ndarray:
        """Evaluate incident multibounce radiance times BSDF*cosine."""

        bsdf_cos = evaluate_bsdf_times_cosine(
            surface_interaction=si,
            world_direction=direction,
        )

        if not np.any(bsdf_cos > 0.0):
            return np.zeros(
                3,
                dtype=np.float64,
            )

        continuation_ray = si.spawn_ray(
            _mi.Vector3f(*direction)
        )

        radiance, _, _ = self.continuation_path.sample(
            scene,
            sampler,
            continuation_ray,
        )

        result = (
            mitsuba_spectrum_to_rgb(radiance)
            * bsdf_cos
        )

        if not np.all(np.isfinite(result)):
            raise ValueError(
                "Non-finite continuation contribution"
            )

        return result

    def sample(
        self,
        scene: Any,
        sampler: Any,
        ray: Any,
        medium: Any = None,
        active: Any = True,
    ) -> tuple[Any, Any, list[Any]]:
        """Estimate radiance for one camera ray."""

        if medium is not None:
            raise NotImplementedError(
                "Participating media are not supported"
            )

        si = scene.ray_intersect(
            ray,
            active,
        )

        # Camera ray missed the scene.
        if not bool(si.is_valid()):
            environment = scene.environment()

            if environment is None:
                return (
                    _mi.Color3f(0.0),
                    _mi.Bool(False),
                    [],
                )

            return (
                environment.eval(si),
                _mi.Bool(False),
                [],
            )

        bsdf = si.bsdf()

        # Use conventional Path Tracing for:
        # - camera-visible emitters
        # - unsupported BSDFs
        # - back-facing interactions
        if (
            si.emitter(scene) is not None
            or not is_supported_diffuse_bsdf(bsdf)
            or float(si.wi.z) <= 0.0
        ):
            return self.fallback_path.sample(
                scene,
                sampler,
                ray,
                medium,
                active,
            )

        # First BSDF proposal.
        bsdf_direction, p_bsdf = self._sample_bsdf(
            si,
            sampler,
            active,
        )

        if self.mode == "bsdf_only":
            # One directional proposal.
            estimate = (
                self._integrand(
                    scene,
                    sampler,
                    si,
                    bsdf_direction,
                )
                / p_bsdf
            )

        else:
            # Prepare uniform or learned directional proposal.
            proposal = self._prepare_proposal(
                si,
                ray,
            )

            # Generate deterministic NumPy RNG state from
            # Mitsuba's sampler for this camera-path sample.
            seed = min(
                int(
                    float(sampler.next_1d())
                    * (2**32)
                ),
                2**32 - 1,
            )

            rng = np.random.default_rng(
                seed
            )

            neural_sample = proposal.sample(
                rng
            )

            neural_direction = (
                neural_sample.direction_world
            )

            p_neural = float(
                neural_sample.pdf
            )

            if not np.isfinite(p_neural) or p_neural <= 0.0:
                raise ValueError(
                    "Invalid neural or uniform sampling PDF"
                )

            # Cross-evaluate BSDF PDF at neural direction.
            p_bsdf_at_neural = evaluate_bsdf_pdf(
                surface_interaction=si,
                world_direction=neural_direction,
            )

            # Cross-evaluate neural PDF at BSDF direction.
            p_neural_at_bsdf = float(
                proposal.pdf(bsdf_direction)
            )

            # Both strategies evaluate the exact same integrand.
            f_neural = self._integrand(
                scene,
                sampler,
                si,
                neural_direction,
            )

            f_bsdf = self._integrand(
                scene,
                sampler,
                si,
                bsdf_direction,
            )

            # Power-heuristic MIS; one proposal per technique.
            estimate = combine_two_technique_samples(
                contribution_a=f_neural,
                pdf_a_at_a=p_neural,
                pdf_b_at_a=p_bsdf_at_neural,
                contribution_b=f_bsdf,
                pdf_a_at_b=p_neural_at_bsdf,
                pdf_b_at_b=p_bsdf,
            )

        if not np.all(np.isfinite(estimate)):
            raise ValueError(
                "Non-finite rendered contribution"
            )

        return (
            _mi.Color3f(*estimate),
            _mi.Bool(True),
            [],
        )


# Register the integrator for mi.load_dict().
_mi.register_integrator(
    "neural_t2_multibounce",
    lambda props: NeuralT2MultibounceIntegrator(props),
)
