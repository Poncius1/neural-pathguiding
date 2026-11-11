"""Scalar RGB path tracer with first-hit directional guiding and NEE.

Modes:
    bsdf    : conventional BSDF sampling at every vertex
    uniform : first eligible vertex uses a BSDF/uniform mixture
    neural  : first eligible vertex uses a BSDF/T2 mixture

This integrator traces the entire path itself. In particular, emitter-hit
MIS and next-event estimation share the same directional sampling PDF.

Scope: scalar_rgb, surface transport, no participating media, no null BSDF
transitions, no polarization, and guiding only at the first interaction on
supported diffuse surfaces. Not a production/GPU integrator.
"""

from __future__ import annotations

from typing import Any

import mitsuba as mi
import numpy as np

from neural_path_guiding.core.bins import HemisphereBins
from neural_path_guiding.core.directional_distribution import (
    DiscreteDirectionalDistribution,
)
from neural_path_guiding.core.mis import power_heuristic
from neural_path_guiding.renderers.mitsuba.adapters import (
    mitsuba_vector_to_numpy,
    normalize_numpy_vector,
)
from neural_path_guiding.renderers.mitsuba.bsdf import (
    evaluate_bsdf_pdf,
    evaluate_bsdf_times_cosine,
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

_mi: Any = mi


def _rgb(value: Any) -> np.ndarray:
    result = np.asarray(mitsuba_spectrum_to_rgb(value), dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"Invalid RGB spectrum: {result}")
    return result


class NPGTransportIntegrator(mi.SamplingIntegrator):
    """Explicit scalar path tracer with consistent NEE / directional MIS."""

    def __init__(self, props: Any) -> None:
        super().__init__(props)
        self.mode = str(props["mode"])
        self.max_depth = int(props["max_depth"])
        self.rr_depth = int(props["rr_depth"])
        self.guiding_probability = float(props["guiding_probability"])

        if self.mode not in ("bsdf", "uniform", "neural"):
            raise ValueError(f"Unknown transport mode: {self.mode}")
        if self.max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        if self.rr_depth < 1:
            raise ValueError("rr_depth must be >= 1")
        if not 0.0 < self.guiding_probability < 1.0:
            raise ValueError("guiding_probability must be in (0, 1)")

        self.runtime = None
        self.uniform_distribution = None
        if self.mode == "neural":
            self.runtime = NeuralMitsubaRuntimeV2.from_checkpoint(
                str(props["checkpoint"]),
                uniform_mix=float(props["uniform_mix"]),
                device="cpu",
            )
        elif self.mode == "uniform":
            bins = HemisphereBins(4, 8)
            self.uniform_distribution = DiscreteDirectionalDistribution(
                bins=bins,
                probabilities=np.full(bins.n_bins, 1.0 / bins.n_bins),
            )

    def _proposal(self, si: Any, ray: Any) -> Any:
        if self.mode == "uniform":
            assert self.uniform_distribution is not None
            return NeuralMitsubaRuntimeV2.proposal_from_distribution(
                si, self.uniform_distribution
            )
        if self.mode == "neural":
            assert self.runtime is not None
            return self.runtime.proposal_at(
                si,
                outgoing_world=normalize_numpy_vector(
                    -mitsuba_vector_to_numpy(ray.d)
                ),
                bounce_depth=0,
            )
        return None

    def _direction_pdf(self, si: Any, direction: np.ndarray, proposal: Any) -> float:
        bsdf_pdf = float(evaluate_bsdf_pdf(
            surface_interaction=si, world_direction=direction
        ))
        if proposal is None:
            return bsdf_pdf
        return (
            (1.0 - self.guiding_probability) * bsdf_pdf
            + self.guiding_probability * float(proposal.pdf(direction))
        )

    def _direct_emitter_sample(
        self, scene: Any, si: Any, sampler: Any, proposal: Any
    ) -> np.ndarray:
        # Mitsuba returns emitter radiance / emitter PDF, including visibility.
        ds, emitter_weight = scene.sample_emitter_direction(
            si, sampler.next_2d(), True
        )
        if float(ds.pdf) <= 0.0:
            return np.zeros(3, dtype=np.float64)
        direction = normalize_numpy_vector(mitsuba_vector_to_numpy(ds.d))
        bsdf_cos = evaluate_bsdf_times_cosine(
            surface_interaction=si, world_direction=direction
        )
        if not np.any(bsdf_cos > 0.0):
            return np.zeros(3, dtype=np.float64)
        other_pdf = 0.0 if bool(ds.delta) else self._direction_pdf(
            si, direction, proposal
        )
        weight = 1.0 if bool(ds.delta) else power_heuristic(
            float(ds.pdf), other_pdf
        )
        return weight * bsdf_cos * _rgb(emitter_weight)

    def _sample_continuation(
        self, si: Any, sampler: Any, proposal: Any
    ) -> tuple[Any, np.ndarray, float, bool, float]:
        """Return ray, throughput multiplier, proposal PDF, delta, eta."""
        bsdf = si.bsdf()
        ctx = _mi.BSDFContext()

        use_guide = (
            proposal is not None
            and float(sampler.next_1d()) < self.guiding_probability
        )
        if use_guide:
            assert proposal is not None
            seed = min(int(float(sampler.next_1d()) * (2**32)), 2**32 - 1)
            directional = proposal.sample(np.random.default_rng(seed))
            direction = np.asarray(directional.direction_world, dtype=np.float64)
            pdf = self._direction_pdf(si, direction, proposal)
            if not np.isfinite(pdf) or pdf <= 0.0:
                raise ValueError("Nonpositive guiding mixture PDF")
            f = evaluate_bsdf_times_cosine(
                surface_interaction=si, world_direction=direction
            )
            return (
                si.spawn_ray(_mi.Vector3f(*direction)),
                np.asarray(f, dtype=np.float64) / pdf,
                pdf,
                False,
                1.0,
            )

        bsdf_sample, bsdf_weight = bsdf.sample(
            ctx, si, sampler.next_1d(), sampler.next_2d(), True
        )
        pdf = float(bsdf_sample.pdf)
        if not np.isfinite(pdf) or pdf < 0.0:
            raise ValueError("Invalid BSDF sampling PDF")
        if pdf == 0.0:
            # A failed BSDF sample contributes nothing. Do not normalize a
            # possibly zero outgoing direction or attempt a new ray cast.
            return None, np.zeros(3, dtype=np.float64), 0.0, False, 1.0

        direction = normalize_numpy_vector(mitsuba_vector_to_numpy(
            si.to_world(bsdf_sample.wo)
        ))
        delta = bool(int(bsdf_sample.sampled_type) & int(_mi.BSDFFlags.Delta))

        if proposal is None or delta:
            # Mitsuba's bsdf_weight handles delta/mixed BSDFs correctly.
            return (
                si.spawn_ray(_mi.Vector3f(*direction)),
                _rgb(bsdf_weight),
                pdf,
                delta,
                float(bsdf_sample.eta),
            )

        # For an eligible continuous diffuse BSDF, use the entire mixture
        # density, including when the BSDF branch generated the direction.
        mixture_pdf = self._direction_pdf(si, direction, proposal)
        if not np.isfinite(mixture_pdf) or mixture_pdf <= 0.0:
            raise ValueError("Nonpositive BSDF mixture PDF")
        f = evaluate_bsdf_times_cosine(
            surface_interaction=si, world_direction=direction
        )
        return (
            si.spawn_ray(_mi.Vector3f(*direction)),
            np.asarray(f, dtype=np.float64) / mixture_pdf,
            mixture_pdf,
            False,
            1.0,
        )

    def sample(
        self,
        scene: Any,
        sampler: Any,
        ray: Any,
        medium: Any = None,
        active: Any = True,
    ) -> tuple[Any, Any, list[Any]]:
        if medium is not None:
            raise NotImplementedError("Participating media are unsupported")
        if not bool(active):
            return _mi.Color3f(0.0), _mi.Bool(False), []

        result = np.zeros(3, dtype=np.float64)
        throughput = np.ones(3, dtype=np.float64)
        eta = 1.0
        previous_si = None
        previous_pdf = 1.0
        previous_delta = True
        valid = False

        for depth in range(self.max_depth):
            si = scene.ray_intersect(ray)
            emitter = si.emitter(scene)

            # Emission on first hit, after a delta event, or after a smooth
            # sample with the correct light-sampling MIS correction.
            if emitter is not None:
                em_weight = 1.0
                if previous_si is not None and not previous_delta:
                    ds = _mi.DirectionSample3f(
                        scene, si=si, ref=previous_si
                    )
                    light_pdf = float(scene.pdf_emitter_direction(
                        previous_si, ds, True
                    ))
                    em_weight = power_heuristic(previous_pdf, light_pdf)
                result += throughput * _rgb(emitter.eval(si)) * em_weight
                valid = True

            if not bool(si.is_valid()):
                break

            # The path counts surface interactions; no new segment can be
            # sampled after the final allowed interaction.
            if depth + 1 >= self.max_depth:
                break

            bsdf = si.bsdf()
            can_nee = bool(_mi.has_flag(bsdf.flags(), _mi.BSDFFlags.Smooth))
            eligible = (
                depth == 0
                and self.mode != "bsdf"
                and bool(si.wi.z > 0.0)
                and is_supported_diffuse_bsdf(bsdf)
                and emitter is None
            )
            proposal = self._proposal(si, ray) if eligible else None

            if can_nee:
                result += throughput * self._direct_emitter_sample(
                    scene, si, sampler, proposal
                )
                valid = True

            ray, multiplier, previous_pdf, previous_delta, eta_change = (
                self._sample_continuation(si, sampler, proposal)
            )
            if not np.all(np.isfinite(multiplier)):
                raise ValueError("Non-finite BSDF throughput")
            throughput *= multiplier
            eta *= eta_change
            if np.max(np.abs(throughput)) <= 0.0:
                break
            valid = True
            previous_si = si

            # Match the scalar form of Mitsuba's usual RR termination rule.
            if depth + 1 >= self.rr_depth:
                probability = min(
                    float(np.max(np.abs(throughput)) * eta * eta), 0.95
                )
                if probability <= 0.0 or float(sampler.next_1d()) >= probability:
                    break
                throughput /= probability

        if not np.all(np.isfinite(result)):
            raise ValueError("Non-finite path radiance")
        return _mi.Color3f(*result), _mi.Bool(valid), []


_mi.register_integrator(
    "npg_transport",
    lambda props: NPGTransportIntegrator(props),
)
