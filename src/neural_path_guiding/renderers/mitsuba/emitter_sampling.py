"""Scalar Mitsuba emitter sampling utilities for direct lighting experiments.

The emitter sampling record and the emitter PDF must be kept together.
This module supports the scalar_rgb diagnostic pipeline.

Delta emitters are explicitly rejected for now because they require
a different MIS treatment from continuous solid-angle PDFs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mitsuba as mi
import numpy as np

from neural_path_guiding.renderers.mitsuba.adapters import (
    mitsuba_vector_to_numpy,
    normalize_numpy_vector,
)
from neural_path_guiding.renderers.mitsuba.bsdf import (
    evaluate_bsdf_pdf,
    evaluate_bsdf_times_cosine,
)
from neural_path_guiding.renderers.mitsuba.radiance import (
    mitsuba_spectrum_to_rgb,
)


@dataclass(frozen=True)
class EmitterDirectionSample:
    direction_world: np.ndarray
    emitter_pdf: float
    bsdf_pdf: float
    neural_pdf: float | None
    contribution_over_pdf: np.ndarray
    is_delta: bool


def _validate_pdf(value: float, name: str) -> float:
    pdf = float(value)

    if not np.isfinite(pdf) or pdf < 0.0:
        raise ValueError(
            f"{name} must be finite and non-negative."
        )

    return pdf


def _validate_rgb(value: Any) -> np.ndarray:
    rgb = np.asarray(value, dtype=np.float64)

    if rgb.shape != (3,):
        raise ValueError(
            f"Expected RGB shape (3,), got {rgb.shape}."
        )

    if not np.all(np.isfinite(rgb)):
        raise ValueError("RGB contribution must be finite.")

    return rgb


def evaluate_sampled_emitter_pdf(
    *,
    scene: Any,
    interaction: Any,
    direction_sample: Any,
) -> float:
    """Re-evaluate the emitter PDF for an existing Mitsuba record.

    The record must come from the same scene and reference interaction.
    """

    pdf = scene.pdf_emitter_direction(
        interaction,
        direction_sample,
        True,
    )

    return _validate_pdf(
        float(pdf),
        "emitter_pdf",
    )


def sample_emitter_direction(
    *,
    scene: Any,
    interaction: Any,
    rng: np.random.Generator,
    proposal: Any | None = None,
) -> EmitterDirectionSample:
    """Sample an emitter and evaluate competing directional PDFs.

    Mitsuba's emitter_weight already includes division by the
    emitter sampling PDF and the optional visibility test.

    The returned contribution_over_pdf therefore must NOT be
    divided by emitter_pdf again.
    """

    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a NumPy Generator.")

    u, v = rng.random(2)

    direction_sample, emitter_weight = (
        scene.sample_emitter_direction(
            interaction,
            mi.Point2f(float(u), float(v)),
            True,
        )
    )

    emitter_pdf = _validate_pdf(
        float(direction_sample.pdf),
        "sampled_emitter_pdf",
    )

    emitter_weight_rgb = _validate_rgb(
        mitsuba_spectrum_to_rgb(emitter_weight)
    )

    is_delta = bool(direction_sample.delta)

    if is_delta:
        raise NotImplementedError(
            "Delta emitters are not supported by this "
            "continuous-density diagnostic."
        )

    if emitter_pdf == 0.0:
        if np.any(emitter_weight_rgb != 0.0):
            raise ValueError(
                "Zero-PDF emitter sample has nonzero weight."
            )

        return EmitterDirectionSample(
            direction_world=np.zeros(3),
            emitter_pdf=0.0,
            bsdf_pdf=0.0,
            neural_pdf=None,
            contribution_over_pdf=np.zeros(3),
            is_delta=False,
        )

    direction_world = normalize_numpy_vector(
        mitsuba_vector_to_numpy(direction_sample.d)
    )

    bsdf_pdf = evaluate_bsdf_pdf(
        surface_interaction=interaction,
        world_direction=direction_world,
    )

    neural_pdf = None

    if proposal is not None:
        neural_pdf = _validate_pdf(
            proposal.pdf(direction_world),
            "neural_pdf",
        )

    bsdf_times_cosine = evaluate_bsdf_times_cosine(
        surface_interaction=interaction,
        world_direction=direction_world,
    )

    contribution = _validate_rgb(
        emitter_weight_rgb * bsdf_times_cosine
    )

    return EmitterDirectionSample(
        direction_world=direction_world,
        emitter_pdf=emitter_pdf,
        bsdf_pdf=bsdf_pdf,
        neural_pdf=neural_pdf,
        contribution_over_pdf=contribution,
        is_delta=False,
    )