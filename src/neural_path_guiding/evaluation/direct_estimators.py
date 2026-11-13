"""Scalar Mitsuba direct-estimator helpers extracted from the legacy CLI.

These helpers preserve the original experimental estimators and their APIs.
Not the canonical image Path Tracer: see transport_integrator.py.
"""
from __future__ import annotations

from typing import Any
from pathlib import Path
import mitsuba as mi
import numpy as np

from neural_path_guiding.core.frames import local_to_world
from neural_path_guiding.core.mis import power_heuristic
from neural_path_guiding.renderers.mitsuba.adapters import (
    mitsuba_vector_to_numpy, normalize_numpy_vector,
)
from neural_path_guiding.renderers.mitsuba.bsdf import (
    evaluate_bsdf_pdf, evaluate_bsdf_times_cosine,
)
from neural_path_guiding.renderers.mitsuba.radiance import mitsuba_spectrum_to_rgb
from neural_path_guiding.evaluation.transport_metrics import file_sha256 as sha256_file

RgbEstimate = np.ndarray
_LUMINANCE_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)

def sample_fixed_surface(
    scene: Any,
    pixel_x: int,
    pixel_y: int,
) -> tuple[Any, Any, tuple[int, int]]:
    """Get the first surface hit for the center of one camera pixel."""

    sensors = scene.sensors()
    if not sensors:
        raise ValueError("The scene does not contain a sensor.")

    sensor = sensors[0]
    film_size = sensor.film().size()
    width = int(film_size[0])
    height = int(film_size[1])

    if not (0 <= pixel_x < width and 0 <= pixel_y < height):
        raise ValueError(
            f"Pixel ({pixel_x}, {pixel_y}) is outside the film "
            f"dimensions ({width}, {height})."
        )

    image_sample = mi.Point2f(
        (pixel_x + 0.5) / width,
        (pixel_y + 0.5) / height,
    )
    ray, _ = sensor.sample_ray(
        0.0,
        0.5,
        image_sample,
        mi.Point2f(0.5, 0.5),
    )
    interaction = scene.ray_intersect(ray)

    if not interaction.is_valid():
        raise ValueError(
            f"Camera ray at pixel ({pixel_x}, {pixel_y}) missed the scene."
        )

    return ray, interaction, (width, height)

def sample_uniform_hemisphere(
    frame: Any,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a direction uniformly in solid angle on the local hemisphere."""

    u_mu, u_phi = rng.random(2)
    mu = float(u_mu)
    radius = np.sqrt(max(0.0, 1.0 - mu * mu))
    phi = 2.0 * np.pi * float(u_phi)

    direction_local = np.array(
        [
            radius * np.cos(phi),
            radius * np.sin(phi),
            mu,
        ],
        dtype=np.float64,
    )
    direction_world = local_to_world(frame, direction_local)
    return normalize_numpy_vector(direction_world)

def sample_bsdf_direction(
    interaction: Any,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Sample one BSDF direction and evaluate its solid-angle PDF."""

    bsdf = interaction.bsdf()
    if bsdf is None:
        raise ValueError("Surface interaction has no BSDF.")

    bsdf_sample, _ = bsdf.sample(
        mi.BSDFContext(),
        interaction,
        float(rng.random()),
        mi.Point2f(
            float(rng.random()),
            float(rng.random()),
        ),
        True,
    )
    direction_world = normalize_numpy_vector(
        mitsuba_vector_to_numpy(
            interaction.to_world(bsdf_sample.wo)
        )
    )
    pdf = evaluate_bsdf_pdf(
        surface_interaction=interaction,
        world_direction=direction_world,
    )

    if not np.isfinite(pdf) or pdf <= 0.0:
        raise ValueError(
            "The BSDF returned a sample with a non-positive or "
            "non-finite PDF. This experiment expects a continuous, "
            "non-delta BSDF."
        )

    return direction_world, float(pdf)

def evaluate_direct_integrand(
    scene: Any,
    interaction: Any,
    direction_world: np.ndarray,
) -> RgbEstimate:
    """Evaluate incident emission times the BSDF*cosine term."""

    direction = normalize_numpy_vector(direction_world)
    ray = interaction.spawn_ray(mi.Vector3f(*direction))
    hit = scene.ray_intersect(ray)

    if not hit.is_valid():
        environment = scene.environment()
        if environment is None:
            return np.zeros(3, dtype=np.float64)

        incident_radiance = mitsuba_spectrum_to_rgb(
            environment.eval(hit)
        )
    else:
        emitter = hit.emitter(scene)
        if emitter is None:
            return np.zeros(3, dtype=np.float64)

        incident_radiance = mitsuba_spectrum_to_rgb(
            emitter.eval(hit)
        )

    bsdf_times_cosine = evaluate_bsdf_times_cosine(
        surface_interaction=interaction,
        world_direction=direction,
    )
    return incident_radiance * bsdf_times_cosine

def sample_emitter_term(
    scene: Any,
    interaction: Any,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Return light direction, PDF, and contribution divided by that PDF."""

    direction_sample, emitter_weight = scene.sample_emitter_direction(
        interaction,
        mi.Point2f(
            float(rng.random()),
            float(rng.random()),
        ),
        True,
    )

    light_pdf = float(direction_sample.pdf)
    if not np.isfinite(light_pdf) or light_pdf < 0.0:
        raise ValueError(
            f"Mitsuba returned an invalid emitter PDF: {light_pdf}."
        )

    emitter_weight_rgb = mitsuba_spectrum_to_rgb(emitter_weight)

    if light_pdf == 0.0:
        if np.any(emitter_weight_rgb > 0.0):
            raise NotImplementedError(
                "This experiment does not support delta emitters."
            )

        return (
            np.zeros(3, dtype=np.float64),
            0.0,
            np.zeros(3, dtype=np.float64),
        )

    direction_world = normalize_numpy_vector(
        mitsuba_vector_to_numpy(direction_sample.d)
    )
    bsdf_times_cosine = evaluate_bsdf_times_cosine(
        surface_interaction=interaction,
        world_direction=direction_world,
    )

    # Mitsuba returns incident radiance divided by the emitter PDF.
    contribution_over_pdf = emitter_weight_rgb * bsdf_times_cosine
    return direction_world, light_pdf, contribution_over_pdf

def sample_emitter_contribution(
    scene: Any,
    interaction: Any,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample one emitter contribution for the light-only estimator."""

    _, _, contribution_over_pdf = sample_emitter_term(
        scene,
        interaction,
        rng,
    )
    return contribution_over_pdf

def estimate_uniform(
    scene: Any,
    interaction: Any,
    frame: Any,
    rng: np.random.Generator,
    sample_count: int,
) -> RgbEstimate:
    """Estimate direct lighting with uniform-hemisphere sampling."""

    pdf = 1.0 / (2.0 * np.pi)
    total = np.zeros(3, dtype=np.float64)

    for _ in range(sample_count):
        direction = sample_uniform_hemisphere(frame, rng)
        total += evaluate_direct_integrand(
            scene,
            interaction,
            direction,
        ) / pdf

    return total / sample_count

def estimate_bsdf(
    scene: Any,
    interaction: Any,
    rng: np.random.Generator,
    sample_count: int,
) -> RgbEstimate:
    """Estimate direct lighting with BSDF sampling."""

    total = np.zeros(3, dtype=np.float64)

    for _ in range(sample_count):
        direction, pdf = sample_bsdf_direction(interaction, rng)
        total += evaluate_direct_integrand(
            scene,
            interaction,
            direction,
        ) / pdf

    return total / sample_count

def estimate_neural(
    scene: Any,
    interaction: Any,
    proposal: Any,
    rng: np.random.Generator,
    sample_count: int,
) -> RgbEstimate:
    """Estimate direct lighting with neural directional sampling."""

    total = np.zeros(3, dtype=np.float64)

    for _ in range(sample_count):
        sample = proposal.sample(rng)
        total += evaluate_direct_integrand(
            scene,
            interaction,
            sample.direction_world,
        ) / sample.pdf

    return total / sample_count

def estimate_light(
    scene: Any,
    interaction: Any,
    rng: np.random.Generator,
    sample_count: int,
) -> RgbEstimate:
    """Estimate direct lighting with emitter sampling."""

    total = np.zeros(3, dtype=np.float64)

    for _ in range(sample_count):
        total += sample_emitter_contribution(
            scene,
            interaction,
            rng,
        )

    return total / sample_count

def estimate_neural_bsdf_mis(
    scene: Any,
    interaction: Any,
    proposal: Any,
    rng: np.random.Generator,
) -> RgbEstimate:
    """Combine one neural and one BSDF sample using the power heuristic."""

    neural_sample = proposal.sample(rng)
    bsdf_direction, bsdf_pdf = sample_bsdf_direction(
        interaction,
        rng,
    )

    neural_contribution = evaluate_direct_integrand(
        scene,
        interaction,
        neural_sample.direction_world,
    )
    bsdf_contribution = evaluate_direct_integrand(
        scene,
        interaction,
        bsdf_direction,
    )

    bsdf_pdf_at_neural = evaluate_bsdf_pdf(
        surface_interaction=interaction,
        world_direction=neural_sample.direction_world,
    )
    neural_pdf_at_bsdf = proposal.pdf(bsdf_direction)

    neural_weight = power_heuristic(
        neural_sample.pdf,
        bsdf_pdf_at_neural,
    )
    bsdf_weight = power_heuristic(
        bsdf_pdf,
        neural_pdf_at_bsdf,
    )

    return (
        neural_weight
        * neural_contribution
        / neural_sample.pdf
        + bsdf_weight
        * bsdf_contribution
        / bsdf_pdf
    )

def summarize_estimates(
    estimates: np.ndarray,
    reference: np.ndarray,
) -> dict[str, Any]:
    """Calculate summary statistics for repeated estimates."""

    mean_rgb = estimates.mean(axis=0)
    variance_rgb = estimates.var(axis=0, ddof=1)
    standard_error_rgb = (
        estimates.std(axis=0, ddof=1)
        / np.sqrt(estimates.shape[0])
    )
    luminance = estimates @ _LUMINANCE_WEIGHTS

    return {
        "mean_rgb": mean_rgb.tolist(),
        "variance_rgb": variance_rgb.tolist(),
        "standard_error_rgb": standard_error_rgb.tolist(),
        "bias_rgb": (mean_rgb - reference).tolist(),
        "rmse_rgb": float(
            np.sqrt(np.mean((estimates - reference) ** 2))
        ),
        "luminance_variance": float(luminance.var(ddof=1)),
    }
