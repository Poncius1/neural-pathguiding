"""Material-aware shading features for T2 datasets.

Feature schema v2:
- Same 14 numeric components as feature schema v1.
- Albedo is derived from the actual Lambertian BSDF.
- Roughness 1.0 is a convention for diffuse materials.
- Only pure diffuse-reflection BSDFs are supported.

Target:
    Python 3.11
    Mitsuba 3.8.0
    scalar_rgb
"""

from __future__ import annotations

from typing import Any

import mitsuba as mi
import numpy as np

from neural_path_guiding.core.features import (
    ShadingFeatures,
)
from neural_path_guiding.renderers.mitsuba.adapters import (
    surface_interaction_to_features,
)
from neural_path_guiding.renderers.mitsuba.bsdf import (
    mitsuba_bsdf_to_rgb,
)


MATERIAL_FEATURE_SCHEMA_VERSION = 2

MATERIAL_FEATURE_MODE = "lambertian_reflectance_v1"


def is_supported_diffuse_bsdf(bsdf: Any) -> bool:
    """Check whether a BSDF has only one diffuse-reflection lobe.

    Mitsuba flags also contain attributes such as FrontSide.
    We therefore mask the scattering-lobe bits before comparing.

    A BSDF with additional glossy, delta, or transmission
    components is rejected.
    """

    if bsdf is None:
        return False

    if int(bsdf.component_count()) != 1:
        return False

    # Mitsuba 3 exposes component flags using flags(index).
    component_flags = int(bsdf.flags(0))

    # Mask out FrontSide, BackSide, SpatiallyVarying,
    # and other non-scattering attributes.
    scattering_mask = int(mi.BSDFFlags.All)

    scattering_flags = (
        component_flags & scattering_mask
    )

    expected_flags = int(
        mi.BSDFFlags.DiffuseReflection
    )

    return scattering_flags == expected_flags


def validate_diffuse_reflectance(
    reflectance: np.ndarray,
) -> np.ndarray:
    """Validate a finite RGB diffuse reflectance.

    A small tolerance accounts for floating-point error.
    """

    values = np.asarray(
        reflectance,
        dtype=np.float64,
    )

    if values.shape != (3,):
        raise ValueError(
            "Reflectance must have shape (3,)."
        )

    if not bool(np.all(np.isfinite(values))):
        raise ValueError(
            "Reflectance contains non-finite values."
        )

    tolerance = 1e-5

    if bool(
        np.any(values < -tolerance)
        or np.any(values > 1.0 + tolerance)
    ):
        raise ValueError(
            "Diffuse reflectance must be within [0, 1]."
        )

    return np.clip(
        values,
        0.0,
        1.0,
    )


def extract_diffuse_reflectance(
    interaction: Any,
) -> np.ndarray:
    """Extract effective reflectance from a Lambertian BSDF.

    Mitsuba BSDF.eval() already includes the cosine term.

    For local outgoing direction (0, 0, 1):

        BSDF.eval = reflectance / pi

    Therefore:

        reflectance = pi * BSDF.eval

    This is valid for supported pure Lambertian surfaces.
    """

    if not bool(interaction.is_valid()):
        raise ValueError(
            "Cannot extract material features "
            "from an invalid surface interaction."
        )

    bsdf = interaction.bsdf()

    if not is_supported_diffuse_bsdf(bsdf):
        raise ValueError(
            "Material feature extraction currently "
            "supports pure diffuse-reflection BSDFs only."
        )

    # Mitsuba stores the incident/view direction
    # in the local shading frame.
    if float(interaction.wi.z) <= 0.0:
        raise ValueError(
            "Diffuse reflection requires a "
            "front-facing surface interaction."
        )

    local_direction = mi.Vector3f(
        0.0,
        0.0,
        1.0,
    )

    context = mi.BSDFContext()

    evaluated_bsdf = bsdf.eval(
        context,
        interaction,
        local_direction,
    )

    bsdf_rgb = mitsuba_bsdf_to_rgb(
        evaluated_bsdf
    )

    reflectance = np.pi * bsdf_rgb

    return validate_diffuse_reflectance(
        reflectance
    )


def surface_to_material_features(
    *,
    surface_interaction: Any,
    outgoing_direction: np.ndarray,
    bounce_depth: int = 0,
) -> ShadingFeatures:
    """Create a 14-dimensional material-aware feature vector.

    The v2 contract differs from v1 in its albedo semantics.
    It is not interchangeable with existing v1 checkpoints.
    """

    albedo = extract_diffuse_reflectance(
        surface_interaction
    )

    return surface_interaction_to_features(
        surface_interaction=surface_interaction,
        outgoing_direction=outgoing_direction,
        bounce_depth=bounce_depth,
        roughness=1.0,
        albedo=albedo,
    )