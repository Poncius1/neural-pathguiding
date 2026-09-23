"""Bridge between Mitsuba surface interactions and neural guiding."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neural_path_guiding.core.directional_distribution import (
    DiscreteDirectionalDistribution,
)
from neural_path_guiding.core.features import (
    FloatArray,
    normalize_vector3,
)
from neural_path_guiding.core.frames import (
    OrthonormalFrame,
    build_frame_from_normal,
    local_to_world,
    world_to_local,
)
from neural_path_guiding.core.sampling import DirectionalSample
from neural_path_guiding.renderers.mitsuba.adapters import (
    surface_interaction_to_features,
)
from neural_path_guiding.training.inference import (
    GuidingInferenceRuntime,
)


@dataclass(frozen=True)
class WorldDirectionalSample:
    """A guiding sample expressed in both local and world coordinates."""

    direction_world: FloatArray
    local_sample: DirectionalSample

    def __post_init__(self) -> None:
        direction = normalize_vector3(
            self.direction_world,
            "direction_world",
        )
        stored_direction = np.array(
            direction,
            dtype=np.float64,
            copy=True,
        )
        stored_direction.setflags(write=False)

        object.__setattr__(
            self,
            "direction_world",
            stored_direction,
        )

    @property
    def direction_local(self) -> FloatArray:
        """Return the sampled direction in the shading frame."""

        return self.local_sample.direction_local

    @property
    def bin_index(self) -> int:
        """Return the selected hemisphere bin."""

        return self.local_sample.bin_index

    @property
    def pdf(self) -> float:
        """Return the solid-angle PDF."""

        return self.local_sample.pdf


@dataclass(frozen=True)
class GuidingProposal:
    """A predicted directional distribution at one shading point."""

    frame: OrthonormalFrame
    distribution: DiscreteDirectionalDistribution

    def sample(
        self,
        rng: np.random.Generator,
    ) -> WorldDirectionalSample:
        """Sample a local direction and transform it to world space."""

        local_sample = self.distribution.sample(rng)
        direction_world = normalize_vector3(
            local_to_world(
                self.frame,
                local_sample.direction_local,
            ),
            "direction_world",
        )

        return WorldDirectionalSample(
            direction_world=direction_world,
            local_sample=local_sample,
        )

    def pdf(
        self,
        direction_world: FloatArray,
    ) -> float:
        """Evaluate the guiding PDF for a world-space direction.

        Directions below the local hemisphere have zero guiding density.
        """

        normalized_world = normalize_vector3(
            direction_world,
            "direction_world",
        )
        direction_local = world_to_local(
            self.frame,
            normalized_world,
        )

        if direction_local[2] < -1e-10:
            return 0.0

        normalized_local = normalize_vector3(
            direction_local,
            "direction_local",
        )

        return self.distribution.pdf(normalized_local)


@dataclass(frozen=True)
class MitsubaGuidingAdapter:
    """Create neural guiding proposals from Mitsuba surface interactions."""

    runtime: GuidingInferenceRuntime

    def prepare_proposal(
        self,
        *,
        surface_interaction: object,
        outgoing_direction: FloatArray,
        bounce_depth: int,
        roughness: float = 1.0,
        albedo: FloatArray | None = None,
    ) -> GuidingProposal:
        """Extract model features and predict a proposal distribution."""

        shading_features = surface_interaction_to_features(
            surface_interaction=surface_interaction,
            outgoing_direction=outgoing_direction,
            bounce_depth=bounce_depth,
            roughness=roughness,
            albedo=albedo,
        )

        feature_vector = shading_features.to_array()
        distribution = self.runtime.predict_distribution(
            feature_vector
        )
        frame = build_frame_from_normal(
            shading_features.normal
        )

        return GuidingProposal(
            frame=frame,
            distribution=distribution,
        )