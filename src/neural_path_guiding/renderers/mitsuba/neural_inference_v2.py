
"""Material-aware T2 inference bridge for Mitsuba 3.8.

The teacher, inference runtime, and renderer share one directional
frame convention, defined by core.frames.build_frame_from_normal().

Supported guiding domain:
    - Pure Lambertian BSDF
    - Front-facing surface interaction
    - Camera-first-hit only (bounce_depth == 0)

This module predicts and samples directional distributions.
It does not implement light transport.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from neural_path_guiding.core.bins import HemisphereBins
from neural_path_guiding.core.directional_distribution import (
    DiscreteDirectionalDistribution,
)
from neural_path_guiding.core.frames import (
    build_frame_from_normal,
)
from neural_path_guiding.renderers.mitsuba.adapters import (
    mitsuba_vector_to_numpy,
    normalize_numpy_vector,
)
from neural_path_guiding.renderers.mitsuba.guiding_adapter import (
    GuidingProposal,
)
from neural_path_guiding.renderers.mitsuba.material_features import (
    MATERIAL_FEATURE_SCHEMA_VERSION,
    is_supported_diffuse_bsdf,
    surface_to_material_features,
)
from neural_path_guiding.training.checkpoint import (
    LoadedCheckpoint,
    load_checkpoint,
)


@dataclass(frozen=True)
class NeuralDirectionSample:
    local: np.ndarray
    world: np.ndarray
    pdf_solid_angle: float
    bin_index: int


@dataclass(frozen=True)
class NeuralMitsubaRuntimeV2:
    checkpoint: LoadedCheckpoint
    bins: HemisphereBins
    uniform_mix: float = 0.05

    def __post_init__(self) -> None:
        if not (
            np.isfinite(self.uniform_mix)
            and 0.0 < self.uniform_mix <= 1.0
        ):
            raise ValueError(
                "uniform_mix must satisfy 0 < alpha <= 1"
            )

        if self.bins.n_bins != self.checkpoint.model.num_bins:
            raise ValueError(
                "Checkpoint bin dimension mismatch"
            )

        if self.checkpoint.model.feature_dimension != 14:
            raise ValueError(
                "Feature dimension must be 14 for schema v2"
            )

        metadata = self.checkpoint.metadata

        if (
            metadata.get("feature_schema_version")
            != MATERIAL_FEATURE_SCHEMA_VERSION
        ):
            raise ValueError(
                "Expected material-aware feature schema version 2"
            )

        if metadata.get("target_type") != "product_integrand_v1":
            raise ValueError(
                "Expected product_integrand_v1 teacher target"
            )

        if metadata.get("feature_ablation", "full") != "full":
            raise ValueError(
                "Ablated checkpoints require their "
                "matching inference feature mask"
            )

        if (
            self.checkpoint.feature_mean.shape != (14,)
            or self.checkpoint.feature_scale.shape != (14,)
        ):
            raise ValueError(
                "Invalid normalization shape"
            )

        if not bool(
            torch.all(
                torch.isfinite(
                    self.checkpoint.feature_mean
                )
            )
        ):
            raise ValueError(
                "Non-finite feature mean"
            )

        scales = self.checkpoint.feature_scale

        if not bool(
            torch.all(
                torch.isfinite(scales) & (scales > 0)
            )
        ):
            raise ValueError(
                "Invalid feature scales"
            )

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        uniform_mix: float = 0.05,
        device: str = "cpu",
    ) -> NeuralMitsubaRuntimeV2:
        checkpoint = load_checkpoint(
            path,
            device=device,
        )

        metadata = checkpoint.metadata
        n_mu = metadata.get("n_mu")
        n_phi = metadata.get("n_phi")

        if (
            isinstance(n_mu, bool)
            or not isinstance(n_mu, int)
            or n_mu <= 0
        ):
            raise ValueError(
                "Checkpoint requires a positive integer n_mu"
            )

        if (
            isinstance(n_phi, bool)
            or not isinstance(n_phi, int)
            or n_phi <= 0
        ):
            raise ValueError(
                "Checkpoint requires a positive integer n_phi"
            )

        return cls(
            checkpoint=checkpoint,
            bins=HemisphereBins(n_mu, n_phi),
            uniform_mix=uniform_mix,
        )

    def distribution_at(
        self,
        interaction: Any,
        *,
        outgoing_world: np.ndarray,
        bounce_depth: int = 0,
    ) -> DiscreteDirectionalDistribution:
        """Predict a distribution in the canonical T2 frame."""

        if bounce_depth != 0:
            raise ValueError(
                "T2 v2 supports camera-first-hit points only"
            )

        if (
            not bool(interaction.is_valid())
            or not is_supported_diffuse_bsdf(
                interaction.bsdf()
            )
        ):
            raise ValueError(
                "Requires a valid pure diffuse interaction"
            )

        if float(interaction.wi.z) <= 0.0:
            raise ValueError(
                "Requires a front-facing interaction"
            )

        outgoing = normalize_numpy_vector(
            outgoing_world
        )

        features = surface_to_material_features(
            surface_interaction=interaction,
            outgoing_direction=outgoing,
            bounce_depth=0,
        ).to_array()

        values = torch.tensor(
            np.asarray(
                features,
                dtype=np.float32,
            ).copy(),
            device=self.checkpoint.feature_mean.device,
            dtype=self.checkpoint.feature_mean.dtype,
        )

        normalized = (
            values - self.checkpoint.feature_mean
        ) / self.checkpoint.feature_scale

        if not bool(torch.isfinite(normalized).all()):
            raise ValueError(
                "Non-finite normalized features"
            )

        with torch.inference_mode():
            logits = self.checkpoint.model(
                normalized.unsqueeze(0)
            )[0]

        if (
            logits.numel() != self.bins.n_bins
            or not bool(torch.isfinite(logits).all())
        ):
            raise ValueError(
                "Invalid neural logits"
            )

        return DiscreteDirectionalDistribution.from_logits(
            bins=self.bins,
            logits=(
                logits.detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            ),
            uniform_mix=float(self.uniform_mix),
        )

    def distribution_from_camera_ray(
        self,
        interaction: Any,
        ray: Any,
    ) -> DiscreteDirectionalDistribution:
        outgoing_world = normalize_numpy_vector(
            -mitsuba_vector_to_numpy(ray.d)
        )

        return self.distribution_at(
            interaction,
            outgoing_world=outgoing_world,
        )

    @staticmethod
    def proposal_from_distribution(
        interaction: Any,
        distribution: DiscreteDirectionalDistribution,
    ) -> GuidingProposal:
        """Create a proposal using the teacher's canonical frame.

        This is the single frame-construction entry point for
        T2 inference, sampling, PDF evaluation and rendering.
        """
        normal = normalize_numpy_vector(
            mitsuba_vector_to_numpy(interaction.n)
        )

        return GuidingProposal(
            frame=build_frame_from_normal(normal),
            distribution=distribution,
        )

    def proposal_at(
        self,
        interaction: Any,
        *,
        outgoing_world: np.ndarray,
        bounce_depth: int = 0,
    ) -> GuidingProposal:
        """Predict and construct a canonical world-space proposal."""
        distribution = self.distribution_at(
            interaction,
            outgoing_world=outgoing_world,
            bounce_depth=bounce_depth,
        )

        return self.proposal_from_distribution(
            interaction,
            distribution,
        )

    @staticmethod
    def sample_world(
        interaction: Any,
        distribution: DiscreteDirectionalDistribution,
        rng: np.random.Generator,
    ) -> NeuralDirectionSample:
        """Sample using exactly the frame used by the teacher."""

        proposal = (
            NeuralMitsubaRuntimeV2.proposal_from_distribution(
                interaction,
                distribution,
            )
        )

        sample = proposal.sample(rng)

        pdf = float(sample.pdf)

        if not np.isfinite(pdf) or pdf <= 0.0:
            raise ValueError(
                "Sample has invalid solid-angle PDF"
            )

        return NeuralDirectionSample(
            local=np.array(
                sample.direction_local,
                dtype=np.float64,
                copy=True,
            ),
            world=np.array(
                sample.direction_world,
                dtype=np.float64,
                copy=True,
            ),
            pdf_solid_angle=pdf,
            bin_index=int(sample.bin_index),
        )

    @staticmethod
    def pdf_world(
        interaction: Any,
        distribution: DiscreteDirectionalDistribution,
        direction_world: np.ndarray,
    ) -> float:
        """Evaluate the same world-space density used for sampling."""

        proposal = (
            NeuralMitsubaRuntimeV2.proposal_from_distribution(
                interaction,
                distribution,
            )
        )

        return float(
            proposal.pdf(direction_world)
        )
