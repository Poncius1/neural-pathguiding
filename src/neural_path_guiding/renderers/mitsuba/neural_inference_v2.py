"""Mitsuba scalar_rgb bridge for material-aware T2 checkpoint inference.

Only pure Lambertian front-facing interactions and depth zero are supported.
This is an inference/sampling bridge, not a Monte Carlo integrator.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mitsuba as mi
import numpy as np
import torch

from neural_path_guiding.core.bins import HemisphereBins
from neural_path_guiding.core.directional_distribution import DiscreteDirectionalDistribution
from neural_path_guiding.renderers.mitsuba.adapters import (
    mitsuba_vector_to_numpy,
    normalize_numpy_vector,
)
from neural_path_guiding.renderers.mitsuba.material_features import (
    MATERIAL_FEATURE_SCHEMA_VERSION,
    is_supported_diffuse_bsdf,
    surface_to_material_features,
)
from neural_path_guiding.training.checkpoint import LoadedCheckpoint, load_checkpoint


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
        if not (np.isfinite(self.uniform_mix) and 0.0 < self.uniform_mix <= 1.0):
            raise ValueError("uniform_mix must satisfy 0 < alpha <= 1")
        if self.bins.n_bins != self.checkpoint.model.num_bins:
            raise ValueError("Checkpoint bin dimension mismatch")
        if self.checkpoint.model.feature_dimension != 14:
            raise ValueError("Feature dimension must be 14 for schema v2")
        md = self.checkpoint.metadata
        if md.get("feature_schema_version") != MATERIAL_FEATURE_SCHEMA_VERSION:
            raise ValueError("Expected material-aware feature schema version 2")
        if md.get("target_type") != "product_integrand_v1":
            raise ValueError("Expected product_integrand_v1 teacher target")
        if md.get("feature_ablation", "full") != "full":
            raise ValueError("Ablated checkpoints need their matching inference mask")
        if self.checkpoint.feature_mean.shape != (14,) or self.checkpoint.feature_scale.shape != (14,):
            raise ValueError("Invalid normalization shape")
        if not bool(torch.all(torch.isfinite(self.checkpoint.feature_mean))):
            raise ValueError("Nonfinite feature mean")
        if not bool(torch.all(torch.isfinite(self.checkpoint.feature_scale) & (self.checkpoint.feature_scale > 0))):
            raise ValueError("Invalid feature scales")

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        uniform_mix: float = 0.05,
        device: str = "cpu",
    ) -> "NeuralMitsubaRuntimeV2":
        loaded = load_checkpoint(path, device=device)
        md = loaded.metadata
        n_mu, n_phi = md.get("n_mu"), md.get("n_phi")
        if isinstance(n_mu, bool) or not isinstance(n_mu, int) or n_mu <= 0:
            raise ValueError("Checkpoint requires positive integer n_mu and n_phi")
        if isinstance(n_phi, bool) or not isinstance(n_phi, int) or n_phi <= 0:
            raise ValueError("Checkpoint requires positive integer n_mu and n_phi")
        return cls(loaded, HemisphereBins(n_mu, n_phi), uniform_mix)

    def distribution_at(
        self,
        interaction: Any,
        *,
        outgoing_world: np.ndarray,
        bounce_depth: int = 0,
    ) -> DiscreteDirectionalDistribution:
        if bounce_depth != 0:
            raise ValueError("Phase 3.3.1 supports camera-first-hit points only")
        if not bool(interaction.is_valid()) or not is_supported_diffuse_bsdf(interaction.bsdf()):
            raise ValueError("Requires a valid pure diffuse interaction")
        if float(interaction.wi.z) <= 0:
            raise ValueError("Requires a front-facing interaction")
        outgoing = normalize_numpy_vector(outgoing_world)
        features = surface_to_material_features(
            surface_interaction=interaction,
            outgoing_direction=outgoing,
            bounce_depth=0,
        ).to_array()
        values = torch.tensor(
            np.asarray(features, dtype=np.float32).copy(),
            device=self.checkpoint.feature_mean.device,
            dtype=self.checkpoint.feature_mean.dtype,
        )
        normalized = (values - self.checkpoint.feature_mean) / self.checkpoint.feature_scale
        if not bool(torch.isfinite(normalized).all()):
            raise ValueError("Nonfinite normalized features")
        with torch.inference_mode():
            logits = self.checkpoint.model(normalized.unsqueeze(0))[0]
        if logits.numel() != self.bins.n_bins or not bool(torch.isfinite(logits).all()):
            raise ValueError("Invalid neural logits")
        return DiscreteDirectionalDistribution.from_logits(
            bins=self.bins,
            logits=logits.detach().cpu().numpy().astype(np.float64),
            uniform_mix=float(self.uniform_mix),
        )

    def distribution_from_camera_ray(
        self, interaction: Any, ray: Any
    ) -> DiscreteDirectionalDistribution:
        outgoing_world = -mitsuba_vector_to_numpy(ray.d)
        return self.distribution_at(interaction, outgoing_world=outgoing_world)

    @staticmethod
    def sample_world(
        interaction: Any,
        distribution: DiscreteDirectionalDistribution,
        rng: np.random.Generator,
    ) -> NeuralDirectionSample:
        sample = distribution.sample(rng)
        local = np.asarray(sample.direction_local, dtype=np.float64)
        world_vector = interaction.to_world(mi.Vector3f(*local))
        world = normalize_numpy_vector(mitsuba_vector_to_numpy(world_vector))
        pdf = distribution.pdf(local)
        if not np.isfinite(pdf) or pdf <= 0:
            raise ValueError("Sample has invalid solid-angle PDF")
        return NeuralDirectionSample(local=local, world=world,
                                     pdf_solid_angle=float(pdf),
                                     bin_index=int(sample.bin_index))

    @staticmethod
    def pdf_world(
        interaction: Any,
        distribution: DiscreteDirectionalDistribution,
        direction_world: np.ndarray,
    ) -> float:
        direction = normalize_numpy_vector(direction_world)
        local = interaction.to_local(mi.Vector3f(*direction))
        return distribution.pdf(mitsuba_vector_to_numpy(local))
