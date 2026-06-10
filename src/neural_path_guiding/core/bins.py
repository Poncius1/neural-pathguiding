"""Hemisphere binning utilities.

This module divides the local upper hemisphere into equal solid-angle bins.

It provides:
- bin indexing
- bin angular bounds
- direction sampling inside a bin
- direction-to-bin lookup

Local convention:
+Z is the surface normal.
omega.z = cos(theta).
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]

_TWO_PI = 2.0 * math.pi
_EPSILON = 1e-12


# Angular bounds of a single hemisphere bin.
@dataclass(frozen=True)
class BinBounds:
    mu_min: float  # Lower bound for mu = cos(theta).
    mu_max: float  # Upper bound for mu = cos(theta).
    phi_min: float  # Lower azimuth bound in radians.
    phi_max: float  # Upper azimuth bound in radians.

    @property
    def solid_angle(self) -> float:
        # Solid angle of a rectangle in (mu, phi).
        return (self.mu_max - self.mu_min) * (self.phi_max - self.phi_min)


# Uniform-solid-angle discretization of the upper hemisphere.
@dataclass(frozen=True)
class HemisphereBins:
    n_mu: int  # Number of bins along mu = cos(theta).
    n_phi: int  # Number of bins along phi.

    def __post_init__(self) -> None:
        # Bin resolution must be positive.
        if self.n_mu <= 0:
            raise ValueError("n_mu must be greater than zero.")

        if self.n_phi <= 0:
            raise ValueError("n_phi must be greater than zero.")

    @property
    def n_bins(self) -> int:
        # Total number of directional bins.
        return self.n_mu * self.n_phi

    @property
    def bin_solid_angle(self) -> float:
        # Every bin has the same solid angle.
        return _TWO_PI / self.n_bins

    def get_bin_indices(self, bin_index: int) -> tuple[int, int]:
        # Converts a flat bin index to (mu_index, phi_index).
        self._validate_bin_index(bin_index)

        mu_index = bin_index // self.n_phi
        phi_index = bin_index % self.n_phi

        return mu_index, phi_index

    def get_flat_index(self, mu_index: int, phi_index: int) -> int:
        # Converts (mu_index, phi_index) to a flat bin index.
        if not 0 <= mu_index < self.n_mu:
            raise ValueError(f"mu_index must be in [0, {self.n_mu}).")

        if not 0 <= phi_index < self.n_phi:
            raise ValueError(f"phi_index must be in [0, {self.n_phi}).")

        return mu_index * self.n_phi + phi_index

    def get_bin_bounds(self, bin_index: int) -> BinBounds:
        # Returns the angular region covered by one bin.
        mu_index, phi_index = self.get_bin_indices(bin_index)

        mu_min = mu_index / self.n_mu
        mu_max = (mu_index + 1) / self.n_mu

        phi_min = _TWO_PI * phi_index / self.n_phi
        phi_max = _TWO_PI * (phi_index + 1) / self.n_phi

        return BinBounds(
            mu_min=mu_min,
            mu_max=mu_max,
            phi_min=phi_min,
            phi_max=phi_max,
        )

    def sample_direction_in_bin(
        self,
        bin_index: int,
        u_mu: float,
        u_phi: float,
    ) -> FloatArray:
        # Samples a local direction uniformly inside one bin.
        self._validate_unit_interval_open(u_mu, "u_mu")
        self._validate_unit_interval_open(u_phi, "u_phi")

        bounds = self.get_bin_bounds(bin_index)

        mu = bounds.mu_min + u_mu * (bounds.mu_max - bounds.mu_min)
        phi = bounds.phi_min + u_phi * (bounds.phi_max - bounds.phi_min)

        sin_theta = math.sqrt(max(0.0, 1.0 - mu * mu))

        x = math.cos(phi) * sin_theta
        y = math.sin(phi) * sin_theta
        z = mu

        return np.array([x, y, z], dtype=np.float64)

    def find_bin_from_direction(self, omega_local: FloatArray) -> int:
        # Finds which bin contains a local upper-hemisphere direction.
        omega = self._normalize_direction(omega_local)

        mu = float(omega[2])
        if mu < -_EPSILON:
            raise ValueError("omega_local must lie in the upper hemisphere.")

        mu = min(max(mu, 0.0), float(np.nextafter(1.0, 0.0)))

        phi = math.atan2(float(omega[1]), float(omega[0]))
        if phi < 0.0:
            phi += _TWO_PI

        phi = min(max(phi, 0.0), float(np.nextafter(_TWO_PI, 0.0)))

        mu_index = min(int(mu * self.n_mu), self.n_mu - 1)
        phi_index = min(int(phi / _TWO_PI * self.n_phi), self.n_phi - 1)

        return self.get_flat_index(mu_index, phi_index)

    def _validate_bin_index(self, bin_index: int) -> None:
        # Checks that the flat bin index is valid.
        if not 0 <= bin_index < self.n_bins:
            raise ValueError(f"bin_index must be in [0, {self.n_bins}).")

    @staticmethod
    def _validate_unit_interval_open(value: float, name: str) -> None:
        # Random numbers must be in [0, 1).
        if not 0.0 <= value < 1.0:
            raise ValueError(f"{name} must be in [0, 1).")

    @staticmethod
    def _normalize_direction(omega_local: FloatArray) -> FloatArray:
        # Normalizes and validates a 3D direction.
        omega = np.asarray(omega_local, dtype=np.float64)

        if omega.shape != (3,):
            raise ValueError(f"omega_local must have shape (3,), got {omega.shape}.")

        norm = float(np.linalg.norm(omega))
        if norm <= 0.0:
            raise ValueError("omega_local must be non-zero.")

        return omega / norm