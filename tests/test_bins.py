"""Tests for hemisphere binning.

These tests validate the geometric part of the method:

- bins cover the upper hemisphere
- every bin has the expected solid angle
- sampled directions are valid
- sampled directions map back to their original bin
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from neural_path_guiding.core.bins import HemisphereBins


class TestHemisphereBins(unittest.TestCase):
    # Shared bin setup for all tests.
    def setUp(self) -> None:
        self.bins = HemisphereBins(n_mu=4, n_phi=8)

    def test_number_of_bins(self) -> None:
        self.assertEqual(self.bins.n_bins, 32)

    def test_bin_solid_angle(self) -> None:
        expected = 2.0 * math.pi / 32

        self.assert_close(self.bins.bin_solid_angle, expected)

    def test_all_bins_have_expected_solid_angle(self) -> None:
        for bin_index in range(self.bins.n_bins):
            bounds = self.bins.get_bin_bounds(bin_index)

            self.assert_close(bounds.solid_angle, self.bins.bin_solid_angle)

    def test_flat_index_round_trip(self) -> None:
        for bin_index in range(self.bins.n_bins):
            mu_index, phi_index = self.bins.get_bin_indices(bin_index)
            recovered_index = self.bins.get_flat_index(mu_index, phi_index)

            self.assertEqual(recovered_index, bin_index)

    def test_sampled_direction_is_valid(self) -> None:
        for bin_index in range(self.bins.n_bins):
            omega = self.bins.sample_direction_in_bin(
                bin_index=bin_index,
                u_mu=0.37,
                u_phi=0.61,
            )

            self.assertEqual(omega.shape, (3,))
            self.assert_close(float(np.linalg.norm(omega)), 1.0)
            self.assertGreaterEqual(float(omega[2]), 0.0)

    def test_sampled_direction_maps_back_to_same_bin(self) -> None:
        for bin_index in range(self.bins.n_bins):
            omega = self.bins.sample_direction_in_bin(
                bin_index=bin_index,
                u_mu=0.37,
                u_phi=0.61,
            )

            recovered_index = self.bins.find_bin_from_direction(omega)

            self.assertEqual(recovered_index, bin_index)

    def assert_close(
        self,
        actual: float,
        expected: float,
        tolerance: float = 1e-12,
    ) -> None:
        # Avoids Pylance overload issues with assertAlmostEqual and NumPy floats.
        difference = abs(float(actual) - float(expected))

        self.assertLessEqual(
            difference,
            tolerance,
            msg=f"Expected {expected}, got {actual}. Difference: {difference}",
        )


if __name__ == "__main__":
    unittest.main()