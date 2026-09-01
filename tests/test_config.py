"""Tests for YAML configuration helpers."""

from __future__ import annotations

import unittest

import numpy as np

from neural_path_guiding.config import (
    optional_float,
    require_float,
    require_float_sequence,
    require_string,
)


class TestConfigHelpers(unittest.TestCase):
    def test_required_float_rejects_non_finite_values(self) -> None:
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    require_float({"value": value}, "value")

    def test_optional_float_rejects_non_finite_values(self) -> None:
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    optional_float({"value": value}, "value", default=1.0)

    def test_float_sequence_rejects_non_finite_values(self) -> None:
        with self.assertRaises(ValueError):
            require_float_sequence({"vector": [0.0, np.nan, 1.0]}, "vector", 3)

    def test_float_sequence_rejects_booleans(self) -> None:
        with self.assertRaises(ValueError):
            require_float_sequence({"vector": [0.0, True, 1.0]}, "vector", 3)

    def test_required_string_rejects_whitespace(self) -> None:
        with self.assertRaises(ValueError):
            require_string({"name": "   "}, "name")


if __name__ == "__main__":
    unittest.main()
