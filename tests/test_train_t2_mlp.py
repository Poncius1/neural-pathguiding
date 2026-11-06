"""Unit tests for Phase 3.2.2 training utilities (no Mitsuba rendering required)."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from scripts.train_t2_mlp import (
    TrainSettings, distribution_metrics, validate_settings,
)


class TestT2Training(unittest.TestCase):
    def test_valid_settings(self):
        validate_settings(TrainSettings())

    def test_invalid_settings(self):
        with self.assertRaises(ValueError):
            validate_settings(TrainSettings(patience=0))
        with self.assertRaises(ValueError):
            validate_settings(TrainSettings(learning_rate=-1))

    def test_identical_distribution(self):
        p = torch.tensor([[0.2, 0.8], [0.4, 0.6]])
        m = distribution_metrics(p, p)
        self.assertAlmostEqual(m["tvd"], 0.0, places=6)
        self.assertAlmostEqual(m["kl"], 0.0, places=6)
        self.assertAlmostEqual(m["jsd"], 0.0, places=6)
        self.assertAlmostEqual(m["cross_entropy"], m["target_entropy"], places=6)

    def test_disjoint_distribution_with_positive_predictions(self):
        p = torch.tensor([[1.0, 0.0]])
        q = torch.tensor([[0.001, 0.999]])
        m = distribution_metrics(p, q)
        self.assertGreater(m["tvd"], 0.99)
        self.assertGreater(m["kl"], 0.0)
        self.assertGreater(m["jsd"], 0.0)

    def test_reject_bad_target(self):
        p = torch.tensor([[0.2, 0.2]])
        q = torch.tensor([[0.5, 0.5]])
        with self.assertRaises(ValueError):
            distribution_metrics(p, q)

    def test_reject_zero_prediction(self):
        p = torch.tensor([[0.0, 1.0]])
        q = torch.tensor([[0.0, 1.0]])
        with self.assertRaises(ValueError):
            distribution_metrics(p, q)

    def test_empty(self):
        p = torch.empty(0, 2)
        with self.assertRaises(ValueError):
            distribution_metrics(p, p)


if __name__ == "__main__":
    unittest.main()
