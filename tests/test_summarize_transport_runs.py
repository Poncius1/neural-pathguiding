"""Pure Python regression tests for aggregated transport experiments."""
from __future__ import annotations

from pathlib import Path
import unittest

from scripts.summarize_transport_runs import summarize_records


def manifest(seed: int, neural: float, bsdf: float) -> dict:
    return {
        "experiment": "single_loop_transport_controls",
        "scene_sha256": "scene", "checkpoint_sha256": "checkpoint",
        "reference_sha256": "reference",
        "settings": {"seed": seed, "width": 32, "height": 32,
                     "spp": 16, "max_depth": 6, "rr_depth": 5,
                     "guiding_probability": .5, "uniform_mix": .05},
        "results": {
            "bsdf": {"rgb_mse": bsdf, "seconds": 3},
            "neural": {"rgb_mse": neural, "seconds": 20},
        },
    }


class TestSummarizeTransportRuns(unittest.TestCase):
    def test_paired_results_keep_seed_alignment(self):
        records = [(Path("seed9"), manifest(9, .2, .3)),
                   (Path("seed2"), manifest(2, .4, .1))]
        result = summarize_records(records)
        paired = result["paired_differences"]["neural_minus_bsdf"]
        self.assertAlmostEqual(paired["mean_mse_difference"], .1)
        self.assertEqual(paired["wins_a"], 1)
        self.assertEqual(result["seeds"], [2, 9])

    def test_duplicate_seeds_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate seed"):
            summarize_records([(Path("a"), manifest(9, .2, .3)),
                               (Path("b"), manifest(9, .4, .3))])

    def test_explicit_latest_deduplication(self):
        records = [(Path("run_2026a/manifest.json"), manifest(42, .5, .3)),
                   (Path("run_2026b/manifest.json"), manifest(42, .2, .3)),
                   (Path("run_2026c/manifest.json"), manifest(123, .4, .3))]
        result = summarize_records(records, duplicate_policy="latest")
        self.assertEqual(result["n_independent_seeds"], 2)
        self.assertEqual(result["methods"]["neural"]["per_seed_mse"]["42"], .2)
        self.assertEqual(len(result["discarded_duplicate_manifests"]), 1)

    def test_mixed_settings_rejected(self):
        first = manifest(1, .2, .3)
        second = manifest(2, .4, .3)
        second["settings"]["max_depth"] = 8
        with self.assertRaisesRegex(ValueError, "Incompatible"):
            summarize_records([(Path("a"), first), (Path("b"), second)])

    def test_missing_reference_metric_rejected(self):
        first = manifest(1, .2, .3)
        second = manifest(2, .4, .3)
        del second["results"]["neural"]["rgb_mse"]
        with self.assertRaisesRegex(ValueError, "MSE absent"):
            summarize_records([(Path("a"), first), (Path("b"), second)])


if __name__ == "__main__":
    unittest.main()
