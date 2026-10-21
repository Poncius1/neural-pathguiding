"""Tests for emitter cross-PDF diagnostic counters."""

from __future__ import annotations

import unittest

import numpy as np

from neural_path_guiding.renderers.mitsuba.emitter_pdf import (
    EmitterPdfQuery,
    EmitterPdfStats,
)


class TestEmitterPdfStats(unittest.TestCase):

    def setUp(self):
        self.stats = EmitterPdfStats()
        self.direction = np.array(
            [0.0, 0.0, 1.0],
            dtype=np.float64,
        )

    def make_query(
        self,
        *,
        pdf=0.1,
        hit_emitter=True,
        environment=False,
        geometric=False,
        fallback=False,
    ):
        return EmitterPdfQuery(
            direction_world=self.direction,
            pdf=pdf,
            hit_emitter=hit_emitter,
            environment=environment,
            used_geometric_hit=geometric,
            used_visibility_fallback=fallback,
        )

    def test_initial_counters_are_zero(self):
        result = self.stats.to_dict()

        self.assertEqual(result["total_queries"], 0)
        self.assertEqual(result["fallback_rate"], 0.0)

    def test_records_geometric_hit(self):
        self.stats.record(
            self.make_query(geometric=True)
        )

        result = self.stats.to_dict()

        self.assertEqual(result["total_queries"], 1)
        self.assertEqual(result["geometric_queries"], 1)

    def test_records_fallback(self):
        self.stats.record(
            self.make_query(fallback=True)
        )

        result = self.stats.to_dict()

        self.assertEqual(result["fallback_queries"], 1)
        self.assertEqual(result["fallback_rate"], 1.0)

    def test_records_environment(self):
        self.stats.record(
            self.make_query(environment=True)
        )

        result = self.stats.to_dict()

        self.assertEqual(result["environment_queries"], 1)

    def test_records_blocked_direction(self):
        self.stats.record(
            self.make_query(
                pdf=0.0,
                hit_emitter=False,
            )
        )

        result = self.stats.to_dict()

        self.assertEqual(result["blocked_queries"], 1)
        self.assertEqual(result["zero_pdf_queries"], 1)

    def test_records_zero_pdf_emitter(self):
        self.stats.record(
            self.make_query(
                pdf=0.0,
                geometric=True,
            )
        )

        result = self.stats.to_dict()

        self.assertEqual(result["geometric_queries"], 1)
        self.assertEqual(result["zero_pdf_queries"], 1)

    def test_calculates_fallback_rates(self):
        self.stats.record(
            self.make_query(geometric=True)
        )
        self.stats.record(
            self.make_query(fallback=True)
        )
        self.stats.record(
            self.make_query(environment=True)
        )
        self.stats.record(
            self.make_query(
                pdf=0.0,
                hit_emitter=False,
            )
        )

        result = self.stats.to_dict()

        self.assertEqual(result["total_queries"], 4)
        self.assertEqual(result["fallback_queries"], 1)
        self.assertAlmostEqual(
            result["fallback_rate"], 0.25
        )
        self.assertAlmostEqual(
            result["fallback_rate_among_area_hits"], 0.5
        )

    def test_rejects_unclassified_area_result(self):
        with self.assertRaisesRegex(
            ValueError,
            "Unclassified",
        ):
            self.stats.record(
                self.make_query()
            )


if __name__ == "__main__":
    unittest.main()