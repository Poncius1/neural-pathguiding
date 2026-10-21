"""Tests for detailed emitter PDF fallback diagnostics.

Validates:
    - Geometric intersection compatibility.
    - Geometric ray misses.
    - Different intersected shapes.
    - Position mismatches.
    - Small numerical ray offsets.
    - Fallback reason counters.
    - Diagnostic example limits.
    - Invalid fallback reasons.
    - Conflicting geometric/fallback modes.

Target:
    Python 3.11
    Mitsuba 3.8.0
    scalar_rgb
"""

from __future__ import annotations

import unittest

import numpy as np

from neural_path_guiding.renderers.mitsuba.emitter_pdf import (
    EmitterPdfQuery,
    EmitterPdfStats,
    _fallback_reason,
)


class FakeShape:
    """Minimal fake shape used to test object identity."""

    pass


class FakeInteraction:
    """Minimal fake Mitsuba surface interaction."""

    def __init__(
        self,
        position,
        *,
        valid: bool = True,
        shape=None,
    ) -> None:
        self.p = np.asarray(
            position,
            dtype=np.float64,
        )

        self._valid = valid
        self.shape = shape

    def is_valid(self) -> bool:
        return self._valid


class TestFallbackReasons(unittest.TestCase):
    """Tests for geometric fallback classification."""

    def setUp(self) -> None:
        self.shape = FakeShape()

        self.origin = FakeInteraction(
            [0.0, 0.0, 0.0],
        )

        self.visibility = FakeInteraction(
            [0.0, 0.0, 2.0],
            shape=self.shape,
        )

    def classify(
        self,
        geometric: FakeInteraction,
    ) -> tuple[str | None, float | None, float | None]:
        """Classify a geometric ray against the visibility ray."""

        return _fallback_reason(
            original_interaction=self.origin,
            visibility_hit=self.visibility,
            geometric_hit=geometric,
            position_tolerance=1e-3,
        )

    def test_compatible_geometry(self) -> None:
        """Identical intersections should be compatible."""

        geometric = FakeInteraction(
            [0.0, 0.0, 2.0],
            shape=self.shape,
        )

        reason, distance, difference = self.classify(
            geometric
        )

        self.assertIsNone(reason)

        # Narrow Optional[float] to float for Pylance.
        assert distance is not None
        assert difference is not None

        self.assertAlmostEqual(
            distance,
            2.0,
        )

        self.assertAlmostEqual(
            difference,
            0.0,
        )

    def test_geometric_miss(self) -> None:
        """A missing geometric hit should be classified."""

        geometric = FakeInteraction(
            [0.0, 0.0, 0.0],
            valid=False,
        )

        reason, distance, difference = self.classify(
            geometric
        )

        self.assertEqual(
            reason,
            "geometric_miss",
        )

        self.assertIsNone(distance)
        self.assertIsNone(difference)

    def test_different_shape(self) -> None:
        """Different intersected shapes should trigger fallback."""

        geometric = FakeInteraction(
            [0.0, 0.0, 2.0],
            shape=FakeShape(),
        )

        reason, distance, difference = self.classify(
            geometric
        )

        self.assertEqual(
            reason,
            "different_shape",
        )

        assert distance is not None

        self.assertAlmostEqual(
            distance,
            2.0,
        )

        self.assertIsNone(difference)

    def test_position_mismatch(self) -> None:
        """A large position discrepancy should be classified."""

        geometric = FakeInteraction(
            [0.0, 0.0, 2.1],
            shape=self.shape,
        )

        reason, distance, difference = self.classify(
            geometric
        )

        self.assertEqual(
            reason,
            "position_mismatch",
        )

        assert distance is not None
        assert difference is not None

        self.assertAlmostEqual(
            distance,
            2.1,
        )

        self.assertAlmostEqual(
            difference,
            0.1,
        )

    def test_small_offset_is_compatible(self) -> None:
        """Small floating-point offsets must be tolerated."""

        geometric = FakeInteraction(
            [0.0, 0.0, 2.0001],
            shape=self.shape,
        )

        reason, distance, difference = self.classify(
            geometric
        )

        self.assertIsNone(reason)

        assert distance is not None
        assert difference is not None

        self.assertAlmostEqual(
            distance,
            2.0001,
        )

        self.assertAlmostEqual(
            difference,
            0.0001,
        )


class TestFallbackCounters(unittest.TestCase):
    """Tests for emitter PDF fallback statistics."""

    def setUp(self) -> None:
        self.direction = np.array(
            [0.0, 0.0, 1.0],
            dtype=np.float64,
        )

    def make_query(
        self,
        reason: str,
    ) -> EmitterPdfQuery:
        """Create a representative fallback PDF query."""

        return EmitterPdfQuery(
            direction_world=self.direction,
            pdf=0.25,
            hit_emitter=True,
            environment=False,
            used_visibility_fallback=True,
            fallback_reason=reason,
            diagnostic={
                "reason": reason,
                "pdf_used": 0.25,
            },
        )

    def test_counts_reasons(self) -> None:
        """Fallback counters must track each cause separately."""

        stats = EmitterPdfStats()

        stats.record(
            self.make_query("geometric_miss")
        )

        stats.record(
            self.make_query("different_shape")
        )

        stats.record(
            self.make_query("different_shape")
        )

        result = stats.to_dict()

        self.assertEqual(
            result["total_queries"],
            3,
        )

        self.assertEqual(
            result["fallback_queries"],
            3,
        )

        reasons = result["fallback_reasons"]

        self.assertEqual(
            reasons["geometric_miss"],
            1,
        )

        self.assertEqual(
            reasons["different_shape"],
            2,
        )

        self.assertEqual(
            sum(reasons.values()),
            3,
        )

        self.assertAlmostEqual(
            result["fallback_rate"],
            1.0,
        )

        self.assertAlmostEqual(
            result["fallback_rate_among_area_hits"],
            1.0,
        )

    def test_limits_examples(self) -> None:
        """Only the configured number of examples is stored."""

        stats = EmitterPdfStats(
            max_diagnostic_examples=2,
        )

        for _ in range(5):
            stats.record(
                self.make_query("position_mismatch")
            )

        result = stats.to_dict()

        self.assertEqual(
            result["total_queries"],
            5,
        )

        self.assertEqual(
            result["fallback_queries"],
            5,
        )

        self.assertEqual(
            result["fallback_reasons"]["position_mismatch"],
            5,
        )

        self.assertEqual(
            len(result["fallback_examples"]),
            2,
        )

    def test_rejects_unknown_reason_without_incrementing(
        self,
    ) -> None:
        """Unknown reasons must not corrupt counters."""

        stats = EmitterPdfStats()

        with self.assertRaises(ValueError):
            stats.record(
                self.make_query("unknown_reason")
            )

        self.assertEqual(
            stats.total_queries,
            0,
        )

        self.assertEqual(
            stats.fallback_queries,
            0,
        )

    def test_rejects_conflicting_modes(self) -> None:
        """A query cannot use exact geometry and fallback together."""

        stats = EmitterPdfStats()

        query = EmitterPdfQuery(
            direction_world=self.direction,
            pdf=0.25,
            hit_emitter=True,
            environment=False,
            used_geometric_hit=True,
            used_visibility_fallback=True,
        )

        with self.assertRaises(ValueError):
            stats.record(query)

        self.assertEqual(
            stats.total_queries,
            0,
        )


if __name__ == "__main__":
    unittest.main()