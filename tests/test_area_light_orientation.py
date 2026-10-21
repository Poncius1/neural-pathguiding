"""Tests for area-emitter orientation diagnostic helpers."""

from __future__ import annotations

import unittest

from scripts.diagnose_area_light_orientation import (
    classify_light_query,
    read_fallback_examples,
)


class TestAreaLightOrientation(unittest.TestCase):

    def test_zero_pdf_zero_emission(self) -> None:
        result = classify_light_query(
            pdf=0.0,
            emitted_luminance=0.0,
        )

        self.assertEqual(
            result,
            "zero_pdf_zero_emission",
        )

    def test_zero_pdf_positive_emission(self) -> None:
        result = classify_light_query(
            pdf=0.0,
            emitted_luminance=2.0,
        )

        self.assertEqual(
            result,
            "zero_pdf_positive_emission",
        )

    def test_positive_pdf_zero_emission(self) -> None:
        result = classify_light_query(
            pdf=0.5,
            emitted_luminance=0.0,
        )

        self.assertEqual(
            result,
            "positive_pdf_zero_emission",
        )

    def test_positive_pdf_positive_emission(self) -> None:
        result = classify_light_query(
            pdf=0.5,
            emitted_luminance=2.0,
        )

        self.assertEqual(
            result,
            "positive_pdf_positive_emission",
        )

    def test_rejects_negative_pdf(self) -> None:
        with self.assertRaises(ValueError):
            classify_light_query(
                pdf=-0.1,
                emitted_luminance=1.0,
            )

    def test_reads_top_level_examples(self) -> None:
        manifest = {
            "emitter_pdf_diagnostics": {
                "fallback_examples": [
                    {"reason": "different_shape"},
                ],
            }
        }

        examples = read_fallback_examples(manifest)

        self.assertEqual(len(examples), 1)
        self.assertEqual(
            examples[0]["reason"],
            "different_shape",
        )

    def test_reads_nested_examples(self) -> None:
        manifest = {
            "estimators": {
                "neural_bsdf_light_mis_3": {
                    "emitter_pdf_diagnostics": {
                        "fallback_examples": [
                            {"reason": "geometric_miss"},
                        ],
                    }
                }
            }
        }

        examples = read_fallback_examples(manifest)

        self.assertEqual(len(examples), 1)
        self.assertEqual(
            examples[0]["reason"],
            "geometric_miss",
        )


if __name__ == "__main__":
    unittest.main()