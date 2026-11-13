"""Diagnose area-emitter PDFs, orientation and emitted radiance.

Mitsuba 3.8.0, scalar_rgb.

Reads fallback_examples from a three-technique MIS manifest.
Replays each stored world-space direction from the original
camera pixel's first surface interaction.

This script does not modify the MIS estimator.

Run:
    python -m scripts.diagnose_area_light_orientation \
        --manifest PATH_TO_MANIFEST_JSON
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from collections.abc import Callable
from pathlib import Path
import json
from typing import Any, cast

import mitsuba as mi
import numpy as np

from neural_path_guiding.evaluation.direct_estimators import (
    evaluate_direct_integrand,
    sample_fixed_surface,
)
from neural_path_guiding.renderers.mitsuba.adapters import (
    mitsuba_vector_to_numpy,
    normalize_numpy_vector,
)
from neural_path_guiding.renderers.mitsuba.radiance import (
    mitsuba_spectrum_to_rgb,
)


LUMINANCE_WEIGHTS = np.array(
    [0.2126, 0.7152, 0.0722],
    dtype=np.float64,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose emitter orientation and PDF "
            "for recorded fallback directions."
        )
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )

    return parser.parse_args()


def classify_light_query(
    *,
    pdf: float,
    emitted_luminance: float,
    tolerance: float = 1e-9,
) -> str:
    """Classify the combination of emission and PDF.

    This classification identifies cases needing attention.
    It does not by itself prove mathematical correctness.
    """

    if (
        not np.isfinite(pdf)
        or not np.isfinite(emitted_luminance)
    ):
        raise ValueError(
            "PDF and emitted luminance must be finite."
        )

    if pdf < 0.0:
        raise ValueError("PDF must be non-negative.")

    if emitted_luminance < -tolerance:
        raise ValueError(
            "Emitted luminance is unexpectedly negative."
        )

    if tolerance < 0.0:
        raise ValueError(
            "Tolerance must be non-negative."
        )

    has_pdf = pdf > tolerance
    has_emission = emitted_luminance > tolerance

    if not has_pdf and not has_emission:
        return "zero_pdf_zero_emission"

    if not has_pdf and has_emission:
        return "zero_pdf_positive_emission"

    if has_pdf and not has_emission:
        return "positive_pdf_zero_emission"

    return "positive_pdf_positive_emission"


def vector3(value: Any) -> np.ndarray:
    """Read a finite three-dimensional vector."""

    result = np.asarray(
        mitsuba_vector_to_numpy(value),
        dtype=np.float64,
    ).reshape(-1)

    if result.shape != (3,):
        raise ValueError(
            "Expected a three-dimensional vector."
        )

    if not np.all(np.isfinite(result)):
        raise ValueError(
            "Encountered a non-finite vector."
        )

    return result


def rgb3(value: Any) -> np.ndarray:
    rgb = np.asarray(
        mitsuba_spectrum_to_rgb(value),
        dtype=np.float64,
    ).reshape(-1)

    if rgb.shape != (3,):
        raise ValueError(
            "Expected RGB radiance with shape (3,)."
        )

    if not np.all(np.isfinite(rgb)):
        raise ValueError(
            "Encountered non-finite radiance."
        )

    return rgb


def object_id(value: Any) -> str:
    if value is None:
        return "none"

    method = getattr(value, "id", None)

    if callable(method):
        identifier = str(method())

        if identifier:
            return identifier

    return type(value).__name__


def read_fallback_examples(
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    """Read diagnostics from the current or older manifest layout."""

    top_level = manifest.get(
        "emitter_pdf_diagnostics",
        {},
    )

    if not top_level:
        top_level = (
            manifest.get("estimators", {})
            .get("neural_bsdf_light_mis_3", {})
            .get("emitter_pdf_diagnostics", {})
        )

    examples = top_level.get(
        "fallback_examples",
        [],
    )

    if not isinstance(examples, list):
        raise ValueError(
            "fallback_examples must be a list."
        )

    return examples


def diagnose_direction(
    *,
    scene: Any,
    interaction: Any,
    example: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    """Replay a recorded fallback direction and inspect its emitter."""

    direction = normalize_numpy_vector(
        np.asarray(
            example["direction_world"],
            dtype=np.float64,
        )
    )

    direction_mi = mi.Vector3f(
        float(direction[0]),
        float(direction[1]),
        float(direction[2]),
    )

    visibility_ray = interaction.spawn_ray(
        direction_mi
    )

    visibility_hit = scene.ray_intersect(
        visibility_ray
    )

    result: dict[str, Any] = {
        "example_index": index,
        "fallback_reason": example.get("reason"),
        "direction_world": direction.tolist(),
        "recorded_pdf": float(
            example.get("pdf_used", 0.0)
        ),
        "recorded_geometric_distance": example.get(
            "geometric_hit_distance"
        ),
        "visibility_valid": bool(
            visibility_hit.is_valid()
        ),
        "visibility_shape": "none",
        "visibility_position": None,
        "geometric_normal": None,
        "front_facing_dot": None,
        "is_front_facing": None,
        "emitter_present": False,
        "emitted_rgb": None,
        "emitted_luminance": None,
        "reconstructed_pdf": None,
        "direct_integrand_luminance": None,
        "classification": "no_surface_hit",
    }

    if not bool(visibility_hit.is_valid()):
        return result

    result["visibility_shape"] = object_id(
        visibility_hit.shape
    )

    result["visibility_position"] = vector3(
        visibility_hit.p
    ).tolist()

    normal = vector3(visibility_hit.n)

    result["geometric_normal"] = normal.tolist()

    # Incoming direction at the emitter points towards
    # the original shading point.
    incoming_to_emitter = -direction

    front_dot = float(
        np.dot(normal, incoming_to_emitter)
    )

    result["front_facing_dot"] = front_dot
    result["is_front_facing"] = front_dot > 0.0

    emitter = visibility_hit.emitter(scene)

    if emitter is None:
        result["classification"] = "non_emissive_surface"
        return result

    result["emitter_present"] = True

    emitted_rgb = rgb3(
        emitter.eval(visibility_hit)
    )

    emitted_luminance = float(
        emitted_rgb @ LUMINANCE_WEIGHTS
    )

    direction_sample = mi.DirectionSample3f(
        scene,
        si=visibility_hit,
        ref=interaction,
    )

    reconstructed_pdf = float(
        scene.pdf_emitter_direction(
            interaction,
            direction_sample,
            True,
        )
    )

    if (
        not np.isfinite(reconstructed_pdf)
        or reconstructed_pdf < 0.0
    ):
        raise ValueError(
            "Invalid reconstructed emitter PDF."
        )

    direct_integrand = np.asarray(
        evaluate_direct_integrand(
            scene,
            interaction,
            direction,
        ),
        dtype=np.float64,
    )

    result["emitted_rgb"] = emitted_rgb.tolist()
    result["emitted_luminance"] = emitted_luminance
    result["reconstructed_pdf"] = reconstructed_pdf
    result["direct_integrand_luminance"] = float(
        direct_integrand @ LUMINANCE_WEIGHTS
    )

    result["classification"] = classify_light_query(
        pdf=reconstructed_pdf,
        emitted_luminance=emitted_luminance,
    )

    return result


def write_csv(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    if not records:
        return

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(records[0].keys()),
        )

        writer.writeheader()

        for record in records:
            row = dict(record)

            for key, value in row.items():
                if isinstance(value, (list, dict)):
                    row[key] = json.dumps(value)

            writer.writerow(row)


def main() -> None:
    args = parse_arguments()

    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)

    with args.manifest.open(
        "r",
        encoding="utf-8",
    ) as file:
        manifest = json.load(file)

    examples = read_fallback_examples(manifest)

    if not examples:
        raise ValueError(
            "The manifest contains no fallback examples."
        )

    scene_path = Path(manifest["scene"])

    if not scene_path.is_file():
        raise FileNotFoundError(scene_path)

    pixel_x, pixel_y = (
        int(value)
        for value in manifest["pixel"]
    )

    mi.set_variant("scalar_rgb")

    load_scene = cast(
        Callable[[str], Any],
        getattr(mi, "load_file"),
    )

    scene = load_scene(
        str(scene_path.resolve())
    )

    _, interaction, _ = sample_fixed_surface(
        scene,
        pixel_x,
        pixel_y,
    )

    records = [
        diagnose_direction(
            scene=scene,
            interaction=interaction,
            example=example,
            index=index,
        )
        for index, example in enumerate(
            examples,
            start=1,
        )
    ]

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else args.manifest.parent / "area_light_diagnostics"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    classification_counts = dict(
        Counter(
            record["classification"]
            for record in records
        )
    )

    summary = {
        "experiment": "area_emitter_fallback_orientation_diagnostic",
        "source_manifest": str(
            args.manifest.resolve()
        ),
        "scene": str(
            scene_path.resolve()
        ),
        "pixel": [pixel_x, pixel_y],
        "examined_examples": len(records),
        "classification_counts": classification_counts,
        "records": records,
        "limitations": [
            "Only previously captured fallback examples are examined.",
            "A zero PDF and zero emission do not alone prove unbiased MIS.",
            "Area emitters may be one-sided.",
            "No delta emitters or participating media.",
        ],
    }

    write_csv(
        output_dir / "orientation_diagnostics.csv",
        records,
    )

    with (
        output_dir / "orientation_diagnostics.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2)

    print()
    print("=== AREA LIGHT ORIENTATION DIAGNOSTICS ===")
    print(f"Pixel: ({pixel_x}, {pixel_y})")
    print(f"Examples: {len(records)}")

    for record in records:
        print(
            f"#{record['example_index']:02d} "
            f"shape={record['visibility_shape']} "
            f"front_dot={record['front_facing_dot']} "
            f"pdf={record['reconstructed_pdf']} "
            f"emission={record['emitted_luminance']} "
            f"class={record['classification']}"
        )

    print()
    print("=== CLASSIFICATION SUMMARY ===")

    for name, count in classification_counts.items():
        print(f"{name}: {count}")

    print()
    print(f"Results: {output_dir.resolve()}")
    print("Diagnostic completed.")


if __name__ == "__main__":
    main()