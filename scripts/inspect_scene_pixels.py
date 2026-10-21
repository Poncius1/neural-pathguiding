"""Inspect visible surface pixels in a Mitsuba scene.

Target: Mitsuba 3.8.0, scalar_rgb.

Uses the same center-of-pixel surface sampling as the
fixed-point MIS experiment.

Example:
    python -u scripts/inspect_scene_pixels.py \
        --scene scenes/simple_diffuse/scene.xml \
        --grid-x 9 \
        --grid-y 9
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Literal, cast

import mitsuba as mi
import numpy as np

from scripts.compare_direct_mis_estimators import (
    sample_fixed_surface,
)
from neural_path_guiding.renderers.mitsuba.adapters import (
    mitsuba_vector_to_numpy,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect Mitsuba surface pixels."
    )

    parser.add_argument(
        "--scene",
        type=Path,
        default=Path("scenes/simple_diffuse/scene.xml"),
    )

    parser.add_argument(
        "--grid-x",
        type=int,
        default=9,
    )

    parser.add_argument(
        "--grid-y",
        type=int,
        default=9,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experiments"),
    )

    return parser.parse_args()


def grid_coordinates(
    size: int,
    count: int,
) -> list[int]:
    """Return unique pixel-center indices along one image axis."""

    if size <= 0 or count <= 0:
        raise ValueError(
            "Grid size and count must be positive."
        )

    values = np.linspace(
        0,
        size - 1,
        count,
    )

    return sorted({
        int(round(value))
        for value in values
    })


def vector_to_list(value: Any) -> list[float]:
    vector = np.asarray(
        mitsuba_vector_to_numpy(value),
        dtype=np.float64,
    ).reshape(-1)

    if vector.size != 3:
        raise ValueError(
            "Expected a three-dimensional vector."
        )

    if not np.all(np.isfinite(vector)):
        raise ValueError(
            "A scene intersection contains non-finite values."
        )

    return [float(component) for component in vector]


def object_id(value: Any) -> str:
    """Extract Mitsuba object identifiers safely."""

    if value is None:
        return "none"

    identifier = getattr(value, "id", None)

    if callable(identifier):
        result = identifier()

        if result:
            return str(result)

    class_method = getattr(value, "class_", None)

    if callable(class_method):
        cls = class_method()
        name_method = getattr(cls, "name", None)

        if callable(name_method):
            return str(name_method())

    return type(value).__name__


def inspect_pixel(
    scene: Any,
    x: int,
    y: int,
) -> dict[str, Any]:
    """Inspect a camera pixel's first surface intersection."""

    try:
        camera_ray, interaction, _ = sample_fixed_surface(
            scene,
            x,
            y,
        )

    except ValueError as exc:
        if "missed the scene" not in str(exc):
            raise

        return {
            "pixel_x": x,
            "pixel_y": y,
            "valid": False,
            "shape": "none",
            "bsdf": "none",
            "emissive": False,
            "position": None,
            "normal": None,
            "shading_normal": None,
            "view_direction": None,
        }

    shape = interaction.shape
    bsdf = interaction.bsdf()
    emitter = interaction.emitter(scene)

    return {
        "pixel_x": x,
        "pixel_y": y,
        "valid": True,
        "shape": object_id(shape),
        "bsdf": object_id(bsdf),
        "emissive": emitter is not None,
        "position": vector_to_list(interaction.p),
        "normal": vector_to_list(interaction.n),
        "shading_normal": vector_to_list(
            interaction.sh_frame.n
        ),
        "view_direction": vector_to_list(-camera_ray.d),
    }


def select_candidates(
    records: list[dict[str, Any]],
    *,
    maximum: int = 8,
) -> list[dict[str, Any]]:
    """Select candidates from different non-emissive shapes.

    This is a deterministic first-pass selection, not a
    learned-distribution or illumination quality ranking.
    """

    candidates = [
        record
        for record in records
        if (
            record["valid"]
            and not record["emissive"]
            and record["bsdf"] != "none"
        )
    ]

    selected: list[dict[str, Any]] = []
    used_shapes: set[str] = set()
    used_pixels: set[tuple[int, int]] = set()

    # Prefer geometrically distinct surfaces.
    for record in candidates:
        shape = record["shape"]

        if shape in used_shapes:
            continue

        selected.append(record)
        used_shapes.add(shape)
        used_pixels.add(
            (record["pixel_x"], record["pixel_y"])
        )

        if len(selected) >= maximum:
            return selected

    # Add more points from surfaces that are already represented.
    if len(selected) < maximum:
        remaining = [
            record
            for record in candidates
            if (
                record["pixel_x"],
                record["pixel_y"],
            ) not in used_pixels
        ]

        if remaining:
            required = min(
                maximum - len(selected),
                len(remaining),
            )

            indices = np.linspace(
                0,
                len(remaining) - 1,
                required,
            )

            for index in indices:
                record = remaining[int(round(index))]
                key = (
                    record["pixel_x"],
                    record["pixel_y"],
                )

                if key not in used_pixels:
                    selected.append(record)
                    used_pixels.add(key)

    return selected


def write_csv(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    fields = (
        "pixel_x",
        "pixel_y",
        "valid",
        "shape",
        "bsdf",
        "emissive",
        "position",
        "normal",
        "shading_normal",
        "view_direction",
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()

        for record in records:
            row = cast(
                dict[
                    Literal[
                        "pixel_x",
                        "pixel_y",
                        "valid",
                        "shape",
                        "bsdf",
                        "emissive",
                        "position",
                        "normal",
                        "shading_normal",
                        "view_direction",
                    ],
                    Any,
                ],
                dict(record),
            )

            for key in (
                "position",
                "normal",
                "shading_normal",
                "view_direction",
            ):
                if row[key] is not None:
                    row[key] = json.dumps(row[key])

            writer.writerow(row)


def main() -> None:
    args = parse_arguments()

    if args.grid_x <= 0 or args.grid_y <= 0:
        raise ValueError(
            "Grid dimensions must be positive."
        )

    if not args.scene.is_file():
        raise FileNotFoundError(args.scene)

    mi.set_variant("scalar_rgb")

    load_scene = cast(
        Callable[[str], Any],
        getattr(mi, "load_file"),
    )

    scene = load_scene(str(args.scene.resolve()))

    sensors = scene.sensors()

    if not sensors:
        raise ValueError(
            "The scene has no camera."
        )

    film_size = sensors[0].film().size()

    width = int(film_size[0])
    height = int(film_size[1])

    x_coords = grid_coordinates(
        width,
        args.grid_x,
    )

    y_coords = grid_coordinates(
        height,
        args.grid_y,
    )

    records: list[dict[str, Any]] = []

    print("Scene pixel inspection started.")
    print(f"Scene: {args.scene}")
    print(f"Film: {width} x {height}")

    for y in y_coords:
        for x in x_coords:
            records.append(
                inspect_pixel(scene, x, y)
            )

    candidates = select_candidates(records)

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )

    output = (
        args.output_dir
        / f"pixel_inspection_{timestamp}"
    )

    output.mkdir(
        parents=True,
        exist_ok=False,
    )

    write_csv(
        output / "scene_pixels.csv",
        records,
    )

    report = {
        "scene": str(args.scene.resolve()),
        "mitsuba_version": mi.__version__,
        "film_size": [width, height],
        "grid": [
            len(x_coords),
            len(y_coords),
        ],
        "total_pixels": len(records),
        "valid_intersections": sum(
            record["valid"]
            for record in records
        ),
        "shape_counts": dict(
            Counter(
                record["shape"]
                for record in records
                if record["valid"]
            )
        ),
        "candidates": candidates,
    }

    with (
        output / "inspection_summary.json"
    ).open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=2)

    print()
    print("=== VISIBLE SHAPES ===")

    for shape, count in report["shape_counts"].items():
        print(f"{shape}: {count}")

    print()
    print("=== PROPOSED PIXELS ===")

    for record in candidates:
        print(
            f"Pixel ({record['pixel_x']}, "
            f"{record['pixel_y']}) | "
            f"Shape: {record['shape']} | "
            f"BSDF: {record['bsdf']} | "
            f"Normal: {record['normal']}"
        )

    print()
    print(f"Results: {output.resolve()}")
    print("Scene inspection completed.")


if __name__ == "__main__":
    main()