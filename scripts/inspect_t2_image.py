
"""Inspect first neural T2 v2 EXR render outputs."""

import argparse
from pathlib import Path

import mitsuba as mi
import numpy as np

mi.set_variant("scalar_rgb")


def inspect(path: Path) -> np.ndarray:
    image = np.asarray(mi.Bitmap(str(path)), dtype=np.float64)

    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"Unexpected EXR shape: {image.shape}")

    rgb = image[:, :, :3]

    print(f"\n{path.name}")
    print(f"  Shape: {rgb.shape}")
    print(f"  Finite: {bool(np.isfinite(rgb).all())}")
    print(f"  Minimum: {rgb.min():.8f}")
    print(f"  Maximum: {rgb.max():.8f}")
    print(f"  Mean RGB: {rgb.mean(axis=(0, 1))}")
    print(f"  Zero pixels: {int(np.all(rgb == 0, axis=2).sum())}")
    print(f"  Negative values: {int((rgb < 0).sum())}")

    if not np.isfinite(rgb).all():
        raise ValueError("Non-finite image values detected")

    return rgb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    baseline = inspect(args.run_dir / "baseline_path.exr")
    neural = inspect(args.run_dir / "neural_t2_multibounce.exr")

    if baseline.shape != neural.shape:
        raise ValueError("Image dimensions do not match")

    difference = neural - baseline

    print("\n=== IMAGE COMPARISON ===")
    print(f"Mean absolute difference: {np.mean(np.abs(difference)):.8f}")
    print(f"Maximum absolute difference: {np.max(np.abs(difference)):.8f}")
    print(f"Baseline mean luminance: {np.mean(baseline @ [0.2126, 0.7152, 0.0722]):.8f}")
    print(f"Neural mean luminance: {np.mean(neural @ [0.2126, 0.7152, 0.0722]):.8f}")
    print("\nDiagnostic completed.")


if __name__ == "__main__":
    main()
