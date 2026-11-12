"""Aggregate independent transport renders without re-rendering.

Usage:
    python -m scripts.summarize_transport_runs --runs outputs/experiments/transport_controls

Each selected directory must contain manifest.json from
scripts.render_transport_controls. Results are grouped by experiment settings,
scene/checkpoint/reference hashes, and the set of methods. Only complete groups
with at least two distinct seeds are summarized. This program intentionally
reads *recorded* per-run RGB MSEs; it does not interpret MSE as variance.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any

MODES = ("path", "bsdf", "uniform", "neural")
PAIRS = (("neural", "bsdf"), ("neural", "path"), ("uniform", "bsdf"))


def _load_manifest(path: Path) -> dict[str, Any] | None:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("experiment") != "single_loop_transport_controls":
        return None
    if not isinstance(data.get("settings"), dict) or not isinstance(data.get("results"), dict):
        raise ValueError(f"Malformed transport manifest: {path}")
    return data


def _signature(data: dict[str, Any]) -> tuple[Any, ...]:
    settings = data["settings"]
    return (
        data.get("scene_sha256"),
        data.get("checkpoint_sha256"),
        data.get("reference_sha256"),
        settings.get("width"), settings.get("height"), settings.get("spp"),
        settings.get("max_depth"), settings.get("rr_depth"),
        settings.get("guiding_probability"), settings.get("uniform_mix"),
        tuple(sorted(data["results"])),
    )


def summarize_records(
    records: list[tuple[Path, dict[str, Any]]],
    *,
    duplicate_policy: str = "error",
) -> dict[str, Any]:
    """Summarize a single compatible group of manifests."""
    if len(records) < 2:
        raise ValueError("At least two independent seeds are needed")
    if duplicate_policy not in ("error", "latest"):
        raise ValueError("Unknown duplicate policy")
    seed_to_entry: dict[int, tuple[Path, dict[str, Any]]] = {}
    discarded: list[str] = []
    signature = _signature(records[0][1])
    for path, data in records:
        if _signature(data) != signature:
            raise ValueError(f"Incompatible experiment settings in {path}")
        seed = data["settings"].get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError(f"Invalid seed in {path}")
        if seed in seed_to_entry:
            if duplicate_policy == "error":
                raise ValueError(f"Duplicate seed {seed}: {path} and {seed_to_entry[seed][0]}")
            # The glob is sorted by run timestamp. Explicit latest policy
            # replaces the older run, without treating it as a new seed.
            discarded.append(str(seed_to_entry[seed][0]))
        seed_to_entry[seed] = (path, data)

    methods = sorted(records[0][1]["results"])
    seed_order = sorted(seed_to_entry)
    methods_summary: dict[str, Any] = {}
    values_by_method: dict[str, list[float]] = {}
    for mode in methods:
        mses: list[float] = []
        times: list[float] = []
        for seed in seed_order:
            entry = seed_to_entry[seed][1]["results"][mode]
            if "rgb_mse" not in entry:
                raise ValueError("MSE absent: provide a high-spp reference for every run")
            mse = float(entry["rgb_mse"])
            seconds = float(entry["seconds"])
            if not (0 <= mse < float("inf") and 0 < seconds < float("inf")):
                raise ValueError(f"Invalid MSE/time for {mode}, seed={seed}")
            mses.append(mse)
            times.append(seconds)
        values_by_method[mode] = mses
        methods_summary[mode] = {
            "mse_mean": mean(mses),
            "mse_sample_std": stdev(mses),
            "seconds_mean": mean(times),
            "seconds_sample_std": stdev(times),
            "per_seed_mse": dict(zip(map(str, seed_order), mses)),
        }
    paired: dict[str, Any] = {}
    for a, b in PAIRS:
        if a in values_by_method and b in values_by_method:
            diffs = [x - y for x, y in zip(values_by_method[a], values_by_method[b])]
            paired[f"{a}_minus_{b}"] = {
                "mean_mse_difference": mean(diffs),
                "sample_std_difference": stdev(diffs),
                "wins_a": sum(x < 0 for x in diffs),
                "ties": sum(x == 0 for x in diffs),
                "n": len(diffs),
            }
    return {
        "n_independent_seeds": len(seed_order),
        "duplicate_policy": duplicate_policy,
        "discarded_duplicate_manifests": discarded,
        "seeds": seed_order,
        "settings": records[0][1]["settings"] | {"seed": None},
        "scene_sha256": records[0][1].get("scene_sha256"),
        "checkpoint_sha256": records[0][1].get("checkpoint_sha256"),
        "reference_sha256": records[0][1].get("reference_sha256"),
        "methods": methods_summary,
        "paired_differences": paired,
        "interpretation_limits": [
            "MSE against a finite-spp reference is not a direct variance estimate",
            "Five seeds are too few to claim small differences statistically significant",
            "Equal spp does not imply equal render time or sampling cost",
            "Per-run sample std measures variability of MSE, not per-pixel Monte Carlo variance",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True,
                        help="Directory containing run_*/manifest.json")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--duplicate-policy", choices=("error", "latest"),
                        default="error", help="Explicitly choose latest run per seed, or fail")
    args = parser.parse_args()
    paths = sorted(args.runs.glob("run_*/manifest.json"))
    groups: dict[tuple[Any, ...], list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    for path in paths:
        data = _load_manifest(path)
        if data is not None:
            groups[_signature(data)].append((path, data))
    complete = [(key, entries) for key, entries in groups.items() if len(entries) >= 2]
    if not complete:
        raise SystemExit("No compatible group of two or more transport runs found")
    report = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input_directory": str(args.runs.resolve()),
        "groups": [],
    }
    for index, (_, entries) in enumerate(complete, start=1):
        summary = summarize_records(entries, duplicate_policy=args.duplicate_policy)
        summary["group_id"] = index
        summary["manifests"] = [str(p.resolve()) for p, _ in entries
                                if str(p) not in summary["discarded_duplicate_manifests"]]
        report["groups"].append(summary)
        print(f"Group {index}: {summary['n_independent_seeds']} seeds")
        for mode, values in summary["methods"].items():
            print(f"  {mode:<8} MSE={values['mse_mean']:.8e} ± {values['mse_sample_std']:.2e}  time={values['seconds_mean']:.3f}s")
        for name, values in summary["paired_differences"].items():
            print(f"  {name}: ΔMSE={values['mean_mse_difference']:+.3e} wins={values['wins_a']}/{values['n']}")
    destination = args.output or args.runs / "transport_summary.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved: {destination.resolve()}")


if __name__ == "__main__":
    main()
