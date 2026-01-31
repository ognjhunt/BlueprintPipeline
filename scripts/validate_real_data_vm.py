#!/usr/bin/env python3
"""
Validate BlueprintPipeline Real Data Usage on VM.

Runs provenance analysis on pipeline outputs to measure real vs heuristic/synthetic
data percentages. Exits non-zero if below threshold.

Usage:
    python scripts/validate_real_data_vm.py \
        --scene-dir ./test_scenes/scenes/lightwheel_kitchen \
        --output-report ./validation_report.json \
        --threshold 95.0

Environment:
    GENIESIM_HOST, GENIESIM_PORT, GEMINI_API_KEY required for pipeline run.
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Provenance scoring weights — higher = more "real"
PROVENANCE_SCORING = {
    "physx_server": 1.0,
    "geniesim_server": 1.0,
    "isaac_sim_camera": 1.0,
    "isaac_sim_replicator": 1.0,
    "simulation": 1.0,
    "object_data": 0.95,
    "scene_graph": 0.95,
    "metadata": 0.9,
    "obb": 0.9,
    "gemini_estimated": 0.85,
    "gemini_calibrated": 0.85,
    "gemini": 0.85,
    "deterministic_default": 0.5,
    "cubic_spline_interpolation": 0.7,
    "finite_difference": 0.7,
    "geometric_goal_region_v2": 0.8,
    "heuristic_default": 0.3,
    "hardcoded_fallback": 0.2,
    "hardcoded_default": 0.2,
    "input_fallback": 0.3,
    "default_placeholder": 0.2,
    "default": 0.2,
    "heuristic_grasp_model_v1": 0.3,
    "synthetic_from_task_config": 0.1,
    "synthetic_fallback": 0.0,
    "synthetic": 0.0,
    "unavailable": 0.0,
}


def score_source(source: str) -> float:
    """Score a provenance source label."""
    if source in PROVENANCE_SCORING:
        return PROVENANCE_SCORING[source]
    # Heuristic matching for unknown labels
    s = source.lower()
    if "gemini" in s:
        return 0.85
    if "physx" in s or "isaac" in s or "server" in s:
        return 1.0
    if "heuristic" in s or "hardcoded" in s or "fallback" in s:
        return 0.3
    if "synthetic" in s:
        return 0.0
    return 0.5  # Unknown — neutral


def extract_provenance_labels(data: Any, labels: Optional[Dict[str, int]] = None) -> Dict[str, int]:
    """Recursively extract all provenance/source labels from nested data."""
    if labels is None:
        labels = defaultdict(int)

    if isinstance(data, dict):
        # Known provenance keys
        for key in (
            "provenance", "estimation_source", "data_source", "source",
            "dimensions_source", "calibration_source", "action_source",
            "velocity_source", "diversity_calibration",
            "reward_weights_source", "reward_thresholds_source",
        ):
            val = data.get(key)
            if isinstance(val, str) and val:
                labels[val] += 1
            elif isinstance(val, dict):
                for _channel, _src in val.items():
                    if isinstance(_src, str) and _src:
                        labels[_src] += 1

        # Check object_property_provenance dict
        opp = data.get("object_property_provenance")
        if isinstance(opp, dict):
            for _key, _src in opp.items():
                if isinstance(_src, str):
                    labels[_src] += 1

        # Recurse
        for v in data.values():
            extract_provenance_labels(v, labels)

    elif isinstance(data, list):
        for item in data:
            extract_provenance_labels(item, labels)

    return dict(labels)


def compute_real_data_stats(provenance_counts: Dict[str, int]) -> Dict[str, Any]:
    """Compute real-data percentage from provenance counts."""
    total = sum(provenance_counts.values())
    if total == 0:
        return {"real_data_pct": 0.0, "total_labels": 0, "warning": "no provenance labels found"}

    weighted_sum = sum(count * score_source(src) for src, count in provenance_counts.items())
    real_pct = (weighted_sum / total) * 100.0

    high_quality = sum(
        count for src, count in provenance_counts.items()
        if score_source(src) >= 0.85
    )
    low_quality = sum(
        count for src, count in provenance_counts.items()
        if score_source(src) < 0.5
    )

    return {
        "real_data_pct": round(real_pct, 2),
        "total_labels": total,
        "high_quality_labels": high_quality,
        "high_quality_pct": round((high_quality / total) * 100, 2),
        "low_quality_labels": low_quality,
        "low_quality_pct": round((low_quality / total) * 100, 2),
        "breakdown": dict(sorted(provenance_counts.items())),
    }


def scan_json_files(directory: Path) -> Dict[str, int]:
    """Scan all JSON files in a directory tree for provenance labels."""
    all_labels: Dict[str, int] = defaultdict(int)
    json_count = 0
    for json_file in directory.rglob("*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
            labels = extract_provenance_labels(data)
            for src, count in labels.items():
                all_labels[src] += count
            json_count += 1
        except (json.JSONDecodeError, OSError):
            continue
    logger.info("Scanned %d JSON files in %s", json_count, directory)
    return dict(all_labels)


def validate_scene_outputs(scene_dir: Path) -> Dict[str, Any]:
    """Validate all pipeline outputs in a scene directory."""
    results = {}

    # Scan known output subdirectories
    for subdir_name in ["simready", "episodes", "usd", "replicator", "geniesim_export"]:
        subdir = scene_dir / subdir_name
        if subdir.exists():
            labels = scan_json_files(subdir)
            if labels:
                results[subdir_name] = compute_real_data_stats(labels)

    # Also scan scene_manifest.json at root
    manifest = scene_dir / "scene_manifest.json"
    if manifest.exists():
        try:
            with open(manifest) as f:
                data = json.load(f)
            labels = extract_provenance_labels(data)
            if labels:
                results["scene_manifest"] = compute_real_data_stats(labels)
        except (json.JSONDecodeError, OSError):
            pass

    # Scan root-level JSON outputs
    root_labels = defaultdict(int)
    for json_file in scene_dir.glob("*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
            labels = extract_provenance_labels(data)
            for src, count in labels.items():
                root_labels[src] += count
        except (json.JSONDecodeError, OSError):
            continue
    if root_labels:
        results["root_metadata"] = compute_real_data_stats(dict(root_labels))

    # Compute overall
    if results:
        all_labels: Dict[str, int] = defaultdict(int)
        for stage_stats in results.values():
            for src, count in stage_stats.get("breakdown", {}).items():
                all_labels[src] += count
        results["overall"] = compute_real_data_stats(dict(all_labels))

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene-dir", type=Path, required=True, help="Scene output directory to validate")
    parser.add_argument("--output-report", type=Path, default=Path("./validation_report.json"), help="Output report path")
    parser.add_argument("--threshold", type=float, default=95.0, help="Minimum real data percentage to pass")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not args.scene_dir.exists():
        logger.error("Scene directory does not exist: %s", args.scene_dir)
        return 1

    logger.info("Validating pipeline outputs in %s", args.scene_dir)
    results = validate_scene_outputs(args.scene_dir)

    if not results:
        logger.error("No provenance data found in %s", args.scene_dir)
        return 1

    # Write report
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_report, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Report written to %s", args.output_report)

    # Print summary
    overall = results.get("overall", {})
    real_pct = overall.get("real_data_pct", 0.0)
    total = overall.get("total_labels", 0)
    print(f"\n{'=' * 60}")
    print(f"REAL DATA VALIDATION REPORT")
    print(f"{'=' * 60}")
    for stage, stats in results.items():
        if stage == "overall":
            continue
        print(f"  {stage:20s}: {stats.get('real_data_pct', 0):.1f}% real ({stats.get('total_labels', 0)} labels)")
    print(f"{'=' * 60}")
    print(f"  {'OVERALL':20s}: {real_pct:.1f}% real ({total} labels)")
    print(f"  Threshold          : {args.threshold:.1f}%")
    print(f"  Result             : {'PASS' if real_pct >= args.threshold else 'FAIL'}")
    print(f"{'=' * 60}")

    if real_pct < args.threshold:
        # Show low-quality breakdown
        breakdown = overall.get("breakdown", {})
        low_sources = {src: count for src, count in breakdown.items() if score_source(src) < 0.5}
        if low_sources:
            print(f"\nLow-quality sources dragging down score:")
            for src, count in sorted(low_sources.items(), key=lambda x: -x[1]):
                print(f"  {src}: {count} labels (score={score_source(src):.1f})")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
