#!/usr/bin/env python3
"""Re-certify existing episode JSON files with the latest physics certification logic.

Usage:
    python scripts/recertify_episodes.py <run_dir> [--out-dir OUT_DIR] [--mode MODE] [--normalize] [--write-normalized-episodes]

Examples:
    # Flat run dir with task_*_ep*.json at top-level
    python scripts/recertify_episodes.py /path/to/run_92e1e2a647e8/

    # Downloaded runs (episodes nested under per-task folders)
    python scripts/recertify_episodes.py downloaded_episodes/run18

This script:
1. Finds episode JSONs matching `task_*_ep*.json` (top-level, with recursive fallback).
2. Optionally normalizes each episode in-memory for certification (fills sparse
   ee_vel/ee_acc, derives object_poses from privileged state, etc).
3. Runs tools/quality_gates/physics_certification.py with the current env vars.
4. Writes updated certification reports under OUT_DIR (default: <run_dir>/recertified).

The original episode files are NOT modified.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure repo root is on PYTHONPATH
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))

from tools.quality_gates.physics_certification import (
    GATE_VERSION,
    run_episode_certification,
    write_run_certification_report,
)
from tools.quality_gates.episode_normalization import normalize_episode_for_certification


def _load_episode(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _extract_task(episode: Dict[str, Any]) -> Dict[str, Any]:
    """Build a task dict from episode metadata (the task isn't stored separately)."""
    target = episode.get("target_object") or episode.get("target_object_id")
    return {
        "task_name": episode.get("task_name"),
        "task_type": episode.get("task_type"),
        "target_object": target,
        "target_object_id": target,
        "goal_region": None,  # Not stored in episode; milestones computed from frames
        "requires_object_motion": episode.get("task_type", "").lower()
        in ("pick_place", "pick", "place", "stack", "organize", "grasp", "interact", "lift", "transport"),
    }


def _load_env_file(path: Path) -> None:
    """Source a KEY=value env file into os.environ (ignoring comments/blanks)."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            os.environ[key] = value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=str, help="Directory containing (or containing subdirs with) task_*_ep*.json")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for recertification reports (default: <run_dir>/recertified)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=("strict", "compat"),
        default="strict",
        help="Physics certification mode (default: strict).",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help=(
            "Normalize episodes in-memory before certification (fills sparse channels, derives object_poses "
            "from privileged state, etc). This is useful for legacy/raw episodes but will change what is "
            "being evaluated."
        ),
    )
    parser.add_argument(
        "--write-normalized-episodes",
        action="store_true",
        help="Also write normalized episode JSONs under OUT_DIR/normalized_episodes/ (originals are untouched)",
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=None,
        help="Path to a KEY=value env file to source before certification (e.g. configs/realism_strict.env)",
    )
    args = parser.parse_args()

    if args.env_file:
        env_path = Path(args.env_file).resolve()
        if not env_path.is_file():
            print(f"ERROR: env file {env_path} not found")
            return 1
        _load_env_file(env_path)
        print(f"Sourced env file: {env_path}")

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        print(f"ERROR: {run_dir} is not a directory")
        return 1

    if args.write_normalized_episodes and not args.normalize:
        print("ERROR: --write-normalized-episodes requires --normalize")
        return 2

    # Find episode files (top-level with recursive fallback for downloaded runs).
    episode_files = sorted(p for p in run_dir.glob("task_*_ep*.json") if "recertified" not in p.parts)
    if not episode_files:
        episode_files = sorted(p for p in run_dir.rglob("task_*_ep*.json") if "recertified" not in p.parts)
        if episode_files:
            print(f"No top-level episodes found; using recursive search under {run_dir}")
    if not episode_files:
        print(f"No episode files matching task_*_ep*.json found under {run_dir}")
        return 1

    print(f"Found {len(episode_files)} episode(s) in {run_dir}")
    print(f"Using physics certification gate version: {GATE_VERSION}")
    print(f"Mode: {args.mode}")
    print(f"Normalize: {args.normalize}")
    print()

    episode_reports: List[Dict[str, Any]] = []
    normalized_out_dir = None
    if args.write_normalized_episodes:
        out_dir = Path(args.out_dir).resolve() if args.out_dir else (run_dir / "recertified")
        normalized_out_dir = out_dir / "normalized_episodes"
        normalized_out_dir.mkdir(parents=True, exist_ok=True)

    for ep_path in episode_files:
        episode = _load_episode(ep_path)
        if args.normalize:
            normalize_episode_for_certification(episode)
        task = _extract_task(episode)
        frames = episode.get("frames", [])
        if not isinstance(frames, list) or not frames:
            print(f"  {ep_path.name}: ❌ FAIL (no frames)")
            continue
        if normalized_out_dir is not None:
            out_path = normalized_out_dir / ep_path.name
            with open(out_path, "w") as f:
                json.dump(episode, f)

        # Run certification
        cert = run_episode_certification(
            frames=frames,
            episode_meta=episode,
            task=task,
            mode=args.mode,
        )

        # Print per-episode result
        status = "✅ PASS" if cert["passed"] else "❌ FAIL"
        failures = cert.get("critical_failures", [])
        metrics = cert.get("metrics", {})
        print(f"  {ep_path.name}: {status}")
        if failures:
            print(f"    Failures: {', '.join(failures)}")
        print(f"    server_backed_ratio: {metrics.get('scene_state_server_backed_ratio', 'N/A')}")
        print(f"    kinematic_ratio: {metrics.get('kinematic_pose_frame_ratio', 'N/A')}")
        print(f"    stale_effort_ratio: {metrics.get('stale_effort_pair_ratio', 'N/A')}")
        print(f"    valid_contact_frames: {metrics.get('valid_contact_frames', 'N/A')}")
        print(f"    target_mass_present: {metrics.get('target_mass_kg_present', 'N/A')}")
        print(f"    target_schema: {metrics.get('target_schema_completeness', 'N/A')}")
        print()

        # Build report entry matching the original report format
        episode_reports.append({
            "episode_id": episode.get("episode_id"),
            "task_name": episode.get("task_name"),
            "robot_type": episode.get("robot_metadata", {}).get("robot_type", "unknown"),
            "dataset_tier": cert.get("dataset_tier"),
            "certification": cert,
        })

    # Write updated reports
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (run_dir / "recertified")
    report = write_run_certification_report(out_dir, episode_reports)

    summary = report.get("summary", {})
    total = summary.get("episodes", 0)
    certified = summary.get("certified", 0)
    pass_rate = summary.get("certification_pass_rate", 0.0)
    gate_hist = summary.get("gate_histogram", {})

    print("=" * 60)
    print(f"RE-CERTIFICATION SUMMARY")
    print(f"  Episodes: {total}")
    print(f"  Certified: {certified}/{total} ({pass_rate*100:.1f}%)")
    print(f"  Gate histogram: {json.dumps(gate_hist, indent=4)}")
    print(f"  Reports written to: {out_dir}")
    print("=" * 60)

    # Compare with original
    orig_report_path = run_dir / "run_certification_report.json"
    if orig_report_path.exists():
        with open(orig_report_path) as f:
            orig = json.load(f)
        orig_summary = orig.get("summary", {})
        orig_pass = orig_summary.get("certification_pass_rate", 0.0)
        orig_hist = orig_summary.get("gate_histogram", {})
        print()
        print("COMPARISON (original → re-certified):")
        print(f"  Pass rate: {orig_pass*100:.1f}% → {pass_rate*100:.1f}%")
        # Show gate changes
        all_gates = sorted(set(list(orig_hist.keys()) + list(gate_hist.keys())))
        for gate in all_gates:
            old_count = orig_hist.get(gate, 0)
            new_count = gate_hist.get(gate, 0)
            delta = new_count - old_count
            arrow = "↓" if delta < 0 else ("↑" if delta > 0 else "=")
            print(f"  {gate}: {old_count} → {new_count} {arrow}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
