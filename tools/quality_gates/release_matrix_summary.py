#!/usr/bin/env python3
"""Summarize diversity/repeatability matrix evidence from repo artifacts."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


THRESHOLDS = {
    "min_scene_count": 12,
    "min_family_count": 4,
    "min_robot_count": 3,
    "max_scene_share_ratio": 0.25,
    "min_per_robot_import_success": 0.90,
    "preprod_certification_pass_rate_min": 0.95,
}


@dataclass
class SceneRecord:
    scene_id: str
    scene_dir: Path
    environment_type: str
    scene_family: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _scene_records(repo_root: Path) -> List[SceneRecord]:
    records: List[SceneRecord] = []
    patterns = (
        "scenes/**/assets/scene_manifest.json",
        "test_scenes/scenes/**/assets/scene_manifest.json",
    )
    for pattern in patterns:
        for path in sorted(repo_root.glob(pattern)):
            payload = _load_json(path)
            if not payload:
                continue
            scene_id = str(payload.get("scene_id") or "").strip()
            if not scene_id:
                continue
            env = payload.get("environment_type")
            family = payload.get("scene_family")
            scene_data = payload.get("scene")
            if not isinstance(env, str) and isinstance(scene_data, dict):
                env = scene_data.get("environment_type")
            if not isinstance(family, str):
                if isinstance(scene_data, dict):
                    family = scene_data.get("family")
            env_norm = str(env).strip().lower() if isinstance(env, str) else ""
            family_norm = str(family).strip().lower() if isinstance(family, str) else ""
            if not family_norm and env_norm:
                family_norm = env_norm
            records.append(
                SceneRecord(
                    scene_id=scene_id,
                    scene_dir=path.parents[1],
                    environment_type=env_norm,
                    scene_family=family_norm,
                )
            )
    return records


def _collect_scene_import_metrics(scene_dir: Path) -> Tuple[int, float]:
    manifests = sorted(scene_dir.glob("episodes/**/import_manifest.json"))
    accepted = 0
    quality_values: List[float] = []
    for path in manifests:
        payload = _load_json(path)
        if not payload:
            continue
        episodes = payload.get("episodes") or {}
        accepted += max(0, _safe_int(episodes.get("passed_validation"), 0))
        quality = payload.get("quality") or {}
        quality_values.append(_safe_float(quality.get("average_score"), 0.0))
    quality_average = (
        round(sum(quality_values) / float(len(quality_values)), 4)
        if quality_values
        else 0.0
    )
    return accepted, quality_average


def _collect_scene_robot_metrics(scene_dir: Path) -> Dict[str, Dict[str, float]]:
    job_path = scene_dir / "geniesim" / "job.json"
    payload = _load_json(job_path)
    if not payload:
        return {}
    robot_metrics = payload.get("job_metrics_by_robot")
    out: Dict[str, Dict[str, float]] = {}
    if isinstance(robot_metrics, dict):
        for robot, metric in robot_metrics.items():
            if not isinstance(metric, dict):
                continue
            total = _safe_int(metric.get("episodes_collected"), _safe_int(metric.get("total_episodes"), 0))
            passed = _safe_int(metric.get("episodes_passed"), 0)
            rate = (passed / float(total)) if total > 0 else 0.0
            out[str(robot).strip().lower()] = {
                "episodes_total": total,
                "episodes_passed": passed,
                "import_success_rate": round(rate, 4),
            }
    generation = payload.get("generation_params")
    if isinstance(generation, dict):
        robots = generation.get("robot_types")
        if isinstance(robots, list):
            for robot in robots:
                robot_key = str(robot).strip().lower()
                if robot_key and robot_key not in out:
                    out[robot_key] = {
                        "episodes_total": 0,
                        "episodes_passed": 0,
                        "import_success_rate": 0.0,
                    }
        robot = generation.get("robot_type")
        if isinstance(robot, str):
            robot_key = robot.strip().lower()
            if robot_key and robot_key not in out:
                out[robot_key] = {
                    "episodes_total": 0,
                    "episodes_passed": 0,
                    "import_success_rate": 0.0,
                }
    return out


def _collect_scene_gate_skip_rate(scene_dir: Path) -> float:
    checkpoints = sorted((scene_dir / ".checkpoints").glob("*.json"))
    if not checkpoints:
        return 0.0
    total = 0
    skipped = 0
    for path in checkpoints:
        payload = _load_json(path)
        if not payload:
            continue
        total += 1
        outputs = payload.get("outputs")
        if isinstance(outputs, dict) and bool(outputs.get("quality_gate_skipped")):
            skipped += 1
        elif bool(payload.get("quality_gate_skipped")):
            skipped += 1
    if total <= 0:
        return 0.0
    return round(skipped / float(total), 4)


def _latest_certification_pass_rate(repo_root: Path) -> float:
    reports = [path for path in repo_root.glob("analysis_outputs/**/run_certification_report.json") if path.is_file()]
    if not reports:
        return 0.0
    latest = max(reports, key=lambda item: item.stat().st_mtime)
    payload = _load_json(latest)
    if not payload:
        return 0.0
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        return 0.0
    return _safe_float(summary.get("certification_pass_rate"), 0.0)


def build_summary(repo_root: Path) -> Dict[str, Any]:
    scenes = _scene_records(repo_root)
    scene_entries: List[Dict[str, Any]] = []
    total_accepted = 0
    families: set[str] = set()
    robots: set[str] = set()
    per_robot_totals: Dict[str, Dict[str, int]] = defaultdict(lambda: {"episodes_total": 0, "episodes_passed": 0})

    for record in scenes:
        accepted, quality_average = _collect_scene_import_metrics(record.scene_dir)
        robot_metrics = _collect_scene_robot_metrics(record.scene_dir)
        gate_skip_rate = _collect_scene_gate_skip_rate(record.scene_dir)

        total_accepted += accepted
        if record.scene_family:
            families.add(record.scene_family)
        for robot, metrics in robot_metrics.items():
            robots.add(robot)
            per_robot_totals[robot]["episodes_total"] += _safe_int(metrics.get("episodes_total"), 0)
            per_robot_totals[robot]["episodes_passed"] += _safe_int(metrics.get("episodes_passed"), 0)

        scene_entries.append(
            {
                "scene_id": record.scene_id,
                "scene_dir": str(record.scene_dir),
                "environment_type": record.environment_type,
                "scene_family": record.scene_family,
                "accepted_episodes": accepted,
                "quality_average_score": quality_average,
                "gate_skip_rate": gate_skip_rate,
                "robots": robot_metrics,
            }
        )

    for entry in scene_entries:
        accepted = _safe_int(entry.get("accepted_episodes"), 0)
        entry["accepted_share_ratio"] = (
            round(accepted / float(total_accepted), 4) if total_accepted > 0 else 0.0
        )

    per_robot_success = {}
    for robot, totals in per_robot_totals.items():
        episodes_total = _safe_int(totals.get("episodes_total"), 0)
        episodes_passed = _safe_int(totals.get("episodes_passed"), 0)
        per_robot_success[robot] = (
            round(episodes_passed / float(episodes_total), 4) if episodes_total > 0 else 0.0
        )

    max_scene_ratio = max((entry["accepted_share_ratio"] for entry in scene_entries), default=0.0)
    cert_pass_rate = _latest_certification_pass_rate(repo_root)

    thresholds = THRESHOLDS.copy()
    checks = {
        "scene_count_ok": len(scene_entries) >= thresholds["min_scene_count"],
        "family_count_ok": len(families) >= thresholds["min_family_count"],
        "robot_count_ok": len(robots) >= thresholds["min_robot_count"],
        "scene_balance_ok": max_scene_ratio <= thresholds["max_scene_share_ratio"],
        "per_robot_import_success_ok": all(
            rate >= thresholds["min_per_robot_import_success"] for rate in per_robot_success.values()
        )
        and len(per_robot_success) > 0,
        "preprod_certification_ok": cert_pass_rate >= thresholds["preprod_certification_pass_rate_min"],
    }
    checks["matrix_ready"] = all(checks.values())

    return {
        "generated_at": _utc_now(),
        "repo_root": str(repo_root),
        "thresholds": thresholds,
        "summary": {
            "scene_count": len(scene_entries),
            "scene_families": sorted(families),
            "robot_types": sorted(robots),
            "total_accepted_episodes": total_accepted,
            "max_scene_contribution_ratio": round(max_scene_ratio, 4),
            "per_robot_import_success": dict(sorted(per_robot_success.items())),
            "preprod_certification_pass_rate": round(cert_pass_rate, 4),
        },
        "checks": checks,
        "scenes": sorted(scene_entries, key=lambda item: item["scene_id"]),
    }


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Build release matrix summary JSON.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=repo_root,
        help="Repository root path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "analysis_outputs" / "release_matrix_summary.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when matrix checks are not satisfied.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    summary = build_summary(repo_root)
    output.write_text(json.dumps(summary, indent=2))
    print(f"[release-matrix-summary] wrote {output}")

    if args.strict and not bool((summary.get("checks") or {}).get("matrix_ready", False)):
        print("[release-matrix-summary] matrix checks failed")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
