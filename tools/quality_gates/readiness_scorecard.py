#!/usr/bin/env python3
"""Generate a machine-readable commercial-readiness scorecard."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET


REQUIRED_ROBOTS = ("franka", "ur5e", "ur10")
SCENE_ID_NULL_TOKENS = {"", "unknown", "none", "null", "n/a", "na"}

RELEASE_THRESHOLDS = {
    "certification_pass_rate_min": 0.98,
    "preprod_certification_pass_rate_min": 0.95,
    "import_quality_average_min": 0.85,
    "min_canonical_scenes": 12,
    "min_scene_families": 4,
    "min_robot_types": 3,
    "max_scene_contribution_ratio": 0.25,
    "min_per_robot_import_success": 0.90,
}


@dataclass
class GateResult:
    name: str
    passed: bool
    details: Dict[str, Any]
    blockers: List[str]


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


def _is_scene_id_valid(scene_id: Any) -> bool:
    if not isinstance(scene_id, str):
        return False
    return scene_id.strip().lower() not in SCENE_ID_NULL_TOKENS


def _latest_file(paths: Iterable[Path]) -> Optional[Path]:
    candidates = [path for path in paths if path.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.stat().st_mtime)


def _find_manifest_paths(repo_root: Path) -> List[Path]:
    patterns = (
        "scenes/**/episodes/**/import_manifest.json",
        "test_scenes/**/episodes/**/import_manifest.json",
    )
    manifests: List[Path] = []
    for pattern in patterns:
        manifests.extend(repo_root.glob(pattern))
    return sorted({path.resolve() for path in manifests})


def _infer_recordings_format(manifest: Dict[str, Any]) -> str:
    declared = manifest.get("recordings_format")
    if isinstance(declared, str) and declared.strip():
        return declared.strip().lower()
    checksums = manifest.get("checksums") or {}
    episodes = checksums.get("episodes")
    if isinstance(episodes, list):
        has_json = False
        has_parquet = False
        for entry in episodes:
            if not isinstance(entry, dict):
                continue
            file_name = str(entry.get("file_name") or "").lower()
            if file_name.endswith(".json"):
                has_json = True
            if file_name.endswith(".parquet"):
                has_parquet = True
        if has_json and has_parquet:
            return "mixed"
        if has_json:
            return "json"
        if has_parquet:
            return "parquet"
    return "unknown"


def _extract_manifest_scene_id(manifest: Dict[str, Any]) -> str:
    scene_id = manifest.get("scene_id")
    if _is_scene_id_valid(scene_id):
        return str(scene_id).strip()
    provenance = manifest.get("provenance")
    if isinstance(provenance, dict):
        prov_scene = provenance.get("scene_id")
        if _is_scene_id_valid(prov_scene):
            return str(prov_scene).strip()
    job_meta = (
        ((provenance or {}).get("config_snapshot") or {}).get("job_metadata")
        if isinstance(provenance, dict)
        else None
    )
    if isinstance(job_meta, dict):
        meta_scene = job_meta.get("scene_id")
        if _is_scene_id_valid(meta_scene):
            return str(meta_scene).strip()
    return ""


def _extract_manifest_robot_types(manifest: Dict[str, Any]) -> List[str]:
    robot_types = manifest.get("robot_types")
    if isinstance(robot_types, list):
        values = [str(item).strip().lower() for item in robot_types if str(item).strip()]
        if values:
            return sorted(set(values))
    provenance = manifest.get("provenance") or {}
    job_meta = (
        ((provenance or {}).get("config_snapshot") or {}).get("job_metadata")
        if isinstance(provenance, dict)
        else None
    )
    if isinstance(job_meta, dict):
        generation_params = job_meta.get("generation_params")
        if isinstance(generation_params, dict):
            values = generation_params.get("robot_types")
            if isinstance(values, list):
                parsed = [str(item).strip().lower() for item in values if str(item).strip()]
                if parsed:
                    return sorted(set(parsed))
            value = generation_params.get("robot_type")
            if isinstance(value, str) and value.strip():
                return [value.strip().lower()]
    return []


def collect_import_summary(repo_root: Path) -> Dict[str, Any]:
    manifests = _find_manifest_paths(repo_root)
    null_scene_paths: List[str] = []
    json_with_parquet_missing_errors: List[str] = []
    per_scene_passed: Dict[str, int] = defaultdict(int)
    robot_types: set[str] = set()

    total_passed_validation = 0
    quality_values: List[float] = []
    successful_manifest_count = 0

    for path in manifests:
        payload = _load_json(path)
        if not payload:
            continue
        scene_id = _extract_manifest_scene_id(payload)
        if not _is_scene_id_valid(scene_id):
            null_scene_paths.append(str(path))

        episodes = payload.get("episodes") or {}
        passed_validation = _safe_int(episodes.get("passed_validation"), 0)
        total_passed_validation += max(0, passed_validation)
        if _is_scene_id_valid(scene_id):
            per_scene_passed[scene_id] += max(0, passed_validation)

        quality = payload.get("quality") or {}
        quality_avg = _safe_float(quality.get("average_score"), 0.0)
        quality_values.append(quality_avg)
        if passed_validation > 0 and quality_avg >= RELEASE_THRESHOLDS["import_quality_average_min"]:
            successful_manifest_count += 1

        recording_format = _infer_recordings_format(payload)
        if recording_format in {"json", "mixed"}:
            validation = payload.get("validation") or {}
            episode_validation = validation.get("episodes") if isinstance(validation, dict) else {}
            episode_results = episode_validation.get("episode_results") if isinstance(episode_validation, dict) else []
            if isinstance(episode_results, list):
                for episode in episode_results:
                    if not isinstance(episode, dict):
                        continue
                    errors = episode.get("errors")
                    if not isinstance(errors, list):
                        continue
                    if any(".parquet" in str(err) and "not found" in str(err).lower() for err in errors):
                        json_with_parquet_missing_errors.append(str(path))
                        break

        robot_types.update(_extract_manifest_robot_types(payload))

    non_null_scene_count = len(manifests) - len(null_scene_paths)
    mean_quality = (
        round(sum(quality_values) / float(len(quality_values)), 4)
        if quality_values
        else 0.0
    )
    return {
        "manifest_count": len(manifests),
        "non_null_scene_count": non_null_scene_count,
        "null_scene_count": len(null_scene_paths),
        "null_scene_paths": sorted(null_scene_paths)[:50],
        "total_passed_validation": total_passed_validation,
        "mean_quality_score": mean_quality,
        "successful_manifest_count": successful_manifest_count,
        "json_with_parquet_missing_errors": sorted(set(json_with_parquet_missing_errors)),
        "accepted_episodes_by_scene": dict(sorted(per_scene_passed.items())),
        "robot_types_from_manifests": sorted(robot_types),
        "manifest_paths": [str(path) for path in manifests],
    }


def collect_quality_gate_summary(repo_root: Path) -> Dict[str, Any]:
    report_paths = sorted(
        set(repo_root.glob("scenes/**/quality_gates/quality_gate_report.json"))
        | set(repo_root.glob("test_scenes/**/quality_gates/quality_gate_report.json"))
    )
    total_gates = 0
    report_count = 0
    reports_with_coverage = 0
    empty_reports = 0
    report_versions: Counter[str] = Counter()

    for path in report_paths:
        payload = _load_json(path)
        if not payload:
            continue
        report_count += 1
        report_versions[str(payload.get("report_version") or "unknown")] += 1
        summary = payload.get("summary") or {}
        gates = _safe_int(summary.get("total_gates"), 0)
        total_gates += gates
        if gates <= 0:
            empty_reports += 1
        required = payload.get("required_checkpoints")
        executed = payload.get("executed_checkpoints")
        skipped = payload.get("skipped_checkpoints")
        if isinstance(required, list) and isinstance(executed, list) and isinstance(skipped, list):
            reports_with_coverage += 1

    checkpoint_files = sorted(
        set(repo_root.glob("scenes/**/.checkpoints/*.json"))
        | set(repo_root.glob("test_scenes/**/.checkpoints/*.json"))
    )
    skipped_entries: List[Dict[str, Any]] = []
    production_skips = 0
    for path in checkpoint_files:
        payload = _load_json(path)
        if not payload:
            continue
        outputs = payload.get("outputs") if isinstance(payload, dict) else {}
        if not isinstance(outputs, dict):
            outputs = {}
        skipped = bool(outputs.get("quality_gate_skipped", payload.get("quality_gate_skipped", False)))
        if not skipped:
            continue
        production_mode = bool(outputs.get("production_mode", payload.get("production_mode", False)))
        if production_mode:
            production_skips += 1
        skipped_entries.append(
            {
                "path": str(path),
                "production_mode": production_mode,
                "skip_reason": outputs.get("quality_gate_skip_reason") or payload.get("quality_gate_skip_reason"),
            }
        )

    return {
        "report_count": report_count,
        "total_gates": total_gates,
        "empty_reports": empty_reports,
        "reports_with_checkpoint_coverage": reports_with_coverage,
        "report_versions": dict(sorted(report_versions.items())),
        "checkpoint_skip_entries": skipped_entries,
        "production_skip_count": production_skips,
        "report_paths": [str(path) for path in report_paths],
    }


def collect_certification_summary(repo_root: Path) -> Dict[str, Any]:
    run_report_paths = list(repo_root.glob("analysis_outputs/**/run_certification_report.json"))
    slo_report_paths = list(repo_root.glob("analysis_outputs/**/certification_slo_gate.json"))
    run_report_path = _latest_file(run_report_paths)
    slo_report_path = _latest_file(slo_report_paths)

    run_report = _load_json(run_report_path) if run_report_path else None
    slo_report = _load_json(slo_report_path) if slo_report_path else None

    summary = run_report.get("summary") if isinstance(run_report, dict) else {}
    if not isinstance(summary, dict):
        summary = {}
    gate_hist = summary.get("gate_histogram")
    if not isinstance(gate_hist, dict):
        gate_hist = {}

    total_failures = sum(_safe_int(val, 0) for val in gate_hist.values())
    dominant_code = None
    dominant_ratio = 0.0
    if total_failures > 0:
        dominant_code, dominant_count = max(
            gate_hist.items(),
            key=lambda item: _safe_int(item[1], 0),
        )
        dominant_ratio = round(_safe_int(dominant_count, 0) / float(total_failures), 4)

    pass_rate = _safe_float(summary.get("certification_pass_rate"), 0.0)
    return {
        "latest_run_report": str(run_report_path) if run_report_path else None,
        "latest_slo_gate": str(slo_report_path) if slo_report_path else None,
        "episodes": _safe_int(summary.get("episodes"), 0),
        "certified": _safe_int(summary.get("certified"), 0),
        "certification_pass_rate": pass_rate,
        "gate_histogram": dict(sorted((str(k), _safe_int(v, 0)) for k, v in gate_hist.items())),
        "dominant_failure_code": dominant_code,
        "dominant_failure_ratio": dominant_ratio,
        "slo_gate": slo_report if isinstance(slo_report, dict) else None,
    }


def _extract_env_and_family(manifest: Dict[str, Any]) -> Tuple[str, str]:
    env = manifest.get("environment_type")
    family = manifest.get("scene_family")
    scene_info = manifest.get("scene")
    if not isinstance(env, str) and isinstance(scene_info, dict):
        env = scene_info.get("environment_type")
    if not isinstance(family, str):
        if isinstance(scene_info, dict):
            family = scene_info.get("family")
    env_norm = str(env).strip().lower() if isinstance(env, str) else ""
    family_norm = str(family).strip().lower() if isinstance(family, str) else ""
    if not family_norm and env_norm:
        family_norm = env_norm
    return env_norm, family_norm


def collect_scene_and_robot_summary(repo_root: Path, import_summary: Dict[str, Any]) -> Dict[str, Any]:
    scene_manifest_paths = sorted(
        set(repo_root.glob("scenes/**/assets/scene_manifest.json"))
        | set(repo_root.glob("test_scenes/**/assets/scene_manifest.json"))
    )
    canonical_scene_count = 0
    canonical_scene_ids: List[str] = []
    families: set[str] = set()
    env_types: set[str] = set()
    missing_metadata_paths: List[str] = []

    for path in scene_manifest_paths:
        payload = _load_json(path)
        if not payload:
            continue
        scene_id = str(payload.get("scene_id") or "").strip()
        env, family = _extract_env_and_family(payload)
        if env:
            env_types.add(env)
        if family:
            families.add(family)
        if not env or not family:
            missing_metadata_paths.append(str(path))
        if "scenes/" in str(path).replace("\\", "/"):
            canonical_scene_count += 1
            if scene_id:
                canonical_scene_ids.append(scene_id)

    robot_types = set(import_summary.get("robot_types_from_manifests", []))
    job_paths = sorted(
        set(repo_root.glob("scenes/**/geniesim/job.json"))
        | set(repo_root.glob("test_scenes/**/geniesim/job.json"))
    )
    for path in job_paths:
        payload = _load_json(path)
        if not payload:
            continue
        generation_params = payload.get("generation_params")
        if isinstance(generation_params, dict):
            values = generation_params.get("robot_types")
            if isinstance(values, list):
                for value in values:
                    robot_value = str(value).strip().lower()
                    if robot_value:
                        robot_types.add(robot_value)
            value = generation_params.get("robot_type")
            if isinstance(value, str) and value.strip():
                robot_types.add(value.strip().lower())

    accepted_by_scene = import_summary.get("accepted_episodes_by_scene") or {}
    total_accepted = sum(_safe_int(v, 0) for v in accepted_by_scene.values())
    max_scene_ratio = 0.0
    max_scene_id = None
    if total_accepted > 0:
        max_scene_id, max_count = max(
            accepted_by_scene.items(),
            key=lambda item: _safe_int(item[1], 0),
        )
        max_scene_ratio = round(_safe_int(max_count, 0) / float(total_accepted), 4)

    return {
        "scene_manifest_count": len(scene_manifest_paths),
        "canonical_scene_count": canonical_scene_count,
        "canonical_scene_ids": sorted(set(canonical_scene_ids)),
        "scene_families": sorted(families),
        "environment_types": sorted(env_types),
        "missing_scene_family_metadata_paths": sorted(missing_metadata_paths),
        "robot_types": sorted(robot_types),
        "accepted_episode_total": total_accepted,
        "max_scene_contribution": {
            "scene_id": max_scene_id,
            "ratio": max_scene_ratio,
        },
        "scene_manifest_paths": [str(path) for path in scene_manifest_paths],
        "job_metadata_paths": [str(path) for path in job_paths],
    }


def collect_production_validation_summary(repo_root: Path) -> Dict[str, Any]:
    paths = sorted(
        set(repo_root.glob("scenes/**/production_validation.json"))
        | set(repo_root.glob("test_scenes/**/production_validation.json"))
    )
    total = 0
    production_true = 0
    ok_count = 0
    with_errors: List[str] = []
    for path in paths:
        payload = _load_json(path)
        if not payload:
            continue
        total += 1
        if bool(payload.get("production_mode", False)):
            production_true += 1
        if bool(payload.get("ok", False)):
            ok_count += 1
        errors = payload.get("errors")
        if isinstance(errors, list) and errors:
            with_errors.append(str(path))
    return {
        "report_count": total,
        "production_mode_true_count": production_true,
        "ok_count": ok_count,
        "reports_with_errors": with_errors,
        "report_paths": [str(path) for path in paths],
    }


def collect_runtime_summary(repo_root: Path) -> Dict[str, Any]:
    summary_path = repo_root / "analysis_outputs" / "runtime_slo_summary.json"
    payload = _load_json(summary_path) if summary_path.is_file() else None
    if not payload:
        return {
            "summary_path": str(summary_path),
            "available": False,
            "stages_with_distributions": 0,
            "complete": False,
            "details": {},
        }
    stages = payload.get("stages")
    stage_count = len(stages) if isinstance(stages, dict) else 0
    complete = bool(payload.get("complete", False))
    return {
        "summary_path": str(summary_path),
        "available": True,
        "stages_with_distributions": stage_count,
        "complete": complete,
        "details": payload,
    }


def collect_ci_summary(repo_root: Path) -> Dict[str, Any]:
    junit_paths = sorted(repo_root.glob("junit*.xml"))
    junit_paths.extend(sorted(repo_root.glob("logs/ci/**/pytest-junit.xml")))

    entrypoint_seen = False
    entrypoint_failures = 0
    for path in junit_paths:
        try:
            tree = ET.parse(path)
        except Exception:
            continue
        root = tree.getroot()
        for testcase in root.iter("testcase"):
            name = str(testcase.attrib.get("name") or "")
            classname = str(testcase.attrib.get("classname") or "")
            if "test_pipeline_data_flow_entrypoints" not in f"{classname}.{name}":
                continue
            entrypoint_seen = True
            if testcase.find("failure") is not None or testcase.find("error") is not None:
                entrypoint_failures += 1

    state = "unknown"
    if entrypoint_seen and entrypoint_failures == 0:
        state = "pass"
    elif entrypoint_seen and entrypoint_failures > 0:
        state = "fail"

    canary_path = repo_root / "analysis_outputs" / "canary_stability_gate.json"
    canary_payload = _load_json(canary_path) if canary_path.is_file() else None
    canary_stable = bool((canary_payload or {}).get("stable_7_day", False))

    return {
        "entrypoint_test_state": state,
        "entrypoint_test_seen": entrypoint_seen,
        "entrypoint_test_failures": entrypoint_failures,
        "junit_paths_scanned": [str(path) for path in junit_paths],
        "canary_stability_path": str(canary_path),
        "canary_stable_7_day": canary_stable,
        "canary_summary": canary_payload,
    }


def _gate_certification(cert: Dict[str, Any]) -> GateResult:
    blockers: List[str] = []
    pass_rate = _safe_float(cert.get("certification_pass_rate"), 0.0)
    if pass_rate < RELEASE_THRESHOLDS["certification_pass_rate_min"]:
        blockers.append(
            f"certification_pass_rate {pass_rate:.4f} below {RELEASE_THRESHOLDS['certification_pass_rate_min']:.4f}"
        )
    dominant_ratio = _safe_float(cert.get("dominant_failure_ratio"), 0.0)
    if dominant_ratio >= 0.5 and cert.get("dominant_failure_code"):
        blockers.append(
            f"dominant critical gate code {cert['dominant_failure_code']} ratio={dominant_ratio:.2f}"
        )
    if _safe_int(cert.get("episodes"), 0) <= 0:
        blockers.append("no certification episodes found")
    return GateResult(
        name="certification",
        passed=not blockers,
        details={
            "pass_rate": pass_rate,
            "episodes": _safe_int(cert.get("episodes"), 0),
            "dominant_failure_code": cert.get("dominant_failure_code"),
            "dominant_failure_ratio": dominant_ratio,
        },
        blockers=blockers,
    )


def _gate_import_viability(imports: Dict[str, Any]) -> GateResult:
    blockers: List[str] = []
    if _safe_int(imports.get("manifest_count"), 0) <= 0:
        blockers.append("no import manifests found")
    if _safe_int(imports.get("null_scene_count"), 0) > 0:
        blockers.append(f"{imports['null_scene_count']} manifests have null/unknown scene_id")
    if _safe_int(imports.get("total_passed_validation"), 0) <= 0:
        blockers.append("episodes.passed_validation is 0")
    if _safe_float(imports.get("mean_quality_score"), 0.0) < RELEASE_THRESHOLDS["import_quality_average_min"]:
        blockers.append(
            f"mean quality {imports['mean_quality_score']:.4f} below "
            f"{RELEASE_THRESHOLDS['import_quality_average_min']:.2f}"
        )
    parquet_mismatch = imports.get("json_with_parquet_missing_errors") or []
    if parquet_mismatch:
        blockers.append("JSON-backed runs still contain missing-parquet validation errors")
    return GateResult(
        name="import_viability",
        passed=not blockers,
        details={
            "manifest_count": _safe_int(imports.get("manifest_count"), 0),
            "null_scene_count": _safe_int(imports.get("null_scene_count"), 0),
            "total_passed_validation": _safe_int(imports.get("total_passed_validation"), 0),
            "mean_quality_score": _safe_float(imports.get("mean_quality_score"), 0.0),
            "successful_manifest_count": _safe_int(imports.get("successful_manifest_count"), 0),
        },
        blockers=blockers,
    )


def _gate_quality_enforcement(quality: Dict[str, Any]) -> GateResult:
    blockers: List[str] = []
    if _safe_int(quality.get("report_count"), 0) <= 0:
        blockers.append("no quality gate reports found")
    if _safe_int(quality.get("total_gates"), 0) <= 0:
        blockers.append("quality gate reports contain zero evaluated gates")
    if _safe_int(quality.get("reports_with_checkpoint_coverage"), 0) <= 0:
        blockers.append("reports missing required/executed/skipped checkpoint coverage fields")
    if _safe_int(quality.get("production_skip_count"), 0) > 0:
        blockers.append("quality_gate_skipped=true observed in production-mode checkpoint outputs")
    return GateResult(
        name="quality_gate_enforcement",
        passed=not blockers,
        details={
            "report_count": _safe_int(quality.get("report_count"), 0),
            "total_gates": _safe_int(quality.get("total_gates"), 0),
            "reports_with_checkpoint_coverage": _safe_int(
                quality.get("reports_with_checkpoint_coverage"), 0
            ),
            "production_skip_count": _safe_int(quality.get("production_skip_count"), 0),
        },
        blockers=blockers,
    )


def _gate_diversity(scene_summary: Dict[str, Any]) -> GateResult:
    blockers: List[str] = []
    canonical_scenes = _safe_int(scene_summary.get("canonical_scene_count"), 0)
    family_count = len(scene_summary.get("scene_families") or [])
    robot_types = set(scene_summary.get("robot_types") or [])
    required_robots = set(REQUIRED_ROBOTS)
    max_scene_ratio = _safe_float(
        ((scene_summary.get("max_scene_contribution") or {}).get("ratio")),
        0.0,
    )

    if canonical_scenes < RELEASE_THRESHOLDS["min_canonical_scenes"]:
        blockers.append(
            f"canonical scene count {canonical_scenes} < {RELEASE_THRESHOLDS['min_canonical_scenes']}"
        )
    if family_count < RELEASE_THRESHOLDS["min_scene_families"]:
        blockers.append(
            f"scene family count {family_count} < {RELEASE_THRESHOLDS['min_scene_families']}"
        )
    if len(robot_types) < RELEASE_THRESHOLDS["min_robot_types"]:
        blockers.append(
            f"robot type count {len(robot_types)} < {RELEASE_THRESHOLDS['min_robot_types']}"
        )
    missing_required = sorted(required_robots - robot_types)
    if missing_required:
        blockers.append(f"missing required robots: {', '.join(missing_required)}")
    if (
        scene_summary.get("accepted_episode_total", 0) > 0
        and max_scene_ratio > RELEASE_THRESHOLDS["max_scene_contribution_ratio"]
    ):
        blockers.append(
            f"scene contribution ratio {max_scene_ratio:.2f} exceeds "
            f"{RELEASE_THRESHOLDS['max_scene_contribution_ratio']:.2f}"
        )

    return GateResult(
        name="diversity",
        passed=not blockers,
        details={
            "canonical_scene_count": canonical_scenes,
            "scene_families": scene_summary.get("scene_families") or [],
            "robot_types": sorted(robot_types),
            "max_scene_contribution_ratio": max_scene_ratio,
        },
        blockers=blockers,
    )


def _gate_production_evidence(production: Dict[str, Any], runtime: Dict[str, Any]) -> GateResult:
    blockers: List[str] = []
    if _safe_int(production.get("production_mode_true_count"), 0) <= 0:
        blockers.append("no production_mode=true validation artifact found")
    if _safe_int(production.get("ok_count"), 0) <= 0:
        blockers.append("no successful production validation artifact found")
    if not bool(runtime.get("available")):
        blockers.append("runtime SLO summary missing")
    elif not bool(runtime.get("complete")):
        blockers.append("runtime SLO summary incomplete (missing distribution/timeout coverage)")

    return GateResult(
        name="production_evidence",
        passed=not blockers,
        details={
            "production_report_count": _safe_int(production.get("report_count"), 0),
            "production_mode_true_count": _safe_int(production.get("production_mode_true_count"), 0),
            "ok_count": _safe_int(production.get("ok_count"), 0),
            "runtime_summary_available": bool(runtime.get("available")),
            "runtime_summary_complete": bool(runtime.get("complete")),
        },
        blockers=blockers,
    )


def _gate_ci_reliability(ci: Dict[str, Any]) -> GateResult:
    blockers: List[str] = []
    state = str(ci.get("entrypoint_test_state") or "unknown")
    if state != "pass":
        blockers.append("test_pipeline_data_flow_entrypoints is not confirmed green")
    if not bool(ci.get("canary_stable_7_day")):
        blockers.append("7-day nightly canary stability gate not satisfied")

    return GateResult(
        name="ci_reliability",
        passed=not blockers,
        details={
            "entrypoint_test_state": state,
            "canary_stable_7_day": bool(ci.get("canary_stable_7_day")),
        },
        blockers=blockers,
    )


def _phase_statuses(
    repo_root: Path,
    gate_results: Sequence[GateResult],
    imports: Dict[str, Any],
    quality: Dict[str, Any],
    cert: Dict[str, Any],
    ci: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    gate_by_name = {gate.name: gate for gate in gate_results}
    ci_wired = "readiness_scorecard.py" in (repo_root / ".github/workflows/test-unit.yml").read_text()
    thresholds_doc = (repo_root / "docs/COMMERCIAL_READINESS_GATES.md").is_file()
    phase0_passed = ci_wired and thresholds_doc

    phase1_blockers = []
    if _safe_int(imports.get("null_scene_count"), 0) > 0:
        phase1_blockers.append("null/unknown scene_id still present in manifests")
    if _safe_int(imports.get("total_passed_validation"), 0) <= 0:
        phase1_blockers.append("no run evidence with episodes.passed_validation > 0")
    if imports.get("json_with_parquet_missing_errors"):
        phase1_blockers.append("JSON runs still failing parquet existence checks")

    phase2_blockers = []
    if _safe_int(quality.get("production_skip_count"), 0) > 0:
        phase2_blockers.append("production checkpoint outputs still show quality_gate_skipped=true")
    if _safe_int(quality.get("reports_with_checkpoint_coverage"), 0) <= 0:
        phase2_blockers.append("checkpoint coverage fields missing from gate reports")
    if _safe_int(quality.get("total_gates"), 0) <= 0:
        phase2_blockers.append("gate reports have zero evaluated gates")

    cert_rate = _safe_float(cert.get("certification_pass_rate"), 0.0)
    phase3_passed = cert_rate >= RELEASE_THRESHOLDS["preprod_certification_pass_rate_min"]
    phase6_blockers = []
    if ci.get("entrypoint_test_state") != "pass":
        phase6_blockers.append("pipeline data-flow entrypoint test not green")
    if not bool(ci.get("canary_stable_7_day")):
        phase6_blockers.append("7-day canary stability signal missing or failing")

    return {
        "phase_0_controls": {
            "passed": phase0_passed,
            "blockers": [] if phase0_passed else [
                "readiness scorecard generation not wired in CI"
                if not ci_wired
                else "thresholds document missing"
            ],
        },
        "phase_1_contract": {
            "passed": len(phase1_blockers) == 0,
            "blockers": phase1_blockers,
        },
        "phase_2_quality_enforcement": {
            "passed": len(phase2_blockers) == 0,
            "blockers": phase2_blockers,
        },
        "phase_3_certification": {
            "passed": phase3_passed,
            "blockers": [] if phase3_passed else [
                f"certification pass rate {cert_rate:.4f} below preprod target "
                f"{RELEASE_THRESHOLDS['preprod_certification_pass_rate_min']:.4f}"
            ],
        },
        "phase_4_diversity": {
            "passed": gate_by_name["diversity"].passed,
            "blockers": gate_by_name["diversity"].blockers,
        },
        "phase_5_production_e2e": {
            "passed": gate_by_name["production_evidence"].passed,
            "blockers": gate_by_name["production_evidence"].blockers,
        },
        "phase_6_reliability": {
            "passed": len(phase6_blockers) == 0,
            "blockers": phase6_blockers,
        },
    }


def build_scorecard(repo_root: Path) -> Dict[str, Any]:
    import_summary = collect_import_summary(repo_root)
    quality_summary = collect_quality_gate_summary(repo_root)
    cert_summary = collect_certification_summary(repo_root)
    scene_summary = collect_scene_and_robot_summary(repo_root, import_summary)
    production_summary = collect_production_validation_summary(repo_root)
    runtime_summary = collect_runtime_summary(repo_root)
    ci_summary = collect_ci_summary(repo_root)

    gates = [
        _gate_certification(cert_summary),
        _gate_import_viability(import_summary),
        _gate_quality_enforcement(quality_summary),
        _gate_diversity(scene_summary),
        _gate_production_evidence(production_summary, runtime_summary),
        _gate_ci_reliability(ci_summary),
    ]
    passed_release = sum(1 for gate in gates if gate.passed)
    release_score = int(round((passed_release / float(len(gates))) * 100)) if gates else 0

    phases = _phase_statuses(
        repo_root,
        gates,
        import_summary,
        quality_summary,
        cert_summary,
        ci_summary,
    )
    completed_phases = sum(1 for phase in phases.values() if phase.get("passed"))
    pipeline_score = int(round((completed_phases / float(len(phases))) * 100)) if phases else 0

    outstanding = []
    for gate in gates:
        if gate.passed:
            continue
        for blocker in gate.blockers:
            outstanding.append(f"{gate.name}: {blocker}")

    return {
        "generated_at": _utc_now(),
        "repo_root": str(repo_root),
        "release_thresholds": RELEASE_THRESHOLDS,
        "scores": {
            "commercial_readiness_score": release_score,
            "pipeline_maturity_score": pipeline_score,
            "release_gates_passed": passed_release,
            "release_gates_total": len(gates),
            "phases_completed": completed_phases,
            "phases_total": len(phases),
        },
        "release_gates": [
            {
                "name": gate.name,
                "passed": gate.passed,
                "details": gate.details,
                "blockers": gate.blockers,
            }
            for gate in gates
        ],
        "phases": phases,
        "evidence": {
            "import": import_summary,
            "quality_gates": quality_summary,
            "certification": cert_summary,
            "diversity": scene_summary,
            "production_validation": production_summary,
            "runtime": runtime_summary,
            "ci": ci_summary,
        },
        "outstanding_work": outstanding,
    }


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Generate commercial-readiness scorecard JSON.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=repo_root,
        help="Repository root (defaults to current script repository).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "analysis_outputs" / "readiness_scorecard.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when any release gate is failing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scorecard = build_scorecard(repo_root)
    output_path.write_text(json.dumps(scorecard, indent=2))
    print(f"[readiness-scorecard] wrote {output_path}")

    failing = [
        gate
        for gate in scorecard.get("release_gates", [])
        if not bool(gate.get("passed", False))
    ]
    if args.strict and failing:
        print("[readiness-scorecard] failing gates:")
        for gate in failing:
            print(f"  - {gate.get('name')}: {'; '.join(gate.get('blockers') or [])}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
