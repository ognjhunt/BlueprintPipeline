#!/usr/bin/env python3
"""Genie Sim readiness probe with strict runtime patch checks."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

try:
    import grpc
except ImportError as exc:  # pragma: no cover - import failure path
    print(f"grpc import failed: {exc}", file=sys.stderr)
    sys.exit(1)


DEFAULT_PATCH_MARKERS: Tuple[Tuple[str, str], ...] = (
    ("source/data_collection/server/grpc_server.py", "BlueprintPipeline contact_report patch"),
    ("source/data_collection/server/command_controller.py", "BlueprintPipeline object_pose patch"),
    ("source/data_collection/server/command_controller.py", "BlueprintPipeline sim_thread_physics_cache patch"),
)

STRICT_REQUIRED_PATCH_MARKERS: Tuple[Tuple[str, str], ...] = (
    ("source/data_collection/server/grpc_server.py", "BlueprintPipeline contact_report patch"),
    ("source/data_collection/server/grpc_server.py", "BlueprintPipeline joint_efforts"),
    ("source/data_collection/server/grpc_server.py", "BPv_dynamic_grasp_toggle"),
    ("source/data_collection/server/command_controller.py", "BlueprintPipeline contact_reporting_on_init patch"),
    ("source/data_collection/server/command_controller.py", "BlueprintPipeline sim_thread_physics_cache patch"),
    ("source/data_collection/server/command_controller.py", "BPv3_pre_play_kinematic"),
    ("source/data_collection/server/command_controller.py", "BPv4_deferred_dynamic_restore"),
    ("source/data_collection/server/command_controller.py", "BPv5_dynamic_teleport_usd_objects"),
    ("source/data_collection/server/command_controller.py", "BPv6_fix_dynamic_prims"),
    ("source/data_collection/server/command_controller.py", "BPv_dynamic_grasp_toggle"),
    ("source/data_collection/server/command_controller.py", "[PATCH] scene_collision_injected"),
    ("source/data_collection/server/command_controller.py", "object_pose_resolver_v4"),
)

STRICT_FORBIDDEN_PATCH_MARKERS: Tuple[Tuple[str, str], ...] = (
    ("source/data_collection/server/command_controller.py", "BPv7_keep_kinematic"),
)

PATCH_MARKER_SETS: Dict[str, Tuple[Tuple[str, str], ...]] = {
    "default": DEFAULT_PATCH_MARKERS,
    "strict": STRICT_REQUIRED_PATCH_MARKERS,
}

FORBIDDEN_PATCH_MARKER_SETS: Dict[str, Tuple[Tuple[str, str], ...]] = {
    "strict": STRICT_FORBIDDEN_PATCH_MARKERS,
}


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_csv(value: str | None) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def check_grpc_ready(host: str, port: str, timeout_s: float) -> Tuple[bool, str]:
    channel = grpc.insecure_channel(f"{host}:{port}")
    try:
        grpc.channel_ready_future(channel).result(timeout=timeout_s)
        return True, "grpc channel ready"
    except grpc.FutureTimeoutError:
        return False, f"grpc channel timeout ({timeout_s:.1f}s)"
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"grpc readiness exception: {exc}"
    finally:
        channel.close()


def _parse_marker_specs(specs: Sequence[str]) -> List[Tuple[str, str]]:
    parsed: List[Tuple[str, str]] = []
    for spec in specs:
        if "::" not in spec:
            continue
        rel_path, marker = spec.split("::", 1)
        rel_path = rel_path.strip()
        marker = marker.strip()
        if rel_path and marker:
            parsed.append((rel_path, marker))
    return parsed


def _expand_marker_sets(
    set_names: Sequence[str],
    marker_sets: Dict[str, Tuple[Tuple[str, str], ...]],
) -> List[Tuple[str, str]]:
    expanded: List[Tuple[str, str]] = []
    for raw_name in set_names:
        name = str(raw_name or "").strip().lower()
        if not name:
            continue
        entries = marker_sets.get(name)
        if not entries:
            continue
        expanded.extend(entries)
    return expanded


def check_patch_markers(
    geniesim_root: Path,
    extra_marker_specs: Sequence[str] = (),
    extra_forbidden_specs: Sequence[str] = (),
    required_marker_sets: Sequence[str] = (),
    forbidden_marker_sets: Sequence[str] = (),
) -> Tuple[bool, List[str], List[str], List[str]]:
    required = list(DEFAULT_PATCH_MARKERS)
    required.extend(_expand_marker_sets(required_marker_sets, PATCH_MARKER_SETS))
    required.extend(_parse_marker_specs(extra_marker_specs))
    forbidden = _expand_marker_sets(forbidden_marker_sets, FORBIDDEN_PATCH_MARKER_SETS)
    forbidden.extend(_parse_marker_specs(extra_forbidden_specs))
    passed = True
    details: List[str] = []
    missing: List[str] = []
    for rel_path, marker in required:
        target = geniesim_root / rel_path
        marker_id = f"{rel_path}::{marker}"
        if not target.is_file():
            passed = False
            missing.append(marker_id)
            details.append(f"missing file: {target}")
            continue
        content = target.read_text(errors="ignore")
        if marker not in content:
            passed = False
            missing.append(marker_id)
            details.append(f"missing marker '{marker}' in {target}")
        else:
            details.append(f"ok marker '{marker}' in {target}")

    forbidden_hits: List[str] = []
    for rel_path, marker in forbidden:
        target = geniesim_root / rel_path
        marker_id = f"{rel_path}::{marker}"
        if not target.is_file():
            details.append(f"forbidden check skipped missing file: {target}")
            continue
        content = target.read_text(errors="ignore")
        if marker in content:
            passed = False
            forbidden_hits.append(marker_id)
            details.append(f"forbidden marker '{marker}' present in {target}")
        else:
            details.append(f"ok forbidden marker absent '{marker}' in {target}")

    return passed, missing, forbidden_hits, details


def check_physics_coverage_report(
    report_path: Path,
    *,
    min_coverage: float,
    strict_collision: bool = False,
) -> Tuple[bool, Dict[str, object], List[str]]:
    errors: List[str] = []
    summary: Dict[str, object] = {
        "path": str(report_path),
        "coverage": 0.0,
        "objects_with_manifest_physics": 0,
        "objects_with_usd_physics": 0,
        "missing_physics_count": 0,
        "mesh_prims_total": 0,
        "mesh_prims_with_collision": 0,
        "mesh_prims_bad_dynamic_approx": 0,
        "collision_coverage": 0.0,
    }
    if not report_path.is_file():
        errors.append(f"physics coverage report not found: {report_path}")
        return False, summary, errors
    try:
        payload = json.loads(report_path.read_text())
    except Exception as exc:
        errors.append(f"failed to parse physics coverage report: {exc}")
        return False, summary, errors

    manifest_count = int(payload.get("objects_with_manifest_physics") or 0)
    usd_count = int(payload.get("objects_with_usd_physics") or 0)
    coverage = float(payload.get("coverage") or payload.get("coverage_ratio") or 0.0)
    missing_count = len(payload.get("missing_physics") or [])
    mesh_total = int(payload.get("mesh_prims_total") or 0)
    mesh_with_collision = int(payload.get("mesh_prims_with_collision") or 0)
    bad_dynamic_approx = int(payload.get("mesh_prims_bad_dynamic_approx") or 0)
    collision_coverage = float(payload.get("collision_coverage") or 0.0)
    summary.update(
        {
            "coverage": round(coverage, 4),
            "objects_with_manifest_physics": manifest_count,
            "objects_with_usd_physics": usd_count,
            "missing_physics_count": missing_count,
            "mesh_prims_total": mesh_total,
            "mesh_prims_with_collision": mesh_with_collision,
            "mesh_prims_bad_dynamic_approx": bad_dynamic_approx,
            "collision_coverage": round(collision_coverage, 4),
        }
    )

    if manifest_count <= 0:
        errors.append("physics coverage report has no manifest physics objects")
    if missing_count > 0:
        errors.append(f"physics coverage report has {missing_count} objects with missing physics fields")
    if coverage < min_coverage:
        errors.append(
            f"physics coverage below threshold ({coverage:.4f} < {min_coverage:.4f})"
        )

    if strict_collision:
        if "collision_coverage" not in payload:
            errors.append("physics coverage report missing collision_coverage")
        elif collision_coverage < 1.0:
            errors.append(
                f"collision coverage below strict threshold ({collision_coverage:.4f} < 1.0000)"
            )
        if "mesh_prims_bad_dynamic_approx" not in payload:
            errors.append("physics coverage report missing mesh_prims_bad_dynamic_approx")
        elif bad_dynamic_approx > 0:
            errors.append(
                f"physics coverage report has {bad_dynamic_approx} mesh prims with invalid dynamic approximation"
            )
        if mesh_total <= 0:
            errors.append("physics coverage report has zero mesh prims")

    return len(errors) == 0, summary, errors


def run_probe(args: argparse.Namespace) -> Tuple[bool, Dict[str, object]]:
    checks: List[Dict[str, object]] = []
    passed = True

    if not args.skip_grpc:
        ok, message = check_grpc_ready(args.host, args.port, args.timeout)
        checks.append({"name": "grpc_ready", "passed": ok, "message": message})
        passed = passed and ok

    must_check_patches = bool(args.check_patches or args.strict_runtime)
    if must_check_patches:
        required_sets = list(args.require_patch_set or [])
        forbidden_sets = list(args.forbid_patch_set or [])
        if args.strict_runtime:
            if "strict" not in required_sets:
                required_sets.append("strict")
            if "strict" not in forbidden_sets:
                forbidden_sets.append("strict")
        ok, missing, forbidden_hits, details = check_patch_markers(
            Path(args.geniesim_root),
            extra_marker_specs=args.require_patch,
            extra_forbidden_specs=args.forbid_patch,
            required_marker_sets=required_sets,
            forbidden_marker_sets=forbidden_sets,
        )
        checks.append(
            {
                "name": "runtime_patch_markers",
                "passed": ok,
                "missing": missing,
                "forbidden_hits": forbidden_hits,
                "required_sets": required_sets,
                "forbidden_sets": forbidden_sets,
                "details": details,
            }
        )
        passed = passed and ok

    physics_report_path = (args.physics_report or "").strip()
    if args.strict_runtime:
        if not physics_report_path:
            checks.append(
                {
                    "name": "physics_coverage_report",
                    "passed": False,
                    "errors": ["strict runtime mode requires --physics-report"],
                }
            )
            passed = False
        else:
            ok, summary, errors = check_physics_coverage_report(
                Path(physics_report_path),
                min_coverage=args.min_physics_coverage,
                strict_collision=True,
            )
            checks.append(
                {
                    "name": "physics_coverage_report",
                    "passed": ok,
                    "summary": summary,
                    "errors": errors,
                }
            )
            passed = passed and ok
    elif physics_report_path:
        ok, summary, errors = check_physics_coverage_report(
            Path(physics_report_path),
            min_coverage=args.min_physics_coverage,
            strict_collision=False,
        )
        checks.append(
            {
                "name": "physics_coverage_report",
                "passed": ok,
                "summary": summary,
                "errors": errors,
            }
        )
        passed = passed and ok

    payload = {
        "passed": passed,
        "strict_runtime": bool(args.strict_runtime),
        "host": args.host,
        "port": args.port,
        "checks": checks,
    }
    return passed, payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Genie Sim runtime readiness probe")
    parser.add_argument("--host", default=os.environ.get("GENIESIM_HOST", "localhost"))
    parser.add_argument("--port", default=os.environ.get("GENIESIM_PORT", "50051"))
    parser.add_argument("--timeout", type=float, default=float(os.environ.get("GENIESIM_READINESS_TIMEOUT_S", "5")))
    parser.add_argument("--geniesim-root", default=os.environ.get("GENIESIM_ROOT", "/opt/geniesim"))
    parser.add_argument("--skip-grpc", action="store_true", help="Skip gRPC channel readiness check.")
    parser.add_argument(
        "--check-patches",
        action="store_true",
        help="Verify required runtime patch markers in GENIESIM_ROOT.",
    )
    parser.add_argument(
        "--require-patch",
        action="append",
        default=[],
        help="Additional required patch marker as 'relative/path.py::marker text'.",
    )
    parser.add_argument(
        "--forbid-patch",
        action="append",
        default=[],
        help="Forbidden patch marker as 'relative/path.py::marker text'.",
    )
    parser.add_argument(
        "--require-patch-set",
        action="append",
        default=_parse_csv(os.environ.get("GENIESIM_REQUIRED_PATCH_SETS")),
        help="Named required marker set (supported: default, strict).",
    )
    parser.add_argument(
        "--forbid-patch-set",
        action="append",
        default=_parse_csv(os.environ.get("GENIESIM_FORBIDDEN_PATCH_SETS")),
        help="Named forbidden marker set (supported: strict).",
    )
    parser.add_argument(
        "--strict-runtime",
        action="store_true",
        default=_parse_bool(os.environ.get("GENIESIM_STRICT_RUNTIME_READINESS"), False),
        help="Enable strict runtime checks (patch markers + physics coverage report).",
    )
    parser.add_argument(
        "--physics-report",
        default=os.environ.get("GENIESIM_USD_PHYSICS_REPORT_PATH", ""),
        help="Path to USD physics coverage report json.",
    )
    parser.add_argument(
        "--min-physics-coverage",
        type=float,
        default=float(os.environ.get("GENIESIM_MIN_PHYSICS_COVERAGE", "0.98")),
    )
    parser.add_argument("--output", default="", help="Optional JSON output path.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    passed, payload = run_probe(args)
    rendered = json.dumps(payload, indent=2)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered + "\n")
    print(rendered)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
