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


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


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


def check_patch_markers(
    geniesim_root: Path,
    extra_marker_specs: Sequence[str] = (),
) -> Tuple[bool, List[str], List[str]]:
    required = list(DEFAULT_PATCH_MARKERS)
    required.extend(_parse_marker_specs(extra_marker_specs))
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
    return passed, missing, details


def check_physics_coverage_report(
    report_path: Path,
    *,
    min_coverage: float,
) -> Tuple[bool, Dict[str, object], List[str]]:
    errors: List[str] = []
    summary: Dict[str, object] = {
        "path": str(report_path),
        "coverage": 0.0,
        "objects_with_manifest_physics": 0,
        "objects_with_usd_physics": 0,
        "missing_physics_count": 0,
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
    coverage = float(payload.get("coverage") or 0.0)
    missing_count = len(payload.get("missing_physics") or [])
    summary.update(
        {
            "coverage": round(coverage, 4),
            "objects_with_manifest_physics": manifest_count,
            "objects_with_usd_physics": usd_count,
            "missing_physics_count": missing_count,
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
        ok, missing, details = check_patch_markers(
            Path(args.geniesim_root),
            extra_marker_specs=args.require_patch,
        )
        checks.append(
            {
                "name": "runtime_patch_markers",
                "passed": ok,
                "missing": missing,
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
