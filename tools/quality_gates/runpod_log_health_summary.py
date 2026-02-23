#!/usr/bin/env python3
"""Summarize key health/failure signatures from a runpod pipeline log."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple


SIGNATURE_PATTERNS: Dict[str, str] = {
    "sensor_qc_failed": r"Sensor QC failed for demo",
    "curobo_plan_single_failed": r"RuntimeError: cuRobo plan_single failed \(success=False\)",
    "rgb_all_black": r"RGB is all-black",
    "called_process_error": r"CalledProcessError",
    "preflight_failed": r"Preflight FAILED",
    "pytorch3d_missing": r"ModuleNotFoundError: No module named 'pytorch3d'",
    "min_required_objects_assertion": r"AssertionError: Minimum required objects not found in policy analysis",
    "xio_fatal_io": r"XIO:  fatal IO error",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def count_signatures(text: str, patterns: Dict[str, str] = SIGNATURE_PATTERNS) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for name, pattern in patterns.items():
        counts[name] = len(re.findall(pattern, text))
    return counts


def _top_buckets(counts: Dict[str, int]) -> List[Tuple[str, int]]:
    ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    return [(name, count) for name, count in ranked if count > 0]


def build_summary(*, log_path: Path, text: str) -> Dict[str, object]:
    counts = count_signatures(text)
    top = _top_buckets(counts)

    blocking = []
    if counts["called_process_error"] > 0:
        blocking.append("called_process_error")
    if counts["rgb_all_black"] > 0:
        blocking.append("rgb_all_black")

    warnings = []
    if counts["sensor_qc_failed"] > 0:
        warnings.append("sensor_qc_failed")
    if counts["curobo_plan_single_failed"] > 0:
        warnings.append("curobo_plan_single_failed")
    if counts["preflight_failed"] > 0:
        warnings.append("preflight_failed")

    status = "pass"
    if blocking:
        status = "fail"
    elif warnings:
        status = "warn"

    return {
        "schema_version": "1.0",
        "generated_at": _utc_now(),
        "log_path": str(log_path),
        "status": status,
        "blocking_signals": blocking,
        "warning_signals": warnings,
        "counts": counts,
        "top_buckets": [{"name": name, "count": count} for name, count in top],
    }


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Summarize health signals from runpod pipeline logs.")
    parser.add_argument("--log-path", type=Path, required=True, help="Path to pipeline log file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "analysis_outputs" / "runpod_log_health_summary.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when status is fail.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log_path = args.log_path.expanduser().resolve()
    if not log_path.is_file():
        raise SystemExit(f"log file not found: {log_path}")

    text = log_path.read_text(encoding="utf-8", errors="ignore")
    summary = build_summary(log_path=log_path, text=text)

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2))
    print(f"[runpod-log-health-summary] wrote {output}")
    print(f"[runpod-log-health-summary] status={summary['status']}")

    if args.strict and summary["status"] == "fail":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
