#!/usr/bin/env python3
"""CI gate for enforcing certification pass-rate SLO from run reports."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SLO_PROFILES = {
    "canary": 0.95,
    "preprod": 0.97,
    "production": 0.98,
}


def _discover_reports(search_roots: List[Path], report_glob: str) -> List[Path]:
    report_paths: List[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        for path in root.glob(report_glob):
            if path.is_file():
                report_paths.append(path)
    report_paths.sort(key=lambda p: p.stat().st_mtime)
    return report_paths


def _load_report(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def evaluate_report(
    report: Dict[str, Any],
    *,
    profile: str,
    min_episodes: int,
    min_pass_rate_override: Optional[float] = None,
) -> Tuple[bool, Dict[str, Any]]:
    profile_norm = (profile or "production").strip().lower()
    if profile_norm not in SLO_PROFILES:
        raise ValueError(f"Unknown profile '{profile}'. Expected one of: {sorted(SLO_PROFILES.keys())}")

    threshold = (
        float(min_pass_rate_override)
        if min_pass_rate_override is not None
        else float(SLO_PROFILES[profile_norm])
    )
    summary = report.get("summary") if isinstance(report, dict) else {}
    if not isinstance(summary, dict):
        summary = {}

    episodes = int(summary.get("episodes", 0))
    pass_rate = float(summary.get("certification_pass_rate", 0.0))
    certified = int(summary.get("certified", 0))
    raw_only = int(summary.get("raw_only", 0))

    reasons: List[str] = []
    if episodes < min_episodes:
        reasons.append(
            f"episodes={episodes} below required minimum {min_episodes}"
        )
    if pass_rate < threshold:
        reasons.append(
            f"certification_pass_rate={pass_rate:.4f} below threshold {threshold:.4f}"
        )

    passed = len(reasons) == 0
    result = {
        "evaluated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "profile": profile_norm,
        "threshold": threshold,
        "min_episodes": min_episodes,
        "episodes": episodes,
        "certified": certified,
        "raw_only": raw_only,
        "certification_pass_rate": pass_rate,
        "passed": passed,
        "reasons": reasons,
        "gate_histogram": summary.get("gate_histogram", {}),
    }
    return passed, result


def main() -> int:
    parser = argparse.ArgumentParser(description="Enforce certification pass-rate SLO from run report.")
    parser.add_argument(
        "--search-root",
        action="append",
        required=True,
        help="Root directory to search for run_certification_report.json.",
    )
    parser.add_argument(
        "--report-glob",
        default="**/run_certification_report.json",
        help="Glob used under each --search-root to find reports.",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(SLO_PROFILES.keys()),
        default="production",
        help="SLO profile threshold set.",
    )
    parser.add_argument(
        "--min-pass-rate",
        type=float,
        default=None,
        help="Optional threshold override.",
    )
    parser.add_argument(
        "--min-episodes",
        type=int,
        default=1,
        help="Minimum episode count required to evaluate pass-rate.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional JSON output file for gate evaluation details.",
    )
    args = parser.parse_args()

    roots = [Path(p).expanduser().resolve() for p in args.search_root]
    reports = _discover_reports(roots, args.report_glob)
    if not reports:
        message = (
            "No run_certification_report.json found under roots: "
            + ", ".join(str(p) for p in roots)
        )
        print(f"::error::{message}")
        return 2

    selected_report_path = reports[-1]
    report_payload = _load_report(selected_report_path)
    passed, result = evaluate_report(
        report_payload,
        profile=args.profile,
        min_episodes=max(1, int(args.min_episodes)),
        min_pass_rate_override=args.min_pass_rate,
    )
    result["selected_report"] = str(selected_report_path)
    result["reports_discovered"] = [str(p) for p in reports]

    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered + "\n")

    if not passed:
        print(
            "::error::Certification SLO gate failed. "
            f"profile={result['profile']} "
            f"pass_rate={result['certification_pass_rate']:.4f} "
            f"threshold={result['threshold']:.4f} "
            f"episodes={result['episodes']} "
            f"report={result['selected_report']}"
        )
        return 1

    print(
        "::notice::Certification SLO gate passed. "
        f"profile={result['profile']} "
        f"pass_rate={result['certification_pass_rate']:.4f} "
        f"threshold={result['threshold']:.4f} "
        f"episodes={result['episodes']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
