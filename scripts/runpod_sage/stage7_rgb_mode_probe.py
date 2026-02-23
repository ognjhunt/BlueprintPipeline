#!/usr/bin/env python3
"""
Standalone Stage 7 RGB mode probe.

Runs small probe collections across candidate Stage 7 render/display modes and
writes a report describing which mode (if any) satisfies the RGB contract.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.runpod_sage import sage_stage567_mobile_franka as stage567


def _env_override(key: str, value: str) -> tuple[str, str | None]:
    prev = os.environ.get(key)
    os.environ[key] = str(value)
    return key, prev


def _restore_env(overrides: list[tuple[str, str | None]]) -> None:
    for key, prev in overrides:
        if prev is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prev


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe Stage 7 RGB capture modes.")
    parser.add_argument("--plan-bundle", required=True, help="Path to plan_bundle.json")
    parser.add_argument("--collector", required=True, help="Path to isaacsim_collect_mobile_franka.py")
    parser.add_argument("--isaacsim-py", required=True, help="Isaac Sim Python binary")
    parser.add_argument("--output-root", required=True, help="Directory for probe attempt outputs")
    parser.add_argument(
        "--mode-order",
        default="auto",
        help="Candidate mode order: auto or csv list (streaming,headless,windowed)",
    )
    parser.add_argument("--probe-demos", type=int, default=1, help="Number of demos per mode probe")
    parser.add_argument("--timeout-s", type=int, default=600, help="Timeout per mode attempt")
    parser.add_argument("--requested-mode", default="auto", choices=["auto", "headless", "windowed", "streaming"])
    parser.add_argument("--strict", action="store_true", help="Run probe collector with --strict")
    parser.add_argument("--layout-id", default="", help="Optional layout id for report metadata")
    parser.add_argument("--report-path", default="", help="Optional output path for stage7_mode_probe.json")
    args = parser.parse_args()

    plan_bundle = Path(args.plan_bundle).expanduser().resolve()
    collector = Path(args.collector).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    report_path = (
        Path(args.report_path).expanduser().resolve()
        if str(args.report_path).strip()
        else (output_root / "stage7_mode_probe.json")
    )
    if not plan_bundle.is_file():
        raise FileNotFoundError(f"Missing plan bundle: {plan_bundle}")
    if not collector.is_file():
        raise FileNotFoundError(f"Missing collector script: {collector}")

    run_id = f"stage7_probe_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{uuid.uuid4().hex[:8]}"
    ns = argparse.Namespace(
        layout_id=str(args.layout_id or ""),
        isaacsim_py=str(args.isaacsim_py),
        headless=bool(args.requested_mode != "windowed"),
        stage7_headless_mode=str(args.requested_mode),
        enable_cameras=True,
        strict=bool(args.strict),
        output_dir=str(output_root / "full_stage7_output"),
    )

    overrides = [
        _env_override("SAGE_STAGE7_MODE_ORDER", str(args.mode_order)),
        _env_override("SAGE_STAGE7_PROBE_DEMOS", str(max(1, int(args.probe_demos)))),
        _env_override("SAGE_STAGE7_PROBE_TIMEOUT_S", str(max(1, int(args.timeout_s)))),
        _env_override("SAGE_REQUIRE_VALID_RGB", "1"),
        _env_override("SAGE_STAGE7_RGB_POLICY", "auto_probe_fail"),
    ]
    try:
        report = stage567.run_stage7_rgb_mode_probe(
            args=ns,
            collector=collector,
            plan_path=plan_bundle,
            output_root=output_root,
            report_path=report_path,
            run_id=run_id,
        )
    finally:
        _restore_env(overrides)

    selected_mode = str(report.get("selected_mode") or "").strip().lower()
    if selected_mode:
        print(
            f"stage7_rgb_probe: PASS selected_mode={selected_mode} report={report_path}",
            flush=True,
        )
        return 0

    attempts = report.get("attempts", [])
    reasons: list[str] = []
    if isinstance(attempts, list):
        for attempt in attempts:
            if not isinstance(attempt, dict):
                continue
            mode = str(attempt.get("mode", "unknown"))
            mode_reasons = attempt.get("failure_reasons", [])
            if isinstance(mode_reasons, list) and mode_reasons:
                reasons.append(f"{mode}[{','.join(str(x) for x in mode_reasons)}]")
            else:
                reasons.append(f"{mode}[unknown_failure]")
    reason_text = "; ".join(reasons) if reasons else "no attempts recorded"
    print(
        f"stage7_rgb_probe: FAIL reason={reason_text} report={report_path}",
        file=sys.stderr,
        flush=True,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
