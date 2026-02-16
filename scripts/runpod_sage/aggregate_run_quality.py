#!/usr/bin/env python3
"""
Aggregate Stage 4/5/7 quality artifacts into one run-level summary.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON at {path}")
    return payload


def _safe_load(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    try:
        payload = _load_json(path)
        payload.setdefault("path", str(path))
        return payload
    except Exception as exc:
        return {"status": "error", "path": str(path), "error": str(exc)}


def _stage_status(payload: Dict[str, Any]) -> str:
    status = str(payload.get("status", "")).strip().lower()
    if status in {"pass", "ok"}:
        return "pass"
    if status in {"missing", "error", "fail"}:
        return status
    all_pass = payload.get("all_pass")
    if isinstance(all_pass, bool):
        return "pass" if all_pass else "fail"
    return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate run quality reports")
    parser.add_argument("--layout-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-path", default="")
    args = parser.parse_args()

    layout_dir = Path(args.layout_dir).expanduser().resolve()
    output_path = (
        Path(args.output_path).expanduser().resolve()
        if str(args.output_path).strip()
        else (layout_dir / "quality" / "run_quality_summary.json")
    )

    stage4_path = layout_dir / "quality" / "scene_quality_report.json"
    stage5_path = layout_dir / "quality" / "stage5_quality_report.json"
    stage7_path = layout_dir / "demos" / "quality_report.json"
    contract_path = layout_dir / "quality" / "stage7_contract_report.json"

    stage4 = _safe_load(stage4_path)
    stage5 = _safe_load(stage5_path)
    stage7 = _safe_load(stage7_path)
    contract = _safe_load(contract_path)

    statuses = {
        "stage4_quality": _stage_status(stage4),
        "stage5_quality": _stage_status(stage5),
        "stage7_quality": _stage_status(stage7),
        "artifact_contract": _stage_status(contract),
    }

    overall = "pass"
    for status in statuses.values():
        if status != "pass":
            overall = "fail"
            break

    summary: Dict[str, Any] = {
        "run_id": str(args.run_id),
        "layout_dir": str(layout_dir),
        "status": overall,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stages": statuses,
        "reports": {
            "stage4_quality": stage4,
            "stage5_quality": stage5,
            "stage7_quality": stage7,
            "artifact_contract": contract,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(str(output_path))
    return 0 if overall == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())
