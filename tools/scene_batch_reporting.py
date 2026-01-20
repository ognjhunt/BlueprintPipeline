"""Helpers for summarizing batch runs and generating reports."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.batch_processing.parallel_runner import SceneStatus


def _collect_quality_failures(report_path: Optional[Path]) -> List[Dict[str, Any]]:
    if not report_path or not report_path.exists():
        return []
    report = json.loads(report_path.read_text())
    failures = []
    for entry in report.get("results", []):
        if not entry.get("passed"):
            failures.append({
                "gate_id": entry.get("gate_id"),
                "checkpoint": entry.get("checkpoint"),
                "severity": entry.get("severity"),
                "message": entry.get("message"),
                "recommendations": entry.get("recommendations", []),
            })
    return failures


def _summarize_batch_results(
    results: List[Any],
    reports_dir: Path,
    dlq_path: Optional[Path] = None,
) -> Dict[str, Any]:
    failures = []
    skipped = []
    dlq_entries: List[Dict[str, Any]] = []
    resolved_dlq_path = dlq_path or (reports_dir / "dead_letter_queue.json" if reports_dir else None)

    for result in results:
        report_path = None
        if result.metadata and result.metadata.get("quality_gate_report"):
            report_path = Path(result.metadata["quality_gate_report"])

        if result.metadata and result.metadata.get("skipped"):
            skipped.append(result.metadata["scene_id"])

        if result.status != SceneStatus.SUCCESS:
            attempt_count = None
            if result.metadata:
                attempt_count = result.metadata.get("attempt") or result.metadata.get("attempts")
            if attempt_count is None and result.error:
                match = re.search(r"Failed after (\d+) attempts", result.error)
                if match:
                    attempt_count = int(match.group(1))

            dlq_entries.append({
                "scene_id": result.scene_id,
                "scene_dir": result.metadata.get("scene_dir") if result.metadata else None,
                "status": result.status.value,
                "error": result.error,
                "quality_gate_failures": _collect_quality_failures(report_path),
                "attempts": attempt_count,
            })
            failures.append({
                "scene_id": result.scene_id,
                "status": result.status.value,
                "error": result.error,
                "quality_gate_failures": _collect_quality_failures(report_path),
            })

    summary = {
        "total": len(results),
        "success": sum(1 for r in results if r.status == SceneStatus.SUCCESS),
        "failed": sum(1 for r in results if r.status == SceneStatus.FAILED),
        "cancelled": sum(1 for r in results if r.status == SceneStatus.CANCELLED),
        "skipped": skipped,
        "failures": failures,
    }

    if reports_dir:
        reports_dir.mkdir(parents=True, exist_ok=True)
        (reports_dir / "batch_report.json").write_text(json.dumps(summary, indent=2))

    if resolved_dlq_path:
        resolved_dlq_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_dlq_path.write_text(json.dumps(dlq_entries, indent=2))

    return summary
