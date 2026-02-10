#!/usr/bin/env python3
"""Build runtime distribution + timeout-usage summary from bp_metric JSONL exports."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _iter_jsonl_records(paths: Iterable[Path]) -> Iterable[Dict[str, Any]]:
    for path in paths:
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                yield payload


def _extract_field(payload: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in payload:
            return payload.get(key)
    json_payload = payload.get("jsonPayload")
    if isinstance(json_payload, dict):
        for key in keys:
            if key in json_payload:
                return json_payload.get(key)
    labels = payload.get("labels")
    if isinstance(labels, dict):
        for key in keys:
            if key in labels:
                return labels.get(key)
    return None


def _extract_stage(payload: Dict[str, Any]) -> str:
    stage = _extract_field(payload, ("stage", "job", "job_name", "step", "pipeline_step"))
    if isinstance(stage, str) and stage.strip():
        return stage.strip()
    metric_name = _extract_field(payload, ("metric", "event", "name"))
    if isinstance(metric_name, str) and metric_name.strip():
        return metric_name.strip()
    return "unknown"


def _extract_event(payload: Dict[str, Any]) -> str:
    event = _extract_field(payload, ("event", "metric", "name", "type"))
    if isinstance(event, str):
        return event.strip().lower()
    return ""


def _extract_duration(payload: Dict[str, Any]) -> Optional[float]:
    return _safe_float(
        _extract_field(
            payload,
            ("duration_seconds", "duration_s", "latency_seconds", "elapsed_seconds"),
        )
    )


def _extract_timeout(payload: Dict[str, Any]) -> Optional[float]:
    return _safe_float(
        _extract_field(
            payload,
            ("timeout_seconds", "configured_timeout_seconds", "timeout_s"),
        )
    )


def _quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.array(values, dtype=float), q))


@dataclass
class StageMetrics:
    durations: List[float]
    timeout_usages: List[float]
    timeout_seconds_observed: List[float]
    timeout_exhausted_events: int = 0


def build_summary(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    by_stage: Dict[str, StageMetrics] = defaultdict(
        lambda: StageMetrics(durations=[], timeout_usages=[], timeout_seconds_observed=[])
    )

    for payload in records:
        event = _extract_event(payload)
        stage = _extract_stage(payload)
        duration = _extract_duration(payload)
        timeout = _extract_timeout(payload)

        is_invocation = (
            "job_invocation" in event
            or event in {"invocation", "duration"}
            or (duration is not None and stage != "unknown")
        )
        if is_invocation and duration is not None:
            by_stage[stage].durations.append(duration)
            if timeout is not None and timeout > 0:
                by_stage[stage].timeout_seconds_observed.append(timeout)
                by_stage[stage].timeout_usages.append(duration / timeout)

        if "retry_exhausted" in event or "timeout_exhausted" in event:
            by_stage[stage].timeout_exhausted_events += 1

    stage_payload: Dict[str, Dict[str, Any]] = {}
    for stage, metrics in sorted(by_stage.items()):
        durations = metrics.durations
        timeout_usages = metrics.timeout_usages
        observed_timeout = metrics.timeout_seconds_observed
        stage_payload[stage] = {
            "samples": len(durations),
            "timeout_samples": len(timeout_usages),
            "configured_timeout_seconds": (
                round(sum(observed_timeout) / float(len(observed_timeout)), 2)
                if observed_timeout
                else None
            ),
            "p50_duration_seconds": round(_quantile(durations, 0.50), 4) if durations else None,
            "p90_duration_seconds": round(_quantile(durations, 0.90), 4) if durations else None,
            "p95_duration_seconds": round(_quantile(durations, 0.95), 4) if durations else None,
            "p99_duration_seconds": round(_quantile(durations, 0.99), 4) if durations else None,
            "timeout_usage_p95": round(_quantile(timeout_usages, 0.95), 4) if timeout_usages else None,
            "timeout_usage_max": round(max(timeout_usages), 4) if timeout_usages else None,
            "timeout_exhausted_events": metrics.timeout_exhausted_events,
        }

    complete = any(
        isinstance(stage_data.get("p95_duration_seconds"), (int, float))
        and isinstance(stage_data.get("timeout_usage_p95"), (int, float))
        for stage_data in stage_payload.values()
    )
    return {
        "generated_at": _utc_now(),
        "stages": stage_payload,
        "stage_count": len(stage_payload),
        "complete": complete,
    }


def _render_markdown_table(summary: Dict[str, Any]) -> str:
    lines = [
        "| Stage | Timeout (s) | P50 Duration (s) | P90 Duration (s) | P95 Duration (s) | P99 Duration (s) | P95 Timeout Usage | Timeout Exhausted |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    stages = summary.get("stages") or {}
    if not isinstance(stages, dict) or not stages:
        lines.append("| N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
        return "\n".join(lines)

    for stage, metrics in sorted(stages.items()):
        if not isinstance(metrics, dict):
            continue
        lines.append(
            "| {stage} | {timeout} | {p50} | {p90} | {p95} | {p99} | {usage} | {exhausted} |".format(
                stage=stage,
                timeout=metrics.get("configured_timeout_seconds", "N/A"),
                p50=metrics.get("p50_duration_seconds", "N/A"),
                p90=metrics.get("p90_duration_seconds", "N/A"),
                p95=metrics.get("p95_duration_seconds", "N/A"),
                p99=metrics.get("p99_duration_seconds", "N/A"),
                usage=metrics.get("timeout_usage_p95", "N/A"),
                exhausted=metrics.get("timeout_exhausted_events", 0),
            )
        )
    return "\n".join(lines)


def _update_ops_runtime_doc(path: Path, table: str) -> None:
    marker_start = "<!-- RUNTIME_TABLE_START -->"
    marker_end = "<!-- RUNTIME_TABLE_END -->"
    original = path.read_text() if path.is_file() else ""
    if marker_start not in original or marker_end not in original:
        return
    before, remainder = original.split(marker_start, 1)
    _, after = remainder.split(marker_end, 1)
    updated = f"{before}{marker_start}\n{table}\n{marker_end}{after}"
    path.write_text(updated)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Summarize runtime SLO evidence from bp_metric JSONL.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        required=True,
        help="One or more JSONL inputs (files).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "analysis_outputs" / "runtime_slo_summary.json",
        help="Output JSON summary path.",
    )
    parser.add_argument(
        "--update-ops-doc",
        action="store_true",
        help="Update docs/OPS_RUNTIME_REPORT.md runtime table block.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when summary is incomplete.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = list(_iter_jsonl_records(args.inputs))
    summary = build_summary(records)

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2))
    print(f"[runtime-slo-summary] wrote {output}")

    if args.update_ops_doc:
        repo_root = Path(__file__).resolve().parents[2]
        doc_path = repo_root / "docs" / "OPS_RUNTIME_REPORT.md"
        table = _render_markdown_table(summary)
        _update_ops_runtime_doc(doc_path, table)
        print(f"[runtime-slo-summary] updated {doc_path}")

    if args.strict and not bool(summary.get("complete")):
        print("[runtime-slo-summary] summary incomplete")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
