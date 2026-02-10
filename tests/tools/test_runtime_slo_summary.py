from __future__ import annotations

from tools.quality_gates.runtime_slo_summary import build_summary


def test_runtime_slo_summary_computes_quantiles_and_timeout_usage() -> None:
    records = [
        {
            "event": "bp_metric:job_invocation",
            "stage": "episode-generation",
            "duration_seconds": 10.0,
            "timeout_seconds": 100.0,
        },
        {
            "event": "bp_metric:job_invocation",
            "stage": "episode-generation",
            "duration_seconds": 20.0,
            "timeout_seconds": 100.0,
        },
        {
            "event": "bp_metric:job_retry_exhausted",
            "stage": "episode-generation",
        },
    ]

    summary = build_summary(records)
    stage = summary["stages"]["episode-generation"]

    assert summary["stage_count"] == 1
    assert stage["samples"] == 2
    assert stage["configured_timeout_seconds"] == 100.0
    assert stage["p50_duration_seconds"] == 15.0
    assert stage["timeout_usage_p95"] is not None
    assert stage["timeout_exhausted_events"] == 1
    assert summary["complete"] is True


def test_runtime_slo_summary_handles_empty_input() -> None:
    summary = build_summary([])
    assert summary["stage_count"] == 0
    assert summary["complete"] is False
