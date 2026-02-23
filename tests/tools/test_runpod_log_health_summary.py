from __future__ import annotations

from pathlib import Path

from tools.quality_gates.runpod_log_health_summary import build_summary, count_signatures


def test_count_signatures_finds_key_failures() -> None:
    text = "\n".join(
        [
            "RuntimeError: cuRobo plan_single failed (success=False)",
            "RuntimeError: cuRobo plan_single failed (success=False)",
            "Sensor QC failed for demo 1",
            "RGB is all-black",
        ]
    )
    counts = count_signatures(text)
    assert counts["curobo_plan_single_failed"] == 2
    assert counts["sensor_qc_failed"] == 1
    assert counts["rgb_all_black"] == 1


def test_build_summary_sets_fail_when_blocking_signals_present() -> None:
    text = "\n".join(
        [
            "Sensor QC failed for demo 1",
            "CalledProcessError: Command '['foo']' returned non-zero exit status 1.",
            "RGB is all-black",
        ]
    )
    summary = build_summary(log_path=Path("/tmp/pipeline.log"), text=text)
    assert summary["status"] == "fail"
    assert "called_process_error" in summary["blocking_signals"]
    assert "rgb_all_black" in summary["blocking_signals"]
