import json
import time
from pathlib import Path

from tools.quality_gates.certification_slo_gate import (
    _discover_reports,
    evaluate_report,
)


def _report(pass_rate: float, episodes: int = 10, certified: int = 0) -> dict:
    if certified == 0:
        certified = int(round(pass_rate * episodes))
    return {
        "summary": {
            "episodes": episodes,
            "certified": certified,
            "raw_only": max(0, episodes - certified),
            "certification_pass_rate": pass_rate,
            "gate_histogram": {},
        }
    }


def test_evaluate_report_passes_when_rate_meets_profile_threshold() -> None:
    passed, result = evaluate_report(
        _report(0.96, episodes=20),
        profile="canary",
        min_episodes=5,
        min_pass_rate_override=None,
    )
    assert passed is True
    assert result["threshold"] == 0.95
    assert result["episodes"] == 20


def test_evaluate_report_fails_when_rate_below_production_threshold() -> None:
    passed, result = evaluate_report(
        _report(0.90, episodes=20),
        profile="production",
        min_episodes=5,
        min_pass_rate_override=None,
    )
    assert passed is False
    assert result["threshold"] == 0.98
    assert any("below threshold" in reason for reason in result["reasons"])


def test_discover_reports_returns_paths_sorted_by_mtime(tmp_path: Path) -> None:
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    run_a.mkdir(parents=True)
    run_b.mkdir(parents=True)
    p1 = run_a / "run_certification_report.json"
    p2 = run_b / "run_certification_report.json"
    p1.write_text(json.dumps(_report(0.5)))
    time.sleep(0.01)
    p2.write_text(json.dumps(_report(0.9)))

    discovered = _discover_reports([tmp_path], "**/run_certification_report.json")
    assert discovered[-1] == p2
