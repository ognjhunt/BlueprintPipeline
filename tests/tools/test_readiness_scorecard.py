from __future__ import annotations

import json
from pathlib import Path

from tools.quality_gates.readiness_scorecard import build_scorecard


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def test_readiness_scorecard_detects_import_and_quality_failures(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "analysis_outputs/run_x/run_certification_report.json",
        {
            "summary": {
                "episodes": 4,
                "certified": 0,
                "certification_pass_rate": 0.0,
                "gate_histogram": {"CHANNEL_INCOMPLETE": 4},
            }
        },
    )
    _write_json(
        tmp_path / "test_scenes/scenes/s1/episodes/run/import_manifest.json",
        {
            "scene_id": None,
            "recordings_format": "json",
            "episodes": {"passed_validation": 0},
            "quality": {"average_score": 0.2},
            "validation": {
                "episodes": {
                    "episode_results": [
                        {
                            "errors": [
                                "Episode file not found: recordings/ep_0001.parquet",
                            ]
                        }
                    ]
                }
            },
        },
    )
    _write_json(
        tmp_path / "test_scenes/scenes/s1/quality_gates/quality_gate_report.json",
        {
            "summary": {
                "total_gates": 0,
                "can_proceed": False,
                "blocking_failures": 1,
            }
        },
    )
    _write_json(
        tmp_path / "test_scenes/scenes/s1/production_validation.json",
        {"production_mode": False, "ok": True, "errors": []},
    )
    _write_json(
        tmp_path / "test_scenes/scenes/s1/assets/scene_manifest.json",
        {"scene_id": "s1", "scene": {"environment_type": "kitchen"}},
    )
    workflow = tmp_path / ".github/workflows/test-unit.yml"
    workflow.parent.mkdir(parents=True, exist_ok=True)
    workflow.write_text("name: unit\n")
    thresholds_doc = tmp_path / "docs/COMMERCIAL_READINESS_GATES.md"
    thresholds_doc.parent.mkdir(parents=True, exist_ok=True)
    thresholds_doc.write_text("# thresholds\n")

    scorecard = build_scorecard(tmp_path)

    gates = {item["name"]: item for item in scorecard["release_gates"]}
    assert gates["import_viability"]["passed"] is False
    assert gates["quality_gate_enforcement"]["passed"] is False
    assert gates["certification"]["passed"] is False
    assert scorecard["scores"]["commercial_readiness_score"] < 100


def test_readiness_scorecard_marks_ci_reliability_when_signals_exist(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "analysis_outputs/run_x/run_certification_report.json",
        {
            "summary": {
                "episodes": 3,
                "certified": 3,
                "certification_pass_rate": 1.0,
                "gate_histogram": {},
            }
        },
    )
    _write_json(
        tmp_path / "test_scenes/scenes/s1/episodes/run/import_manifest.json",
        {
            "schema_version": "1.3",
            "scene_id": "s1",
            "run_id": "run-1",
            "status": "completed",
            "episodes": {"passed_validation": 2},
            "quality": {"average_score": 0.9},
            "recordings_format": "json",
            "validation": {"episodes": {}},
            "robot_types": ["franka", "ur5e", "ur10"],
        },
    )
    _write_json(
        tmp_path / "test_scenes/scenes/s1/quality_gates/quality_gate_report.json",
        {
            "report_version": "2.0",
            "required_checkpoints": ["geniesim_import_complete"],
            "executed_checkpoints": ["geniesim_import_complete"],
            "skipped_checkpoints": [],
            "skip_reasons": {},
            "summary": {"total_gates": 1, "can_proceed": True, "blocking_failures": 0},
        },
    )
    _write_json(
        tmp_path / "test_scenes/scenes/s1/production_validation.json",
        {"production_mode": True, "ok": True, "errors": []},
    )
    _write_json(
        tmp_path / "analysis_outputs/runtime_slo_summary.json",
        {
            "complete": True,
            "stages": {
                "episode-generation": {
                    "p50_duration_seconds": 10.0,
                    "p90_duration_seconds": 20.0,
                    "p95_duration_seconds": 25.0,
                    "p99_duration_seconds": 30.0,
                    "timeout_usage_p95": 0.5,
                }
            },
        },
    )
    _write_json(
        tmp_path / "analysis_outputs/canary_stability_gate.json",
        {"stable_7_day": True},
    )
    _write_json(
        tmp_path / "test_scenes/scenes/s1/assets/scene_manifest.json",
        {
            "scene_id": "s1",
            "scene_family": "kitchen",
            "environment_type": "kitchen",
        },
    )
    workflow = tmp_path / ".github/workflows/test-unit.yml"
    workflow.parent.mkdir(parents=True, exist_ok=True)
    workflow.write_text("python tools/quality_gates/readiness_scorecard.py\n")
    thresholds_doc = tmp_path / "docs/COMMERCIAL_READINESS_GATES.md"
    thresholds_doc.parent.mkdir(parents=True, exist_ok=True)
    thresholds_doc.write_text("# thresholds\n")

    junit = tmp_path / "junit.xml"
    junit.write_text(
        "<testsuite><testcase classname='tests.test_pipeline_data_flow' "
        "name='test_pipeline_data_flow_entrypoints'/></testsuite>"
    )

    scorecard = build_scorecard(tmp_path)
    gates = {item["name"]: item for item in scorecard["release_gates"]}
    assert gates["ci_reliability"]["passed"] is True
    assert scorecard["phases"]["phase_6_reliability"]["passed"] is True
