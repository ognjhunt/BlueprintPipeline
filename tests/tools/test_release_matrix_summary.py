from __future__ import annotations

import json
from pathlib import Path

from tools.quality_gates.release_matrix_summary import build_summary


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def test_release_matrix_summary_collects_scene_and_robot_metrics(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "test_scenes/scenes/scene_a/assets/scene_manifest.json",
        {
            "scene_id": "scene_a",
            "scene_family": "kitchen",
            "environment_type": "kitchen",
        },
    )
    _write_json(
        tmp_path / "test_scenes/scenes/scene_a/episodes/run/import_manifest.json",
        {
            "scene_id": "scene_a",
            "episodes": {"passed_validation": 3},
            "quality": {"average_score": 0.91},
        },
    )
    _write_json(
        tmp_path / "test_scenes/scenes/scene_a/geniesim/job.json",
        {
            "generation_params": {"robot_types": ["franka", "ur5e"]},
            "job_metrics_by_robot": {
                "franka": {"episodes_collected": 5, "episodes_passed": 5},
                "ur5e": {"episodes_collected": 4, "episodes_passed": 3},
            },
        },
    )
    _write_json(
        tmp_path / "analysis_outputs/run_z/run_certification_report.json",
        {"summary": {"certification_pass_rate": 0.96}},
    )

    summary = build_summary(tmp_path)

    assert summary["summary"]["scene_count"] == 1
    assert summary["summary"]["per_robot_import_success"]["franka"] == 1.0
    assert summary["summary"]["per_robot_import_success"]["ur5e"] == 0.75
    assert summary["summary"]["preprod_certification_pass_rate"] == 0.96
    assert summary["checks"]["preprod_certification_ok"] is True
    assert summary["checks"]["scene_count_ok"] is False


def test_release_matrix_summary_flags_scene_balance_violation(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "test_scenes/scenes/scene_a/assets/scene_manifest.json",
        {"scene_id": "scene_a", "scene_family": "kitchen", "environment_type": "kitchen"},
    )
    _write_json(
        tmp_path / "test_scenes/scenes/scene_b/assets/scene_manifest.json",
        {"scene_id": "scene_b", "scene_family": "lab", "environment_type": "lab"},
    )
    _write_json(
        tmp_path / "test_scenes/scenes/scene_a/episodes/run/import_manifest.json",
        {"scene_id": "scene_a", "episodes": {"passed_validation": 9}, "quality": {"average_score": 0.9}},
    )
    _write_json(
        tmp_path / "test_scenes/scenes/scene_b/episodes/run/import_manifest.json",
        {"scene_id": "scene_b", "episodes": {"passed_validation": 1}, "quality": {"average_score": 0.9}},
    )

    summary = build_summary(tmp_path)

    assert summary["summary"]["max_scene_contribution_ratio"] == 0.9
    assert summary["checks"]["scene_balance_ok"] is False
