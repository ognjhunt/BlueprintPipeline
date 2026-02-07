from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from tools.geniesim_adapter.deployment import readiness_probe


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_check_patch_markers_detects_missing_marker(tmp_path: Path) -> None:
    root = tmp_path / "geniesim"
    grpc_file = root / "source/data_collection/server/grpc_server.py"
    cmd_file = root / "source/data_collection/server/command_controller.py"

    _write(grpc_file, "# BlueprintPipeline contact_report patch\n")
    _write(
        cmd_file,
        "# BlueprintPipeline object_pose patch\n# BlueprintPipeline sim_thread_physics_cache patch\n",
    )

    ok, missing, details = readiness_probe.check_patch_markers(root)
    assert ok is True
    assert missing == []
    assert any("contact_report patch" in d for d in details)

    _write(cmd_file, "# BlueprintPipeline object_pose patch\n")
    ok2, missing2, _ = readiness_probe.check_patch_markers(root)
    assert ok2 is False
    assert any("sim_thread_physics_cache" in m for m in missing2)


def test_check_physics_coverage_report_thresholds(tmp_path: Path) -> None:
    report = tmp_path / "scene.physics_report.json"
    report.write_text(
        '{"objects_with_manifest_physics": 10, "objects_with_usd_physics": 10, "coverage": 1.0, "missing_physics": []}'
    )
    ok, summary, errors = readiness_probe.check_physics_coverage_report(report, min_coverage=0.98)
    assert ok is True
    assert errors == []
    assert summary["coverage"] == 1.0

    report.write_text(
        '{"objects_with_manifest_physics": 10, "objects_with_usd_physics": 5, "coverage": 0.5, "missing_physics": [{"id":"x"}]}'
    )
    ok2, _, errors2 = readiness_probe.check_physics_coverage_report(report, min_coverage=0.98)
    assert ok2 is False
    assert len(errors2) >= 2


def test_run_probe_strict_runtime_requires_physics_report() -> None:
    args = Namespace(
        host="localhost",
        port="50051",
        timeout=0.01,
        geniesim_root="/tmp/does-not-matter",
        skip_grpc=True,
        check_patches=False,
        require_patch=[],
        strict_runtime=True,
        physics_report="",
        min_physics_coverage=0.98,
        output="",
    )
    passed, payload = readiness_probe.run_probe(args)
    assert passed is False
    checks = payload.get("checks", [])
    physics_checks = [c for c in checks if c.get("name") == "physics_coverage_report"]
    assert len(physics_checks) == 1
    assert physics_checks[0]["passed"] is False
