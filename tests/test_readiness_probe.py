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

    ok, missing, forbidden_hits, details = readiness_probe.check_patch_markers(root)
    assert ok is True
    assert missing == []
    assert forbidden_hits == []
    assert any("contact_report patch" in d for d in details)

    _write(cmd_file, "# BlueprintPipeline object_pose patch\n")
    ok2, missing2, forbidden_hits2, _ = readiness_probe.check_patch_markers(root)
    assert ok2 is False
    assert any("sim_thread_physics_cache" in m for m in missing2)
    assert forbidden_hits2 == []


def test_check_patch_markers_forbidden_marker_fails(tmp_path: Path) -> None:
    root = tmp_path / "geniesim"
    grpc_file = root / "source/data_collection/server/grpc_server.py"
    cmd_file = root / "source/data_collection/server/command_controller.py"

    _write(grpc_file, "# BlueprintPipeline contact_report patch\n# BlueprintPipeline joint_efforts handler patch\n")
    _write(
        cmd_file,
        (
            "# BlueprintPipeline object_pose patch\n"
            "# BlueprintPipeline sim_thread_physics_cache patch\n"
            "# BPv7_keep_kinematic\n"
        ),
    )

    ok, missing, forbidden_hits, details = readiness_probe.check_patch_markers(
        root,
        forbidden_marker_sets=["strict"],
    )
    assert ok is False
    assert missing == []
    assert any("BPv7_keep_kinematic" in hit for hit in forbidden_hits)
    assert any("forbidden marker" in detail for detail in details)


def test_run_probe_strict_patch_sets_include_required_and_forbidden(tmp_path: Path) -> None:
    root = tmp_path / "geniesim"
    grpc_file = root / "source/data_collection/server/grpc_server.py"
    cmd_file = root / "source/data_collection/server/command_controller.py"

    _write(
        grpc_file,
        (
            "# BlueprintPipeline contact_report patch\n"
            "# BlueprintPipeline joint_efforts patch\n"
        ),
    )
    _write(
        cmd_file,
        (
            "# BlueprintPipeline object_pose patch\n"
            "# BlueprintPipeline sim_thread_physics_cache patch\n"
            "# BlueprintPipeline contact_reporting_on_init patch\n"
            "# BPv3_pre_play_kinematic\n"
            "# BPv4_deferred_dynamic_restore\n"
            "# BPv5_dynamic_teleport_usd_objects\n"
            "# BPv6_fix_dynamic_prims\n"
            "# [PATCH] scene_collision_injected\n"
            "# object_pose_resolver_v4\n"
        ),
    )
    physics_report = tmp_path / "physics_report.json"
    physics_report.write_text(
        '{"objects_with_manifest_physics": 5, "objects_with_usd_physics": 5, "coverage": 1.0, "missing_physics": [], "mesh_prims_total": 20, "mesh_prims_with_collision": 20, "mesh_prims_bad_dynamic_approx": 0, "collision_coverage": 1.0}'
    )

    args = Namespace(
        host="localhost",
        port="50051",
        timeout=0.01,
        geniesim_root=str(root),
        skip_grpc=True,
        check_patches=True,
        require_patch=[],
        forbid_patch=[],
        require_patch_set=[],
        forbid_patch_set=[],
        strict_runtime=True,
        physics_report=str(physics_report),
        min_physics_coverage=0.98,
        output="",
    )
    passed, payload = readiness_probe.run_probe(args)
    assert passed is True
    checks = payload.get("checks", [])
    patch_checks = [c for c in checks if c.get("name") == "runtime_patch_markers"]
    assert len(patch_checks) == 1
    assert patch_checks[0]["passed"] is True
    assert "strict" in patch_checks[0].get("required_sets", [])
    assert "strict" in patch_checks[0].get("forbidden_sets", [])
    assert patch_checks[0].get("forbidden_hits") == []


def test_run_probe_strict_patch_sets_fail_without_object_pose_resolver_v4(tmp_path: Path) -> None:
    root = tmp_path / "geniesim"
    grpc_file = root / "source/data_collection/server/grpc_server.py"
    cmd_file = root / "source/data_collection/server/command_controller.py"

    _write(
        grpc_file,
        (
            "# BlueprintPipeline contact_report patch\n"
            "# BlueprintPipeline joint_efforts patch\n"
        ),
    )
    _write(
        cmd_file,
        (
            "# BlueprintPipeline object_pose patch\n"
            "# BlueprintPipeline sim_thread_physics_cache patch\n"
            "# BlueprintPipeline contact_reporting_on_init patch\n"
            "# BPv3_pre_play_kinematic\n"
            "# BPv4_deferred_dynamic_restore\n"
            "# BPv5_dynamic_teleport_usd_objects\n"
            "# BPv6_fix_dynamic_prims\n"
            "# [PATCH] scene_collision_injected\n"
        ),
    )
    physics_report = tmp_path / "physics_report.json"
    physics_report.write_text(
        '{"objects_with_manifest_physics": 2, "objects_with_usd_physics": 2, "coverage": 1.0, "missing_physics": [], "mesh_prims_total": 4, "mesh_prims_with_collision": 4, "mesh_prims_bad_dynamic_approx": 0, "collision_coverage": 1.0}'
    )

    args = Namespace(
        host="localhost",
        port="50051",
        timeout=0.01,
        geniesim_root=str(root),
        skip_grpc=True,
        check_patches=True,
        require_patch=[],
        forbid_patch=[],
        require_patch_set=[],
        forbid_patch_set=[],
        strict_runtime=True,
        physics_report=str(physics_report),
        min_physics_coverage=0.98,
        output="",
    )
    passed, payload = readiness_probe.run_probe(args)
    assert passed is False
    checks = payload.get("checks", [])
    patch_checks = [c for c in checks if c.get("name") == "runtime_patch_markers"]
    assert len(patch_checks) == 1
    assert patch_checks[0]["passed"] is False
    missing = patch_checks[0].get("missing", [])
    assert any("object_pose_resolver_v4" in marker for marker in missing)


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


def test_check_physics_coverage_report_strict_collision_gate(tmp_path: Path) -> None:
    report = tmp_path / "scene.physics_report.json"
    report.write_text(
        '{"objects_with_manifest_physics": 10, "objects_with_usd_physics": 10, "coverage": 1.0, "missing_physics": [], "mesh_prims_total": 50, "mesh_prims_with_collision": 49, "mesh_prims_bad_dynamic_approx": 2, "collision_coverage": 0.98}'
    )
    ok, _, errors = readiness_probe.check_physics_coverage_report(
        report,
        min_coverage=0.98,
        strict_collision=True,
    )
    assert ok is False
    assert any("collision coverage below strict threshold" in e for e in errors)
    assert any("invalid dynamic approximation" in e for e in errors)


def test_run_probe_strict_runtime_requires_physics_report() -> None:
    args = Namespace(
        host="localhost",
        port="50051",
        timeout=0.01,
        geniesim_root="/tmp/does-not-matter",
        skip_grpc=True,
        check_patches=False,
        require_patch=[],
        forbid_patch=[],
        require_patch_set=[],
        forbid_patch_set=[],
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
