import argparse
from pathlib import Path

import numpy as np
import pytest


class _AnnotatorStub:
    def __init__(self, value):
        self._value = value

    def get_data(self):
        return self._value


@pytest.mark.unit
def test_extract_room_dict_room_payload_passthrough(tmp_path):
    from scripts.runpod_sage import isaacsim_collect_mobile_franka as c

    room = {"objects": [{"source_id": "obj_1"}], "dimensions": {"width": 1}}
    out = c._extract_room_dict(room, demo={}, variant_path=tmp_path / "variant.json")
    assert out is room


@pytest.mark.unit
def test_extract_room_dict_single_room_layout(tmp_path):
    from scripts.runpod_sage import isaacsim_collect_mobile_franka as c

    payload = {"rooms": [{"id": "room_0", "objects": [{"source_id": "salt_0"}]}]}
    out = c._extract_room_dict(payload, demo={}, variant_path=tmp_path / "variant.json")
    assert out["id"] == "room_0"


@pytest.mark.unit
def test_extract_room_dict_multi_room_by_pick_place_ids(tmp_path):
    from scripts.runpod_sage import isaacsim_collect_mobile_franka as c

    payload = {
        "rooms": [
            {"id": "room_a", "objects": [{"source_id": "chair_0"}]},
            {"id": "room_b", "objects": [{"source_id": "salt_0"}, {"source_id": "table_0"}]},
        ]
    }
    demo = {
        "pick_object": {"source_id": "salt_0"},
        "place_surface": {"source_id": "table_0"},
    }
    out = c._extract_room_dict(payload, demo=demo, variant_path=tmp_path / "variant.json")
    assert out["id"] == "room_b"


@pytest.mark.unit
def test_extract_room_dict_empty_rooms_raises(tmp_path):
    from scripts.runpod_sage import isaacsim_collect_mobile_franka as c

    with pytest.raises(RuntimeError):
        c._extract_room_dict({"rooms": []}, demo={}, variant_path=tmp_path / "variant.json")


@pytest.mark.unit
def test_annotator_data_supports_ndarray_and_dict():
    from scripts.runpod_sage import isaacsim_collect_mobile_franka as c

    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    assert c._annotator_data(_AnnotatorStub(arr), name="rgb").shape == (4, 4, 3)
    assert c._annotator_data(_AnnotatorStub({"data": arr}), name="rgb").shape == (4, 4, 3)
    assert c._annotator_data(_AnnotatorStub({"alt": arr}), name="rgb").shape == (4, 4, 3)


@pytest.mark.unit
def test_annotator_data_invalid_dict_raises():
    from scripts.runpod_sage import isaacsim_collect_mobile_franka as c

    with pytest.raises(RuntimeError):
        c._annotator_data(_AnnotatorStub({"meta": "x"}), name="rgb")


@pytest.mark.unit
def test_sensor_failure_policy_resolution():
    from scripts.runpod_sage import isaacsim_collect_mobile_franka as c

    assert c._resolve_sensor_failure_policy(strict=True, requested="auto") == "fail"
    assert c._resolve_sensor_failure_policy(strict=False, requested="auto") == "warn"
    assert c._resolve_sensor_failure_policy(strict=True, requested="warn") == "warn"


@pytest.mark.unit
def test_resolve_ridgeback_franka_usd_strict_local(monkeypatch, tmp_path):
    from scripts.runpod_sage import isaacsim_collect_mobile_franka as c

    root = tmp_path / "Assets" / "Isaac" / "5.1"
    monkeypatch.setenv("ISAAC_ASSETS_ROOT", str(root))
    monkeypatch.setenv("SAGE_ALLOW_REMOTE_ISAAC_ASSETS", "0")
    with pytest.raises(FileNotFoundError):
        c._resolve_ridgeback_franka_usd()


@pytest.mark.unit
def test_resolve_ridgeback_franka_usd_remote_opt_in(monkeypatch, tmp_path):
    from scripts.runpod_sage import isaacsim_collect_mobile_franka as c

    root = tmp_path / "Assets" / "Isaac" / "5.1"
    monkeypatch.setenv("ISAAC_ASSETS_ROOT", str(root))
    monkeypatch.setenv("SAGE_ALLOW_REMOTE_ISAAC_ASSETS", "1")
    assert c._resolve_ridgeback_franka_usd() == "/Isaac/Robots/Clearpath/RidgebackFranka/ridgeback_franka.usd"


@pytest.mark.unit
def test_resolve_ridgeback_franka_usd_local(monkeypatch, tmp_path):
    from scripts.runpod_sage import isaacsim_collect_mobile_franka as c

    root = tmp_path / "Assets" / "Isaac" / "5.1"
    robot = root / "Isaac" / "Robots" / "Clearpath" / "RidgebackFranka" / "ridgeback_franka.usd"
    robot.parent.mkdir(parents=True, exist_ok=True)
    robot.write_text("#usda 1.0\n", encoding="utf-8")
    monkeypatch.setenv("ISAAC_ASSETS_ROOT", str(root))
    monkeypatch.setenv("SAGE_ALLOW_REMOTE_ISAAC_ASSETS", "0")
    assert c._resolve_ridgeback_franka_usd() == str(robot)


@pytest.mark.unit
def test_mcp_path_resolution_new_path(tmp_path):
    from scripts.runpod_sage import mcp_extension_paths as m

    sage_dir = tmp_path / "SAGE"
    new_path = sage_dir / "server" / "isaacsim_mcp_ext" / "isaac.sim.mcp_extension"
    new_path.mkdir(parents=True, exist_ok=True)
    result = m.resolve_mcp_extension_path(sage_dir, migrate_legacy=True)
    assert result.resolved_path == new_path
    assert result.state in {"new", "new_with_legacy"}


@pytest.mark.unit
def test_mcp_path_resolution_migrates_legacy(tmp_path):
    from scripts.runpod_sage import mcp_extension_paths as m

    sage_dir = tmp_path / "SAGE"
    legacy_path = sage_dir / "server" / "isaacsim" / "isaac.sim.mcp_extension"
    legacy_path.mkdir(parents=True, exist_ok=True)
    (legacy_path / "placeholder.txt").write_text("ok\n", encoding="utf-8")

    result = m.resolve_mcp_extension_path(sage_dir, migrate_legacy=True)
    expected_new = sage_dir / "server" / "isaacsim_mcp_ext" / "isaac.sim.mcp_extension"
    assert result.resolved_path == expected_new
    assert result.state == "migrated"
    assert expected_new.exists()


@pytest.mark.unit
def test_mcp_path_resolution_legacy_unmigrated_on_move_error(tmp_path, monkeypatch):
    from scripts.runpod_sage import mcp_extension_paths as m

    sage_dir = tmp_path / "SAGE"
    legacy_path = sage_dir / "server" / "isaacsim" / "isaac.sim.mcp_extension"
    legacy_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(m.shutil, "move", lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("denied")))

    result = m.resolve_mcp_extension_path(sage_dir, migrate_legacy=True)
    assert result.resolved_path == legacy_path
    assert result.state == "legacy_unmigrated"


@pytest.mark.unit
def test_validate_ridgeback_assets_transitive_success(tmp_path):
    from scripts.runpod_sage import validate_ridgeback_usd_assets as v

    assets_root = tmp_path / "Assets" / "Isaac" / "5.1"
    root = assets_root / "Isaac" / "Robots" / "Clearpath" / "RidgebackFranka" / "ridgeback_franka.usd"
    child = root.parent / "child.usda"
    tex = root.parent / "tex.png"
    root.parent.mkdir(parents=True, exist_ok=True)
    root.write_text('#usda 1.0\ndef Xform "Root" (references = @child.usda@) {}\n', encoding="utf-8")
    child.write_text('#usda 1.0\ndef Xform "Child" (asset inputs:file = @tex.png@) {}\n', encoding="utf-8")
    tex.write_bytes(b"PNG")

    report = v.validate_ridgeback_assets(root, assets_root=assets_root)
    assert report["ok"] is True
    assert report["missing_count"] == 0
    assert report["visited_usd_count"] >= 2


@pytest.mark.unit
def test_validate_ridgeback_assets_transitive_missing(tmp_path):
    from scripts.runpod_sage import validate_ridgeback_usd_assets as v

    assets_root = tmp_path / "Assets" / "Isaac" / "5.1"
    root = assets_root / "Isaac" / "Robots" / "Clearpath" / "RidgebackFranka" / "ridgeback_franka.usd"
    root.parent.mkdir(parents=True, exist_ok=True)
    root.write_text('#usda 1.0\ndef Xform "Root" (references = @missing_child.usda@) {}\n', encoding="utf-8")

    report = v.validate_ridgeback_assets(root, assets_root=assets_root)
    assert report["ok"] is False
    assert report["missing_count"] >= 1


@pytest.mark.unit
def test_stage7_command_env_assembly_defaults(monkeypatch, tmp_path):
    from scripts.runpod_sage import sage_stage567_mobile_franka as s

    monkeypatch.delenv("SAGE_ALLOW_REMOTE_ISAAC_ASSETS", raising=False)
    monkeypatch.delenv("SAGE_SENSOR_FAILURE_POLICY", raising=False)
    monkeypatch.delenv("SAGE_RENDER_WARMUP_FRAMES", raising=False)
    monkeypatch.delenv("SAGE_SENSOR_MIN_RGB_STD", raising=False)
    monkeypatch.delenv("SAGE_SENSOR_MIN_DEPTH_STD", raising=False)
    monkeypatch.delenv("SAGE_MIN_VALID_DEPTH_PX", raising=False)
    monkeypatch.delenv("SAGE_SENSOR_CHECK_FRAME", raising=False)
    monkeypatch.delenv("SAGE_ENFORCE_BUNDLE_STRICT", raising=False)
    monkeypatch.delenv("SAGE_RUN_ID", raising=False)

    args = argparse.Namespace(
        isaacsim_py="/tmp/isaacsim_env/bin/python3",
        headless=True,
        enable_cameras=True,
        strict=True,
    )
    collector = tmp_path / "collector.py"
    collector.write_text("print('collector')\n", encoding="utf-8")
    plan_path = tmp_path / "plan.json"
    plan_path.write_text("{}\n", encoding="utf-8")
    output_dir = tmp_path / "demos"

    cmd, env = s._build_stage7_command_and_env(
        args=args,
        collector=collector,
        plan_path=plan_path,
        output_dir=output_dir,
        run_id="run_test_123",
    )

    assert cmd[:3] == ["/tmp/isaacsim_env/bin/python3", "-P", str(collector)]
    assert "--strict" in cmd
    assert "--headless" in cmd
    assert "--enable_cameras" in cmd
    assert env["SAGE_ALLOW_REMOTE_ISAAC_ASSETS"] == "0"
    assert env["SAGE_SENSOR_FAILURE_POLICY"] == "fail"
    assert env["SAGE_RENDER_WARMUP_FRAMES"] == "100"
    assert env["SAGE_SENSOR_MIN_RGB_STD"] == "0.01"
    assert env["SAGE_SENSOR_MIN_DEPTH_STD"] == "0.0001"
    assert env["SAGE_MIN_VALID_DEPTH_PX"] == "1024"
    assert env["SAGE_SENSOR_CHECK_FRAME"] == "10"
    assert env["SAGE_ENFORCE_BUNDLE_STRICT"] == "1"
    assert env["SAGE_RUN_ID"] == "run_test_123"


@pytest.mark.unit
def test_bundle_runtime_mismatch_detection_and_enforcement(monkeypatch):
    from scripts.runpod_sage import isaacsim_collect_mobile_franka as c

    bundle = {"strict": False, "headless": True, "enable_cameras": True}
    args = argparse.Namespace(strict=True, headless=True, enable_cameras=True)
    mismatches = c._bundle_runtime_mismatches(bundle, args)
    assert mismatches
    assert all(c.BUNDLE_RUNTIME_MISMATCH_MARKER in m for m in mismatches)
    with pytest.raises(RuntimeError):
        c._enforce_bundle_runtime_parity(bundle, args)

    monkeypatch.setenv("SAGE_ENFORCE_BUNDLE_STRICT", "0")
    c._enforce_bundle_runtime_parity(bundle, argparse.Namespace(strict=False, headless=True, enable_cameras=True))

    monkeypatch.setenv("SAGE_ENFORCE_BUNDLE_STRICT", "1")
    mismatch_bundle = {"strict": True, "headless": True, "enable_cameras": True}
    with pytest.raises(RuntimeError):
        c._enforce_bundle_runtime_parity(
            mismatch_bundle,
            argparse.Namespace(strict=False, headless=True, enable_cameras=True),
        )


@pytest.mark.unit
def test_dual_camera_sensor_qc_pass_and_fail_cases():
    from scripts.runpod_sage import isaacsim_collect_mobile_franka as c

    rgb_ok = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb_ok[..., 0] = np.arange(4, dtype=np.uint8)[:, None]
    depth_ok = np.linspace(0.5, 2.0, 16, dtype=np.float32).reshape(4, 4)

    qc_ok = c._evaluate_dual_camera_sensor_qc(
        agentview_rgb=rgb_ok,
        agentview_depth_raw=depth_ok,
        agentview2_rgb=rgb_ok,
        agentview2_depth_raw=depth_ok,
        min_valid_depth_px=4,
        min_rgb_std=0.01,
        min_depth_std=0.0001,
    )
    assert qc_ok["status"] == "pass"
    assert qc_ok["failures"] == []

    depth_bad = np.zeros((4, 4), dtype=np.float32)
    qc_bad = c._evaluate_dual_camera_sensor_qc(
        agentview_rgb=rgb_ok,
        agentview_depth_raw=depth_ok,
        agentview2_rgb=rgb_ok,
        agentview2_depth_raw=depth_bad,
        min_valid_depth_px=4,
        min_rgb_std=0.01,
        min_depth_std=0.0001,
    )
    assert qc_bad["status"] == "fail"
    assert "agentview_2:degenerate_depth_valid_px" in qc_bad["failures"]


@pytest.mark.unit
def test_collision_approximation_normalization_and_validation():
    from scripts.runpod_sage import isaacsim_collect_mobile_franka as c

    assert c._normalize_dynamic_collision_approximation("convexHull") == "convexHull"
    assert c._normalize_dynamic_collision_approximation("convex_decomposition") == "convexDecomposition"
    assert c._normalize_dynamic_collision_approximation("MeshSimplification") == "convexDecomposition"
    assert c._is_valid_dynamic_collision_approximation("convexHull")
    assert c._is_valid_dynamic_collision_approximation("convexDecomposition")
    assert not c._is_valid_dynamic_collision_approximation("MeshSimplification")
