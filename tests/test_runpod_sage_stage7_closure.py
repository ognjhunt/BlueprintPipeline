import argparse
import json
import os
import subprocess
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
def test_collector_resolve_headless_mode_streaming(monkeypatch):
    from scripts.runpod_sage import isaacsim_collect_mobile_franka as c

    monkeypatch.delenv("DISPLAY", raising=False)
    headless, mode = c._resolve_headless_mode("streaming")
    assert headless is True
    assert mode == "streaming"


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

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.delenv("SAGE_KEEP_CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("SAGE_ALLOW_REMOTE_ISAAC_ASSETS", raising=False)
    monkeypatch.delenv("SAGE_SENSOR_FAILURE_POLICY", raising=False)
    monkeypatch.delenv("SAGE_STRICT_SENSORS", raising=False)
    monkeypatch.delenv("SAGE_RENDER_WARMUP_FRAMES", raising=False)
    monkeypatch.delenv("SAGE_SENSOR_MIN_RGB_STD", raising=False)
    monkeypatch.delenv("SAGE_SENSOR_MIN_DEPTH_STD", raising=False)
    monkeypatch.delenv("SAGE_MIN_DEPTH_FINITE_RATIO", raising=False)
    monkeypatch.delenv("SAGE_MAX_RGB_SATURATION_RATIO", raising=False)
    monkeypatch.delenv("SAGE_MIN_DEPTH_RANGE_M", raising=False)
    monkeypatch.delenv("SAGE_MIN_VALID_DEPTH_PX", raising=False)
    monkeypatch.delenv("SAGE_SENSOR_CHECK_FRAME", raising=False)
    monkeypatch.delenv("SAGE_EXPORT_SCENE_USD", raising=False)
    monkeypatch.delenv("SAGE_EXPORT_DEMO_VIDEOS", raising=False)
    monkeypatch.delenv("SAGE_QUALITY_REPORT_PATH", raising=False)
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
    assert "--strict-sensors" in cmd
    assert "--headless-mode" in cmd
    assert "--headless" in cmd
    assert "--enable_cameras" in cmd
    assert "--export-scene-usd" in cmd
    assert "--export-demo-videos" in cmd
    assert "--quality-report-path" in cmd
    assert env["SAGE_ALLOW_REMOTE_ISAAC_ASSETS"] == "0"
    assert env["SAGE_SENSOR_FAILURE_POLICY"] == "fail"
    assert env["SAGE_STRICT_SENSORS"] == "1"
    assert env["SAGE_RENDER_WARMUP_FRAMES"] == "100"
    assert env["SAGE_SENSOR_MIN_RGB_STD"] == "5.0"
    assert env["SAGE_SENSOR_MIN_DEPTH_STD"] == "0.0001"
    assert env["SAGE_MIN_DEPTH_FINITE_RATIO"] == "0.98"
    assert env["SAGE_MAX_RGB_SATURATION_RATIO"] == "0.85"
    assert env["SAGE_MIN_DEPTH_RANGE_M"] == "0.05"
    assert env["SAGE_MIN_VALID_DEPTH_PX"] == "1024"
    assert env["SAGE_SENSOR_CHECK_FRAME"] == "10"
    assert env["SAGE_EXPORT_SCENE_USD"] == "1"
    assert env["SAGE_EXPORT_DEMO_VIDEOS"] == "1"
    assert env["SAGE_QUALITY_REPORT_PATH"].endswith("/quality_report.json")
    assert env["SAGE_ENFORCE_BUNDLE_STRICT"] == "1"
    assert env["SAGE_RUN_ID"] == "run_test_123"
    assert "CUDA_VISIBLE_DEVICES" not in env


@pytest.mark.unit
def test_stage7_command_env_streaming_mode(monkeypatch, tmp_path):
    from scripts.runpod_sage import sage_stage567_mobile_franka as s

    monkeypatch.setenv("SAGE_STAGE7_HEADLESS_MODE", "streaming")
    monkeypatch.setenv("SAGE_STAGE7_STREAMING_PORT", "49100")
    args = argparse.Namespace(
        isaacsim_py="/tmp/isaacsim_env/bin/python3",
        headless=True,
        stage7_headless_mode="streaming",
        enable_cameras=True,
        strict=False,
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
        run_id="run_test_streaming",
    )

    assert "--headless-mode" in cmd
    assert "streaming" in cmd
    assert "--headless" in cmd
    assert env["SAGE_STAGE7_HEADLESS_MODE"] == "streaming"


@pytest.mark.unit
def test_stage7_probe_mode_order_auto_without_display(monkeypatch):
    from scripts.runpod_sage import sage_stage567_mobile_franka as s

    monkeypatch.delenv("DISPLAY", raising=False)
    modes = s._resolve_stage7_probe_mode_order(
        requested_mode="auto",
        mode_order_raw="auto",
        env={},
        streaming_enabled=True,
    )
    assert modes == ["streaming", "headless"]


@pytest.mark.unit
def test_stage7_probe_mode_order_auto_with_display(monkeypatch):
    from scripts.runpod_sage import sage_stage567_mobile_franka as s

    env = {"DISPLAY": ":0"}
    monkeypatch.setattr(s, "_display_server_ready_for_mode_probe", lambda _env: True)
    modes = s._resolve_stage7_probe_mode_order(
        requested_mode="auto",
        mode_order_raw="auto",
        env=env,
        streaming_enabled=True,
    )
    assert modes == ["windowed", "streaming", "headless"]


@pytest.mark.unit
def test_stage7_probe_mode_order_explicit_streaming_disabled_raises():
    from scripts.runpod_sage import sage_stage567_mobile_franka as s

    with pytest.raises(RuntimeError):
        s._resolve_stage7_probe_mode_order(
            requested_mode="streaming",
            mode_order_raw="auto",
            env={},
            streaming_enabled=False,
        )


@pytest.mark.unit
def test_stage7_probe_selects_first_passing_mode(monkeypatch, tmp_path):
    from scripts.runpod_sage import sage_stage567_mobile_franka as s

    plan_path = tmp_path / "plan_bundle.json"
    plan_path.write_text(
        json.dumps(
            {
                "layout_id": "layout_x",
                "strict": True,
                "enable_cameras": True,
                "demos": [{"demo_idx": 0}, {"demo_idx": 1}],
            }
        ),
        encoding="utf-8",
    )
    collector = tmp_path / "collector.py"
    collector.write_text("print('collector')\n", encoding="utf-8")
    output_root = tmp_path / "probe"
    report_path = tmp_path / "quality" / "stage7_mode_probe.json"
    args = argparse.Namespace(
        layout_id="layout_x",
        isaacsim_py="/tmp/isaacsim_env/bin/python3",
        headless=True,
        stage7_headless_mode="auto",
        enable_cameras=True,
        strict=True,
    )

    monkeypatch.setenv("SAGE_STAGE7_MODE_ORDER", "streaming,headless")
    monkeypatch.setenv("SAGE_STAGE7_PROBE_DEMOS", "1")
    monkeypatch.setenv("SAGE_STAGE7_PROBE_TIMEOUT_S", "10")
    monkeypatch.setenv("SAGE_STAGE7_PROBE_KEEP_ARTIFACTS", "1")
    monkeypatch.setenv("SAGE_STAGE7_STREAMING_ENABLED", "1")

    def _fake_build(*, args, collector, plan_path, output_dir, run_id):
        return (
            ["/tmp/fake_python", "-P", str(collector), "--output_dir", str(output_dir), "--headless-mode", str(args.stage7_headless_mode)],
            {"SAGE_STAGE7_HEADLESS_MODE": str(args.stage7_headless_mode)},
        )

    outcomes = {"streaming": (1, False, 1.2), "headless": (0, False, 1.0)}

    def _fake_run(*, cmd, env, cwd, log_path, timeout_s):
        mode = str(env.get("SAGE_STAGE7_HEADLESS_MODE", ""))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("probe-log\n", encoding="utf-8")
        return outcomes[mode]

    def _fake_contract(*, output_dir, expected_demos, enable_cameras, require_valid_rgb):
        mode = Path(output_dir).name.replace("mode_", "")
        if mode == "headless":
            return True, []
        return False, ["rgb_std_min=0.0"]

    monkeypatch.setattr(s, "_build_stage7_command_and_env", _fake_build)
    monkeypatch.setattr(s, "_run_stage7_probe_process", _fake_run)
    monkeypatch.setattr(s, "_stage7_output_contract_ok", _fake_contract)
    monkeypatch.setattr(
        s,
        "_read_json_if_exists",
        lambda path: {"status": "pass", "summary": {"rgb_std_min": 12.0}, "video_report": {"exported_demos": 1}},
    )

    report = s.run_stage7_rgb_mode_probe(
        args=args,
        collector=collector,
        plan_path=plan_path,
        output_root=output_root,
        report_path=report_path,
        run_id="run_probe_pass",
    )

    assert report["status"] == "pass"
    assert report["selected_mode"] == "headless"
    assert report_path.exists()


@pytest.mark.unit
def test_stage7_probe_all_modes_fail(monkeypatch, tmp_path):
    from scripts.runpod_sage import sage_stage567_mobile_franka as s

    plan_path = tmp_path / "plan_bundle.json"
    plan_path.write_text(
        json.dumps({"layout_id": "layout_x", "strict": True, "enable_cameras": True, "demos": [{"demo_idx": 0}]}),
        encoding="utf-8",
    )
    collector = tmp_path / "collector.py"
    collector.write_text("print('collector')\n", encoding="utf-8")
    output_root = tmp_path / "probe"
    report_path = tmp_path / "quality" / "stage7_mode_probe.json"
    args = argparse.Namespace(
        layout_id="layout_x",
        isaacsim_py="/tmp/isaacsim_env/bin/python3",
        headless=True,
        stage7_headless_mode="auto",
        enable_cameras=True,
        strict=True,
    )

    monkeypatch.setenv("SAGE_STAGE7_MODE_ORDER", "streaming,headless")
    monkeypatch.setenv("SAGE_STAGE7_PROBE_DEMOS", "1")
    monkeypatch.setenv("SAGE_STAGE7_PROBE_TIMEOUT_S", "10")
    monkeypatch.setenv("SAGE_STAGE7_PROBE_KEEP_ARTIFACTS", "1")
    monkeypatch.setenv("SAGE_STAGE7_STREAMING_ENABLED", "1")

    monkeypatch.setattr(
        s,
        "_build_stage7_command_and_env",
        lambda **kwargs: (["/tmp/fake_python"], {"SAGE_STAGE7_HEADLESS_MODE": str(kwargs["args"].stage7_headless_mode)}),
    )
    monkeypatch.setattr(
        s,
        "_run_stage7_probe_process",
        lambda **kwargs: (1, False, 1.0),
    )
    monkeypatch.setattr(
        s,
        "_stage7_output_contract_ok",
        lambda **kwargs: (False, ["rgb_std_min=0.0", "missing_videos=1"]),
    )
    monkeypatch.setattr(s, "_read_json_if_exists", lambda path: None)

    report = s.run_stage7_rgb_mode_probe(
        args=args,
        collector=collector,
        plan_path=plan_path,
        output_root=output_root,
        report_path=report_path,
        run_id="run_probe_fail",
    )
    assert report["status"] == "fail"
    assert report["selected_mode"] is None
    assert report_path.exists()


@pytest.mark.unit
def test_stage7_command_env_respects_keep_cuda_visible_devices(monkeypatch, tmp_path):
    from scripts.runpod_sage import sage_stage567_mobile_franka as s

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("SAGE_KEEP_CUDA_VISIBLE_DEVICES", "1")

    args = argparse.Namespace(
        isaacsim_py="/tmp/isaacsim_env/bin/python3",
        headless=True,
        enable_cameras=True,
        strict=False,
    )
    collector = tmp_path / "collector.py"
    collector.write_text("print('collector')\n", encoding="utf-8")
    plan_path = tmp_path / "plan.json"
    plan_path.write_text("{}\n", encoding="utf-8")
    output_dir = tmp_path / "demos"

    _cmd, env = s._build_stage7_command_and_env(
        args=args,
        collector=collector,
        plan_path=plan_path,
        output_dir=output_dir,
        run_id="run_test_keep_cuda",
    )
    assert "--headless-mode" in _cmd
    assert env["CUDA_VISIBLE_DEVICES"] == "0"


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
        min_depth_finite_ratio=0.5,
        min_valid_depth_px=4,
        min_rgb_std=0.01,
        max_rgb_saturation_ratio=0.95,
        min_depth_std=0.0001,
        min_depth_range_m=0.1,
    )
    assert qc_ok["status"] == "pass"
    assert qc_ok["failures"] == []

    depth_bad = np.zeros((4, 4), dtype=np.float32)
    qc_bad = c._evaluate_dual_camera_sensor_qc(
        agentview_rgb=rgb_ok,
        agentview_depth_raw=depth_ok,
        agentview2_rgb=rgb_ok,
        agentview2_depth_raw=depth_bad,
        min_depth_finite_ratio=0.5,
        min_valid_depth_px=4,
        min_rgb_std=0.01,
        max_rgb_saturation_ratio=0.95,
        min_depth_std=0.0001,
        min_depth_range_m=0.1,
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


@pytest.mark.unit
def test_stage5_grasp_retry_meets_threshold(monkeypatch, tmp_path):
    from scripts.runpod_sage import sage_stage567_mobile_franka as s

    calls = {"count": 0}

    def _fake_infer(*, layout_id, layout_dir, variant_layout_json, pick_obj_id, base_pos, room_id, num_views, strict, model, cfg):
        calls["count"] += 1
        if calls["count"] == 1:
            return np.zeros((2, 4, 4), dtype=np.float32), np.zeros((2, 3), dtype=np.float32)
        return np.zeros((12, 4, 4), dtype=np.float32), np.zeros((12, 3), dtype=np.float32)

    monkeypatch.setattr(s, "_infer_grasps_for_variant", _fake_infer)
    grasps, contacts, attempts = s._infer_grasps_with_retries(
        layout_id="layout_x",
        layout_dir=tmp_path,
        variant_layout_json=tmp_path / "variant.json",
        pick_obj_id="obj_1",
        base_pos=(1.0, 2.0, 0.0),
        room_id="room_0",
        num_views=1,
        strict=True,
        model=object(),
        cfg=object(),
        min_grasps_per_object=10,
        max_retries=3,
    )
    assert grasps.shape[0] == 12
    assert contacts.shape[0] == 12
    assert len(attempts) == 2
    assert attempts[0]["num_grasps"] == 2
    assert attempts[1]["num_grasps"] == 12


@pytest.mark.unit
def test_fix_layout_json_synthesizes_stage4_fields():
    from scripts.runpod_sage import fix_layout_json as f

    layout = {
        "rooms": [
            {
                "id": "room_0",
                "dimensions": {"width": 6.0, "length": 6.0, "height": 3.0},
                "objects": [
                    {"id": "salt_0", "type": "salt_shaker", "position": {"x": 2.0, "y": 2.0, "z": 0.8}},
                    {"id": "counter_0", "type": "kitchen_counter", "position": {"x": 2.0, "y": 2.0, "z": 0.0}},
                    {"id": "table_0", "type": "dining_table", "position": {"x": 4.0, "y": 4.0, "z": 0.0}},
                ],
            }
        ]
    }
    out = f.patch_layout_dict(
        layout,
        task_desc="Pick up the salt from the counter and place it on the dining table",
        require_stage4_fields=True,
    )
    pa = out["policy_analysis"]
    assert isinstance(pa.get("minimum_required_objects"), list) and pa["minimum_required_objects"]
    assert isinstance(pa.get("task_decomposition"), list) and pa["task_decomposition"]
    utd = pa.get("updated_task_decomposition")
    assert isinstance(utd, list) and utd
    pick_steps = [s for s in utd if str(s.get("action", "")).lower() == "pick"]
    place_steps = [s for s in utd if str(s.get("action", "")).lower() == "place"]
    assert pick_steps and place_steps
    assert pick_steps[0].get("target_object_id") == "salt_0"
    assert place_steps[0].get("location_object_id") == "table_0"


@pytest.mark.unit
def test_validate_ridgeback_assets_handles_pathological_binary_refs(tmp_path):
    from scripts.runpod_sage import validate_ridgeback_usd_assets as v

    assets_root = tmp_path / "Assets" / "Isaac" / "5.1"
    root = assets_root / "Isaac" / "Robots" / "Clearpath" / "RidgebackFranka" / "ridgeback_franka.usdc"
    root.parent.mkdir(parents=True, exist_ok=True)
    # Include an absurdly long token that previously triggered path-length issues.
    long_ref = "A" * 12000
    root.write_bytes(b"PXR-USDC\n@" + long_ref.encode("ascii") + b"@\n")
    report = v.validate_ridgeback_assets(root, assets_root=assets_root)
    assert isinstance(report, dict)
    assert report["visited_usd_count"] >= 1
    assert report["checked_references_count"] >= 0


@pytest.mark.unit
def test_nav_gate_prefers_explicit_task_anchors():
    from scripts.runpod_sage import robot_nav_gate as g

    layout = {
        "rooms": [
            {
                "id": "room_0",
                "dimensions": {"width": 6.0, "length": 6.0, "height": 3.0},
                "objects": [
                    {
                        "id": "salt_0",
                        "type": "salt_shaker",
                        "dimensions": {"width": 0.08, "length": 0.08, "height": 0.16},
                        "position": {"x": 1.8, "y": 1.7, "z": 0.9},
                    },
                    {
                        "id": "kitchen_island_0",
                        "type": "kitchen_island",
                        "dimensions": {"width": 1.2, "length": 0.8, "height": 0.9},
                        "position": {"x": 1.8, "y": 1.7, "z": 0.0},
                    },
                    {
                        "id": "dining_table_0",
                        "type": "dining_table",
                        "dimensions": {"width": 1.4, "length": 0.9, "height": 0.75},
                        "position": {"x": 4.2, "y": 4.2, "z": 0.0},
                    },
                ],
            }
        ]
    }
    report = g.run_gate(
        layout=layout,
        grid_res_m=0.05,
        pick_radius_min_m=0.55,
        pick_radius_max_m=1.40,
        robot_radius_m=0.32,
        obstacle_min_size_m=0.25,
        pick_object_id="salt_0",
        place_surface_id="dining_table_0",
        task_desc="Pick up salt and place it on the dining table",
        require_non_heuristic=True,
    )
    assert report["status"] == "pass"
    assert report["pick_object_id"] == "salt_0"
    assert report["place_surface_id"] == "dining_table_0"
    assert report["anchor_source"] == "cli_override"


@pytest.mark.unit
def test_stage7_nonzero_teardown_only_is_downgraded_when_contract_passes(tmp_path):
    from scripts.runpod_sage import sage_stage567_mobile_franka as s

    output_dir = tmp_path / "demos"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dataset.hdf5").write_bytes(b"ok")
    (output_dir / "artifact_manifest.json").write_text("{}", encoding="utf-8")
    (output_dir / "quality_report.json").write_text(
        (
            "{\n"
            '  "status": "pass",\n'
            '  "num_demos": 1,\n'
            '  "summary": {"rgb_std_min": 10.0, "frozen_observations": {"rgb_all_black_demos": 0}},\n'
            '  "artifact_contract": {"missing_videos": []},\n'
            '  "video_report": {"exported_demos": 1}\n'
            "}\n"
        ),
        encoding="utf-8",
    )
    log_path = tmp_path / "stage7.log"
    s._run_stage7_collector_with_tee(
        cmd=["bash", "-lc", "echo 'XIO:  fatal IO error 11'; exit 1"],
        env={"PATH": os.environ.get("PATH", "")},
        cwd=tmp_path,
        log_path=log_path,
        output_dir=output_dir,
        expected_demos=1,
        enable_cameras=True,
        require_valid_rgb=True,
    )


@pytest.mark.unit
def test_stage7_nonzero_still_fails_when_contract_invalid(tmp_path):
    from scripts.runpod_sage import sage_stage567_mobile_franka as s

    output_dir = tmp_path / "demos"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dataset.hdf5").write_bytes(b"ok")
    (output_dir / "artifact_manifest.json").write_text("{}", encoding="utf-8")
    (output_dir / "quality_report.json").write_text(
        (
            "{\n"
            '  "status": "fail",\n'
            '  "num_demos": 1,\n'
            '  "summary": {"rgb_std_min": 0.0, "frozen_observations": {"rgb_all_black_demos": 1}},\n'
            '  "artifact_contract": {"missing_videos": ["demo_0.mp4"]},\n'
            '  "video_report": {"exported_demos": 0}\n'
            "}\n"
        ),
        encoding="utf-8",
    )
    with pytest.raises(subprocess.CalledProcessError):
        s._run_stage7_collector_with_tee(
            cmd=["bash", "-lc", "echo 'XIO:  fatal IO error 11'; exit 1"],
            env={"PATH": os.environ.get("PATH", "")},
            cwd=tmp_path,
            log_path=tmp_path / "stage7_fail.log",
            output_dir=output_dir,
            expected_demos=1,
            enable_cameras=True,
            require_valid_rgb=True,
        )
