from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_command_controller(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """class CommandController:
    def _init_robot_cfg(self, stage):
            self._play()
            articulation = type("A", (), {"dof_names": []})()
            self.dof_names = articulation.dof_names

    def _set_object_pose(self, poses):
        for pose in poses:
            if pose["prim_path"] in self.usd_objects:
                self.usd_objects[pose["prim_path"]].set_world_pose(
                    pose["position"], pose["rotation"]
                )
            else:
                stage = omni.usd.get_context().get_stage()
                if not stage:
                    return
                prim = stage.GetPrimAtPath(pose["prim_path"])
                if not prim.IsValid():
                    continue
                translate_attr = prim.GetAttribute("xformOp:translate")
                orient_attr = prim.GetAttribute("xformOp:orient")
                if translate_attr:
                    translate_attr.Set(pose["position"])
                if orient_attr:
                    rotation_data = pose["rotation"]
                    quat_type = type(rotation_data)
                    orient_attr.Set(quat_type(*rotation_data))

    def on_physics_step(self, dt):
        pass

    def on_command_step(self):
        pass

    def handle_init_robot(self, req):
            self.data_to_send = "success"
"""
    )


def _write_grpc_server(path: Path, *, legacy_dispatch: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if legacy_dispatch:
        path.write_text(
            """class GrpcServer:
    def _bp_handle_dynamic_toggle(self, prim_path, is_dynamic):
        return True

    def set_task_metric(self, req, context):
        # BPv_dynamic_grasp_toggle — set_task_metric dispatch
        _metric_str = str(getattr(req, 'metric', '') or '')
        if _metric_str.startswith('bp::set_object_dynamic::'):
            return SetTaskMetricRsp(msg='bp::dynamic_toggle::ok::dummy')
        return SetTaskMetricRsp(msg='metric set')
"""
        )
    else:
        path.write_text(
            """class GrpcServer:
    def set_task_metric(self, req, context):
        return SetTaskMetricRsp(msg='metric set')
"""
        )


@pytest.mark.unit
def test_patch_register_scene_objects_idempotent_and_no_runtime_collision_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    geniesim_root = tmp_path / "geniesim"
    cc_path = geniesim_root / "source/data_collection/server/command_controller.py"
    _write_command_controller(cc_path)

    patch_path = (
        REPO_ROOT
        / "tools/geniesim_adapter/deployment/patches/patch_register_scene_objects.py"
    )
    mod = _load_module("patch_register_scene_objects_test", patch_path)
    monkeypatch.setenv("GENIESIM_ROOT", str(geniesim_root))
    mod.CC_PATH = str(cc_path)

    assert mod.apply() is True
    assert mod.apply() is True

    patched = cc_path.read_text()
    assert patched.count("BPv3_pre_play_kinematic") == 1
    assert "_kar.Set(False)" in patched
    assert "Pre-play collision: SKIPPED" not in patched
    assert "_bp_collision_applied_pre_play = True" not in patched


@pytest.mark.unit
def test_patch_scene_collision_validator_only_and_stable_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    geniesim_root = tmp_path / "geniesim"
    cc_path = geniesim_root / "source/data_collection/server/command_controller.py"
    _write_command_controller(cc_path)

    reg_patch_path = (
        REPO_ROOT
        / "tools/geniesim_adapter/deployment/patches/patch_register_scene_objects.py"
    )
    reg_mod = _load_module("patch_register_scene_objects_for_scene_collision_test", reg_patch_path)
    monkeypatch.setenv("GENIESIM_ROOT", str(geniesim_root))
    reg_mod.CC_PATH = str(cc_path)
    assert reg_mod.apply() is True

    scene_patch_path = (
        REPO_ROOT
        / "tools/geniesim_adapter/deployment/patches/patch_scene_collision.py"
    )
    scene_mod = _load_module("patch_scene_collision_test", scene_patch_path)
    scene_mod.TARGET = str(cc_path)

    scene_mod.apply()
    scene_mod.apply()

    patched = cc_path.read_text()
    assert patched.count("[PATCH] scene_collision_injected") == 1
    assert "CollisionAPI.Apply(" not in patched
    assert 'CreateAttribute("physics:approximation"' not in patched
    assert "strict collision validation failed" in patched

    hook_idx = patched.find("# [PATCH] scene_collision_hook")
    restore_idx = patched.find("# Restore dynamic objects after articulation is stable")
    assert hook_idx != -1 and restore_idx != -1
    assert hook_idx < restore_idx


@pytest.mark.unit
def test_patch_dynamic_grasp_toggle_injects_rsp_guard_fresh(tmp_path: Path) -> None:
    cc_path = tmp_path / "source/data_collection/server/command_controller.py"
    grpc_path = tmp_path / "source/data_collection/server/grpc_server.py"
    _write_command_controller(cc_path)
    _write_grpc_server(grpc_path, legacy_dispatch=False)

    patch_path = (
        REPO_ROOT
        / "tools/geniesim_adapter/deployment/patches/patch_dynamic_grasp_toggle.py"
    )
    mod = _load_module("patch_dynamic_grasp_toggle_fresh_test", patch_path)
    mod.CC_PATH = str(cc_path)
    mod.GRPC_PATH = str(grpc_path)

    assert mod.apply() is True
    assert mod.apply() is True

    patched = grpc_path.read_text()
    assert "# BPv_dynamic_grasp_toggle_rsp_guard" in patched
    assert "SetTaskMetricRsp = globals().get('SetTaskMetricRsp')" in patched
    assert patched.count("# BPv_dynamic_grasp_toggle_rsp_guard") == 1


@pytest.mark.unit
def test_patch_dynamic_grasp_toggle_upgrades_legacy_dispatch(tmp_path: Path) -> None:
    cc_path = tmp_path / "source/data_collection/server/command_controller.py"
    grpc_path = tmp_path / "source/data_collection/server/grpc_server.py"
    _write_command_controller(cc_path)
    _write_grpc_server(grpc_path, legacy_dispatch=True)

    patch_path = (
        REPO_ROOT
        / "tools/geniesim_adapter/deployment/patches/patch_dynamic_grasp_toggle.py"
    )
    mod = _load_module("patch_dynamic_grasp_toggle_upgrade_test", patch_path)
    mod.CC_PATH = str(cc_path)
    mod.GRPC_PATH = str(grpc_path)

    assert mod.apply() is True

    patched = grpc_path.read_text()
    assert "# BPv_dynamic_grasp_toggle_rsp_guard" in patched
    assert "SetTaskMetricRsp = globals().get('SetTaskMetricRsp')" in patched
    assert patched.count("# BPv_dynamic_grasp_toggle_rsp_guard") == 1
