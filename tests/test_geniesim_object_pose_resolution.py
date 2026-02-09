import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.geniesim_adapter import local_framework as lf


def test_resolve_object_prim_candidates_aliases(monkeypatch):
    monkeypatch.delenv("GENIESIM_OBJECT_PRIM_OVERRIDES_JSON", raising=False)
    monkeypatch.delenv("GENIESIM_OBJECT_PRIM_INDEX_MAX", raising=False)

    client = lf.GenieSimGRPCClient(host="localhost", port=1234, timeout=1.0)
    client._object_prim_aliases = {
        "lightwheel_kitchen_obj_Pot057": ["Pot057Alias"],
    }

    candidates = client._resolve_object_prim_candidates("lightwheel_kitchen_obj_Pot057")

    assert "/World/Scene/obj_lightwheel_kitchen_obj_Pot057" in candidates
    assert "/World/Scene/obj_Pot057" in candidates
    assert "/World/Scene/obj_Pot057Alias" in candidates


def test_build_object_metadata_from_scene_config_objects(tmp_path):
    recordings_dir = tmp_path / "recordings"
    log_dir = tmp_path / "logs"
    recordings_dir.mkdir()
    log_dir.mkdir()

    config = lf.GenieSimConfig(
        geniesim_root=tmp_path / "missing_geniesim",
        isaac_sim_path=tmp_path / "missing_isaac",
        recording_dir=recordings_dir,
        log_dir=log_dir,
    )
    framework = lf.GenieSimLocalFramework(config=config, verbose=False)

    scene_config = {
        "objects": [
            {
                "id": "ObjStatic",
                "category": "table",
                "sim_role": "static",
                "asset": {"path": "assets/obj_ObjStatic/ObjStatic.usd"},
            },
            {
                "id": "variation_pot",
                "category": "pot",
                "sim_role": "manipulable_object",
                "is_variation_asset": True,
                "asset": {"path": "assets/obj_Pot057/Pot057.usd"},
            },
        ]
    }
    task_config = {
        "suggested_tasks": [
            {"target_object": "TaskObj"},
        ]
    }

    meta = framework._build_object_metadata_from_scene_config(task_config, scene_config)

    assert "ObjStatic" in meta["static_ids"]
    assert "TaskObj" in meta["dynamic_ids"]
    assert "variation_pot" in meta["variation_ids"]
    assert "variation_pot" in meta["object_prim_aliases"]
    assert "Pot057" in meta["object_prim_aliases"]["variation_pot"]
    assert "pot" in meta["object_prim_aliases"]["variation_pot"]


def test_get_observation_uses_logical_object_ids(monkeypatch):
    client = lf.GenieSimGRPCClient(host="localhost", port=1234, timeout=1.0)
    client._channel = object()
    client._scene_object_ids_dynamic = ["lightwheel_kitchen_obj_Test"]
    client._scene_object_ids_static = []
    client._scene_variation_object_ids = []

    def _jp(*_args, **_kwargs):
        return lf.GrpcCallResult(success=True, available=True, payload=[0.0])

    def _ee(*_args, **_kwargs):
        return lf.GrpcCallResult(success=True, available=True, payload={})

    client.get_joint_position = _jp
    client.get_ee_pose = _ee
    client._resolve_object_prim = lambda obj_id: ("/World/Scene/obj_Test", {"position": {"x": 1.0, "y": 2.0, "z": 3.0}})

    obs_result = client.get_observation()

    assert obs_result.success
    objects = obs_result.payload["scene_state"]["objects"]
    assert objects
    obj = objects[0]
    assert obj["object_id"] == "lightwheel_kitchen_obj_Test"
    assert obj["prim_path"] == "/World/Scene/obj_Test"


def test_set_object_dynamic_uses_task_metric_shim():
    client = lf.GenieSimGRPCClient(host="localhost", port=1234, timeout=1.0)
    client._connected = True

    captured = {}

    class _Stub:
        def set_task_metric(self, request, timeout=None):
            captured["metric"] = request.metric
            captured["timeout"] = timeout
            return SimpleNamespace(msg="queued")

    client._stub = _Stub()

    result_on = client.set_object_dynamic("/World/Scene/obj_Toaster003", is_dynamic=True)
    assert result_on.success is True
    assert "bp::set_object_dynamic::" in captured.get("metric", "")
    assert client._active_dynamic_grasp_prim == "/World/Scene/obj_Toaster003"

    result_off = client.set_object_dynamic("/World/Scene/obj_Toaster003", is_dynamic=False)
    assert result_off.success is True
    assert client._active_dynamic_grasp_prim is None


def test_get_observation_strict_target_manifest_drift_fails(monkeypatch):
    monkeypatch.setenv("STRICT_REALISM", "1")
    monkeypatch.setenv("STRICT_REAL_ONLY", "1")
    monkeypatch.setenv("STRICT_FAIL_ON_MANIFEST_DRIFT", "1")
    monkeypatch.setenv("SKIP_RGB_CAPTURE", "1")

    client = lf.GenieSimGRPCClient(host="localhost", port=1234, timeout=1.0)
    client._channel = object()
    client._scene_object_ids_dynamic = ["lightwheel_kitchen_obj_Toaster003"]
    client._scene_object_ids_static = []
    client._scene_variation_object_ids = []
    client._active_target_object = "lightwheel_kitchen_obj_Toaster003"
    client._active_target_object_prim = "/World/Scene/obj_Toaster003"

    client.get_joint_position = lambda *_args, **_kwargs: lf.GrpcCallResult(
        success=True, available=True, payload=[0.0]
    )
    client.get_ee_pose = lambda *_args, **_kwargs: lf.GrpcCallResult(
        success=True, available=True, payload={}
    )
    client._resolve_object_prim = lambda _obj_id: (
        "/World/Scene/obj_Toaster003",
        {"position": {"x": 3.5, "y": 0.0, "z": 0.0}},
    )
    client._get_manifest_transform_fallback = lambda _obj_id: {
        "position": {"x": 0.0, "y": 0.0, "z": 0.0}
    }

    with pytest.raises(lf.FatalRealismError) as exc:
        client.get_observation()
    assert exc.value.reason_code == "STRICT_TARGET_MANIFEST_DRIFT"
