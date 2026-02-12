from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("flask")


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name: str, relative_path: str):
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module: {relative_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_scenesmith_service_internal_generate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEXT_GEN_USE_LLM", "false")
    monkeypatch.setenv("SCENESMITH_SERVICE_MODE", "internal")

    module = _load_module("scenesmith_service_module", "scenesmith-service/scenesmith_service.py")
    client = module.app.test_client()

    response = client.post(
        "/v1/generate",
        json={
            "scene_id": "svc_scene_001",
            "prompt": "A kitchen with a table and mug",
            "quality_tier": "standard",
            "seed": 1,
            "constraints": {},
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["schema_version"] == "v1"
    assert "request_id" in payload
    assert isinstance(payload.get("package"), dict)
    package = payload["package"]
    assert package["text_backend"] == "scenesmith"
    assert isinstance(package.get("objects"), list)
    assert len(package["objects"]) > 0


@pytest.mark.unit
def test_sage_service_internal_refine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEXT_GEN_USE_LLM", "false")
    monkeypatch.setenv("SAGE_SERVICE_MODE", "internal")

    module = _load_module("sage_service_module", "sage-service/sage_service.py")
    client = module.app.test_client()

    response = client.post(
        "/v1/refine",
        json={
            "scene_id": "svc_scene_002",
            "prompt": "Move the bowl to the shelf",
            "quality_tier": "standard",
            "seed": 3,
            "constraints": {},
            "source_backend": "scenesmith",
            "base_scene": {
                "room_type": "kitchen",
                "objects": [
                    {
                        "id": "counter_1",
                        "name": "counter",
                        "category": "counter",
                        "sim_role": "static",
                        "transform": {
                            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                        },
                        "dimensions_est": {"width": 1.0, "height": 0.9, "depth": 0.6},
                    },
                    {
                        "id": "bowl_1",
                        "name": "bowl",
                        "category": "bowl",
                        "sim_role": "manipulable_object",
                        "transform": {
                            "position": {"x": 0.2, "y": 0.9, "z": 0.1},
                        },
                        "dimensions_est": {"width": 0.18, "height": 0.08, "depth": 0.18},
                    },
                ],
            },
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["schema_version"] == "v1"
    assert isinstance(payload.get("package"), dict)
    package = payload["package"]
    assert package["text_backend"] == "sage"
    assert isinstance(package.get("objects"), list)
    assert len(package["objects"]) >= 2


@pytest.mark.unit
def test_scenesmith_service_command_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cmd_script = tmp_path / "scenesmith_cmd.py"
    cmd_script.write_text(
        "#!/usr/bin/env python3\n"
        "import json,sys\n"
        "req=json.load(sys.stdin)\n"
        "resp={'schema_version':'v1','request_id':'cmd-1','objects':[{'id':'obj_001','name':'mug','category':'mug','sim_role':'manipulable_object','transform':{'position':{'x':0,'y':0,'z':0}}}]}\n"
        "print(json.dumps(resp))\n",
        encoding="utf-8",
    )
    cmd_script.chmod(0o755)

    monkeypatch.setenv("SCENESMITH_SERVICE_MODE", "command")
    monkeypatch.setenv("SCENESMITH_COMMAND", str(cmd_script))

    module = _load_module("scenesmith_service_cmd_module", "scenesmith-service/scenesmith_service.py")
    client = module.app.test_client()

    response = client.post(
        "/v1/generate",
        json={
            "scene_id": "svc_scene_003",
            "prompt": "A test prompt",
            "quality_tier": "standard",
            "seed": 11,
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["request_id"] == "cmd-1"
    assert isinstance(payload.get("objects"), list)


@pytest.mark.unit
def test_scenesmith_service_paper_stack_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCENESMITH_SERVICE_MODE", "paper_stack")
    monkeypatch.setenv("SCENESMITH_PAPER_COMMAND", "python /tmp/paper_cmd.py")

    module = _load_module("scenesmith_service_paper_module", "scenesmith-service/scenesmith_service.py")

    called = {"command": None}

    def _fake_run_command(payload):  # type: ignore[no-untyped-def]
        called["command"] = module.os.environ.get("SCENESMITH_COMMAND")
        return {
            "schema_version": "v1",
            "request_id": "paper-1",
            "objects": [
                {
                    "id": "obj_200",
                    "name": "plate",
                    "category": "plate",
                    "sim_role": "manipulable_object",
                    "transform": {"position": {"x": 0, "y": 0, "z": 0}},
                }
            ],
        }

    monkeypatch.setattr(module, "_run_command_backend", _fake_run_command)

    response = module.app.test_client().post(
        "/v1/generate",
        json={
            "scene_id": "svc_scene_005",
            "prompt": "A test prompt",
            "quality_tier": "standard",
            "seed": 7,
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["request_id"] == "paper-1"
    assert isinstance(payload.get("objects"), list)
    assert called["command"] == "python /tmp/paper_cmd.py"


@pytest.mark.unit
def test_sage_service_http_forward_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SAGE_SERVICE_MODE", "http_forward")
    monkeypatch.setenv("SAGE_UPSTREAM_URL", "https://sage-upstream.example/v1/refine")
    monkeypatch.setenv("SAGE_SERVICE_TIMEOUT_SECONDS", "12")

    module = _load_module("sage_service_forward_module", "sage-service/sage_service.py")

    class _FakeResponse:
        def __init__(self, payload: dict) -> None:
            self._payload = payload

        def read(self) -> bytes:
            return json.dumps(self._payload).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    def _fake_urlopen(req, timeout=0):  # type: ignore[no-untyped-def]
        assert req.full_url == "https://sage-upstream.example/v1/refine"
        assert timeout == 12.0
        request_payload = json.loads(req.data.decode("utf-8"))
        assert request_payload["scene_id"] == "svc_scene_004"
        return _FakeResponse(
            {
                "schema_version": "v1",
                "request_id": "forward-1",
                "objects": [
                    {
                        "id": "obj_100",
                        "name": "tray",
                        "category": "tray",
                        "sim_role": "manipulable_object",
                        "transform": {"position": {"x": 0, "y": 0, "z": 0}},
                    }
                ],
            }
        )

    monkeypatch.setattr(module.url_request, "urlopen", _fake_urlopen)

    client = module.app.test_client()
    response = client.post(
        "/v1/refine",
        json={
            "scene_id": "svc_scene_004",
            "prompt": "A test prompt",
            "quality_tier": "standard",
            "seed": 4,
            "constraints": {},
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["request_id"] == "forward-1"
    assert isinstance(payload.get("objects"), list)


@pytest.mark.unit
def test_backend_services_health_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCENESMITH_SERVICE_MODE", "internal")
    monkeypatch.setenv("SAGE_SERVICE_MODE", "internal")

    scenesmith = _load_module("scenesmith_service_health_module", "scenesmith-service/scenesmith_service.py")
    sage = _load_module("sage_service_health_module", "sage-service/sage_service.py")

    scenesmith_resp = scenesmith.app.test_client().get("/healthz")
    sage_resp = sage.app.test_client().get("/healthz")

    assert scenesmith_resp.status_code == 200
    assert scenesmith_resp.get_json()["service"] == "scenesmith"
    assert sage_resp.status_code == 200
    assert sage_resp.get_json()["service"] == "sage"
