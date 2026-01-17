import base64
import importlib
import sys
import threading
from pathlib import Path

import pytest


class _NoOpThread:
    def __init__(self, *args, **kwargs) -> None:
        self._target = kwargs.get("target")

    def start(self) -> None:
        return None


def _import_service(monkeypatch: pytest.MonkeyPatch, max_bytes: int):
    monkeypatch.setenv("PARTICULATE_MAX_GLB_BYTES", str(max_bytes))
    monkeypatch.setattr(threading, "Thread", _NoOpThread)

    repo_root = Path(__file__).resolve().parents[1]
    service_dir = repo_root / "particulate-service"
    monkeypatch.syspath_prepend(str(service_dir))

    sys.modules.pop("particulate_service", None)
    return importlib.import_module("particulate_service")


@pytest.mark.unit
def test_payload_too_large_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    particulate_service = _import_service(monkeypatch, max_bytes=10)
    particulate_service._models_ready.set()
    particulate_service._warmup_error = None

    def _should_not_run(*args, **kwargs):
        raise AssertionError("run_particulate should not be called for large payloads")

    monkeypatch.setattr(particulate_service, "run_particulate", _should_not_run)

    client = particulate_service.app.test_client()
    glb_b64 = base64.b64encode(b"x" * 20).decode("ascii")
    response = client.post("/", json={"glb_base64": glb_b64})

    assert response.status_code == 413


@pytest.mark.unit
def test_payload_under_limit_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    particulate_service = _import_service(monkeypatch, max_bytes=1024)
    particulate_service._models_ready.set()
    particulate_service._warmup_error = None

    def _fake_run(glb_bytes: bytes, request_id: str):
        return (
            b"mesh",
            b"urdf",
            {"joint_count": 1, "part_count": 1, "is_articulated": True},
        )

    monkeypatch.setattr(particulate_service, "run_particulate", _fake_run)

    client = particulate_service.app.test_client()
    glb_b64 = base64.b64encode(b"ok").decode("ascii")
    response = client.post("/", json={"glb_base64": glb_b64})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["mesh_base64"] == base64.b64encode(b"mesh").decode("ascii")
    assert payload["urdf_base64"] == base64.b64encode(b"urdf").decode("ascii")
    assert payload["articulation"] == {
        "joint_count": 1,
        "part_count": 1,
        "is_articulated": True,
    }
