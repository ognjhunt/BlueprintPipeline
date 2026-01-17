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


@pytest.mark.unit
def test_debug_endpoint_forbidden_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PARTICULATE_DEBUG", raising=False)
    monkeypatch.delenv("PARTICULATE_DEBUG_TOKEN", raising=False)
    monkeypatch.setattr(threading, "Thread", _NoOpThread)

    repo_root = Path(__file__).resolve().parents[1]
    service_dir = repo_root / "particulate-service"
    monkeypatch.syspath_prepend(str(service_dir))

    sys.modules.pop("particulate_service", None)
    particulate_service = importlib.import_module("particulate_service")

    client = particulate_service.app.test_client()
    response = client.get("/debug")

    assert response.status_code == 403
