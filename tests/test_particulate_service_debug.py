import importlib
import sys
import threading
from pathlib import Path

import pytest
pytest.importorskip("flask")

import pytest


class _NoOpThread:
    def __init__(self, *args, **kwargs) -> None:
        self._target = kwargs.get("target")

    def start(self) -> None:
        return None


def _import_particulate_service(
    monkeypatch: pytest.MonkeyPatch,
    *,
    get_secret_or_env=None,
):
    monkeypatch.setattr(threading, "Thread", _NoOpThread)

    repo_root = Path(__file__).resolve().parents[1]
    service_dir = repo_root / "particulate-service"
    monkeypatch.syspath_prepend(str(service_dir))

    if get_secret_or_env is not None:
        import tools.secret_store

        monkeypatch.setattr(tools.secret_store, "get_secret_or_env", get_secret_or_env)

    sys.modules.pop("particulate_service", None)
    return importlib.import_module("particulate_service")


@pytest.mark.unit
def test_debug_endpoint_forbidden_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PARTICULATE_DEBUG", raising=False)
    monkeypatch.delenv("PARTICULATE_DEBUG_TOKEN", raising=False)

    particulate_service = _import_particulate_service(monkeypatch)

    client = particulate_service.app.test_client()
    response = client.get("/debug")

    assert response.status_code == 403


@pytest.mark.unit
def test_debug_token_uses_secret_manager_in_non_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PIPELINE_ENV", raising=False)
    monkeypatch.delenv("DEBUG_TOKEN", raising=False)
    monkeypatch.delenv("PARTICULATE_DEBUG_TOKEN", raising=False)

    calls = {}

    def fake_get_secret_or_env(secret_id: str, env_var: str, fallback_to_env: bool) -> str:
        calls["secret_id"] = secret_id
        calls["env_var"] = env_var
        calls["fallback_to_env"] = fallback_to_env
        return "secret-token"

    particulate_service = _import_particulate_service(
        monkeypatch,
        get_secret_or_env=fake_get_secret_or_env,
    )

    assert particulate_service.DEBUG_TOKEN == "secret-token"
    assert calls == {
        "secret_id": "particulate-debug-token",
        "env_var": "DEBUG_TOKEN",
        "fallback_to_env": True,
    }


@pytest.mark.unit
def test_debug_token_env_rejected_in_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PIPELINE_ENV", "production")
    monkeypatch.setenv("DEBUG_TOKEN", "env-token")
    monkeypatch.delenv("PARTICULATE_DEBUG_TOKEN", raising=False)

    calls = {}

    def fake_get_secret_or_env(secret_id: str, env_var: str, fallback_to_env: bool):
        calls["secret_id"] = secret_id
        calls["env_var"] = env_var
        calls["fallback_to_env"] = fallback_to_env
        raise RuntimeError("missing secret")

    particulate_service = _import_particulate_service(
        monkeypatch,
        get_secret_or_env=fake_get_secret_or_env,
    )

    assert particulate_service.DEBUG_TOKEN is None
    assert calls == {
        "secret_id": "particulate-debug-token",
        "env_var": "DEBUG_TOKEN",
        "fallback_to_env": False,
    }
