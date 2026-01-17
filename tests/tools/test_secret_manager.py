import sys
import types

import pytest

from tools.secrets import secret_manager

pytestmark = pytest.mark.usefixtures("add_repo_to_path")


def _install_fake_secretmanager(secret_value: str = "secret-value") -> None:
    payload = types.SimpleNamespace(data=secret_value.encode("utf-8"))
    response = types.SimpleNamespace(payload=payload)

    class FakeClient:
        def __init__(self) -> None:
            self.request = None

        def access_secret_version(self, request):
            self.request = request
            return response

    secretmanager_module = types.ModuleType("google.cloud.secretmanager")
    secretmanager_module.SecretManagerServiceClient = FakeClient

    google_module = types.ModuleType("google")
    cloud_module = types.ModuleType("google.cloud")
    cloud_module.secretmanager = secretmanager_module

    sys.modules["google"] = google_module
    sys.modules["google.cloud"] = cloud_module
    sys.modules["google.cloud.secretmanager"] = secretmanager_module


@pytest.mark.unit
def test_get_secret_uses_project_env(monkeypatch) -> None:
    secret_manager.get_secret.cache_clear()
    _install_fake_secretmanager("value-123")
    monkeypatch.setenv("GCP_PROJECT", "test-project")

    value = secret_manager.get_secret("my-secret")

    assert value == "value-123"


@pytest.mark.unit
def test_get_secret_requires_project_id(monkeypatch) -> None:
    secret_manager.get_secret.cache_clear()
    _install_fake_secretmanager()
    monkeypatch.delenv("GCP_PROJECT", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)

    with pytest.raises(secret_manager.SecretManagerError):
        secret_manager.get_secret("my-secret")


@pytest.mark.unit
def test_get_secret_or_env_falls_back(monkeypatch) -> None:
    def _raise(*_args, **_kwargs):
        raise secret_manager.SecretManagerError("nope")

    monkeypatch.setattr(secret_manager, "get_secret", _raise)
    monkeypatch.setenv("API_KEY", "fallback")

    value = secret_manager.get_secret_or_env("secret-id", "API_KEY")

    assert value == "fallback"


@pytest.mark.unit
def test_get_secret_or_env_no_fallback(monkeypatch) -> None:
    def _raise(*_args, **_kwargs):
        raise secret_manager.SecretManagerError("nope")

    monkeypatch.setattr(secret_manager, "get_secret", _raise)

    with pytest.raises(secret_manager.SecretManagerError):
        secret_manager.get_secret_or_env("secret-id", "API_KEY", fallback_to_env=False)


@pytest.mark.unit
def test_secret_cache_loads_and_reads(monkeypatch) -> None:
    def _fake_get_secret(secret_id: str, **_kwargs):
        return f"value-for-{secret_id}"

    monkeypatch.setattr(secret_manager, "get_secret", _fake_get_secret)

    cache = secret_manager.SecretCache(project_id="project")
    cache.load("secret-a")
    cache.load("secret-b")

    assert cache.get("secret-a") == "value-for-secret-a"
    assert cache.get_or_load("secret-c") == "value-for-secret-c"


@pytest.mark.unit
def test_secret_cache_load_all_ignores_failures(monkeypatch) -> None:
    def _fake_get_secret(secret_id: str, **_kwargs):
        if secret_id == "bad":
            raise secret_manager.SecretManagerError("boom")
        return "ok"

    monkeypatch.setattr(secret_manager, "get_secret", _fake_get_secret)

    cache = secret_manager.SecretCache(project_id="project")
    cache.load_all(["good", "bad"])

    assert cache.get("good") == "ok"
    assert cache.get("bad") is None
