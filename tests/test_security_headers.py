import importlib.util
from pathlib import Path

import pytest
pytest.importorskip("flask")


def _load_module(module_name: str, relative_path: str):
    module_path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module: {relative_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_security_headers(response):
    assert response.headers.get("Content-Security-Policy")
    assert response.headers.get("X-Content-Type-Options") == "nosniff"
    assert response.headers.get("Referrer-Policy") == "no-referrer"


def test_webhook_security_headers(monkeypatch):
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "https://example.com")
    monkeypatch.setenv("ENV", "test")

    module = _load_module("webhook_main", "genie-sim-import-webhook/main.py")

    client = module.app.test_client()
    response = client.get("/healthz", headers={"Origin": "https://example.com"})

    assert response.status_code == 200
    _assert_security_headers(response)
    assert response.headers.get("Access-Control-Allow-Origin") == "https://example.com"


def test_particulate_security_headers(monkeypatch):
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "https://example.com")
    monkeypatch.setenv("GPU_HEALTH_REQUIRED", "false")
    monkeypatch.setenv("PARTICULATE_SKIP_WARMUP", "true")

    module = _load_module("particulate_service_module", "particulate-service/particulate_service.py")

    client = module.app.test_client()
    response = client.get("/healthz", headers={"Origin": "https://example.com"})

    assert response.status_code == 200
    _assert_security_headers(response)
    assert response.headers.get("Access-Control-Allow-Origin") == "https://example.com"
