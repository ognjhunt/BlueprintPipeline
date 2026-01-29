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


def test_webhook_rejects_missing_job_id(monkeypatch):
    module = _load_module("webhook_payload_missing_job", "genie-sim-import-webhook/main.py")
    monkeypatch.setattr(module, "_is_authenticated", lambda body: True)

    client = module.app.test_client()
    response = client.post(
        "/webhooks/geniesim/job-complete",
        json={"status": "completed"},
    )

    assert response.status_code == 400
    payload = response.get_json()
    assert payload["error"] == "invalid_payload"
    assert any("job_id" in detail for detail in payload.get("details", []))


def test_webhook_rejects_non_json_payload(monkeypatch):
    module = _load_module("webhook_payload_non_json", "genie-sim-import-webhook/main.py")
    monkeypatch.setattr(module, "_is_authenticated", lambda body: True)

    client = module.app.test_client()
    response = client.post(
        "/webhooks/geniesim/job-complete",
        data="not-json",
        content_type="text/plain",
    )

    assert response.status_code == 400
    payload = response.get_json()
    assert payload["error"] == "invalid_payload"
