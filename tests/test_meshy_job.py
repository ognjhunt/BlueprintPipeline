import base64
import importlib.util
from pathlib import Path

import pytest
import requests

MODULE_PATH = Path(__file__).resolve().parents[1] / "meshy-job" / "run_meshy_from_assets.py"
SPEC = importlib.util.spec_from_file_location("run_meshy_from_assets", MODULE_PATH)
run_meshy_from_assets = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_meshy_from_assets)


class FakeResponse:
    def __init__(self, payload=None, raise_exc=None):
        self._payload = payload or {}
        self._raise_exc = raise_exc
        self.raise_called = False

    def raise_for_status(self):
        self.raise_called = True
        if self._raise_exc:
            raise self._raise_exc

    def json(self):
        return self._payload


def test_image_to_data_uri_png(tmp_path):
    img_path = tmp_path / "sample.png"
    img_path.write_bytes(b"png-bytes")

    data_uri = run_meshy_from_assets.image_to_data_uri(img_path)

    encoded = base64.b64encode(b"png-bytes").decode("ascii")
    assert data_uri == f"data:image/png;base64,{encoded}"


def test_image_to_data_uri_jpg(tmp_path):
    img_path = tmp_path / "sample.jpg"
    img_path.write_bytes(b"jpeg-bytes")

    data_uri = run_meshy_from_assets.image_to_data_uri(img_path)

    encoded = base64.b64encode(b"jpeg-bytes").decode("ascii")
    assert data_uri == f"data:image/jpeg;base64,{encoded}"


def test_load_assets_plan_errors_when_missing_manifest(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run_meshy_from_assets, "load_manifest_or_scene_assets", lambda _: None)

    with pytest.raises(SystemExit) as excinfo:
        run_meshy_from_assets.load_assets_plan(tmp_path)

    assert excinfo.value.code == 1
    stderr = capsys.readouterr().err
    assert "expected manifest" in stderr


def test_create_image_to_3d_task_posts_payload(monkeypatch):
    captured = {}

    def fake_post(url, headers=None, json=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return FakeResponse({"result": "task-123"})

    monkeypatch.setattr(run_meshy_from_assets.requests, "post", fake_post)

    task_id = run_meshy_from_assets.create_image_to_3d_task("api-key", "data:uri")

    assert task_id == "task-123"
    assert captured["headers"]["Authorization"] == "Bearer api-key"
    assert captured["json"]["image_url"] == "data:uri"


def test_create_image_to_3d_task_uses_configured_base_url(monkeypatch):
    captured = {}

    def fake_post(url, headers=None, json=None):
        captured["url"] = url
        return FakeResponse({"result": "task-123"})

    monkeypatch.setenv("MESHY_BASE_URL", "https://example.meshy.local/")
    monkeypatch.setattr(run_meshy_from_assets.requests, "post", fake_post)

    run_meshy_from_assets.create_image_to_3d_task("api-key", "data:uri")

    assert captured["url"] == "https://example.meshy.local/openapi/v1/image-to-3d"


def test_create_image_to_3d_task_propagates_http_error(monkeypatch):
    def fake_post(*_args, **_kwargs):
        return FakeResponse(raise_exc=requests.HTTPError("boom"))

    monkeypatch.setattr(run_meshy_from_assets.requests, "post", fake_post)

    with pytest.raises(requests.HTTPError):
        run_meshy_from_assets.create_image_to_3d_task("api-key", "data:uri")


def test_wait_for_task_retries_until_success(monkeypatch):
    responses = [
        FakeResponse({"status": "PENDING", "progress": 10}),
        FakeResponse({"status": "SUCCEEDED", "progress": 100, "result": "done"}),
    ]

    def fake_get(*_args, **_kwargs):
        return responses.pop(0)

    monkeypatch.setattr(run_meshy_from_assets.requests, "get", fake_get)
    monkeypatch.setattr(run_meshy_from_assets.time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_meshy_from_assets.time, "time", lambda: 0.0)

    data = run_meshy_from_assets.wait_for_task("api-key", "task-123", poll_seconds=0.0, timeout=10.0)

    assert data["status"] == "SUCCEEDED"
    assert data["result"] == "done"


def test_wait_for_task_uses_configured_base_url(monkeypatch):
    captured = {}

    def fake_get(url, **_kwargs):
        captured["url"] = url
        return FakeResponse({"status": "SUCCEEDED", "progress": 100, "result": "done"})

    monkeypatch.setenv("MESHY_API_BASE", "https://example.meshy.internal")
    monkeypatch.setattr(run_meshy_from_assets.requests, "get", fake_get)
    monkeypatch.setattr(run_meshy_from_assets.time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_meshy_from_assets.time, "time", lambda: 0.0)

    run_meshy_from_assets.wait_for_task("api-key", "task-123", poll_seconds=0.0, timeout=10.0)

    assert captured["url"] == "https://example.meshy.internal/openapi/v1/image-to-3d/task-123"


def test_wait_for_task_timeout(monkeypatch):
    response = FakeResponse({"status": "PENDING", "progress": 10})

    def fake_get(*_args, **_kwargs):
        return response

    times = iter([0.0, 11.0])

    def fake_time():
        return next(times, 11.0)

    monkeypatch.setattr(run_meshy_from_assets.requests, "get", fake_get)
    monkeypatch.setattr(run_meshy_from_assets.time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_meshy_from_assets.time, "time", fake_time)

    with pytest.raises(TimeoutError):
        run_meshy_from_assets.wait_for_task("api-key", "task-123", poll_seconds=0.0, timeout=10.0)


def test_wait_for_task_failure_status(monkeypatch):
    def fake_get(*_args, **_kwargs):
        return FakeResponse({"status": "FAILED", "task_error": "bad input"})

    monkeypatch.setattr(run_meshy_from_assets.requests, "get", fake_get)
    monkeypatch.setattr(run_meshy_from_assets.time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_meshy_from_assets.time, "time", lambda: 0.0)

    with pytest.raises(RuntimeError, match="bad input"):
        run_meshy_from_assets.wait_for_task("api-key", "task-123", poll_seconds=0.0, timeout=10.0)
