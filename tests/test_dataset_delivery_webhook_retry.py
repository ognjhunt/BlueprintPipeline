import importlib.util
import sys
import types
from pathlib import Path

import pytest
import requests


def _load_retry_module():
    existing = sys.modules.get("tools.error_handling.retry")
    if existing is not None:
        return existing
    module_path = Path(__file__).resolve().parents[1] / "tools/error_handling/retry.py"
    spec = importlib.util.spec_from_file_location("tools.error_handling.retry", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load retry module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    package = types.ModuleType("tools.error_handling")
    package.__path__ = [str(module_path.parent)]
    sys.modules.setdefault("tools.error_handling", package)
    sys.modules["tools.error_handling.retry"] = module
    return module


def _load_module(module_name: str, relative_path: str):
    _load_retry_module()
    module_path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module: {relative_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeResponse:
    def __init__(self, status_code: int, text: str):
        self.status_code = status_code
        self.text = text


def _make_post_stub(results, calls):
    def _post(*args, **kwargs):
        calls.append((args, kwargs))
        result = results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    return _post


def test_send_webhook_retries_on_5xx(monkeypatch):
    retry_module = _load_retry_module()
    module = _load_module("dataset_delivery_retry_5xx", "dataset-delivery-job/dataset_delivery.py")
    calls = []
    results = [
        FakeResponse(500, "server error"),
        FakeResponse(502, "bad gateway"),
        FakeResponse(200, "ok"),
    ]
    monkeypatch.setattr(module.requests, "post", _make_post_stub(results, calls))
    monkeypatch.setattr(retry_module.time, "sleep", lambda *_: None)

    module.send_webhook("https://example.com/webhook", {"payload": "ok"})

    assert len(calls) == 3


def test_send_webhook_retries_on_timeout(monkeypatch):
    retry_module = _load_retry_module()
    module = _load_module("dataset_delivery_retry_timeout", "dataset-delivery-job/dataset_delivery.py")
    calls = []
    results = [
        requests.exceptions.Timeout("timed out"),
        FakeResponse(200, "ok"),
    ]
    monkeypatch.setattr(module.requests, "post", _make_post_stub(results, calls))
    monkeypatch.setattr(retry_module.time, "sleep", lambda *_: None)

    module.send_webhook("https://example.com/webhook", {"payload": "ok"})

    assert len(calls) == 2


def test_send_webhook_does_not_retry_on_4xx(monkeypatch):
    retry_module = _load_retry_module()
    module = _load_module("dataset_delivery_retry_4xx", "dataset-delivery-job/dataset_delivery.py")
    non_retryable_error = retry_module.NonRetryableError
    calls = []
    results = [FakeResponse(400, "bad request")]
    monkeypatch.setattr(module.requests, "post", _make_post_stub(results, calls))
    monkeypatch.setattr(retry_module.time, "sleep", lambda *_: None)

    with pytest.raises(non_retryable_error):
        module.send_webhook("https://example.com/webhook", {"payload": "bad"})

    assert len(calls) == 1
