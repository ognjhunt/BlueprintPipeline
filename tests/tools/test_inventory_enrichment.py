import json
import sys
import warnings

import pytest
import requests

warnings.filterwarnings(
    "ignore",
    message="Type google._upb._message.*",
    category=DeprecationWarning,
)

from tools.inventory_enrichment.enricher import (
    ExternalInventoryEnricher,
    InventoryEnrichmentAuthError,
    InventoryEnrichmentValidationError,
)

pytestmark = pytest.mark.usefixtures("add_repo_to_path")


def _inventory_payload() -> dict:
    return {"inventory_id": "inv-123", "items": [{"sku": "abc"}]}


def _response_payload() -> dict:
    return {
        "request_id": "req-456",
        "provider": "external-service",
        "enrichment": {"score": 0.91},
    }


def _disable_retry_sleep(monkeypatch) -> None:
    retry_module = sys.modules["tools.error_handling.retry"]
    monkeypatch.setattr(retry_module.time, "sleep", lambda *_args, **_kwargs: None)


@pytest.mark.unit
def test_external_enricher_success(monkeypatch) -> None:
    captured = {}

    def _fake_post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        captured["timeout"] = timeout

        class Response:
            status_code = 200

            @staticmethod
            def json():
                return _response_payload()

        return Response()

    monkeypatch.setattr(requests, "post", _fake_post)
    _disable_retry_sleep(monkeypatch)

    enricher = ExternalInventoryEnricher(api_key="key-123", endpoint="https://example.test")
    result = enricher.enrich(_inventory_payload())

    assert captured["url"] == "https://example.test"
    assert captured["headers"]["Authorization"] == "Bearer key-123"
    assert captured["headers"]["Content-Type"] == "application/json"
    assert captured["timeout"] == enricher._timeout_seconds
    assert captured["json"]["inventory"]["inventory_id"] == "inv-123"
    assert "requested_at" in captured["json"]

    enrichment = result["metadata"]["inventory_enrichment"]
    assert enrichment["provider"] == "external-service"
    assert enrichment["request_id"] == "req-456"
    assert enrichment["status"] == "success"
    assert enrichment["data"] == {"score": 0.91}


@pytest.mark.unit
def test_external_enricher_timeout_retries(monkeypatch) -> None:
    attempts = {"count": 0}

    def _fake_post(*_args, **_kwargs):
        attempts["count"] += 1
        raise requests.Timeout("timeout")

    monkeypatch.setattr(requests, "post", _fake_post)
    _disable_retry_sleep(monkeypatch)

    enricher = ExternalInventoryEnricher(api_key="key-123", endpoint="https://example.test")

    with pytest.raises(requests.Timeout):
        enricher.enrich(_inventory_payload())

    assert attempts["count"] == 3


@pytest.mark.unit
def test_external_enricher_invalid_schema(monkeypatch) -> None:
    calls = {"count": 0}

    def _fake_post(*_args, **_kwargs):
        calls["count"] += 1

        class Response:
            status_code = 200

            @staticmethod
            def json():
                return {"provider": "external-service", "enrichment": {"score": 0.5}}

        return Response()

    monkeypatch.setattr(requests, "post", _fake_post)
    _disable_retry_sleep(monkeypatch)

    enricher = ExternalInventoryEnricher(api_key="key-123", endpoint="https://example.test")

    with pytest.raises(InventoryEnrichmentValidationError):
        enricher.enrich(_inventory_payload())

    assert calls["count"] == 1


@pytest.mark.unit
def test_external_enricher_auth_failure(monkeypatch) -> None:
    calls = {"count": 0}

    def _fake_post(*_args, **_kwargs):
        calls["count"] += 1

        class Response:
            status_code = 401

            @staticmethod
            def json():
                return json.loads("{}")

        return Response()

    monkeypatch.setattr(requests, "post", _fake_post)
    _disable_retry_sleep(monkeypatch)

    enricher = ExternalInventoryEnricher(api_key="key-123", endpoint="https://example.test")

    with pytest.raises(InventoryEnrichmentAuthError):
        enricher.enrich(_inventory_payload())

    assert calls["count"] == 1
