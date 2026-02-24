from __future__ import annotations

import sys
import types

from tools.llm_client import client as llm_client


def test_openai_client_passes_base_url_and_headers(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))

    client = llm_client.OpenAIClient(
        api_key="test-key",
        model="qwen/qwen3.5-397b-a17b",
        base_url="https://openrouter.ai/api/v1",
        default_headers={"X-Title": "BlueprintPipeline"},
    )

    assert client.base_url == "https://openrouter.ai/api/v1"
    assert captured["api_key"] == "test-key"
    assert captured["base_url"] == "https://openrouter.ai/api/v1"
    assert captured["default_headers"] == {"X-Title": "BlueprintPipeline"}


def test_openai_client_forwards_websocket_base_url(monkeypatch) -> None:
    captured: list[dict[str, object]] = []

    class FakeOpenAI:
        def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
            captured.append(kwargs)

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("OPENAI_WEBSOCKET_BASE_URL", "wss://api.openai.com/ws")
    monkeypatch.setenv("OPENAI_USE_WEBSOCKET", "1")

    client = llm_client.OpenAIClient(api_key="test-key", model="gpt-5.1")

    assert client._client is not None
    assert any("websocket_base_url" in kwargs for kwargs in captured)
    assert captured[0]["websocket_base_url"] == "wss://api.openai.com/ws"
    assert captured[0]["api_key"] == "test-key"


def test_openai_client_falls_back_when_websocket_kwarg_unsupported(monkeypatch) -> None:
    captured: list[dict[str, object]] = []

    class FakeOpenAI:
        def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
            if "websocket_base_url" in kwargs:
                raise TypeError("unexpected keyword argument: websocket_base_url")
            captured.append(kwargs)

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("OPENAI_WEBSOCKET_BASE_URL", "wss://api.openai.com/ws")
    monkeypatch.setenv("OPENAI_USE_WEBSOCKET", "1")

    client = llm_client.OpenAIClient(api_key="test-key", model="gpt-5.1")

    assert client._client is not None
    assert len(captured) == 1
    assert "websocket_base_url" not in captured[0]
    assert captured[0]["api_key"] == "test-key"
