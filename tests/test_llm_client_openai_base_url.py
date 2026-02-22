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
