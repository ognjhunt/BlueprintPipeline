from __future__ import annotations

import sys
import types

from tools.asset_catalog.embeddings import AssetEmbeddings, EmbeddingConfig


def test_asset_embeddings_openai_client_forwards_websocket_base_url(monkeypatch) -> None:
    captured: list[dict[str, object]] = []

    class FakeOpenAI:
        def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
            captured.append(kwargs)
            self.embeddings = types.SimpleNamespace(
                create=lambda **_: types.SimpleNamespace(data=[types.SimpleNamespace(embedding=[1.0, 2.0, 3.0)])
            )

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openai.com/v1")
    monkeypatch.setenv("OPENAI_WEBSOCKET_BASE_URL", "wss://api.openai.com/ws")
    monkeypatch.setenv("OPENAI_USE_WEBSOCKET", "1")

    ae = AssetEmbeddings(config=EmbeddingConfig(backend="openai", api_key="test-key"))
    vec = ae.embed_text("modern chair")

    assert any("websocket_base_url" in kwargs for kwargs in captured)
    assert captured[0]["api_key"] == "test-key"
    assert captured[0]["websocket_base_url"] == "wss://api.openai.com/ws"
    assert vec.tolist() == [1.0, 2.0, 3.0]


def test_asset_embeddings_openai_client_falls_back_without_websocket_arg(monkeypatch) -> None:
    captured: list[dict[str, object]] = []

    class FakeOpenAI:
        def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
            if "websocket_base_url" in kwargs:
                raise TypeError("unexpected keyword argument: websocket_base_url")
            captured.append(kwargs)
            self.embeddings = types.SimpleNamespace(
                create=lambda **_: types.SimpleNamespace(data=[types.SimpleNamespace(embedding=[3.0, 2.0, 1.0)])
            )

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("OPENAI_WEBSOCKET_BASE_URL", "wss://api.openai.com/ws")
    monkeypatch.setenv("OPENAI_USE_WEBSOCKET", "1")

    ae = AssetEmbeddings(config=EmbeddingConfig(backend="openai", api_key="test-key"))
    vec = ae.embed_text("modern table")

    assert len(captured) == 1
    assert "websocket_base_url" not in captured[0]
    assert captured[0]["api_key"] == "test-key"
    assert vec.tolist() == [3.0, 2.0, 1.0]
