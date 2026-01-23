from __future__ import annotations

import base64

import pytest

from tools.audio import tts_providers


pytestmark = pytest.mark.usefixtures("add_repo_to_path")


class DummyResponse:
    def __init__(
        self,
        status_code: int,
        *,
        content: bytes = b"",
        json_payload: object | None = None,
        text: str = "",
        raise_json: bool = False,
    ) -> None:
        self.status_code = status_code
        self.content = content
        self._json_payload = json_payload
        self.text = text
        self._raise_json = raise_json

    def json(self) -> object:
        if self._raise_json:
            raise ValueError("invalid json")
        return self._json_payload


def test_google_tts_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/tmp/creds.json")
    expected_audio = b"google-audio"
    payloads: list[dict[str, object]] = []

    class FakeSession:
        def __init__(self, _credentials: object) -> None:
            self.credentials = _credentials

        def post(self, url: str, json: dict[str, object], timeout: int) -> DummyResponse:
            payloads.append({"url": url, "json": json, "timeout": timeout})
            return DummyResponse(
                200,
                json_payload={
                    "audioContent": base64.b64encode(expected_audio).decode("utf-8"),
                },
            )

    monkeypatch.setattr(
        tts_providers.service_account.Credentials,
        "from_service_account_file",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(tts_providers, "AuthorizedSession", FakeSession)

    provider = tts_providers.GoogleTTSProvider()
    audio = provider.generate_audio("Hello world", "en-US-Neural2-D", "mp3")

    assert audio == expected_audio
    assert payloads[0]["url"] == "https://texttospeech.googleapis.com/v1/text:synthesize"


def test_google_tts_maps_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/tmp/creds.json")

    class FakeSession:
        def __init__(self, _credentials: object) -> None:
            self.credentials = _credentials

        def post(self, url: str, json: dict[str, object], timeout: int) -> DummyResponse:
            return DummyResponse(401, json_payload={"error": "unauthorized"})

    monkeypatch.setattr(
        tts_providers.service_account.Credentials,
        "from_service_account_file",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(tts_providers, "AuthorizedSession", FakeSession)

    provider = tts_providers.GoogleTTSProvider()
    with pytest.raises(tts_providers.TTSProviderError) as excinfo:
        provider.generate_audio("Hello world", "en-US-Neural2-D", "mp3")

    assert excinfo.value.error_type == "unauthorized"


def test_openai_tts_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    expected_audio = b"openai-audio"
    captured: dict[str, object] = {}

    def fake_post(url: str, headers: dict[str, str], json: dict[str, object], timeout: int) -> DummyResponse:
        captured.update({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return DummyResponse(200, content=expected_audio)

    monkeypatch.setattr(tts_providers.requests, "post", fake_post)

    provider = tts_providers.OpenAITTSProvider()
    audio = provider.generate_audio("Speak", "alloy", "wav")

    assert audio == expected_audio
    assert captured["url"] == "https://api.openai.com/v1/audio/speech"


def test_openai_tts_rate_limit_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def fake_post(url: str, headers: dict[str, str], json: dict[str, object], timeout: int) -> DummyResponse:
        return DummyResponse(429, json_payload={"error": {"message": "rate limit"}})

    monkeypatch.setattr(tts_providers.requests, "post", fake_post)

    provider = tts_providers.OpenAITTSProvider()
    with pytest.raises(tts_providers.TTSProviderError) as excinfo:
        provider.generate_audio("Speak", "alloy", "mp3")

    assert excinfo.value.error_type == "rate_limited"


def test_azure_tts_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_SPEECH_KEY", "key")
    monkeypatch.setenv("AZURE_SPEECH_REGION", "westus")
    expected_audio = b"azure-audio"
    captured: dict[str, object] = {}

    def fake_post(url: str, headers: dict[str, str], data: bytes, timeout: int) -> DummyResponse:
        captured.update({"url": url, "headers": headers, "data": data, "timeout": timeout})
        return DummyResponse(200, content=expected_audio)

    monkeypatch.setattr(tts_providers.requests, "post", fake_post)

    provider = tts_providers.AzureTTSProvider()
    audio = provider.generate_audio("Hello", "en-US-AriaNeural", "mp3")

    assert audio == expected_audio
    assert captured["url"] == "https://westus.tts.speech.microsoft.com/cognitiveservices/v1"


def test_azure_tts_server_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_SPEECH_KEY", "key")
    monkeypatch.setenv("AZURE_SPEECH_REGION", "westus")

    def fake_post(url: str, headers: dict[str, str], data: bytes, timeout: int) -> DummyResponse:
        return DummyResponse(500, text="server error")

    monkeypatch.setattr(tts_providers.requests, "post", fake_post)

    provider = tts_providers.AzureTTSProvider()
    with pytest.raises(tts_providers.TTSProviderError) as excinfo:
        provider.generate_audio("Hello", "en-US-AriaNeural", "mp3")

    assert excinfo.value.error_type == "provider_unavailable"
