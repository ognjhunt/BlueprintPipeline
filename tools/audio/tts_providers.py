"""TTS provider adapters and capability metadata."""
from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional
from xml.sax.saxutils import escape

import requests
from google.auth.transport.requests import AuthorizedSession
from google.oauth2 import service_account


@dataclass(frozen=True)
class TTSProviderCapability:
    provider: str
    display_name: str
    env_vars: List[str]
    supported_voices: List[str]
    supported_formats: List[str]
    default_voice: str
    notes: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "provider": self.provider,
            "display_name": self.display_name,
            "env_vars": self.env_vars,
            "supported_voices": self.supported_voices,
            "supported_formats": self.supported_formats,
            "default_voice": self.default_voice,
            "notes": self.notes,
        }


class TTSProviderError(RuntimeError):
    """Structured error for TTS provider failures."""

    def __init__(
        self,
        provider: str,
        message: str,
        error_type: str = "provider_error",
        details: Optional[Dict[str, object]] = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.error_type = error_type
        self.details = details or {}

    def to_dict(self) -> Dict[str, object]:
        return {
            "provider": self.provider,
            "error_type": self.error_type,
            "message": str(self),
            "details": self.details,
        }


PROVIDER_CAPABILITIES: Dict[str, TTSProviderCapability] = {
    "google": TTSProviderCapability(
        provider="google",
        display_name="Google Cloud Text-to-Speech",
        env_vars=["GOOGLE_APPLICATION_CREDENTIALS"],
        supported_voices=[
            "en-US-Neural2-D",
            "en-US-Neural2-F",
            "en-US-Neural2-J",
            "en-US-Standard-B",
        ],
        supported_formats=["mp3", "wav", "ogg"],
        default_voice="en-US-Neural2-D",
        notes="Requires a service account JSON credential.",
    ),
    "openai": TTSProviderCapability(
        provider="openai",
        display_name="OpenAI Text-to-Speech",
        env_vars=["OPENAI_API_KEY"],
        supported_voices=["alloy", "verse", "nova", "echo"],
        supported_formats=["mp3", "wav", "ogg"],
        default_voice="alloy",
        notes="Uses OpenAI's TTS API with API key.",
    ),
    "azure": TTSProviderCapability(
        provider="azure",
        display_name="Azure Speech Service",
        env_vars=["AZURE_SPEECH_KEY", "AZURE_SPEECH_REGION"],
        supported_voices=["en-US-AriaNeural", "en-US-GuyNeural"],
        supported_formats=["mp3", "wav", "ogg"],
        default_voice="en-US-AriaNeural",
        notes="Requires Azure Speech key and region.",
    ),
    "local": TTSProviderCapability(
        provider="local",
        display_name="Local TTS",
        env_vars=[],
        supported_voices=["default"],
        supported_formats=["wav", "mp3"],
        default_voice="default",
        notes="Offline placeholder audio for local pipelines.",
    ),
    "mock": TTSProviderCapability(
        provider="mock",
        display_name="Mock TTS",
        env_vars=[],
        supported_voices=["mock"],
        supported_formats=["wav", "mp3", "ogg"],
        default_voice="mock",
        notes="Always succeeds with placeholder audio.",
    ),
}

DEFAULT_FALLBACK_ORDER: List[str] = ["google", "openai", "azure", "local", "mock"]


def provider_metadata() -> Dict[str, Dict[str, object]]:
    return {name: capability.to_dict() for name, capability in PROVIDER_CAPABILITIES.items()}


def detect_available_providers(fallback_order: Iterable[str]) -> List[str]:
    available: List[str] = []
    for provider in fallback_order:
        if provider in ("local", "mock"):
            available.append(provider)
            continue
        capability = PROVIDER_CAPABILITIES.get(provider)
        if not capability:
            continue
        if all(os.getenv(var) for var in capability.env_vars):
            available.append(provider)
    return available


def _ensure_format_supported(provider: str, audio_format: str) -> None:
    capability = PROVIDER_CAPABILITIES.get(provider)
    if not capability:
        raise TTSProviderError(
            provider,
            f"Unknown provider '{provider}'",
            error_type="unknown_provider",
        )
    if audio_format not in capability.supported_formats:
        raise TTSProviderError(
            provider,
            f"Format '{audio_format}' is not supported by {provider}",
            error_type="unsupported_format",
            details={"supported_formats": capability.supported_formats},
        )


def _generate_silence_frames(duration_s: float = 1.0, sample_rate: int = 22050, channels: int = 1) -> bytes:
    frames = int(duration_s * sample_rate)
    return b"\x00\x00" * frames * channels


def _language_code_from_voice(voice: str) -> str:
    parts = voice.split("-")
    if len(parts) >= 2:
        return "-".join(parts[:2])
    return voice


def _map_status_to_error_type(status_code: int) -> str:
    if status_code in {401, 403}:
        return "unauthorized"
    if status_code == 400:
        return "bad_request"
    if status_code == 404:
        return "not_found"
    if status_code == 429:
        return "rate_limited"
    if status_code >= 500:
        return "provider_unavailable"
    return "provider_error"


def _raise_response_error(provider: str, response: requests.Response) -> None:
    details: Dict[str, object] = {"status_code": response.status_code}
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if payload is not None:
        details["response_json"] = payload
    elif response.text:
        details["response_text"] = response.text[:500]
    raise TTSProviderError(
        provider,
        f"{provider} TTS request failed with status {response.status_code}",
        error_type=_map_status_to_error_type(response.status_code),
        details=details,
    )


class BaseTTSProvider:
    provider: str

    def generate_audio(self, text: str, voice: str, audio_format: str) -> bytes:
        raise NotImplementedError

    def _resolve_voice(self, voice: str) -> str:
        capability = PROVIDER_CAPABILITIES[self.provider]
        if voice in capability.supported_voices:
            return voice
        return capability.default_voice


class GoogleTTSProvider(BaseTTSProvider):
    provider = "google"

    def generate_audio(self, text: str, voice: str, audio_format: str) -> bytes:
        _ensure_format_supported(self.provider, audio_format)
        if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            raise TTSProviderError(
                self.provider,
                "Missing Google application credentials",
                error_type="credentials_missing",
                details={"env_vars": PROVIDER_CAPABILITIES[self.provider].env_vars},
            )
        resolved_voice = self._resolve_voice(voice)
        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
        try:
            credentials = service_account.Credentials.from_service_account_file(
                credentials_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
        except Exception as exc:  # noqa: BLE001
            raise TTSProviderError(
                self.provider,
                "Invalid Google application credentials",
                error_type="credentials_invalid",
                details={"path": credentials_path, "reason": str(exc)},
            ) from exc
        session = AuthorizedSession(credentials)
        payload = {
            "input": {"text": text},
            "voice": {
                "languageCode": _language_code_from_voice(resolved_voice),
                "name": resolved_voice,
            },
            "audioConfig": {
                "audioEncoding": {
                    "mp3": "MP3",
                    "wav": "LINEAR16",
                    "ogg": "OGG_OPUS",
                }[audio_format],
            },
        }
        try:
            response = session.post(
                "https://texttospeech.googleapis.com/v1/text:synthesize",
                json=payload,
                timeout=30,
            )
        except requests.RequestException as exc:
            raise TTSProviderError(
                self.provider,
                "Google TTS request failed",
                error_type="request_failed",
                details={"reason": str(exc)},
            ) from exc
        if response.status_code != 200:
            _raise_response_error(self.provider, response)
        try:
            audio_content = response.json().get("audioContent")
        except ValueError as exc:
            raise TTSProviderError(
                self.provider,
                "Google TTS response was not valid JSON",
                error_type="invalid_response",
                details={"status_code": response.status_code},
            ) from exc
        if not audio_content:
            raise TTSProviderError(
                self.provider,
                "Google TTS response missing audio content",
                error_type="invalid_response",
                details={"status_code": response.status_code},
            )
        return base64.b64decode(audio_content)


class OpenAITTSProvider(BaseTTSProvider):
    provider = "openai"

    def generate_audio(self, text: str, voice: str, audio_format: str) -> bytes:
        _ensure_format_supported(self.provider, audio_format)
        if not os.getenv("OPENAI_API_KEY"):
            raise TTSProviderError(
                self.provider,
                "Missing OpenAI API key",
                error_type="credentials_missing",
                details={"env_vars": PROVIDER_CAPABILITIES[self.provider].env_vars},
            )
        resolved_voice = self._resolve_voice(voice)
        headers = {
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "gpt-4o-mini-tts",
            "input": text,
            "voice": resolved_voice,
            "format": audio_format,
        }
        try:
            response = requests.post(
                "https://api.openai.com/v1/audio/speech",
                headers=headers,
                json=payload,
                timeout=30,
            )
        except requests.RequestException as exc:
            raise TTSProviderError(
                self.provider,
                "OpenAI TTS request failed",
                error_type="request_failed",
                details={"reason": str(exc)},
            ) from exc
        if response.status_code != 200:
            _raise_response_error(self.provider, response)
        return response.content


class AzureTTSProvider(BaseTTSProvider):
    provider = "azure"

    def generate_audio(self, text: str, voice: str, audio_format: str) -> bytes:
        _ensure_format_supported(self.provider, audio_format)
        if not (os.getenv("AZURE_SPEECH_KEY") and os.getenv("AZURE_SPEECH_REGION")):
            raise TTSProviderError(
                self.provider,
                "Missing Azure Speech credentials",
                error_type="credentials_missing",
                details={"env_vars": PROVIDER_CAPABILITIES[self.provider].env_vars},
            )
        resolved_voice = self._resolve_voice(voice)
        region = os.getenv("AZURE_SPEECH_REGION", "")
        endpoint = f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1"
        output_format = {
            "mp3": "audio-16khz-128kbitrate-mono-mp3",
            "wav": "riff-16khz-16bit-mono-pcm",
            "ogg": "ogg-16khz-16bit-mono-opus",
        }[audio_format]
        ssml = (
            "<speak version='1.0' xml:lang='{}'>"
            "<voice name='{}'>{}</voice>"
            "</speak>"
        ).format(
            _language_code_from_voice(resolved_voice),
            escape(resolved_voice),
            escape(text),
        )
        headers = {
            "Ocp-Apim-Subscription-Key": os.getenv("AZURE_SPEECH_KEY", ""),
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": output_format,
        }
        try:
            response = requests.post(
                endpoint,
                headers=headers,
                data=ssml.encode("utf-8"),
                timeout=30,
            )
        except requests.RequestException as exc:
            raise TTSProviderError(
                self.provider,
                "Azure TTS request failed",
                error_type="request_failed",
                details={"reason": str(exc)},
            ) from exc
        if response.status_code != 200:
            _raise_response_error(self.provider, response)
        return response.content


class LocalTTSProvider(BaseTTSProvider):
    provider = "local"

    def generate_audio(self, text: str, voice: str, audio_format: str) -> bytes:
        _ensure_format_supported(self.provider, audio_format)
        if audio_format == "wav":
            return _generate_silence_frames()
        return f"LOCAL_TTS_PLACEHOLDER:{text}".encode("utf-8")


class MockTTSProvider(BaseTTSProvider):
    provider = "mock"

    def generate_audio(self, text: str, voice: str, audio_format: str) -> bytes:
        _ensure_format_supported(self.provider, audio_format)
        if audio_format == "wav":
            return _generate_silence_frames()
        return f"MOCK_TTS_PLACEHOLDER:{text}".encode("utf-8")


_PROVIDER_MAP = {
    "google": GoogleTTSProvider,
    "openai": OpenAITTSProvider,
    "azure": AzureTTSProvider,
    "local": LocalTTSProvider,
    "mock": MockTTSProvider,
}


def get_provider_adapter(provider: str) -> BaseTTSProvider:
    provider_cls = _PROVIDER_MAP.get(provider)
    if not provider_cls:
        raise TTSProviderError(
            provider,
            f"Unknown provider '{provider}'",
            error_type="unknown_provider",
        )
    return provider_cls()
