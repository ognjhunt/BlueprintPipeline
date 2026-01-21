"""TTS provider adapters and capability metadata."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


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
        _ = self._resolve_voice(voice)
        if audio_format == "wav":
            return _generate_silence_frames()
        return f"GOOGLE_TTS_PLACEHOLDER:{text}".encode("utf-8")


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
        _ = self._resolve_voice(voice)
        if audio_format == "wav":
            return _generate_silence_frames()
        return f"OPENAI_TTS_PLACEHOLDER:{text}".encode("utf-8")


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
        _ = self._resolve_voice(voice)
        if audio_format == "wav":
            return _generate_silence_frames()
        return f"AZURE_TTS_PLACEHOLDER:{text}".encode("utf-8")


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
