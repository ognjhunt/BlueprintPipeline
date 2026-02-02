"""
Unified LLM Client Implementation.

Supports:
- Google Gemini 3.0 Pro (with Google Search grounding) - DEFAULT
- Anthropic Claude Sonnet 4.5 (with extended thinking)
- OpenAI GPT-5.1 (with adaptive reasoning)

Environment Variables:
    LLM_PROVIDER: "gemini" | "anthropic" | "openai" | "mock" | "auto" (default: gemini)
    GEMINI_API_KEY: API key for Google Gemini
    ANTHROPIC_API_KEY: API key for Anthropic Claude
    OPENAI_API_KEY: API key for OpenAI
    LLM_MOCK_RESPONSE_PATH: JSON response path for mock provider
    LLM_FALLBACK_ENABLED: "true" | "false" (default: true)
    LLM_MAX_RETRIES: Number of retries (default: 3)

    # Model overrides (optional)
    GEMINI_MODEL: Override default Gemini model (default: gemini-3-flash-preview)
    ANTHROPIC_MODEL: Override default Anthropic model (default: claude-sonnet-4-5-20250929)
    OPENAI_MODEL: Override default OpenAI model (default: gpt-5.1)
"""

from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import os
import sys
import threading
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from tools.config.env import parse_bool_env
from tools.config.production_mode import resolve_production_mode
try:
    from PIL import Image
except ImportError:
    Image = None

# Secret Manager for secure API key storage
try:
    from tools.secret_store import get_secret_or_env, SecretIds
    HAVE_SECRET_MANAGER = True
except ImportError:
    HAVE_SECRET_MANAGER = False


def _get_secret_or_env_with_log(
    secret_id: str,
    env_var: str,
    label: str,
) -> Optional[str]:
    env_value = os.getenv(env_var)
    production_mode = _is_production_env()
    if HAVE_SECRET_MANAGER:
        try:
            value = get_secret_or_env(
                secret_id,
                env_var=env_var,
                fallback_to_env=not production_mode,
            )
        except Exception as exc:  # pragma: no cover - defensive
            if production_mode:
                raise ValueError(
                    f"{label} credentials must be stored in Secret Manager in production. "
                    f"Missing secret '{secret_id}' (env var '{env_var}' is not allowed)."
                ) from exc
            print(
                f"[LLM] WARNING: Failed to fetch secret for {label}: {exc}",
                file=sys.stderr,
            )
            value = env_value
        if value and not env_value:
            print(
                f"[LLM] Using Secret Manager for {label} credentials.",
                file=sys.stderr,
            )
        return value
    if production_mode:
        raise ValueError(
            f"{label} credentials must be stored in Secret Manager in production. "
            f"Secret Manager is unavailable; env var '{env_var}' is not allowed."
        )
    return env_value


def _has_secret_or_env(secret_id: str, env_var: str) -> bool:
    if os.getenv(env_var):
        return True
    if HAVE_SECRET_MANAGER:
        try:
            return bool(get_secret_or_env(secret_id, env_var=env_var))
        except Exception:  # pragma: no cover - defensive
            return False
    return False


def _is_production_env() -> bool:
    return (
        resolve_production_mode()
        or os.getenv("K_SERVICE") is not None
        or os.getenv("KUBERNETES_SERVICE_HOST") is not None
    )


# =============================================================================
# Enums and Data Classes
# =============================================================================


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    MOCK = "mock"
    AUTO = "auto"


@dataclass
class LLMResponse:
    """Unified response from any LLM provider."""
    text: str
    provider: LLMProvider
    model: str
    raw_response: Any = None
    error_message: Optional[str] = None

    # Parsed data (if JSON was requested)
    data: Optional[Dict[str, Any]] = None

    # Metadata
    usage: Dict[str, int] = field(default_factory=dict)
    latency_seconds: float = 0.0

    # For multi-modal responses
    images: List[bytes] = field(default_factory=list)

    # Web search sources (if enabled)
    sources: List[Dict[str, str]] = field(default_factory=list)

    def parse_json(self) -> Dict[str, Any]:
        """Parse response text as JSON."""
        if self.data is not None:
            return self.data

        text = self.text.strip()

        # Strip markdown code blocks if present
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]

        text = text.strip()
        self.data = json.loads(text)
        return self.data


# =============================================================================
# Rate Limiting, Concurrency, and Caching
# =============================================================================


class _LeakyBucketRateLimiter:
    def __init__(self, rate_qps: float):
        if rate_qps <= 0:
            raise ValueError("rate_qps must be > 0")
        self._interval = 1.0 / rate_qps
        self._next_available = time.monotonic()
        self._lock = threading.Lock()

    def reserve_delay(self) -> float:
        with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next_available - now)
            self._next_available = max(self._next_available, now) + self._interval
            return wait


class _LLMResponseCache:
    def __init__(self, ttl_seconds: float):
        self._ttl_seconds = ttl_seconds
        self._cache: Dict[str, tuple[float, LLMResponse]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[LLMResponse]:
        now = time.monotonic()
        with self._lock:
            entry = self._cache.get(key)
            if not entry:
                return None
            expires_at, response = entry
            if expires_at < now:
                self._cache.pop(key, None)
                return None
            return copy.deepcopy(response)

    def set(self, key: str, response: LLMResponse) -> None:
        expires_at = time.monotonic() + self._ttl_seconds
        cached_response = LLMResponse(
            text=response.text,
            provider=response.provider,
            model=response.model,
            raw_response=None,
            error_message=response.error_message,
            data=copy.deepcopy(response.data),
            usage=copy.deepcopy(response.usage),
            latency_seconds=0.0,
            images=copy.deepcopy(response.images),
            sources=copy.deepcopy(response.sources),
        )
        with self._lock:
            self._cache[key] = (expires_at, cached_response)


class _LLMRequestManager:
    def __init__(self) -> None:
        self._rate_limiters: Dict[LLMProvider, Optional[_LeakyBucketRateLimiter]] = {}
        self._rate_lock = threading.Lock()
        cache_ttl = self._get_float_env("LLM_CACHE_TTL_SECONDS", 0.0)
        cache_enabled = parse_bool_env(os.getenv("LLM_CACHE_ENABLED"), default=True)
        self._cache = _LLMResponseCache(cache_ttl) if cache_ttl > 0 and cache_enabled else None
        max_concurrency = self._get_int_env("LLM_MAX_CONCURRENCY", 0)
        self._semaphore = (
            threading.BoundedSemaphore(max_concurrency) if max_concurrency > 0 else None
        )

    @staticmethod
    def _get_float_env(name: str, default: float) -> float:
        value = os.getenv(name)
        if value is None or value == "":
            return default
        try:
            return float(value)
        except ValueError:
            print(f"[LLM] WARNING: Invalid {name}={value}, using default {default}", file=sys.stderr)
            return default

    @staticmethod
    def _get_int_env(name: str, default: int) -> int:
        value = os.getenv(name)
        if value is None or value == "":
            return default
        try:
            return int(value)
        except ValueError:
            print(f"[LLM] WARNING: Invalid {name}={value}, using default {default}", file=sys.stderr)
            return default

    def _resolve_rate_limit(self, provider: LLMProvider) -> Optional[float]:
        provider_key = f"LLM_RATE_LIMIT_QPS_{provider.value.upper()}"
        for key in (provider_key, "LLM_RATE_LIMIT_QPS"):
            value = os.getenv(key)
            if value is None or value == "":
                continue
            try:
                qps = float(value)
            except ValueError:
                print(f"[LLM] WARNING: Invalid {key}={value}; ignoring", file=sys.stderr)
                return None
            return qps if qps > 0 else None
        return None

    def _get_rate_limiter(self, provider: LLMProvider) -> Optional[_LeakyBucketRateLimiter]:
        with self._rate_lock:
            if provider in self._rate_limiters:
                return self._rate_limiters[provider]
            qps = self._resolve_rate_limit(provider)
            limiter = _LeakyBucketRateLimiter(qps) if qps else None
            self._rate_limiters[provider] = limiter
            return limiter

    def _apply_rate_limit(self, provider: LLMProvider) -> None:
        limiter = self._get_rate_limiter(provider)
        if limiter is None:
            return
        delay = limiter.reserve_delay()
        if delay > 0:
            print(
                f"[LLM] Throttling {provider.value} for {delay:.2f}s (rate limit).",
                file=sys.stderr,
            )
            time.sleep(delay)

    @contextmanager
    def request_context(self, provider: LLMProvider):
        self._apply_rate_limit(provider)
        if self._semaphore is None:
            yield
            return
        start = time.monotonic()
        self._semaphore.acquire()
        wait = time.monotonic() - start
        if wait > 0:
            print(f"[LLM] Throttling for {wait:.2f}s (max concurrency).", file=sys.stderr)
        try:
            yield
        finally:
            self._semaphore.release()

    def build_cache_key(
        self,
        *,
        provider: LLMProvider,
        model: str,
        prompt: str,
        json_output: bool,
        use_web_search: bool,
        temperature: float,
        max_tokens: Optional[int],
        extra: Dict[str, Any],
        image: Optional[Any],
        images: Optional[List[Any]],
    ) -> Optional[str]:
        if self._cache is None:
            return None
        if image is not None or images:
            return None
        payload = {
            "provider": provider.value,
            "model": model,
            "prompt": prompt,
            "json_output": json_output,
            "use_web_search": use_web_search,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "extra": extra,
        }
        encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def get_cached_response(self, key: Optional[str]) -> Optional[LLMResponse]:
        if self._cache is None or key is None:
            return None
        return self._cache.get(key)

    def set_cached_response(self, key: Optional[str], response: LLMResponse) -> None:
        if self._cache is None or key is None:
            return
        if response.error_message:
            return
        self._cache.set(key, response)


_REQUEST_MANAGER = _LLMRequestManager()


# =============================================================================
# Abstract Base Client
# =============================================================================


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    provider: LLMProvider
    model: str

    def __init__(
        self,
        model: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._model = model

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Default model ID for this provider."""
        pass

    @property
    def model(self) -> str:
        return self._model or self.default_model

    @property
    def supports_image_generation(self) -> bool:
        return False

    @abstractmethod
    def generate(
        self,
        prompt: str,
        image: Optional[Any] = None,
        images: Optional[List[Any]] = None,
        json_output: bool = False,
        use_web_search: bool = False,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate content from the LLM."""
        pass

    def generate_image(
        self,
        prompt: str,
        size: str = "1024x1024",
        **kwargs
    ) -> LLMResponse:
        """Generate an image from the LLM (if supported)."""
        if not self.supports_image_generation:
            return self._unsupported_image_response()
        return self._generate_image(prompt=prompt, size=size, **kwargs)

    @abstractmethod
    def _generate_image(
        self,
        prompt: str,
        size: str = "1024x1024",
        **kwargs
    ) -> LLMResponse:
        """Provider-specific image generation implementation."""
        pass

    def _unsupported_image_response(self) -> LLMResponse:
        return LLMResponse(
            text="",
            provider=self.provider,
            model=self.model,
            error_message=(
                f"{self.provider.value} does not support image generation. "
                "Use a provider that supports image outputs."
            ),
        )

    def _encode_image(self, image: Any) -> str:
        """Encode image to base64."""
        if Image is None:
            raise ImportError("PIL is required for image processing")

        if isinstance(image, (str, Path)):
            # Load from path
            img = Image.open(str(image))
        elif hasattr(image, 'read'):
            # File-like object
            img = Image.open(image)
        elif isinstance(image, Image.Image):
            img = image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        # Convert to RGB if needed
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Encode to base64
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


# =============================================================================
# Mock Client
# =============================================================================


class MockLLMClient(LLMClient):
    """Mock client for deterministic, credential-free LLM responses."""

    provider = LLMProvider.MOCK

    @property
    def default_model(self) -> str:
        return "mock"

    def generate(
        self,
        prompt: str,
        image: Optional[Any] = None,
        images: Optional[List[Any]] = None,
        json_output: bool = False,
        use_web_search: bool = False,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        cache_key = _REQUEST_MANAGER.build_cache_key(
            provider=self.provider,
            model=self.model,
            prompt=prompt,
            json_output=json_output,
            use_web_search=use_web_search,
            temperature=temperature,
            max_tokens=max_tokens,
            extra={},
            image=image,
            images=images,
        )
        cached = _REQUEST_MANAGER.get_cached_response(cache_key)
        if cached:
            return cached

        response_data = self._load_mock_response() or self._default_analysis()
        response_text = json.dumps(response_data, indent=2)
        response = LLMResponse(
            text=response_text,
            provider=self.provider,
            model=self.model,
            raw_response=response_data,
            data=response_data,
        )
        _REQUEST_MANAGER.set_cached_response(cache_key, response)
        return response

    def _generate_image(
        self,
        prompt: str,
        size: str = "1024x1024",
        **kwargs
    ) -> LLMResponse:
        return self._unsupported_image_response()

    def _load_mock_response(self) -> Optional[Dict[str, Any]]:
        path_value = os.getenv("LLM_MOCK_RESPONSE_PATH")
        if not path_value:
            return None
        path = Path(path_value)
        if not path.is_file():
            raise FileNotFoundError(f"Mock response path not found: {path}")
        with path.open("r", encoding="utf-8") as file_handle:
            return json.load(file_handle)

    def _default_analysis(self) -> Dict[str, Any]:
        return {
            "analysis": {
                "scene_summary": "Mock analysis response.",
                "recommended_policies": ["general_manipulation"],
            },
            "placement_regions": [
                {
                    "name": "mock_table_region",
                    "description": "Mock horizontal region on a table.",
                    "surface_type": "horizontal",
                    "parent_object_id": "table_01",
                    "position": [0.0, 0.0, 0.75],
                    "size": [0.8, 1.2, 0.02],
                    "rotation": [0.0, 0.0, 0.0],
                    "semantic_tags": ["table", "surface"],
                    "suitable_for": ["object", "props"],
                }
            ],
            "variation_assets": [
                {
                    "name": "mock_box",
                    "category": "props",
                    "description": "Mock cardboard box",
                    "semantic_class": "box",
                    "priority": "recommended",
                    "source_hint": "library",
                    "example_variants": ["small box", "medium box"],
                    "physics_hints": {
                        "mass_range_kg": [0.2, 1.0],
                        "friction": 0.6,
                        "collision_shape": "convex",
                    },
                }
            ],
            "policy_configs": [
                {
                    "policy_id": "general_manipulation",
                    "policy_name": "General Manipulation",
                    "description": "Mock policy configuration.",
                    "placement_regions_used": ["mock_table_region"],
                    "variation_assets_used": ["mock_box"],
                    "randomizers": [
                        {
                            "name": "light_randomizer",
                            "enabled": True,
                            "frequency": "per_episode",
                            "parameters": {"intensity_range": [0.8, 1.2]},
                        }
                    ],
                    "capture_config": {"annotations": ["rgb", "depth"]},
                    "scene_modifications": {
                        "hide_objects": [],
                        "spawn_additional": False,
                        "modify_existing": False,
                    },
                }
            ],
        }


# =============================================================================
# Gemini Client
# =============================================================================


class GeminiClient(LLMClient):
    """Google Gemini client implementation."""

    provider = LLMProvider.GEMINI

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model=model, **kwargs)

        # GAP-SEC-001 FIX: Use Secret Manager for API key (with fallback to env var)
        if api_key is None:
            self.api_key = _get_secret_or_env_with_log(
                SecretIds.GEMINI_API_KEY if HAVE_SECRET_MANAGER else "gemini-api-key",
                env_var="GEMINI_API_KEY",
                label="Gemini",
            )
        else:
            self.api_key = api_key

        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")

        try:
            from google import genai
            from google.genai import types
            self._genai = genai
            self._types = types
            self._client = genai.Client(
                api_key=self.api_key,
                http_options={"timeout": 120_000},  # 120s timeout for API calls
            )
        except ImportError:
            raise ImportError("google-genai package is required for Gemini support")

    @property
    def default_model(self) -> str:
        return os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")

    @property
    def default_image_model(self) -> str:
        return os.getenv("GEMINI_IMAGE_MODEL", "gemini-3-pro-image-preview")

    @property
    def supports_image_generation(self) -> bool:
        return True

    def generate(
        self,
        prompt: str,
        image: Optional[Any] = None,
        images: Optional[List[Any]] = None,
        json_output: bool = False,
        use_web_search: bool = False,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate content using Gemini."""

        cache_key = _REQUEST_MANAGER.build_cache_key(
            provider=self.provider,
            model=self.model,
            prompt=prompt,
            json_output=json_output,
            use_web_search=use_web_search,
            temperature=temperature,
            max_tokens=max_tokens,
            extra={},
            image=image,
            images=images,
        )
        cached = _REQUEST_MANAGER.get_cached_response(cache_key)
        if cached:
            return cached

        start_time = time.time()

        # Build contents
        contents = []

        # Add images first (if any)
        all_images = []
        if image is not None:
            all_images.append(image)
        if images:
            all_images.extend(images)

        for img in all_images:
            if isinstance(img, (str, Path)):
                # Load PIL image
                if Image is not None:
                    contents.append(Image.open(str(img)).convert("RGB"))
                else:
                    raise ImportError("PIL required for image input")
            elif Image is not None and isinstance(img, Image.Image):
                contents.append(img)
            else:
                contents.append(img)

        # Add prompt
        contents.append(prompt)

        # Build config
        config_kwargs: Dict[str, Any] = {
            "temperature": temperature,
        }

        if max_tokens:
            config_kwargs["max_output_tokens"] = max_tokens

        if json_output:
            config_kwargs["response_mime_type"] = "application/json"

        # Enable thinking, grounding, and URL context for Gemini 3.x models
        # Callers can pass disable_tools=True to skip AFC/thinking (e.g. simple factual prompts)
        if self.model.startswith("gemini-3") and not kwargs.get("disable_tools", False):
            config_kwargs["thinking_config"] = self._types.ThinkingConfig(thinking_level="HIGH")
            tools = [
                self._types.Tool(url_context=self._types.UrlContext()),
                self._types.Tool(googleSearch=self._types.GoogleSearch()),
            ]
            config_kwargs["tools"] = tools
        elif use_web_search:
            if hasattr(self._types, "Tool") and hasattr(self._types, "GoogleSearch"):
                config_kwargs["tools"] = [
                    self._types.Tool(googleSearch=self._types.GoogleSearch())
                ]

        config = self._types.GenerateContentConfig(**config_kwargs)

        # Execute with retries
        last_error = None
        for attempt in range(self.max_retries):
            try:
                with _REQUEST_MANAGER.request_context(self.provider):
                    response = self._client.models.generate_content(
                        model=self.model,
                        contents=contents,
                        config=config,
                    )

                latency = time.time() - start_time

                # Extract sources from grounding metadata if available
                sources = []
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'grounding_metadata'):
                        meta = candidate.grounding_metadata
                        if hasattr(meta, 'search_entry_point') and meta.search_entry_point:
                            sources.append({
                                "title": "Google Search",
                                "url": getattr(meta.search_entry_point, 'uri', ''),
                            })

                llm_response = LLMResponse(
                    text=response.text or "",
                    provider=self.provider,
                    model=self.model,
                    raw_response=response,
                    latency_seconds=latency,
                    sources=sources,
                )
                _REQUEST_MANAGER.set_cached_response(cache_key, llm_response)
                return llm_response

            except Exception as e:
                last_error = e
                # Don't retry rate-limit errors — let caller handle cooldown/cascade
                _err_lower = str(e).lower()
                if any(t in _err_lower for t in ("429", "rate limit", "too many requests", "quota", "resource_exhausted")):
                    raise
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))

        raise RuntimeError(f"Gemini generation failed after {self.max_retries} attempts: {last_error}")

    def _generate_image(
        self,
        prompt: str,
        size: str = "1024x1024",
        **kwargs
    ) -> LLMResponse:
        """Generate an image using Gemini 3.0 Pro Image."""

        start_time = time.time()

        config = self._types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
            temperature=kwargs.get("temperature", 0.8),
        )

        last_error = None
        for attempt in range(self.max_retries):
            try:
                with _REQUEST_MANAGER.request_context(self.provider):
                    response = self._client.models.generate_content(
                        model=self.default_image_model,
                        contents=prompt,
                        config=config,
                    )

                # Extract images from response
                images = []
                if response.candidates and len(response.candidates) > 0:
                    candidate = response.candidates[0]
                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            if hasattr(part, 'inline_data') and part.inline_data:
                                image_data = part.inline_data.data
                                if isinstance(image_data, str):
                                    images.append(base64.b64decode(image_data))
                                else:
                                    images.append(image_data)

                return LLMResponse(
                    text=response.text or "",
                    provider=self.provider,
                    model=self.default_image_model,
                    raw_response=response,
                    images=images,
                    latency_seconds=time.time() - start_time,
                )

            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))

        raise RuntimeError(f"Gemini image generation failed: {last_error}")


# =============================================================================
# OpenAI Client (GPT-5.1 with Adaptive Reasoning)
# =============================================================================


class OpenAIClient(LLMClient):
    """OpenAI GPT-5.1 client implementation with adaptive reasoning.

    Based on OpenAI latest models documentation (2025):
    https://platform.openai.com/docs/models/gpt-5.1

    GPT-5.1 features:
    - Adaptive reasoning (configurable effort: none, low, medium, high)
    - Faster and more token-efficient than GPT-5
    - Web browsing enabled
    - Enhanced vision understanding
    - Structured outputs
    """

    provider = LLMProvider.OPENAI

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        reasoning_effort: str = "high",
        **kwargs
    ):
        super().__init__(model=model, **kwargs)

        # GAP-SEC-001 FIX: Use Secret Manager for API key (with fallback to env var)
        if api_key is None:
            self.api_key = _get_secret_or_env_with_log(
                SecretIds.OPENAI_API_KEY if HAVE_SECRET_MANAGER else "openai-api-key",
                env_var="OPENAI_API_KEY",
                label="OpenAI",
            )
        else:
            self.api_key = api_key

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        # Reasoning effort: none, low, medium, high
        self.reasoning_effort = reasoning_effort

        try:
            from openai import OpenAI
            self._openai = OpenAI
            self._client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("openai package is required for OpenAI support")

    @property
    def default_model(self) -> str:
        # GPT-5.1 with adaptive reasoning
        return os.getenv("OPENAI_MODEL", "gpt-5.1")

    @property
    def default_image_model(self) -> str:
        return os.getenv("OPENAI_IMAGE_MODEL", "dall-e-3")

    @property
    def supports_image_generation(self) -> bool:
        return True

    def generate(
        self,
        prompt: str,
        image: Optional[Any] = None,
        images: Optional[List[Any]] = None,
        json_output: bool = False,
        use_web_search: bool = False,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate content using GPT-5.2 Thinking."""

        # Enable adaptive reasoning for GPT-5.1
        # Use reasoning_effort from kwargs, instance default, or "high"
        effort = kwargs.get("reasoning_effort", self.reasoning_effort)

        cache_key = _REQUEST_MANAGER.build_cache_key(
            provider=self.provider,
            model=self.model,
            prompt=prompt,
            json_output=json_output,
            use_web_search=use_web_search,
            temperature=temperature,
            max_tokens=max_tokens,
            extra={"reasoning_effort": effort},
            image=image,
            images=images,
        )
        cached = _REQUEST_MANAGER.get_cached_response(cache_key)
        if cached:
            return cached

        start_time = time.time()

        # Build messages
        messages = []

        # System message for better reasoning
        system_msg = "You are an expert assistant for robotics simulation and scene analysis."
        if json_output:
            system_msg += " Always respond with valid JSON only, no additional text."

        messages.append({"role": "system", "content": system_msg})

        # Build user message content
        content = []

        # Add images
        all_images = []
        if image is not None:
            all_images.append(image)
        if images:
            all_images.extend(images)

        for img in all_images:
            img_base64 = self._encode_image(img)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_base64}",
                    "detail": "high"
                }
            })

        # Add text
        content.append({"type": "text", "text": prompt})

        messages.append({"role": "user", "content": content})

        # Build request kwargs
        request_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }

        if max_tokens:
            request_kwargs["max_completion_tokens"] = max_tokens

        # Enable web search for GPT-5.2 Thinking
        if use_web_search:
            request_kwargs["tools"] = [{"type": "web_search"}]
            request_kwargs["tool_choice"] = "auto"

        # JSON mode
        if json_output:
            request_kwargs["response_format"] = {"type": "json_object"}

        if effort and effort != "none":
            request_kwargs["reasoning_effort"] = effort

        # Execute with retries
        last_error = None
        for attempt in range(self.max_retries):
            try:
                with _REQUEST_MANAGER.request_context(self.provider):
                    response = self._client.chat.completions.create(**request_kwargs)

                latency = time.time() - start_time

                # Extract text
                text = response.choices[0].message.content or ""

                # Extract sources from web search if available
                sources = []
                if hasattr(response.choices[0].message, 'tool_calls'):
                    tool_calls = response.choices[0].message.tool_calls or []
                    for tc in tool_calls:
                        if tc.type == "web_search" and hasattr(tc, 'results'):
                            for result in tc.results:
                                sources.append({
                                    "title": getattr(result, 'title', ''),
                                    "url": getattr(result, 'url', ''),
                                })

                # Usage stats
                usage = {}
                if response.usage:
                    usage = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    }
                    if hasattr(response.usage, 'reasoning_tokens'):
                        usage["reasoning_tokens"] = response.usage.reasoning_tokens

                llm_response = LLMResponse(
                    text=text,
                    provider=self.provider,
                    model=self.model,
                    raw_response=response,
                    usage=usage,
                    latency_seconds=latency,
                    sources=sources,
                )
                _REQUEST_MANAGER.set_cached_response(cache_key, llm_response)
                return llm_response

            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))

        raise RuntimeError(f"OpenAI generation failed after {self.max_retries} attempts: {last_error}")

    def _generate_image(
        self,
        prompt: str,
        size: str = "1024x1024",
        **kwargs
    ) -> LLMResponse:
        """Generate an image using DALL-E."""

        start_time = time.time()

        last_error = None
        for attempt in range(self.max_retries):
            try:
                with _REQUEST_MANAGER.request_context(self.provider):
                    response = self._client.images.generate(
                        model=self.default_image_model,
                        prompt=prompt,
                        size=size,
                        quality=kwargs.get("quality", "hd"),
                        n=1,
                        response_format="b64_json",
                    )

                # Extract image data
                images = []
                if response.data:
                    for img_data in response.data:
                        if img_data.b64_json:
                            images.append(base64.b64decode(img_data.b64_json))

                return LLMResponse(
                    text="",
                    provider=self.provider,
                    model=self.default_image_model,
                    raw_response=response,
                    images=images,
                    latency_seconds=time.time() - start_time,
                )

            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))

        raise RuntimeError(f"OpenAI image generation failed: {last_error}")


# =============================================================================
# Anthropic Client (Claude Sonnet 4.5 with Extended Thinking)
# =============================================================================


class AnthropicClient(LLMClient):
    """Anthropic Claude Sonnet 4.5 client implementation with extended thinking.

    Based on Anthropic latest models documentation (2025):
    https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking

    Claude Sonnet 4.5 features:
    - Extended thinking with configurable budget
    - Interleaved thinking for agentic workflows
    - Best-in-class coding capabilities
    - 1M token context window (with beta header)
    """

    provider = LLMProvider.ANTHROPIC

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        thinking_budget: int = 16000,
        enable_thinking: bool = True,
        **kwargs
    ):
        super().__init__(model=model, **kwargs)

        # GAP-SEC-001 FIX: Use Secret Manager for API key (with fallback to env var)
        if api_key is None:
            self.api_key = _get_secret_or_env_with_log(
                SecretIds.ANTHROPIC_API_KEY if HAVE_SECRET_MANAGER else "anthropic-api-key",
                env_var="ANTHROPIC_API_KEY",
                label="Anthropic",
            )
        else:
            self.api_key = api_key

        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")

        # Extended thinking configuration
        # Minimum budget is 1024 tokens per Anthropic docs
        self.thinking_budget = max(1024, thinking_budget)
        self.enable_thinking = enable_thinking

        try:
            from anthropic import Anthropic
            self._anthropic = Anthropic
            self._client = Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("anthropic package is required for Anthropic support")

    @property
    def default_model(self) -> str:
        # Claude Sonnet 4.5 with specific version for production stability
        return os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")

    @property
    def supports_image_generation(self) -> bool:
        return False

    def generate(
        self,
        prompt: str,
        image: Optional[Any] = None,
        images: Optional[List[Any]] = None,
        json_output: bool = False,
        use_web_search: bool = False,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate content using Claude Sonnet 4.5 with extended thinking."""

        # Default max_tokens needs to accommodate thinking budget
        thinking_enabled = kwargs.get("enable_thinking", self.enable_thinking)
        thinking_budget = kwargs.get("thinking_budget", self.thinking_budget)

        cache_key = _REQUEST_MANAGER.build_cache_key(
            provider=self.provider,
            model=self.model,
            prompt=prompt,
            json_output=json_output,
            use_web_search=use_web_search,
            temperature=temperature,
            max_tokens=max_tokens,
            extra={
                "enable_thinking": thinking_enabled,
                "thinking_budget": thinking_budget,
            },
            image=image,
            images=images,
        )
        cached = _REQUEST_MANAGER.get_cached_response(cache_key)
        if cached:
            return cached

        start_time = time.time()

        # Build messages
        messages = []

        # Build user message content
        content = []

        # Add images (Claude uses base64 encoded images)
        all_images = []
        if image is not None:
            all_images.append(image)
        if images:
            all_images.extend(images)

        for img in all_images:
            img_base64 = self._encode_image(img)
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": img_base64,
                }
            })

        # Add text prompt
        if json_output:
            prompt = f"{prompt}\n\nRespond with valid JSON only, no additional text."
        content.append({"type": "text", "text": prompt})

        messages.append({"role": "user", "content": content})

        # Build request kwargs
        # When thinking is enabled, max_tokens must be > thinking_budget
        default_max = 20000 if thinking_enabled else 8192
        actual_max_tokens = max_tokens or default_max

        request_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": actual_max_tokens,
        }

        # Extended thinking configuration
        if thinking_enabled:
            request_kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": min(thinking_budget, actual_max_tokens - 1000),
            }
        else:
            # When thinking is disabled, we can use temperature
            request_kwargs["temperature"] = temperature

        # Note: Claude doesn't have built-in web search like Gemini/OpenAI
        # If use_web_search is True, we add guidance to the prompt instead
        if use_web_search:
            # Claude doesn't support web search natively
            # This is a no-op, but we note it for compatibility
            pass

        # Execute with retries
        last_error = None
        for attempt in range(self.max_retries):
            try:
                with _REQUEST_MANAGER.request_context(self.provider):
                    response = self._client.messages.create(**request_kwargs)

                latency = time.time() - start_time

                # Extract text from response
                text_parts = []
                for block in response.content:
                    if hasattr(block, 'type'):
                        if block.type == "text":
                            text_parts.append(block.text)
                        # Skip thinking blocks in the final output
                        # (they contain internal reasoning)

                text = "\n".join(text_parts)

                # Usage stats
                usage = {}
                if response.usage:
                    usage = {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                    }

                llm_response = LLMResponse(
                    text=text,
                    provider=self.provider,
                    model=self.model,
                    raw_response=response,
                    usage=usage,
                    latency_seconds=latency,
                    sources=[],  # Claude doesn't have web search sources
                )
                _REQUEST_MANAGER.set_cached_response(cache_key, llm_response)
                return llm_response

            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))

        raise RuntimeError(f"Anthropic generation failed after {self.max_retries} attempts: {last_error}")

    def _generate_image(
        self,
        prompt: str,
        size: str = "1024x1024",
        **kwargs
    ) -> LLMResponse:
        """Claude doesn't support image generation natively."""
        return self._unsupported_image_response()


# =============================================================================
# Factory Functions
# =============================================================================


def get_default_provider() -> LLMProvider:
    """Get the default LLM provider based on environment.

    Priority order when LLM_PROVIDER is "auto" or not set:
    1. Gemini (if GEMINI_API_KEY is set) - DEFAULT
    2. Anthropic (if ANTHROPIC_API_KEY is set)
    3. OpenAI (if OPENAI_API_KEY is set)
    """
    provider_str = os.getenv("LLM_PROVIDER", "gemini").lower()

    if provider_str == "openai":
        return LLMProvider.OPENAI
    elif provider_str == "anthropic":
        return LLMProvider.ANTHROPIC
    elif provider_str == "gemini":
        return LLMProvider.GEMINI
    elif provider_str == "mock":
        return LLMProvider.MOCK
    else:
        # Auto-detect based on available API keys (Gemini preferred)
        if _has_secret_or_env(
            SecretIds.GEMINI_API_KEY if HAVE_SECRET_MANAGER else "gemini-api-key",
            "GEMINI_API_KEY",
        ):
            return LLMProvider.GEMINI
        elif _has_secret_or_env(
            SecretIds.ANTHROPIC_API_KEY if HAVE_SECRET_MANAGER else "anthropic-api-key",
            "ANTHROPIC_API_KEY",
        ):
            return LLMProvider.ANTHROPIC
        elif _has_secret_or_env(
            SecretIds.OPENAI_API_KEY if HAVE_SECRET_MANAGER else "openai-api-key",
            "OPENAI_API_KEY",
        ):
            return LLMProvider.OPENAI
        else:
            if _is_production_env():
                raise ValueError(
                    "No LLM provider credentials found in Secret Manager or environment variables."
                )
            # Default to Gemini even without key (will fail with helpful error)
            return LLMProvider.GEMINI


def _create_client_for_provider(
    provider: LLMProvider,
    model: Optional[str] = None,
    **kwargs
) -> LLMClient:
    """Create a client for a specific provider."""
    if provider == LLMProvider.MOCK:
        return MockLLMClient(model=model, **kwargs)
    if provider == LLMProvider.OPENAI:
        return OpenAIClient(model=model, **kwargs)
    elif provider == LLMProvider.ANTHROPIC:
        return AnthropicClient(model=model, **kwargs)
    else:
        return GeminiClient(model=model, **kwargs)


def _get_fallback_order(primary: LLMProvider) -> List[LLMProvider]:
    """Get fallback provider order based on primary provider."""
    all_providers = [LLMProvider.GEMINI, LLMProvider.ANTHROPIC, LLMProvider.OPENAI]
    return [p for p in all_providers if p != primary]


def _is_rate_limit_error(exc: Exception) -> bool:
    """Check if an exception indicates a rate limit / quota error."""
    text = str(exc).lower()
    return any(t in text for t in ("429", "rate limit", "too many requests", "quota", "resource_exhausted"))


class FallbackLLMClient(LLMClient):
    """Wraps multiple LLM clients with automatic rate-limit fallback.

    On generate(), tries the primary client first. If rate-limited (429/quota),
    cascades to fallback clients in order. Per-provider cooldowns prevent
    repeatedly hitting a rate-limited provider.

    Environment Variables:
        LLM_FALLBACK_MODELS: Comma-separated fallback model specs
            (default: "gemini-2.5-flash,openai:gpt-5.1")
        LLM_FALLBACK_COOLDOWN_S: Cooldown seconds per provider after rate limit
            (default: 300)
    """

    provider = LLMProvider.GEMINI  # default; overridden by primary

    def __init__(
        self,
        primary: LLMClient,
        fallbacks: Optional[List[LLMClient]] = None,
        cooldown_s: float = 300.0,
    ):
        # Don't call super().__init__ with model — we delegate to primary
        self.max_retries = primary.max_retries
        self.retry_delay = primary.retry_delay
        self._model = None
        self._primary = primary
        self._fallbacks = fallbacks or []
        self._clients: List[LLMClient] = [primary] + self._fallbacks
        self._cooldowns: Dict[str, float] = {}
        self._cooldown_s = cooldown_s
        self._cooldown_logged: Dict[str, float] = {}

    @property
    def default_model(self) -> str:
        return self._primary.default_model

    @property
    def model(self) -> str:
        return self._primary.model

    def generate(
        self,
        prompt: str,
        image: Optional[Any] = None,
        images: Optional[List[Any]] = None,
        json_output: bool = False,
        use_web_search: bool = False,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        last_error: Optional[Exception] = None
        now = time.time()
        for client in self._clients:
            client_key = f"{getattr(client, 'provider', 'unknown')}:{client.model}"
            cooldown_until = self._cooldowns.get(client_key, 0.0)
            if now < cooldown_until:
                # Log once per cooldown period
                if now >= self._cooldown_logged.get(client_key, 0.0):
                    self._cooldown_logged[client_key] = now + 60.0
                    print(
                        f"[LLM_FALLBACK] Skipping {client_key} (on cooldown until "
                        f"{time.strftime('%H:%M:%S', time.localtime(cooldown_until))})",
                        file=sys.stderr,
                    )
                continue
            try:
                return client.generate(
                    prompt=prompt,
                    image=image,
                    images=images,
                    json_output=json_output,
                    use_web_search=use_web_search,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )
            except Exception as e:
                last_error = e
                if _is_rate_limit_error(e):
                    self._cooldowns[client_key] = time.time() + self._cooldown_s
                    print(
                        f"[LLM_FALLBACK] {client_key} rate-limited, trying next provider",
                        file=sys.stderr,
                    )
                    continue
                # Non-rate-limit errors: propagate immediately
                raise
        # All clients exhausted (all on cooldown or rate-limited)
        if last_error:
            raise last_error
        raise RuntimeError("All LLM providers are on cooldown")

    def _generate_image(
        self,
        prompt: str,
        size: str = "1024x1024",
        **kwargs,
    ) -> LLMResponse:
        """Delegate image generation to the primary client."""
        return self._primary._generate_image(prompt=prompt, size=size, **kwargs)


def create_llm_client(
    provider: Optional[LLMProvider] = None,
    model: Optional[str] = None,
    fallback_enabled: bool = True,
    **kwargs
) -> LLMClient:
    """Create an LLM client for the specified or auto-detected provider.

    Args:
        provider: LLM provider to use (default: Gemini)
        model: Specific model to use (default: provider's default)
        fallback_enabled: Enable fallback to other providers on error
        **kwargs: Additional arguments passed to the client

            For Anthropic:
                thinking_budget (int): Token budget for extended thinking (default: 16000)
                enable_thinking (bool): Enable extended thinking (default: True)

            For OpenAI:
                reasoning_effort (str): Reasoning level - none, low, medium, high (default: high)

    Returns:
        LLMClient instance

    Raises:
        ValueError: If no valid provider can be initialized

    Example:
        # Use default provider (Gemini)
        client = create_llm_client()

        # Use specific provider
        client = create_llm_client(provider=LLMProvider.ANTHROPIC)

        # Use Anthropic with custom thinking budget
        client = create_llm_client(
            provider=LLMProvider.ANTHROPIC,
            thinking_budget=32000
        )

        # Use OpenAI with medium reasoning
        client = create_llm_client(
            provider=LLMProvider.OPENAI,
            reasoning_effort="medium"
        )

        # Generate content
        response = client.generate(
            prompt="Analyze this scene",
            image=pil_image,
            json_output=True
        )
    """
    if provider is None or provider == LLMProvider.AUTO:
        provider = get_default_provider()

    fallback_enabled = fallback_enabled and parse_bool_env(
        os.getenv("LLM_FALLBACK_ENABLED"),
        default=True,
    )

    # Build the primary client
    primary_client: Optional[LLMClient] = None
    try:
        primary_client = _create_client_for_provider(provider, model, **kwargs)
    except Exception as e:
        if fallback_enabled:
            print(f"[LLM] Primary provider {provider.value} failed: {e}", file=sys.stderr)
            print(f"[LLM] Attempting fallback...", file=sys.stderr)

            fallback_order = _get_fallback_order(provider)
            errors = {provider.value: str(e)}

            for fallback_provider in fallback_order:
                try:
                    print(f"[LLM] Trying {fallback_provider.value}...", file=sys.stderr)
                    primary_client = _create_client_for_provider(fallback_provider, model, **kwargs)
                    break
                except Exception as e2:
                    errors[fallback_provider.value] = str(e2)
                    continue

            if primary_client is None:
                error_details = ", ".join(f"{k} ({v})" for k, v in errors.items())
                raise ValueError(f"Failed to initialize any provider: {error_details}")
        else:
            raise

    # Build runtime fallback chain for rate-limit resilience
    if fallback_enabled:
        _fallback_specs = os.getenv(
            "LLM_FALLBACK_MODELS", "gemini-2.5-flash,openai:gpt-5.1"
        ).split(",")
        _cooldown_s = float(os.getenv("LLM_FALLBACK_COOLDOWN_S", "300"))
        _fallback_clients: List[LLMClient] = []
        for spec in _fallback_specs:
            spec = spec.strip()
            if not spec:
                continue
            try:
                if spec.startswith("openai:"):
                    _fb_model = spec[len("openai:"):]
                    _fb = _create_client_for_provider(LLMProvider.OPENAI, _fb_model, **kwargs)
                elif spec.startswith("anthropic:"):
                    _fb_model = spec[len("anthropic:"):]
                    _fb = _create_client_for_provider(LLMProvider.ANTHROPIC, _fb_model, **kwargs)
                else:
                    # Default: Gemini model
                    _fb = _create_client_for_provider(LLMProvider.GEMINI, spec, **kwargs)
                # Skip if same as primary
                if (getattr(_fb, 'provider', None) == getattr(primary_client, 'provider', None)
                        and _fb.model == primary_client.model):
                    continue
                _fallback_clients.append(_fb)
                print(f"[LLM_FALLBACK] Registered fallback: {spec}", file=sys.stderr)
            except Exception as _fb_err:
                print(f"[LLM_FALLBACK] Skipping fallback '{spec}': {_fb_err}", file=sys.stderr)

        if _fallback_clients:
            return FallbackLLMClient(
                primary=primary_client,
                fallbacks=_fallback_clients,
                cooldown_s=_cooldown_s,
            )

    return primary_client
