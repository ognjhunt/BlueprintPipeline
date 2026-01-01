"""
Unified LLM Client Implementation.

Supports:
- Google Gemini 3.0 Pro (with Google Search grounding) - DEFAULT
- Anthropic Claude Sonnet 4.5 (with extended thinking)
- OpenAI GPT-5.1 (with adaptive reasoning)

Environment Variables:
    LLM_PROVIDER: "gemini" | "anthropic" | "openai" | "auto" (default: gemini)
    GEMINI_API_KEY: API key for Google Gemini
    ANTHROPIC_API_KEY: API key for Anthropic Claude
    OPENAI_API_KEY: API key for OpenAI
    LLM_FALLBACK_ENABLED: "true" | "false" (default: true)
    LLM_MAX_RETRIES: Number of retries (default: 3)

    # Model overrides (optional)
    GEMINI_MODEL: Override default Gemini model (default: gemini-3-pro-preview)
    ANTHROPIC_MODEL: Override default Anthropic model (default: claude-sonnet-4-5-20250929)
    OPENAI_MODEL: Override default OpenAI model (default: gpt-5.1)
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    from PIL import Image
except ImportError:
    Image = None


# =============================================================================
# Enums and Data Classes
# =============================================================================


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    AUTO = "auto"


@dataclass
class LLMResponse:
    """Unified response from any LLM provider."""
    text: str
    provider: LLMProvider
    model: str
    raw_response: Any = None

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

    @abstractmethod
    def generate_image(
        self,
        prompt: str,
        size: str = "1024x1024",
        **kwargs
    ) -> LLMResponse:
        """Generate an image from the LLM (if supported)."""
        pass

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

        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key required. Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable."
            )

        try:
            from google import genai
            from google.genai import types
            self._genai = genai
            self._types = types
            self._client = genai.Client(api_key=self.api_key)
        except ImportError:
            raise ImportError("google-genai package is required for Gemini support")

    @property
    def default_model(self) -> str:
        return os.getenv("GEMINI_MODEL", "gemini-2.5-pro-preview-06-05")

    @property
    def default_image_model(self) -> str:
        return os.getenv("GEMINI_IMAGE_MODEL", "gemini-3-pro-image-preview")

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

        # Add web search grounding
        if use_web_search:
            if hasattr(self._types, "Tool") and hasattr(self._types, "GoogleSearch"):
                config_kwargs["tools"] = [
                    self._types.Tool(googleSearch=self._types.GoogleSearch())
                ]

        config = self._types.GenerateContentConfig(**config_kwargs)

        # Execute with retries
        last_error = None
        for attempt in range(self.max_retries):
            try:
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

                return LLMResponse(
                    text=response.text or "",
                    provider=self.provider,
                    model=self.model,
                    raw_response=response,
                    latency_seconds=latency,
                    sources=sources,
                )

            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))

        raise RuntimeError(f"Gemini generation failed after {self.max_retries} attempts: {last_error}")

    def generate_image(
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

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
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

        # Enable adaptive reasoning for GPT-5.1
        # Use reasoning_effort from kwargs, instance default, or "high"
        effort = kwargs.get("reasoning_effort", self.reasoning_effort)
        if effort and effort != "none":
            request_kwargs["reasoning_effort"] = effort

        # Execute with retries
        last_error = None
        for attempt in range(self.max_retries):
            try:
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

                return LLMResponse(
                    text=text,
                    provider=self.provider,
                    model=self.model,
                    raw_response=response,
                    usage=usage,
                    latency_seconds=latency,
                    sources=sources,
                )

            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))

        raise RuntimeError(f"OpenAI generation failed after {self.max_retries} attempts: {last_error}")

    def generate_image(
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

        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
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
        # Default max_tokens needs to accommodate thinking budget
        thinking_enabled = kwargs.get("enable_thinking", self.enable_thinking)
        thinking_budget = kwargs.get("thinking_budget", self.thinking_budget)

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

                return LLMResponse(
                    text=text,
                    provider=self.provider,
                    model=self.model,
                    raw_response=response,
                    usage=usage,
                    latency_seconds=latency,
                    sources=[],  # Claude doesn't have web search sources
                )

            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))

        raise RuntimeError(f"Anthropic generation failed after {self.max_retries} attempts: {last_error}")

    def generate_image(
        self,
        prompt: str,
        size: str = "1024x1024",
        **kwargs
    ) -> LLMResponse:
        """Claude doesn't support image generation natively."""
        raise NotImplementedError(
            "Anthropic Claude does not support image generation. "
            "Use Gemini or OpenAI for image generation tasks."
        )


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
    else:
        # Auto-detect based on available API keys (Gemini preferred)
        if os.getenv("GEMINI_API_KEY"):
            return LLMProvider.GEMINI
        elif os.getenv("ANTHROPIC_API_KEY"):
            return LLMProvider.ANTHROPIC
        elif os.getenv("OPENAI_API_KEY"):
            return LLMProvider.OPENAI
        else:
            # Default to Gemini even without key (will fail with helpful error)
            return LLMProvider.GEMINI


def _create_client_for_provider(
    provider: LLMProvider,
    model: Optional[str] = None,
    **kwargs
) -> LLMClient:
    """Create a client for a specific provider."""
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

    fallback_enabled = fallback_enabled and os.getenv("LLM_FALLBACK_ENABLED", "true").lower() == "true"

    try:
        return _create_client_for_provider(provider, model, **kwargs)
    except Exception as e:
        if fallback_enabled:
            print(f"[LLM] Primary provider {provider.value} failed: {e}", file=sys.stderr)
            print(f"[LLM] Attempting fallback...", file=sys.stderr)

            fallback_order = _get_fallback_order(provider)
            errors = {provider.value: str(e)}

            for fallback_provider in fallback_order:
                try:
                    print(f"[LLM] Trying {fallback_provider.value}...", file=sys.stderr)
                    return _create_client_for_provider(fallback_provider, model, **kwargs)
                except Exception as e2:
                    errors[fallback_provider.value] = str(e2)
                    continue

            # All providers failed
            error_details = ", ".join(f"{k} ({v})" for k, v in errors.items())
            raise ValueError(f"Failed to initialize any provider: {error_details}")
        else:
            raise
