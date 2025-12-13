"""
Unified LLM Client Implementation.

Supports:
- Google Gemini 3.0 Pro (with Google Search grounding)
- OpenAI GPT-5.2 Thinking (with web browsing)

Environment Variables:
    LLM_PROVIDER: "gemini" | "openai" | "auto" (default: auto)
    GEMINI_API_KEY: API key for Google Gemini
    OPENAI_API_KEY: API key for OpenAI
    LLM_FALLBACK_ENABLED: "true" | "false" (default: true)
    LLM_MAX_RETRIES: Number of retries (default: 3)
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

        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")

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
        return os.getenv("GEMINI_MODEL", "gemini-3-pro-preview")

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
# OpenAI Client (GPT-5.2 Thinking)
# =============================================================================


class OpenAIClient(LLMClient):
    """OpenAI GPT-5.2 Thinking client implementation.

    Based on OpenAI latest models documentation (2025):
    https://platform.openai.com/docs/guides/latest-model

    GPT-5.2 Thinking features:
    - Extended reasoning capabilities
    - Web browsing enabled
    - Enhanced vision understanding
    - Structured outputs
    """

    provider = LLMProvider.OPENAI

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model=model, **kwargs)

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        try:
            from openai import OpenAI
            self._openai = OpenAI
            self._client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("openai package is required for OpenAI support")

    @property
    def default_model(self) -> str:
        # GPT-5.2 Thinking is the latest model as of Dec 2025
        return os.getenv("OPENAI_MODEL", "gpt-5.2-thinking")

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

        # Enable thinking/reasoning for complex tasks
        if "thinking" in self.model.lower():
            # GPT-5.2 Thinking has built-in reasoning
            request_kwargs["reasoning_effort"] = kwargs.get("reasoning_effort", "high")

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
# Factory Functions
# =============================================================================


def get_default_provider() -> LLMProvider:
    """Get the default LLM provider based on environment."""
    provider_str = os.getenv("LLM_PROVIDER", "auto").lower()

    if provider_str == "openai":
        return LLMProvider.OPENAI
    elif provider_str == "gemini":
        return LLMProvider.GEMINI
    else:
        # Auto-detect based on available API keys
        if os.getenv("OPENAI_API_KEY") and not os.getenv("GEMINI_API_KEY"):
            return LLMProvider.OPENAI
        else:
            # Default to Gemini (existing behavior)
            return LLMProvider.GEMINI


def create_llm_client(
    provider: Optional[LLMProvider] = None,
    model: Optional[str] = None,
    fallback_enabled: bool = True,
    **kwargs
) -> LLMClient:
    """Create an LLM client for the specified or auto-detected provider.

    Args:
        provider: LLM provider to use (default: auto-detect)
        model: Specific model to use (default: provider's default)
        fallback_enabled: Enable fallback to other provider on error
        **kwargs: Additional arguments passed to the client

    Returns:
        LLMClient instance

    Raises:
        ValueError: If no valid provider can be initialized

    Example:
        # Auto-detect provider
        client = create_llm_client()

        # Use specific provider
        client = create_llm_client(provider=LLMProvider.OPENAI)

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
        if provider == LLMProvider.OPENAI:
            return OpenAIClient(model=model, **kwargs)
        else:
            return GeminiClient(model=model, **kwargs)
    except Exception as e:
        if fallback_enabled:
            # Try the other provider
            print(f"[LLM] Primary provider {provider.value} failed: {e}", file=sys.stderr)
            print(f"[LLM] Attempting fallback...", file=sys.stderr)

            fallback_provider = (
                LLMProvider.GEMINI if provider == LLMProvider.OPENAI
                else LLMProvider.OPENAI
            )

            try:
                if fallback_provider == LLMProvider.OPENAI:
                    return OpenAIClient(model=model, **kwargs)
                else:
                    return GeminiClient(model=model, **kwargs)
            except Exception as e2:
                raise ValueError(
                    f"Failed to initialize both providers: "
                    f"{provider.value} ({e}), {fallback_provider.value} ({e2})"
                )
        else:
            raise
