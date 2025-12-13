"""
Unified LLM Client for BlueprintPipeline.

Provides a single abstraction layer supporting multiple LLM providers:
- Google Gemini (default, 3.0 Pro)
- OpenAI GPT-5.2 Thinking (with web enabled)

This module allows jobs to seamlessly switch between providers based on
environment variables or explicit configuration, enabling fallback and
A/B testing of different models.

Usage:
    from tools.llm_client import create_llm_client, LLMProvider

    # Auto-detect provider from environment
    client = create_llm_client()

    # Or specify explicitly
    client = create_llm_client(provider=LLMProvider.OPENAI)

    # Generate content (unified API)
    response = client.generate(
        prompt="Analyze this scene",
        image=pil_image,  # optional
        json_output=True,
        use_web_search=True
    )
"""

from .client import (
    LLMClient,
    LLMProvider,
    LLMResponse,
    GeminiClient,
    OpenAIClient,
    create_llm_client,
    get_default_provider,
)

__all__ = [
    "LLMClient",
    "LLMProvider",
    "LLMResponse",
    "GeminiClient",
    "OpenAIClient",
    "create_llm_client",
    "get_default_provider",
]
