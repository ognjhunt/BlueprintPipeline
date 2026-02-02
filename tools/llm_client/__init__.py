"""
Unified LLM Client for BlueprintPipeline.

Provides a single abstraction layer supporting multiple LLM providers:
- Google Gemini (DEFAULT, 3.0 Pro with Google Search grounding)
- Anthropic Claude (Sonnet 4.5 with extended thinking)
- OpenAI GPT-5.1 (with adaptive reasoning)

This module allows jobs to seamlessly switch between providers based on
environment variables or explicit configuration, enabling fallback and
A/B testing of different models.

Environment Variables:
    LLM_PROVIDER: "gemini" | "anthropic" | "openai" | "mock" | "auto" (default: gemini)
    GEMINI_API_KEY: API key for Google Gemini
    ANTHROPIC_API_KEY: API key for Anthropic Claude
    OPENAI_API_KEY: API key for OpenAI
    LLM_MOCK_RESPONSE_PATH: JSON response path for mock provider

Usage:
    from tools.llm_client import create_llm_client, LLMProvider

    # Use default provider (Gemini)
    client = create_llm_client()

    # Or specify explicitly
    client = create_llm_client(provider=LLMProvider.ANTHROPIC)

    # Anthropic with extended thinking
    client = create_llm_client(
        provider=LLMProvider.ANTHROPIC,
        thinking_budget=32000  # tokens for thinking
    )

    # OpenAI with configurable reasoning
    client = create_llm_client(
        provider=LLMProvider.OPENAI,
        reasoning_effort="high"  # none, low, medium, high
    )

    # Generate content (unified API)
    response = client.generate(
        prompt="Analyze this scene",
        image=pil_image,  # optional
        json_output=True,
        use_web_search=True  # Gemini/OpenAI only
    )
"""

from .client import (
    LLMClient,
    LLMProvider,
    LLMResponse,
    GeminiClient,
    AnthropicClient,
    OpenAIClient,
    MockLLMClient,
    FallbackLLMClient,
    create_llm_client,
    get_default_provider,
)

__all__ = [
    "LLMClient",
    "LLMProvider",
    "LLMResponse",
    "GeminiClient",
    "AnthropicClient",
    "OpenAIClient",
    "MockLLMClient",
    "FallbackLLMClient",
    "create_llm_client",
    "get_default_provider",
]
