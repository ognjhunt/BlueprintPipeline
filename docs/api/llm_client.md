# `tools/llm_client` API

## Purpose

`tools/llm_client` provides a unified interface for LLM providers (Gemini, Anthropic, OpenAI) with consistent request/response handling, retry settings, and provider selection. It also integrates Secret Manager in production while supporting environment-based credentials locally.【F:tools/llm_client/__init__.py†L1-L47】【F:tools/llm_client/client.py†L1-L53】

## Public entrypoints

- Factory and provider helpers:
  - `create_llm_client`, `get_default_provider`
  - `LLMProvider` (enum)
- Client classes:
  - `LLMClient` (abstract base)
  - `GeminiClient`, `AnthropicClient`, `OpenAIClient`
- Response model:
  - `LLMResponse`【F:tools/llm_client/__init__.py†L40-L63】

## Configuration / environment variables

Supported environment variables (read by `LLMClient` helpers):

- `LLM_PROVIDER`: `gemini | anthropic | openai | auto` (default: `gemini`)
- `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`
- `LLM_FALLBACK_ENABLED`: `true | false` (default: `true`)
- `LLM_MAX_RETRIES`: retries before surfacing failure (default: `3`)
- Model overrides:
  - `GEMINI_MODEL` (default `gemini-3-pro-preview`)
  - `ANTHROPIC_MODEL` (default `claude-sonnet-4-5-20250929`)
  - `OPENAI_MODEL` (default `gpt-5.1`)【F:tools/llm_client/client.py†L10-L43】

Secret Manager integration is enforced in production via `_get_secret_or_env_with_log`, which requires secrets in Secret Manager when `PIPELINE_ENV` resolves to production (via `resolve_production_mode`) or when `K_SERVICE`/`KUBERNETES_SERVICE_HOST` indicates a production runtime.【F:tools/llm_client/client.py†L55-L124】

## Request/response payloads & data models

### Request model

All providers implement `LLMClient.generate`:

- `prompt`: string prompt
- `image` / `images`: optional PIL images
- `json_output`: request JSON-structured output
- `use_web_search`: enables search grounding for supported providers
- `temperature`, `max_tokens`, plus provider-specific kwargs

See the abstract method signature in `LLMClient.generate` for the common request shape.【F:tools/llm_client/client.py†L140-L201】

### Response model

`LLMResponse` normalizes provider output:

- `text`, `provider`, `model`
- `usage`, `latency_seconds`
- `data`: parsed JSON if `json_output=True`
- `images`: for multimodal outputs
- `sources`: web search citations when applicable
- `parse_json()` helper to parse JSON safely from the text response【F:tools/llm_client/client.py†L90-L132】

## Example usage

```python
from tools.llm_client import create_llm_client, LLMProvider

client = create_llm_client(provider=LLMProvider.GEMINI)
response = client.generate(
    prompt="Describe the lighting and layout of this scene.",
    json_output=True,
    use_web_search=False,
)
print(response.text)
print(response.parse_json())
```
