# `tools/external_services/service_client.py` API

## Purpose

`ServiceClient` is the unified wrapper for external service calls. It provides retries, circuit breaking, timeouts, and optional rate limiting so downstream integrations (e.g., Gemini, Genie Sim, GCS, Particulate) behave consistently under failure conditions.【F:tools/external_services/service_client.py†L1-L33】【F:tools/external_services/service_client.py†L54-L138】

## Public entrypoints

- Core classes:
  - `ServiceClientConfig`
  - `ServiceClient`
  - `RateLimiter`
- Factory helpers:
  - `create_gemini_client`
  - `create_genie_sim_client`
  - `create_gcs_client`
  - `create_particulate_client`【F:tools/external_services/service_client.py†L20-L220】

## Configuration / environment variables

`ServiceClientConfig` is created in code rather than from environment variables. Consumers can wire environment variables into config values as needed.

Key configuration fields:

- Retry: `max_retries`, `base_delay`, `max_delay`, `backoff_factor`
- Circuit breaker: `circuit_breaker_enabled`, `failure_threshold`, `recovery_timeout`
- Timeout: `default_timeout`
- Rate limiting: `rate_limit_enabled`, `calls_per_second`
- Retry filters: `retryable_status_codes`, `retryable_exceptions`【F:tools/external_services/service_client.py†L27-L69】

## Request/response payloads & data models

### Request model

`ServiceClient.call` accepts:

- `func`: a no-argument callable returning a response or data object
- `timeout`: optional per-call timeout override
- `operation_name`: optional label used for logging

Under the hood, the call is wrapped with timeout handling and retry logic. The callable is responsible for constructing the HTTP request (e.g., `requests.post`).【F:tools/external_services/service_client.py†L70-L159】

### Response model

The return value is whatever `func` returns (generic `T`). If the object has a `status_code` attribute and it is retryable, `ServiceClient` will raise an HTTP error to trigger retry logic. Errors after retry exhaustion are surfaced to callers, while circuit breaker and rate-limiter stats can be observed via `get_stats()`.【F:tools/external_services/service_client.py†L160-L292】

## Example usage

```python
import requests
from tools.external_services.service_client import ServiceClient, ServiceClientConfig

client = ServiceClient(
    ServiceClientConfig(service_name="vector_api", max_retries=4, default_timeout=15.0)
)

response = client.call(
    func=lambda: requests.post("https://api.example.com/embeddings", json={"text": "chair"}),
    operation_name="create_embeddings",
)
print(response.status_code)
```

