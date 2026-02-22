# text-request-emitter-job

Emits daily autonomous text `scene_request.json` files for `source-orchestrator`.

## Behavior
- Acquires a per-day lock: `automation/text_daily/locks/<YYYY-MM-DD>.lock`
- Reads/writes autonomy state: `automation/text_daily/state.json`
- Generates weighted + LLM-expanded prompts via `tools/source_pipeline/prompt_engine.py`
- Writes requests to trigger path: `scenes/<scene_id>/prompts/scene_request.json`
- Writes run outputs:
  - `automation/text_daily/runs/<YYYY-MM-DD>/emitted_requests.json`
  - `automation/text_daily/runs/<YYYY-MM-DD>/emit_manifest.json`

## Required env
- `BUCKET`: target bucket name

## Optional env
- `TEXT_DAILY_QUOTA` (default `1`)
- `TEXT_AUTONOMY_STATE_PREFIX` (default `automation/text_daily`)
- `TEXT_AUTONOMY_TIMEZONE` (default `America/New_York`)
- `TEXT_AUTONOMY_RUN_DATE` (optional override, `YYYY-MM-DD`)
- `TEXT_AUTONOMY_PROVIDER_POLICY` (default `openrouter_qwen_primary`)
- `TEXT_AUTONOMY_TEXT_BACKEND` (default `hybrid_serial`, one of `scenesmith|sage|hybrid_serial`)
- `TEXT_AUTONOMY_QUALITY_TIER` (default `premium`)
- `TEXT_AUTONOMY_ALLOW_IMAGE_FALLBACK` (default `false`)
- `TEXT_AUTONOMY_SEED_COUNT` (default `1`)
- `TEXT_AUTONOMY_STORAGE_MODE` (`auto` | `gcs` | `filesystem`, default `auto`)

Prompt engine controls:
- `TEXT_PROMPT_USE_LLM` (default `true`)
- `TEXT_PROMPT_LLM_MAX_ATTEMPTS` (default `3`)
- `TEXT_PROMPT_LLM_RETRY_BACKOFF_SECONDS` (default `2`)
- `TEXT_PROMPT_LLM_REASONING_EFFORT` (default `high`)
- `TEXT_OPENROUTER_API_KEY` (fallback to `OPENROUTER_API_KEY`)
- `TEXT_OPENROUTER_BASE_URL` (default `https://openrouter.ai/api/v1`)
- `TEXT_OPENROUTER_MODEL_CHAIN` (default `qwen/qwen3.5-397b-a17b,moonshotai/kimi-k2.5`)
- `TEXT_OPENROUTER_INCLUDE_LEGACY_FALLBACK` (default `true`)
