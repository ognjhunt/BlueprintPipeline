# Tools

This directory hosts shared utilities and pipeline helpers used by jobs across BlueprintPipeline (LLM access, asset cataloging, quality gates, adapters, and operational helpers).

## Key references

- [API Reference](../docs/api/README.md)
- [Environment variables](config/ENVIRONMENT_VARIABLES.md)
- [Pipeline configuration](config/pipeline_config.json)

## Common entrypoints

- `tools/run_local_pipeline.py` — local end-to-end pipeline runner
- `tools/geniesim_adapter/` — Genie Sim export + local framework client
- `tools/llm_client/` — unified LLM provider client
- `tools/asset_catalog/` — asset indexing, embeddings, vector stores
- `tools/quality_gates/` — quality gate checks and SLI runner
- `tools/external_services/service_client.py` — resilient external API wrapper
