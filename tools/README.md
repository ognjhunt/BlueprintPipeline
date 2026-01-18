# Tools

This directory hosts shared utilities and pipeline helpers used by jobs across BlueprintPipeline (LLM access, asset cataloging, quality gates, adapters, and operational helpers).

## Key references

- [API Reference](../docs/api/README.md)
- [Environment variables](config/ENVIRONMENT_VARIABLES.md)
- [Pipeline configuration](config/pipeline_config.json)
- Scene graph + health check defaults are defined under `scene_graph` and `health_checks` in `config/pipeline_config.json`.

## Common entrypoints

- `tools/run_local_pipeline.py` — local end-to-end pipeline runner
- `tools/geniesim_adapter/` — Genie Sim export + local framework client
- `tools/llm_client/` — unified LLM provider client
- `tools/asset_catalog/` — asset indexing, embeddings, vector stores
- `tools/quality_gates/` — quality gate checks and SLI runner
- `tools/external_services/service_client.py` — resilient external API wrapper
# tools

## Purpose / scope
Shared tooling and libraries that support pipeline jobs, validation, and integrations.

## Index
- `arena_integration/` Arena integration helpers.
- `asset_catalog/` asset catalog tooling.
- `batch_processing/` batch utilities.
- `geniesim_adapter/` GenieSim adapter helpers.
- `isaac_lab_tasks/` Isaac Lab task builders.
- `job_registry/` job registration utilities.
- `llm_client/` LLM client utilities.
- `pipeline_selector/` pipeline selection logic.
- `quality_gates/` quality gate checks.
- `scene_manifest/` scene manifest helpers.
- `storage_layout/` storage layout utilities.
- `workflow/` workflow helpers.
- `run_local_pipeline.py` local pipeline runner.
- `run_full_isaacsim_pipeline.py` full IsaacSim pipeline runner.
- `startup_validation.py` startup checks.

## Primary entrypoints
- Python modules under the directories above.
- Runner scripts in the repository root of `tools/`.

## Required inputs / outputs
- **Inputs:** configuration files, manifests, and job-specific parameters consumed by each tool.
- **Outputs:** generated manifests, reports, or validation results.

## Key environment variables
- Environment variables specific to each tool (e.g., storage paths, credentials, and service endpoints).

## How to run locally
- Run helper scripts directly, for example: `python run_local_pipeline.py`.
- Import modules in your own scripts using `from tools.<module> import ...`.
