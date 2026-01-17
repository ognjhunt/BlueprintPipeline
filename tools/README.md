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

