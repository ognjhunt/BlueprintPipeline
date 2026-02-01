# VM pipeline run report (2026-02-01)

## Request
Run the VM pipeline with the patched image that includes PR #1034 changes, allow `init_robot` to warm up, and validate episode JSON outputs.

## Execution summary
- **Start time (UTC):** 2026-02-01T22:31:40Z
- **End time (UTC):** 2026-02-01T22:31:41Z
- **Environment:** /workspace/BlueprintPipeline

## Steps attempted
1. `bash scripts/vm-start.sh`

## Outcome
- The VM start script failed immediately because Docker is not available in this environment.
- As a result, the full pipeline could not be executed and no episode JSONs were produced for inspection.

## Errors observed
- `sudo: docker: command not found`

## Follow-up needed
- Re-run the pipeline on a VM with Docker + GPU/Isaac Sim runtime available.
- Ensure the patched image containing PR #1034 changes is used for the episode-generation pipeline.
- After a successful run, inspect episode JSON outputs for lock-release errors, observation counts, and per-frame fields.
