# scene-batch-job

## Purpose / scope
Runs batch scene processing by forwarding requests to `tools/run_scene_batch.py`.

> **Note:** This job is intentionally a lightweight wrapper around the shared batch
> tooling, keeping the container focused on configuration and execution.

## Primary entrypoints
- `entrypoint.py` Cloud Run entrypoint.
- `Dockerfile` container image definition.

## Key environment variables
- `SCENE_LIST_JSON`: JSON array (or `{ "scenes": [...] }`) of scene IDs to process.
- `SCENE_ROOT`: Root folder containing scenes (defaults to `/mnt/gcs/scenes` when mounted).
- `REPORTS_DIR`: Output directory for batch reports (defaults to `/mnt/gcs/batch_reports` when mounted).
- `MAX_CONCURRENT`, `RETRY_ATTEMPTS`, `RETRY_DELAY`, `RATE_LIMIT`: Optional batch controls.
