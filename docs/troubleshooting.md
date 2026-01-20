# Troubleshooting

Common failure modes in BlueprintPipeline and recommended fixes.

## Pipeline orchestration

### Cloud Run job never starts
**Symptoms**: Workflow stuck in `RUNNING`, no logs in Cloud Run.

**Likely causes**
- Missing EventArc trigger or wrong bucket/prefix.
- Service account lacks `run.jobs.run` or `storage.objects.get` permissions.

**Fixes**
- Verify EventArc trigger filters and that `.regen3d_complete` markers exist.
- Ensure the workflow service account has Cloud Run Invoker + Storage Viewer/Writer roles.

### GCS inputs not found
**Symptoms**: Job exits with `404` or `No such object` when loading scene assets.

**Likely causes**
- `SCENE_ID` mismatch or wrong prefix variables (`ASSETS_PREFIX`, `LAYOUT_PREFIX`, etc.).
- Files generated in a different bucket or path.

**Fixes**
- Confirm environment variables and final GCS paths.
- Re-run `fixtures/generate_mock_regen3d.py` or `tools/run_local_pipeline.py` with the expected output directory.

## regen3d-job

### Missing `scene_layout_scaled.json`
**Symptoms**: Downstream jobs fail validation or crash on layout load.

**Likely causes**
- 3D-RE-GEN export incomplete or corrupted.

**Fixes**
- Re-run the regen3d export and verify `scene_info.json`, `pose.json`, and `bounds.json` exist for each object.
- Validate the output with `python tools/run_local_pipeline.py --validate`.

## simready-job

### Physics proxies not generated
**Symptoms**: `simready.usda` loads but objects have no collisions.

**Likely causes**
- Meshes missing or invalid (non-manifold or zero-area faces).
- Physics proxy generation disabled or failed.

**Fixes**
- Inspect meshes in `assets/obj_{id}/asset.glb` for geometry issues.
- Re-run with validation enabled and fix any geometry errors.

## usd-assembly-job

### OpenUSD (`pxr`) import fails
**Symptoms**: Job logs show `ImportError: No module named 'pxr'` or USD assembly exits early.

**Likely causes**
- OpenUSD bindings were not installed in the usd-assembly-job image.
- A custom image was built without `usd-core`.

**Fixes**
- Confirm the image installs `usd-core` (OpenUSD) and rebuild if needed.
- Run this smoke check inside the job image:

```bash
docker build -t usd-assembly-job:smoke usd-assembly-job
docker run --rm usd-assembly-job:smoke python -c "from pxr import Usd, UsdGeom, Sdf; print('pxr import OK')"
```

### `scene.usda` missing or empty
**Symptoms**: Final USD not created or size is 0 bytes.

**Likely causes**
- Manifest references assets that do not exist.
- Layout transforms are invalid (NaNs or extreme scales).

**Fixes**
- Check `assets/scene_manifest.json` for bad paths.
- Validate numeric values in `layout/scene_layout_scaled.json`.

## genie-sim services

### Genie Sim gRPC health check fails
**Symptoms**: gRPC client cannot connect, `UNAVAILABLE`, or pipeline stages report the Genie Sim server is not ready.

**Health check commands**
- **Preflight CLI (recommended)**:
  ```bash
  python -m tools.geniesim_adapter.geniesim_healthcheck --json
  ```
  **Expected response (healthy)**: `ok: true`, `status.server_running: true`, `server_ready: true`, with exit code `0`.
- **Direct gRPC check (local server)**:
  ```bash
  python -m tools.geniesim_adapter.geniesim_server --health-check --host <host> --port 50051
  ```
  **Expected response (healthy)**: logs `Health check status: SERVING`, exit code `0`.

**Common failure modes**
- **Server not running**: `status.server_running: false` in the preflight report.
- **Isaac Sim not found**: `missing` contains `ISAAC_SIM_PATH` guidance (wrong or missing install path).
- **gRPC packages/stubs missing**: `missing` lists `grpcio` or gRPC stubs; regenerate or install dependencies.
- **Server not ready**: `server_ready: false` after a successful socket connection (server still loading or stuck).

**Related tooling**
- Preflight CLI: [`tools/geniesim_adapter/geniesim_healthcheck.py`](../tools/geniesim_adapter/geniesim_healthcheck.py)
- Local gRPC server + health check: [`tools/geniesim_adapter/geniesim_server.py`](../tools/geniesim_adapter/geniesim_server.py)

## particulate-service

### Health checks failing or timing out
**Symptoms**: Cloud Run or GKE health checks return `503`, or upstream callers get `Service warming up` / `Warmup failed` errors.

**Health check URLs/commands**
- **Liveness**:
  ```bash
  curl -s -i http://<host>:<port>/
  ```
  **Expected response (healthy)**: `HTTP/1.1 200` with JSON body:
  ```json
  {
    "status": "ok",
    "ready": true,
    "service": "particulate"
  }
  ```
  **Expected response (warming up/failed)**: `HTTP/1.1 503` with JSON body containing `status: "warming_up"` or `status: "error"`.
- **Readiness**:
  ```bash
  curl -s -i http://<host>:<port>/ready
  ```
  **Expected response (healthy)**: `HTTP/1.1 200` with body `ready`.
  **Expected response (loading/error)**: `HTTP/1.1 503` with body `not ready: <reason>`.
- **Debug metadata** (requires debug token):
  ```bash
  curl -s -H "Authorization: Bearer <shared-secret>" http://<host>:<port>/debug | jq .
  ```
  **Expected response**: `HTTP/1.1 200` with JSON fields `models_ready`, `warmup_error`, `warmup_details`, and `installation_validation` when
  debug is explicitly enabled via `PARTICULATE_DEBUG=1` and `PARTICULATE_DEBUG_TOKEN=<shared-secret>` in non-production. In production,
  the debug token must be stored in Secret Manager under `particulate-debug-token`, and env var tokens are rejected; otherwise `HTTP/1.1 403`.

**Expected behavior**
- `GET /` returns `200` once model warmup completes; `503` during warmup or if warmup fails.
- `GET /ready` returns `200` when warmup is complete; `503` while loading or on warmup failure.
- `GET /debug` returns `200` with detailed warmup/validation metadata when debug access is explicitly enabled and authorized.

**Likely causes**
- Model warmup still running (first deploy or cold start).
- Missing Particulate files or CUDA/GPU misconfiguration during warmup.
- Installation validation failures (missing model files or invalid paths).

**Fixes**
- Wait for warmup to finish and re-check `GET /` or `GET /ready`.
- Use `GET /debug` (with the configured `Authorization` header) to confirm installation validation and CUDA checks, then fix missing files or GPU config.

## replicator-job

### Replicator script errors
**Symptoms**: `replicator/` folder exists but scripts fail to execute in Isaac Sim.

**Likely causes**
- Missing placement regions or incorrect material bindings.
- Unsupported primitives in asset USDs.

**Fixes**
- Ensure `replicator/placement_regions.usda` exists and references valid prims.
- Convert incompatible assets to USD with supported materials.

## isaac-lab-job

### Import errors in Isaac Lab
**Symptoms**: `ModuleNotFoundError` or missing task classes.

**Likely causes**
- Package not on `PYTHONPATH` or version mismatch with Isaac Lab.

**Fixes**
- Add the scene package root to `PYTHONPATH`.
- Confirm Isaac Lab version compatibility with the generated configs.

## Isaac Sim runtime

### `scene.usda` fails to load
**Symptoms**: Isaac Sim logs warnings about invalid prims or missing assets.

**Likely causes**
- Asset paths are relative to an unexpected root.
- Stage has invalid or corrupt prims.

**Fixes**
- Open the USD in a clean Isaac Sim stage and verify asset references.
- Ensure all paths are correct and accessible on disk or via Nucleus.

### Simulation unstable or exploding physics
**Symptoms**: Objects jitter or fly apart.

**Likely causes**
- Invalid mass/inertia, penetration at start pose, or unit scale issues.

**Fixes**
- Re-check scale assumptions and align object origin to the floor.
- Increase solver iterations or reduce time step in Isaac Sim.

## Local pipeline execution

### Validation fails in `run_local_pipeline.py`
**Symptoms**: Validation errors during local pipeline run.

**Likely causes**
- Missing required files in the scene directory.
- Invalid JSON in manifests.

**Fixes**
- Re-generate fixtures and confirm file presence.
- Reformat or regenerate JSON manifests with proper schemas.

## Quick checks

- Verify `.regen3d_complete` exists before running downstream jobs.
- Confirm `scene_manifest.json`, `scene_layout_scaled.json`, and `scene.usda` are created.
- Run `python tests/test_pipeline_e2e.py` for end-to-end validation.
