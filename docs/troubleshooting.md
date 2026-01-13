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

### `scene.usda` missing or empty
**Symptoms**: Final USD not created or size is 0 bytes.

**Likely causes**
- Manifest references assets that do not exist.
- Layout transforms are invalid (NaNs or extreme scales).

**Fixes**
- Check `assets/scene_manifest.json` for bad paths.
- Validate numeric values in `layout/scene_layout_scaled.json`.

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
