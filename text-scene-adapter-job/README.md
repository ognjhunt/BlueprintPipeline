# text-scene-adapter-job

## Purpose / scope
Converts text-generation Stage 1 outputs into BlueprintPipeline canonical artifacts:

- `scenes/<scene_id>/assets/scene_manifest.json`
- `scenes/<scene_id>/layout/scene_layout_scaled.json`
- `scenes/<scene_id>/seg/inventory.json`
- `scenes/<scene_id>/assets/.regen3d_complete`

This keeps downstream Stage 2+ workflows unchanged.

## Primary entrypoints
- `adapt_text_scene.py` job entrypoint.
- `Dockerfile` container definition.

## Required inputs / outputs
- **Inputs:** `scenes/<scene_id>/textgen/package.json`, `scene_request.json`.
- **Outputs:** canonical assets/layout/seg artifacts plus completion markers.

## Key environment variables
- `BUCKET` (required)
- `SCENE_ID` (required)
- `REQUEST_OBJECT` (default: `scenes/<scene_id>/prompts/scene_request.json`)
- `TEXTGEN_PREFIX` (default: `scenes/<scene_id>/textgen`)
- `ASSETS_PREFIX` (default: `scenes/<scene_id>/assets`)
- `LAYOUT_PREFIX` (default: `scenes/<scene_id>/layout`)
- `SEG_PREFIX` (default: `scenes/<scene_id>/seg`)
