# text-scene-gen-job

## Purpose / scope
Generates Stage 1 text scene packages under `scenes/<scene_id>/textgen/` from
`scene_request.json` (`schema_version=v1`).

The output is intentionally intermediate and feeds `text-scene-adapter-job`,
which converts it into canonical Blueprint artifacts for Stage 2+.

## Primary entrypoints
- `generate_text_scene.py` job entrypoint.
- `Dockerfile` container definition.

## Required inputs / outputs
- **Inputs:** `scenes/<scene_id>/prompts/scene_request.json`.
- **Outputs:** `textgen/package.json`, placement graph, quality report, markers.

## Key environment variables
- `BUCKET` (required)
- `SCENE_ID` (required)
- `REQUEST_OBJECT` (default: `scenes/<scene_id>/prompts/scene_request.json`)
- `TEXTGEN_PREFIX` (default: `scenes/<scene_id>/textgen`)
- `TEXT_SEED` (default: `1`)
- `DEFAULT_SOURCE_MODE` (default: `text`)
- `TEXT_GEN_MAX_SEEDS` (default: `16`)
- `TEXT_GEN_STANDARD_PROFILE` (default: `standard_v1`)
- `TEXT_GEN_PREMIUM_PROFILE` (default: `premium_v1`)
- `TEXT_GEN_QUALITY_TIER` (optional override)
