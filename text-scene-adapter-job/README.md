# text-scene-adapter-job

## Purpose / scope
Converts text-generation Stage 1 outputs into BlueprintPipeline canonical artifacts:

- `scenes/<scene_id>/assets/scene_manifest.json`
- `scenes/<scene_id>/layout/scene_layout_scaled.json`
- `scenes/<scene_id>/seg/inventory.json`
- `scenes/<scene_id>/assets/.stage1_complete`

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

Asset materialization controls:
- `TEXT_ASSET_RETRIEVAL_ENABLED` (default: `true`) enable retrieval before placeholder fallback
- `TEXT_ASSET_LIBRARY_PREFIXES` (default: `scenes`) comma-separated search roots under `/mnt/gcs`
- `TEXT_ASSET_LIBRARY_MAX_FILES` (default: `2500`) scan cap for index build
- `TEXT_ASSET_LIBRARY_MIN_SCORE` (default: `0.25`) minimum token-match score for retrieval
- `TEXT_ASSET_RETRIEVAL_MODE` (default: `ann_shadow`) retrieval mode (`lexical_primary`, `ann_shadow`, `ann_primary`)
- `TEXT_ASSET_ANN_ENABLED` (default: `true`) enable ANN semantic retrieval
- `TEXT_ASSET_ANN_TOP_K` (default: `40`) ANN top-k query size
- `TEXT_ASSET_ANN_MIN_SCORE` (default: `0.28`) semantic similarity floor
- `TEXT_ASSET_ANN_MAX_RERANK` (default: `20`) max ANN candidates reranked
- `TEXT_ASSET_ANN_NAMESPACE` (default: `assets-v1`) ANN namespace/collection
- `TEXT_ASSET_LEXICAL_FALLBACK_ENABLED` (default: `true`) lexical fallback when ANN misses/fails
- `TEXT_ASSET_ROLLOUT_STATE_PREFIX` (default: `automation/asset_retrieval_rollout`) rollout state object path
- `TEXT_ASSET_ROLLOUT_MIN_DECISIONS` (default: `500`) min decisions window for promotion
- `TEXT_ASSET_ROLLOUT_MIN_HIT_RATE` (default: `0.95`) promotion threshold
- `TEXT_ASSET_ROLLOUT_MAX_ERROR_RATE` (default: `0.01`) promotion threshold
- `TEXT_ASSET_ROLLOUT_MAX_P95_MS` (default: `400`) promotion threshold
- `TEXT_ASSET_CATALOG_ENABLED` (default: `true`) publish asset + scene metadata to Firestore asset catalog
- `TEXT_ASSET_EMBEDDING_QUEUE_PREFIX` (default: `automation/asset_embedding/queue`) queue embedding requests
- `TEXT_ASSET_EMBEDDING_PROCESSED_PREFIX` (default: `automation/asset_embedding/processed`) embedding results path
- `TEXT_ASSET_EMBEDDING_FAILED_PREFIX` (default: `automation/asset_embedding/failed`) embedding failures path
- `TEXT_ASSET_EMBEDDING_MODEL` (default: `text-embedding-3-small`) embedding model for indexing
- `TEXT_ASSET_REPLICATION_ENABLED` (default: `true`) enqueue async replication requests
- `TEXT_ASSET_REPLICATION_QUEUE_PREFIX` (default: `automation/asset_replication/queue`) queue object prefix
- `TEXT_ASSET_REPLICATION_TARGET` (default: `backblaze_b2`) replication target name
- `TEXT_ASSET_REPLICATION_TARGET_PREFIX` (default: `assets`) target key prefix
- `TEXT_ASSET_GENERATION_ENABLED` (default: `true`) enable provider fallback when retrieval misses
- `TEXT_ASSET_GENERATION_PROVIDER` (default: `sam3d`) generation provider
- `TEXT_ASSET_GENERATION_PROVIDER_CHAIN` (default: `sam3d,hunyuan3d`) ordered provider fallback chain
- `TEXT_ASSET_GENERATED_CACHE_ENABLED` (default: `true`) cache generated bundles for reuse
- `TEXT_ASSET_GENERATED_CACHE_PREFIX` (default: `asset-library/generated-text`) cache path under `/mnt/gcs`
- `SAM3D_API_KEY` (or `TEXT_SAM3D_API_KEY`) required for SAM3D generation
- `TEXT_SAM3D_API_HOST` base URL for SAM3D provider API
- `TEXT_SAM3D_TEXT_ENDPOINTS` (default: `/openapi/v1/text-to-3d,/v1/text-to-3d`) endpoint candidates
- `TEXT_SAM3D_TIMEOUT_SECONDS` (default: `1800`) provider task timeout
- `TEXT_SAM3D_POLL_SECONDS` (default: `10`) provider polling interval
- `HUNYUAN_API_KEY` (or `TEXT_HUNYUAN_API_KEY`) optional fallback provider key
- `TEXT_HUNYUAN_API_HOST` base URL for Hunyuan provider API
- `TEXT_HUNYUAN_TEXT_ENDPOINTS` (default: `/openapi/v1/text-to-3d,/v1/text-to-3d`) endpoint candidates
- `TEXT_HUNYUAN_TIMEOUT_SECONDS` (default: `1800`) provider task timeout
- `TEXT_HUNYUAN_POLL_SECONDS` (default: `10`) provider polling interval
- `VECTOR_STORE_PROVIDER` (default: `vertex`) ANN vector backend
- `VECTOR_STORE_PROJECT_ID` GCP project for vector backend
- `VECTOR_STORE_LOCATION` vector backend location
- `VECTOR_STORE_NAMESPACE` (default: `assets-v1`) vector namespace
- `VECTOR_STORE_DIMENSION` (default: `1536`) embedding dimension
- `VERTEX_INDEX_ENDPOINT` Vertex Matching Engine endpoint resource
- `VERTEX_DEPLOYED_INDEX_ID` Vertex deployed index ID
