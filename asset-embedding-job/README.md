# asset-embedding-job

Processes `automation/asset_embedding/queue/*.json` and upserts semantic vectors
for Stage 1 asset retrieval.

## Required env vars

- `BUCKET`
- `VECTOR_STORE_PROVIDER` (default `vertex`)
- `VECTOR_STORE_PROJECT_ID`
- `VECTOR_STORE_LOCATION`
- `VERTEX_INDEX_ENDPOINT`
- `VERTEX_DEPLOYED_INDEX_ID`
- `OPENAI_API_KEY` (unless `TEXT_ASSET_EMBEDDING_BACKEND=deterministic`)

## Optional env vars

- `QUEUE_OBJECT`
- `TEXT_ASSET_EMBEDDING_QUEUE_PREFIX` (default `automation/asset_embedding/queue`)
- `TEXT_ASSET_EMBEDDING_PROCESSED_PREFIX` (default `automation/asset_embedding/processed`)
- `TEXT_ASSET_EMBEDDING_FAILED_PREFIX` (default `automation/asset_embedding/failed`)
- `TEXT_ASSET_EMBEDDING_MAX_ITEMS` (default `20`)
- `TEXT_ASSET_EMBEDDING_BATCH_SIZE` (default `32`)
- `TEXT_ASSET_EMBEDDING_MODEL` (default `text-embedding-3-small`)
- `TEXT_ASSET_EMBEDDING_BACKEND` (default `openai`)
- `TEXT_ASSET_EMBEDDING_DRY_RUN` (default `false`)
- `TEXT_ASSET_EMBEDDING_FAIL_ON_ERROR` (default `true`)
- `TEXT_ASSET_ANN_NAMESPACE` (default `assets-v1`)

