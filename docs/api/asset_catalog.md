# `tools/asset_catalog` API

## Purpose

The asset catalog subsystem indexes NVIDIA/Stage 1 text generation asset packs, builds searchable metadata catalogs, generates embeddings for semantic search, and ingests structured asset metadata into Firestore/vector stores. It powers asset lookup and semantic matching in asset-heavy pipeline steps.【F:tools/asset_catalog/__init__.py†L1-L23】【F:tools/asset_catalog/catalog_builder.py†L1-L39】

## Public entrypoints

Exported via `tools.asset_catalog`:

- Catalog indexing:
  - `AssetCatalogBuilder`, `AssetCatalog`, `AssetEntry`
- Matching utilities:
  - `AssetMatcher`, `AssetMatch`, `MatchResult`
- Embeddings:
  - `AssetEmbeddings`, `EmbeddingConfig`
  - `caption_thumbnail` (image caption helper)
- Ingestion:
  - `AssetIngestionService`, `StorageURIs`
- Vector store:
  - `VectorStoreClient`, `VectorStoreConfig`, `VectorRecord`【F:tools/asset_catalog/__init__.py†L1-L27】

## Configuration / environment variables

### Embedding configuration

`EmbeddingConfig` controls embedding backends and keys:

- `backend`: `sentence-transformers`, `vertex-ai`/`gemini`, or `openai`
- `model_name`, `dimension`
- `api_key`, `project_id`
- Image embedding overrides: `image_model_name`, `image_backend`, `image_dimension`【F:tools/asset_catalog/embeddings.py†L22-L43】

### Vector store configuration

`VectorStoreConfig.from_env()` reads the following environment variables:

- `VECTOR_STORE_PROVIDER`
- `VECTOR_STORE_COLLECTION`
- `VECTOR_STORE_PROJECT_ID`
- `VECTOR_STORE_LOCATION`
- `VECTOR_STORE_CONNECTION_URI`
- `VECTOR_STORE_NAMESPACE`
- `VECTOR_STORE_DIMENSION`【F:tools/asset_catalog/vector_store.py†L20-L55】

When using the Vertex AI provider, `VectorStoreClient` also expects:

- `VERTEX_INDEX_ENDPOINT`
- `VERTEX_DEPLOYED_INDEX_ID`
- `VERTEX_INDEX_NAME` (optional)【F:tools/asset_catalog/vector_store.py†L753-L781】

## Request/response payloads & data models

### Catalog indexing

- **Request**: `AssetCatalogBuilder.build()` crawls a pack directory, optionally generating thumbnails, and builds `AssetCatalog` with a list of `AssetEntry` records (asset IDs, tags, USD metadata, and dimensions).【F:tools/asset_catalog/catalog_builder.py†L52-L149】
- **Response**: `AssetCatalog` includes pack metadata and the list of indexed assets, usable for downstream ingestion or search.【F:tools/asset_catalog/catalog_builder.py†L40-L51】

### Embedding & vector store

- **Request**: `AssetEmbeddings.embed_text` / `embed_image` produce vectors for text or image inputs. `VectorStoreClient.upsert` writes `VectorRecord` entries to the backing store for search.【F:tools/asset_catalog/embeddings.py†L86-L185】【F:tools/asset_catalog/vector_store.py†L57-L139】
- **Response**: `VectorRecord` entries include `id`, `embedding`, `metadata`, and optional similarity `score` for search results.【F:tools/asset_catalog/vector_store.py†L57-L69】

### Ingestion

- **Request**: `AssetIngestionService.ingest_pack_asset` accepts catalog items, `StorageURIs` pointing to GCS assets, and optional embeddings to create Firestore documents and vector store entries.【F:tools/asset_catalog/ingestion.py†L49-L190】
- **Response**: returns a dictionary including embedding references and any metadata that was persisted to storage.【F:tools/asset_catalog/ingestion.py†L120-L186】

## Example usage

```python
from tools.asset_catalog import (
    AssetCatalogBuilder,
    AssetEmbeddings,
    EmbeddingConfig,
    VectorStoreConfig,
    VectorStoreClient,
)

# Build catalog from an asset pack
builder = AssetCatalogBuilder(
    pack_path="/data/NVIDIA/ResidentialAssetsPack",
    pack_name="ResidentialAssetsPack",
)
catalog = builder.build()

# Create embeddings and search
vector_config = VectorStoreConfig.from_env(provider="in-memory")
vector_store = VectorStoreClient(vector_config)
embeddings = AssetEmbeddings(
    config=EmbeddingConfig(backend="sentence-transformers"),
    vector_store=vector_store,
)
embeddings.build_index(catalog)
results = embeddings.search("modern dining table", top_k=5)
for result in results:
    print(result.asset_id, result.score)
```
