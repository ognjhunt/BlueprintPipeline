"""Asset Catalog - Tools for indexing and searching NVIDIA/ZeroScene asset packs."""
from .catalog_builder import AssetCatalogBuilder
from .asset_matcher import AssetMatcher, AssetMatch, MatchResult
from .embeddings import AssetEmbeddings, EmbeddingConfig
from .image_captioning import caption_thumbnail
from .ingestion import AssetIngestionService, StorageURIs
from .vector_store import VectorStoreClient, VectorStoreConfig, VectorRecord

__all__ = [
    "AssetCatalogBuilder",
    "AssetMatcher",
    "AssetMatch",
    "MatchResult",
    "AssetEmbeddings",
    "EmbeddingConfig",
    "AssetIngestionService",
    "StorageURIs",
    "VectorStoreClient",
    "VectorStoreConfig",
    "VectorRecord",
    "caption_thumbnail",
]
