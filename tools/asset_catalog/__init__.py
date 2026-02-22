"""Asset Catalog - Tools for indexing and searching NVIDIA/Stage 1 text generation asset packs."""
import importlib.util
import sys
from pathlib import Path

from .catalog_builder import AssetCatalogBuilder
from .asset_matcher import AssetMatcher, AssetMatch, MatchResult
from .embeddings import AssetEmbeddings, EmbeddingConfig
from .image_captioning import caption_thumbnail
from .ingestion import AssetIngestionService, StorageURIs
from .vector_store import VectorStoreClient, VectorStoreConfig, VectorRecord

# Re-export AssetCatalogClient from the legacy tools/asset_catalog.py module
# which is shadowed by this package directory.
_legacy_module_path = Path(__file__).resolve().parent.parent / "asset_catalog.py"
_spec = importlib.util.spec_from_file_location("tools._asset_catalog_legacy", _legacy_module_path)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)
AssetCatalogClient = _mod.AssetCatalogClient
AssetCatalogConfig = _mod.AssetCatalogConfig

__all__ = [
    "AssetCatalogBuilder",
    "AssetCatalogClient",
    "AssetCatalogConfig",
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
