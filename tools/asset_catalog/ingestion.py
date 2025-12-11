"""Asset ingestion helpers for registering catalog entries and embeddings."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

try:  # pragma: no cover - optional dependency
    from google.cloud import firestore  # type: ignore
except Exception:  # pragma: no cover - firestore optional
    firestore = None


@dataclass
class StorageURIs:
    """Container for asset storage locations."""

    usd_uri: Optional[str] = None
    gcs_uri: Optional[str] = None
    thumbnail_uri: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class AssetIngestionService:
    """Registers assets into Firestore (when configured) or a local cache."""

    def __init__(self, collection: str = "asset_metadata", project_id: Optional[str] = None, local_cache: str = "asset_ingestion_cache.json"):
        self.collection = collection
        self.local_cache = Path(local_cache)
        self._client = None
        if firestore is not None:
            try:  # pragma: no cover - environment dependent
                self._client = firestore.Client(project=project_id)
            except Exception:
                self._client = None

    def upsert_asset(
        self,
        asset_id: str,
        payload: Dict[str, Any],
        storage: Optional[StorageURIs] = None,
        embedding: Optional[np.ndarray] = None,
        embedding_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Persist an asset record.

        Args:
            asset_id: Unique ID to store the asset under.
            payload: Base metadata for the asset.
            storage: Optional storage references (USD path, thumbnail, etc.).
            embedding: Optional embedding to attach to the record.
            embedding_model: Model name used to generate the embedding.
        """

        record: Dict[str, Any] = {"asset_id": asset_id, **payload}
        if storage:
            record.update({
                "usd_path": storage.usd_uri,
                "gcs_uri": storage.gcs_uri,
                "thumbnail_uri": storage.thumbnail_uri,
                **(storage.extra or {}),
            })

        if embedding is not None:
            record["embedding"] = {
                "model": embedding_model or "unknown",
                "vector": embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
            }

        if self._client:
            # Firestore writes
            self._client.collection(self.collection).document(asset_id).set(record)
        else:
            # Persist to a local cache for offline development
            existing: Dict[str, Any] = {}
            if self.local_cache.exists():
                try:
                    existing = json.loads(self.local_cache.read_text())
                except json.JSONDecodeError:
                    existing = {}
            existing[asset_id] = record
            self.local_cache.write_text(json.dumps(existing, indent=2))

        return record

