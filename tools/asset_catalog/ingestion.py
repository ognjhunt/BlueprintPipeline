"""Asset ingestion helpers for Firestore and vector stores.

The ingestion service writes asset metadata to Firestore using the
schemas/firestore_asset_schema.json contract and pushes embeddings to a
configured vector database (Vertex AI Vector Search, pgvector, or the
in-memory helper used for local testing).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np

try:  # Firestore is optional in some environments
    from google.cloud import firestore
except Exception:  # pragma: no cover - dependency guard
    firestore = None

from .vector_store import VectorRecord, VectorStoreClient, VectorStoreConfig


@dataclass
class StorageURIs:
    """Pointers to authored asset artifacts in GCS."""

    usd_uri: str
    thumbnail_uri: Optional[str] = None
    gcs_bucket: Optional[str] = None


class AssetIngestionService:
    """Coordinates metadata and embedding ingestion for assets."""

    def __init__(
        self,
        firestore_client: Optional[Any] = None,
        vector_store: Optional[VectorStoreClient] = None,
        vector_store_config: Optional[VectorStoreConfig] = None,
        collection_path: str = "assets",
    ) -> None:
        self.firestore = firestore_client or (firestore.Client() if firestore else None)
        self.vector_store = vector_store or (
            VectorStoreClient(vector_store_config) if vector_store_config else None
        )
        self.collection_path = collection_path

    def _normalize_dimensions(self, dims: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
        if not dims:
            return None
        rename = {"width": "width_m", "height": "height_m", "depth": "depth_m"}
        return {rename.get(k, k): v for k, v in dims.items() if v is not None}

    def _upsert_embeddings(
        self,
        asset_id: str,
        text_embedding: Optional[np.ndarray],
        thumbnail_embedding: Optional[np.ndarray],
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, dict[str, Any]]:
        if not self.vector_store:
            return {}

        metadata = metadata or {}
        records: list[VectorRecord] = []
        vector_ids: dict[str, dict[str, Any]] = {}
        provider = self.vector_store.config.provider if self.vector_store else None

        if text_embedding is not None:
            text_id = f"{asset_id}:text"
            records.append(
                VectorRecord(id=text_id, embedding=text_embedding, metadata={"kind": "text", **metadata})
            )
            vector_ids["text"] = {
                "vector_id": text_id,
                "provider": provider,
                "dimension": int(text_embedding.shape[-1]),
            }

        if thumbnail_embedding is not None:
            thumb_id = f"{asset_id}:thumbnail"
            records.append(
                VectorRecord(
                    id=thumb_id, embedding=thumbnail_embedding, metadata={"kind": "thumbnail", **metadata}
                )
            )
            vector_ids["thumbnail"] = {
                "vector_id": thumb_id,
                "provider": provider,
                "dimension": int(thumbnail_embedding.shape[-1]),
            }

        if records:
            self.vector_store.upsert(records, namespace=self.vector_store.config.collection)

        return vector_ids

    def _build_document(
        self,
        asset_id: str,
        pack_name: str,
        display_name: Optional[str],
        version_hash: Optional[str],
        category: str,
        tags: list[str],
        storage: StorageURIs,
        source: str,
        description: Optional[str] = None,
        subcategory: Optional[str] = None,
        dimensions: Optional[dict[str, float]] = None,
        simready: Optional[dict[str, Any]] = None,
        licensing: Optional[dict[str, Any]] = None,
        embedding_refs: Optional[dict[str, dict[str, Any]]] = None,
        attributes: Optional[dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        now = datetime.utcnow().isoformat() + "Z"
        embedding_payload = None
        if embedding_refs:
            embedding_payload = {}
            for kind, ref in embedding_refs.items():
                if not ref:
                    continue
                embedding_payload[kind] = {
                    "vector_id": ref.get("vector_id"),
                    "provider": ref.get("provider") or None,
                    "dimension": ref.get("dimension"),
                }

        document = {
            "asset_id": asset_id,
            "source": source,
            "pack": {
                "name": pack_name,
                "display_name": display_name or pack_name,
                "version_hash": version_hash,
            },
            "category": category,
            "subcategory": subcategory,
            "display_name": display_name or asset_id,
            "description": description,
            "tags": tags or [],
            "dimensions": self._normalize_dimensions(dimensions),
            "simready": simready or {},
            "licensing": licensing or {},
            "media": {
                "usd_uri": storage.usd_uri,
                "thumbnail_uri": storage.thumbnail_uri,
                "gcs_bucket": storage.gcs_bucket,
            },
            "embeddings": embedding_payload,
            "attributes": attributes or {},
            "created_at": now,
            "updated_at": now,
        }
        return document

    def ingest_pack_asset(
        self,
        asset: Any,
        catalog: Any,
        storage: StorageURIs,
        text_embedding: Optional[np.ndarray] = None,
        thumbnail_embedding: Optional[np.ndarray] = None,
        licensing: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        metadata = {
            "asset_id": asset.asset_id,
            "pack": catalog.pack_name,
            "source": "nvidia-pack",
        }
        embedding_refs = self._upsert_embeddings(
            asset.asset_id, text_embedding, thumbnail_embedding, metadata=metadata
        )

        document = self._build_document(
            asset_id=asset.asset_id,
            pack_name=catalog.pack_name,
            display_name=catalog.display_name,
            version_hash=getattr(catalog, "version_hash", None),
            category=asset.category,
            tags=asset.tags,
            storage=storage,
            source="nvidia-pack",
            description=getattr(asset, "description", None),
            subcategory=asset.subcategory,
            dimensions=getattr(asset, "dimensions", None),
            simready=getattr(asset, "simready_metadata", None),
            licensing=licensing,
            embedding_refs=embedding_refs,
            attributes={
                "file_type": asset.file_type,
                "default_prim": getattr(asset, "default_prim", None),
            },
        )

        if self.firestore:
            try:
                self.firestore.collection(self.collection_path).document(asset.asset_id).set(
                    document, merge=True
                )
            except Exception as exc:  # pragma: no cover - best effort ingest
                logging.error("Failed to upsert asset %s to Firestore: %s", asset.asset_id, exc)

        return document

    def ingest_zeroscene_asset(
        self,
        payload: dict[str, Any],
        storage: StorageURIs,
        text_embedding: Optional[np.ndarray] = None,
        thumbnail_embedding: Optional[np.ndarray] = None,
    ) -> dict[str, Any]:
        asset_id = payload.get("asset_id") or payload.get("id")
        if not asset_id:
            raise ValueError("ZeroScene payload must include 'asset_id' or 'id'")

        metadata = {"asset_id": asset_id, "source": "zeroscene"}
        embedding_refs = self._upsert_embeddings(
            asset_id, text_embedding, thumbnail_embedding, metadata=metadata
        )

        document = self._build_document(
            asset_id=asset_id,
            pack_name=payload.get("pack_name", "ZeroScene"),
            display_name=payload.get("display_name") or payload.get("title") or asset_id,
            version_hash=payload.get("version_hash"),
            category=payload.get("category", "uncategorized"),
            tags=payload.get("tags", []),
            storage=storage,
            source="zeroscene",
            description=payload.get("description"),
            subcategory=payload.get("subcategory"),
            dimensions=payload.get("dimensions"),
            simready=payload.get("simready"),
            licensing=payload.get("licensing"),
            embedding_refs=embedding_refs,
            attributes=payload.get("attributes"),
        )

        if self.firestore:
            try:
                self.firestore.collection(self.collection_path).document(asset_id).set(
                    document, merge=True
                )
            except Exception as exc:  # pragma: no cover - best effort ingest
                logging.error("Failed to upsert ZeroScene asset %s to Firestore: %s", asset_id, exc)

        return document
