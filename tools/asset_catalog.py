"""
Helpers for reading/writing asset metadata from a central catalog.

The catalog is intended to be a canonical Firestore collection that stores
mesh bounds, physics hints, and material names keyed by asset ID. Each entry
can also carry an ``asset_path`` or GCS URI so downstream jobs can correlate
files written by different pipelines.

The helpers are best-effort and designed to fail open: if Firestore is not
available or configuration is missing, lookups return ``None`` and publishes
are skipped so offline workflows continue to function.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional

try:  # Optional dependency
    from google.cloud import firestore  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    firestore = None

try:  # Optional dependency
    from google.oauth2 import service_account  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    service_account = None


@dataclass
class AssetCatalogConfig:
    """Lightweight config holder for the catalog client."""

    project_id: Optional[str] = None
    collection: str = "asset_metadata"
    credentials_path: Optional[str] = None
    emulator_host: Optional[str] = None

    @classmethod
    def from_env(cls) -> "AssetCatalogConfig":
        return cls(
            project_id=os.getenv("ASSET_CATALOG_PROJECT"),
            collection=os.getenv("ASSET_CATALOG_COLLECTION", "asset_metadata"),
            credentials_path=os.getenv("ASSET_CATALOG_CREDENTIALS"),
            emulator_host=os.getenv("ASSET_CATALOG_EMULATOR_HOST"),
        )


class AssetCatalogClient:
    """Small helper for catalog lookups and publishes."""

    def __init__(self, config: Optional[AssetCatalogConfig] = None):
        self.config = config or AssetCatalogConfig.from_env()
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return self._client

        if firestore is None:
            return None

        kwargs = {}
        if self.config.emulator_host:
            os.environ.setdefault("FIRESTORE_EMULATOR_HOST", self.config.emulator_host)

        if self.config.credentials_path and service_account is not None:
            try:
                kwargs["credentials"] = service_account.Credentials.from_service_account_file(
                    self.config.credentials_path
                )
            except Exception as exc:  # pragma: no cover - credential parsing errors
                print(f"[CATALOG] WARNING: failed to load credentials: {exc}", file=sys.stderr)

        try:
            self._client = firestore.Client(project=self.config.project_id, **kwargs)
        except Exception as exc:  # pragma: no cover - client creation errors
            print(f"[CATALOG] WARNING: failed to initialize Firestore client: {exc}", file=sys.stderr)
            self._client = None

        return self._client

    def lookup_metadata(self, asset_id: Optional[Any] = None, asset_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Best-effort lookup by asset_id or asset_path."""

        client = self._ensure_client()
        if client is None:
            return None

        try:
            coll = client.collection(self.config.collection)

            if asset_id is not None:
                doc = coll.document(str(asset_id)).get()
                if doc.exists:
                    return doc.to_dict()

            if asset_path:
                query = coll.where("asset_path", "==", asset_path).limit(1)
                for doc in query.stream():
                    return doc.to_dict()

        except Exception as exc:  # pragma: no cover - network/firestore errors
            print(f"[CATALOG] WARNING: catalog lookup failed: {exc}", file=sys.stderr)

        return None

    def publish_metadata(
        self,
        asset_id: Any,
        metadata: Dict[str, Any],
        *,
        asset_path: Optional[str] = None,
    ) -> None:
        """
        Publish merged metadata for the asset.

        The document is keyed by ``asset_id`` and can include ``asset_path`` to make
        lookups by relative path possible.
        """

        client = self._ensure_client()
        if client is None:
            return

        try:
            coll = client.collection(self.config.collection)
            doc_ref = coll.document(str(asset_id))

            payload: Dict[str, Any] = dict(metadata)
            payload["asset_id"] = asset_id
            if asset_path:
                payload.setdefault("asset_path", asset_path)

            doc_ref.set(payload, merge=True)
        except Exception as exc:  # pragma: no cover - network/firestore errors
            print(f"[CATALOG] WARNING: failed to publish metadata: {exc}", file=sys.stderr)
