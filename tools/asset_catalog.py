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

import datetime
import importlib
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional

ASSET_CATALOG_SCHEMA_VERSION = 1

firestore_spec = importlib.util.find_spec("google.cloud.firestore")
firestore = importlib.import_module("google.cloud.firestore") if firestore_spec else None

service_account_spec = importlib.util.find_spec("google.oauth2.service_account")
service_account = (
    importlib.import_module("google.oauth2.service_account") if service_account_spec else None
)


@dataclass
class EmbeddingInfo:
    """Metadata describing a stored embedding vector."""

    model: str
    text: str
    vector: List[float]
    updated_at: str = field(
        default_factory=lambda: datetime.datetime.utcnow().isoformat() + "Z"
    )

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "EmbeddingInfo":
        return cls(
            model=payload.get("model", ""),
            text=payload.get("text", ""),
            vector=list(payload.get("vector", [])),
            updated_at=payload.get("updated_at", payload.get("timestamp", "")),
        )

    def to_payload(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AssetDocument:
    """Canonical document stored in the ``assets`` collection."""

    asset_id: str
    schema_version: int = ASSET_CATALOG_SCHEMA_VERSION
    logical_id: Optional[str] = None
    source: str = "unknown"  # "nvidia_pack", "regen3d", "generated", ...
    usd_path: Optional[str] = None
    gcs_uri: Optional[str] = None
    thumbnail_uri: Optional[str] = None
    sim_roles: List[str] = field(default_factory=list)
    class_name: Optional[str] = None
    description: Optional[str] = None
    physics_profile: Optional[Dict[str, Any]] = None
    articulation_profile: Optional[Dict[str, Any]] = None
    embedding: Optional[EmbeddingInfo] = None
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_firestore(cls, payload: Dict[str, Any]) -> "AssetDocument":
        embedding = None
        if payload.get("embedding"):
            embedding = EmbeddingInfo.from_payload(payload["embedding"])

        return cls(
            asset_id=str(payload.get("asset_id") or payload.get("id")),
            schema_version=int(payload.get("schema_version", ASSET_CATALOG_SCHEMA_VERSION)),
            logical_id=payload.get("logical_id"),
            source=payload.get("source", "unknown"),
            usd_path=payload.get("usd_path") or payload.get("asset_path"),
            gcs_uri=payload.get("gcs_uri"),
            thumbnail_uri=payload.get("thumbnail_uri"),
            sim_roles=list(payload.get("sim_roles", []) or []),
            class_name=payload.get("class_name"),
            description=payload.get("description") or payload.get("summary"),
            physics_profile=payload.get("physics") or payload.get("physics_profile"),
            articulation_profile=payload.get("articulation")
            or payload.get("articulation_profile"),
            embedding=embedding,
            extra_metadata={k: v for k, v in payload.items() if k not in {
                "asset_id",
                "id",
                "schema_version",
                "logical_id",
                "source",
                "usd_path",
                "asset_path",
                "gcs_uri",
                "thumbnail_uri",
                "sim_roles",
                "class_name",
                "description",
                "physics",
                "physics_profile",
                "articulation",
                "articulation_profile",
                "embedding",
            }},
        )

    def to_firestore(self) -> Dict[str, Any]:
        payload = {
            "asset_id": self.asset_id,
            "schema_version": self.schema_version,
            "logical_id": self.logical_id,
            "source": self.source,
            "usd_path": self.usd_path,
            "gcs_uri": self.gcs_uri,
            "thumbnail_uri": self.thumbnail_uri,
            "sim_roles": self.sim_roles,
            "class_name": self.class_name,
            "description": self.description,
            "physics_profile": self.physics_profile,
            "articulation_profile": self.articulation_profile,
        }

        if self.embedding:
            payload["embedding"] = self.embedding.to_payload()

        payload.update(self.extra_metadata)
        return {k: v for k, v in payload.items() if v is not None}

    def to_legacy_metadata(self) -> Dict[str, Any]:
        """Return a metadata dict compatible with existing jobs."""

        legacy = dict(self.extra_metadata)
        if self.physics_profile:
            legacy.setdefault("physics", self.physics_profile)
        if self.articulation_profile:
            legacy.setdefault("articulation", self.articulation_profile)
        if self.usd_path:
            legacy.setdefault("asset_path", self.usd_path)
        if self.class_name:
            legacy.setdefault("class_name", self.class_name)
        if self.embedding:
            legacy.setdefault("embedding", self.embedding.to_payload())
        return legacy


@dataclass
class SceneDocument:
    """Scene-level metadata stored in the ``scenes`` collection."""

    scene_id: str
    schema_version: int = ASSET_CATALOG_SCHEMA_VERSION
    assets_prefix: Optional[str] = None
    usd_path: Optional[str] = None
    thumbnail_uri: Optional[str] = None
    asset_ids: List[str] = field(default_factory=list)
    description: Optional[str] = None
    embedding: Optional[EmbeddingInfo] = None
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_firestore(cls, payload: Dict[str, Any]) -> "SceneDocument":
        embedding = None
        if payload.get("embedding"):
            embedding = EmbeddingInfo.from_payload(payload["embedding"])

        return cls(
            scene_id=str(payload.get("scene_id") or payload.get("id")),
            schema_version=int(payload.get("schema_version", ASSET_CATALOG_SCHEMA_VERSION)),
            assets_prefix=payload.get("assets_prefix"),
            usd_path=payload.get("usd_path"),
            thumbnail_uri=payload.get("thumbnail_uri"),
            asset_ids=list(payload.get("asset_ids", []) or []),
            description=payload.get("description"),
            embedding=embedding,
            extra_metadata={k: v for k, v in payload.items() if k not in {
                "scene_id",
                "id",
                "schema_version",
                "assets_prefix",
                "usd_path",
                "thumbnail_uri",
                "asset_ids",
                "description",
                "embedding",
            }},
        )

    def to_firestore(self) -> Dict[str, Any]:
        payload = {
            "scene_id": self.scene_id,
            "schema_version": self.schema_version,
            "assets_prefix": self.assets_prefix,
            "usd_path": self.usd_path,
            "thumbnail_uri": self.thumbnail_uri,
            "asset_ids": self.asset_ids,
            "description": self.description,
        }
        if self.embedding:
            payload["embedding"] = self.embedding.to_payload()
        payload.update(self.extra_metadata)
        return {k: v for k, v in payload.items() if v is not None}


@dataclass
class AssetCatalogConfig:
    """Lightweight config holder for the catalog client."""

    project_id: Optional[str] = None
    assets_collection: str = "assets"
    scenes_collection: str = "scenes"
    legacy_collection: str = "asset_metadata"
    credentials_path: Optional[str] = None
    emulator_host: Optional[str] = None

    @classmethod
    def from_env(cls) -> "AssetCatalogConfig":
        return cls(
            project_id=os.getenv("ASSET_CATALOG_PROJECT"),
            assets_collection=os.getenv("ASSET_CATALOG_ASSETS", "assets"),
            scenes_collection=os.getenv("ASSET_CATALOG_SCENES", "scenes"),
            legacy_collection=os.getenv("ASSET_CATALOG_COLLECTION", "asset_metadata"),
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
                creds_cls = getattr(service_account, "Credentials", None)
                if creds_cls:
                    kwargs["credentials"] = creds_cls.from_service_account_file(
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

    def _collection(self, name: str):
        client = self._ensure_client()
        if client is None:
            return None
        return client.collection(name)

    def lookup_metadata(
        self,
        asset_id: Optional[Any] = None,
        asset_path: Optional[str] = None,
        logical_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Best-effort lookup by asset_id, logical_id or asset_path.

        Legacy lookups hit the ``legacy_collection`` while newer lookups
        prefer the ``assets`` collection and normalize results to the
        metadata structure expected by downstream jobs.
        """

        doc = self.fetch_asset_document(asset_id=asset_id, asset_path=asset_path, logical_id=logical_id)
        if doc:
            return doc.to_legacy_metadata()

        client = self._ensure_client()
        if client is None:
            return None

        try:
            coll = client.collection(self.config.legacy_collection)

            if asset_id is not None:
                legacy_doc = coll.document(str(asset_id)).get()
                if legacy_doc.exists:
                    return legacy_doc.to_dict()

            if asset_path:
                query = coll.where("asset_path", "==", asset_path).limit(1)
                for legacy_doc in query.stream():
                    return legacy_doc.to_dict()

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
        """Publish merged metadata for the asset (legacy compatibility)."""

        client = self._ensure_client()
        if client is None:
            return

        try:
            coll = client.collection(self.config.legacy_collection)
            doc_ref = coll.document(str(asset_id))

            payload: Dict[str, Any] = dict(metadata)
            payload["asset_id"] = asset_id
            payload.setdefault("schema_version", ASSET_CATALOG_SCHEMA_VERSION)
            if asset_path:
                payload.setdefault("asset_path", asset_path)

            doc_ref.set(payload, merge=True)
        except Exception as exc:  # pragma: no cover - network/firestore errors
            print(f"[CATALOG] WARNING: failed to publish metadata: {exc}", file=sys.stderr)

    # ------------------------------------------------------------------
    # New collection helpers
    # ------------------------------------------------------------------

    def fetch_asset_document(
        self,
        *,
        asset_id: Optional[Any] = None,
        logical_id: Optional[str] = None,
        asset_path: Optional[str] = None,
    ) -> Optional[AssetDocument]:
        coll = self._collection(self.config.assets_collection)
        if coll is None:
            return None

        try:
            if asset_id is not None:
                doc = coll.document(str(asset_id)).get()
                if doc.exists:
                    return AssetDocument.from_firestore(doc.to_dict())

            if logical_id:
                for doc in coll.where("logical_id", "==", logical_id).limit(1).stream():
                    return AssetDocument.from_firestore(doc.to_dict())

            if asset_path:
                for doc in coll.where("usd_path", "==", asset_path).limit(1).stream():
                    return AssetDocument.from_firestore(doc.to_dict())
        except Exception as exc:  # pragma: no cover - network/firestore errors
            print(f"[CATALOG] WARNING: asset fetch failed: {exc}", file=sys.stderr)

        return None

    def upsert_asset_document(self, asset: AssetDocument) -> None:
        coll = self._collection(self.config.assets_collection)
        if coll is None:
            return

        try:
            coll.document(asset.asset_id).set(asset.to_firestore(), merge=True)
        except Exception as exc:  # pragma: no cover - network/firestore errors
            print(f"[CATALOG] WARNING: failed to upsert asset {asset.asset_id}: {exc}", file=sys.stderr)

    def fetch_scene_document(self, scene_id: str) -> Optional[SceneDocument]:
        coll = self._collection(self.config.scenes_collection)
        if coll is None:
            return None

        try:
            doc = coll.document(str(scene_id)).get()
            if doc.exists:
                return SceneDocument.from_firestore(doc.to_dict())
        except Exception as exc:  # pragma: no cover - network/firestore errors
            print(f"[CATALOG] WARNING: scene fetch failed: {exc}", file=sys.stderr)

        return None

    def upsert_scene_document(self, scene: SceneDocument) -> None:
        coll = self._collection(self.config.scenes_collection)
        if coll is None:
            return

        try:
            coll.document(scene.scene_id).set(scene.to_firestore(), merge=True)
        except Exception as exc:  # pragma: no cover - network/firestore errors
            print(f"[CATALOG] WARNING: failed to upsert scene {scene.scene_id}: {exc}", file=sys.stderr)

    def query_assets(
        self,
        *,
        sim_roles: Optional[Iterable[str]] = None,
        source: Optional[Iterable[str]] = None,
        class_name: Optional[str] = None,
        limit: int = 25,
    ) -> List[AssetDocument]:
        coll = self._collection(self.config.assets_collection)
        if coll is None:
            return []

        query = coll.limit(limit)

        if sim_roles:
            query = query.where("sim_roles", "array_contains_any", list(sim_roles))
        if source:
            query = query.where("source", "in", list(source))
        if class_name:
            query = query.where("class_name", "==", class_name)

        try:
            return [AssetDocument.from_firestore(doc.to_dict()) for doc in query.stream()]
        except Exception as exc:  # pragma: no cover - network/firestore errors
            print(f"[CATALOG] WARNING: asset query failed: {exc}", file=sys.stderr)
            return []
