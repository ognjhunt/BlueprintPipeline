"""Vector store helpers for asset embeddings.

This module centralizes how BlueprintPipeline writes and queries embeddings
from vector databases such as Vertex AI Vector Search or pgvector. The default
implementation ships with an in-memory store to keep local execution simple
while providing the same API shape that cloud backends use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


@dataclass
class VectorStoreConfig:
    """Configuration for connecting to a vector database."""

    provider: str = "in-memory"
    collection: str = "asset-embeddings"
    project_id: Optional[str] = None
    location: Optional[str] = None
    connection_uri: Optional[str] = None
    namespace: Optional[str] = None
    dimension: Optional[int] = None


@dataclass
class VectorRecord:
    """Single embedding record stored in a vector DB."""

    id: str
    embedding: np.ndarray
    metadata: Dict[str, Any]
    score: Optional[float] = None


class BaseVectorStore:
    """Abstract interface for all vector store providers."""

    def upsert(self, records: Iterable[VectorRecord], namespace: Optional[str] = None) -> None:
        raise NotImplementedError

    def query(
        self,
        embedding: np.ndarray,
        top_k: int = 10,
        namespace: Optional[str] = None,
        filter_metadata: Optional[dict[str, Any]] = None,
    ) -> List[VectorRecord]:
        raise NotImplementedError

    def fetch(self, ids: Iterable[str], namespace: Optional[str] = None) -> List[VectorRecord]:
        raise NotImplementedError

    def list(
        self, namespace: Optional[str] = None, filter_metadata: Optional[dict[str, Any]] = None
    ) -> List[VectorRecord]:
        raise NotImplementedError


class InMemoryVectorStore(BaseVectorStore):
    """Lightweight in-memory vector DB used for local execution and tests."""

    def __init__(self) -> None:
        self._storage: dict[str, list[VectorRecord]] = {}

    def _namespace_bucket(self, namespace: Optional[str]) -> list[VectorRecord]:
        bucket_key = namespace or "default"
        if bucket_key not in self._storage:
            self._storage[bucket_key] = []
        return self._storage[bucket_key]

    def upsert(self, records: Iterable[VectorRecord], namespace: Optional[str] = None) -> None:
        bucket = self._namespace_bucket(namespace)
        bucket_index = {rec.id: rec for rec in bucket}
        for record in records:
            bucket_index[record.id] = record
        self._storage[namespace or "default"] = list(bucket_index.values())

    def query(
        self,
        embedding: np.ndarray,
        top_k: int = 10,
        namespace: Optional[str] = None,
        filter_metadata: Optional[dict[str, Any]] = None,
    ) -> List[VectorRecord]:
        bucket = self._namespace_bucket(namespace)
        if not bucket:
            return []

        def _matches_filter(record: VectorRecord) -> bool:
            if not filter_metadata:
                return True
            for key, value in filter_metadata.items():
                if record.metadata.get(key) != value:
                    return False
            return True

        filtered = [rec for rec in bucket if _matches_filter(rec)]
        if not filtered:
            return []

        emb_norm = np.linalg.norm(embedding) + 1e-8
        scored: list[VectorRecord] = []
        for rec in filtered:
            denom = (np.linalg.norm(rec.embedding) * emb_norm) + 1e-8
            score = float(np.dot(rec.embedding, embedding) / denom)
            scored.append(
                VectorRecord(id=rec.id, embedding=rec.embedding, metadata=rec.metadata, score=score)
            )

        scored.sort(key=lambda r: r.score or 0.0, reverse=True)
        return scored[:top_k]

    def fetch(self, ids: Iterable[str], namespace: Optional[str] = None) -> List[VectorRecord]:
        bucket = self._namespace_bucket(namespace)
        id_set = set(ids)
        return [rec for rec in bucket if rec.id in id_set]

    def list(
        self, namespace: Optional[str] = None, filter_metadata: Optional[dict[str, Any]] = None
    ) -> List[VectorRecord]:
        bucket = self._namespace_bucket(namespace)
        if not filter_metadata:
            return list(bucket)
        return [rec for rec in bucket if all(rec.metadata.get(k) == v for k, v in filter_metadata.items())]


class VectorStoreClient:
    """Convenience wrapper that hides provider-specific details."""

    def __init__(self, config: VectorStoreConfig, store: Optional[BaseVectorStore] = None):
        self.config = config
        self.store = store or self._create_store(config)

    def _create_store(self, config: VectorStoreConfig) -> BaseVectorStore:
        provider = (config.provider or "in-memory").lower()
        if provider == "in-memory":
            return InMemoryVectorStore()

        if provider == "pgvector":
            raise NotImplementedError(
                "pgvector provider requires a connection URI and integration with a Postgres driver"
            )

        if provider in {"vertex", "vertex-ai", "vertexai"}:
            raise NotImplementedError(
                "Vertex AI Vector Search requires the google-cloud-aiplatform dependency and project configuration"
            )

        raise ValueError(f"Unsupported vector store provider: {config.provider}")

    def upsert(self, records: Iterable[VectorRecord], namespace: Optional[str] = None) -> None:
        self.store.upsert(records, namespace=namespace or self.config.collection)

    def query(
        self,
        embedding: np.ndarray,
        top_k: int = 10,
        namespace: Optional[str] = None,
        filter_metadata: Optional[dict[str, Any]] = None,
    ) -> List[VectorRecord]:
        return self.store.query(
            embedding=embedding,
            top_k=top_k,
            namespace=namespace or self.config.collection,
            filter_metadata=filter_metadata,
        )

    def fetch(self, ids: Iterable[str], namespace: Optional[str] = None) -> List[VectorRecord]:
        return self.store.fetch(ids, namespace=namespace or self.config.collection)

    def list(
        self, namespace: Optional[str] = None, filter_metadata: Optional[dict[str, Any]] = None
    ) -> List[VectorRecord]:
        return self.store.list(namespace=namespace or self.config.collection, filter_metadata=filter_metadata)
