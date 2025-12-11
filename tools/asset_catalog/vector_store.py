"""Lightweight vector store client used by the asset catalog utilities.

The implementation is intentionally simple and dependency-free so it can run
in both local notebooks and Cloud Run jobs without additional services. If a
real vector database is desired, you can swap the backend by subclassing
:class:`VectorStoreClient`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


@dataclass
class VectorStoreConfig:
    """Configuration for initializing a vector store client."""

    backend: str = "in-memory"
    namespace: str = "default"


@dataclass
class VectorRecord:
    """Single record stored in the vector store."""

    id: str
    embedding: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: Optional[float] = None


class VectorStoreClient:
    """Tiny, dependency-light vector store for similarity search.

    The client keeps embeddings in-memory grouped by namespace. It supports a
    subset of the operations used by the asset catalog utilities: upsert,
    query (cosine similarity), fetch by ID, and list by metadata filter.
    """

    def __init__(self, config: Optional[VectorStoreConfig] = None):
        self.config = config or VectorStoreConfig()
        self._store: dict[str, dict[str, VectorRecord]] = {}

    # ------------------------ CRUD ------------------------
    def upsert(self, records: Iterable[VectorRecord], namespace: Optional[str] = None) -> None:
        ns = namespace or self.config.namespace
        bucket = self._store.setdefault(ns, {})
        for rec in records:
            bucket[rec.id] = rec

    def fetch(self, ids: Iterable[str], namespace: Optional[str] = None) -> List[VectorRecord]:
        ns = namespace or self.config.namespace
        bucket = self._store.get(ns, {})
        return [bucket[i] for i in ids if i in bucket]

    def list(self, namespace: Optional[str] = None, filter_metadata: Optional[Dict[str, Any]] = None) -> List[VectorRecord]:
        ns = namespace or self.config.namespace
        bucket = list(self._store.get(ns, {}).values())
        if not filter_metadata:
            return bucket

        filtered: list[VectorRecord] = []
        for rec in bucket:
            if all(rec.metadata.get(k) == v for k, v in filter_metadata.items()):
                filtered.append(rec)
        return filtered

    # ------------------------ Query ------------------------
    def query(
        self,
        embedding: np.ndarray,
        top_k: int = 10,
        namespace: Optional[str] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[VectorRecord]:
        ns = namespace or self.config.namespace
        bucket = self.list(namespace=ns, filter_metadata=filter_metadata)
        if not bucket:
            return []

        query_norm = embedding / (np.linalg.norm(embedding) + 1e-8)

        scored: list[VectorRecord] = []
        for rec in bucket:
            emb = rec.embedding
            denom = (np.linalg.norm(emb) * np.linalg.norm(query_norm)) + 1e-8
            score = float(np.dot(emb, query_norm) / denom) if denom != 0 else 0.0
            scored.append(VectorRecord(id=rec.id, embedding=rec.embedding, metadata=rec.metadata, score=score))

        scored.sort(key=lambda r: r.score or 0.0, reverse=True)
        return scored[:top_k]

    # ------------------------ Convenience ------------------------
    def clear(self, namespace: Optional[str] = None) -> None:
        ns = namespace or self.config.namespace
        self._store.pop(ns, None)

