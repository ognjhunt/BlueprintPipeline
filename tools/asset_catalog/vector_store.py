"""Vector store helpers for asset embeddings.

This module centralizes how BlueprintPipeline writes and queries embeddings
from vector databases such as Vertex AI Vector Search or pgvector. The default
implementation ships with an in-memory store to keep local execution simple
while providing the same API shape that cloud backends use.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
import os
import sqlite3
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

    @classmethod
    def from_env(cls, **overrides: Any) -> "VectorStoreConfig":
        """Create a config using VECTOR_STORE_* environment variables."""
        env_config = cls(
            provider=os.getenv("VECTOR_STORE_PROVIDER", cls.provider),
            collection=os.getenv("VECTOR_STORE_COLLECTION", cls.collection),
            project_id=os.getenv("VECTOR_STORE_PROJECT_ID"),
            location=os.getenv("VECTOR_STORE_LOCATION"),
            connection_uri=os.getenv("VECTOR_STORE_CONNECTION_URI"),
            namespace=os.getenv("VECTOR_STORE_NAMESPACE"),
            dimension=_optional_int(os.getenv("VECTOR_STORE_DIMENSION")),
        )
        for key, value in overrides.items():
            setattr(env_config, key, value)
        return env_config


@dataclass
class VectorRecord:
    """Single embedding record stored in a vector DB."""

    id: str
    embedding: np.ndarray
    metadata: Dict[str, Any]
    score: Optional[float] = None


class BaseVectorStore(ABC):
    """Abstract interface for all vector store providers."""

    @abstractmethod
    def upsert(self, records: Iterable[VectorRecord], namespace: Optional[str] = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def query(
        self,
        embedding: np.ndarray,
        top_k: int = 10,
        namespace: Optional[str] = None,
        filter_metadata: Optional[dict[str, Any]] = None,
    ) -> List[VectorRecord]:
        raise NotImplementedError

    @abstractmethod
    def fetch(self, ids: Iterable[str], namespace: Optional[str] = None) -> List[VectorRecord]:
        raise NotImplementedError

    @abstractmethod
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


class PgVectorStore(BaseVectorStore):
    """
    PostgreSQL + pgvector based vector store.

    Requires:
    - PostgreSQL with pgvector extension installed
    - psycopg2 or asyncpg driver
    """

    def __init__(self, connection_uri: str, collection: str = "embeddings", dimension: int = 1536):
        self.connection_uri = connection_uri
        self.collection = collection
        self.dimension = dimension
        self._conn = None

    def _get_connection(self):
        """Get or create database connection."""
        if self._conn is not None:
            return self._conn

        try:
            import psycopg2
            from psycopg2.extras import execute_values
        except ImportError:
            raise RuntimeError("psycopg2 is required for pgvector. Install with: pip install psycopg2-binary")

        self._conn = psycopg2.connect(self.connection_uri)

        # Ensure table exists
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.collection} (
                    id TEXT PRIMARY KEY,
                    embedding vector({self.dimension}),
                    metadata JSONB
                )
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.collection}_embedding_idx
                ON {self.collection} USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
            """)
            self._conn.commit()

        return self._conn

    def upsert(self, records: Iterable[VectorRecord], namespace: Optional[str] = None) -> None:
        """Upsert records into pgvector."""
        import json as json_module
        from psycopg2.extras import execute_values

        conn = self._get_connection()
        records_list = list(records)
        if not records_list:
            return

        with conn.cursor() as cur:
            rows = []
            for record in records_list:
                embedding_str = "[" + ",".join(str(x) for x in record.embedding.tolist()) + "]"
                metadata_json = json_module.dumps(record.metadata)
                rows.append((record.id, embedding_str, metadata_json))
            execute_values(
                cur,
                f"""
                INSERT INTO {self.collection} (id, embedding, metadata)
                VALUES %s
                ON CONFLICT (id) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata
                """,
                rows,
                template="(%s, %s::vector, %s::jsonb)",
            )
            conn.commit()

    def query(
        self,
        embedding: np.ndarray,
        top_k: int = 10,
        namespace: Optional[str] = None,
        filter_metadata: Optional[dict[str, Any]] = None,
    ) -> List[VectorRecord]:
        """Query similar vectors using cosine similarity."""
        import json as json_module

        conn = self._get_connection()
        embedding_str = "[" + ",".join(str(x) for x in embedding.tolist()) + "]"

        # Build filter clause
        filter_clause = ""
        filter_params: List[Any] = [embedding_str, top_k]
        if filter_metadata:
            conditions = []
            for key, value in filter_metadata.items():
                conditions.append(f"metadata->>%s = %s")
                filter_params.insert(-1, key)
                filter_params.insert(-1, str(value))
            if conditions:
                filter_clause = "WHERE " + " AND ".join(conditions)

        query = f"""
            SELECT id, embedding, metadata, 1 - (embedding <=> %s::vector) as score
            FROM {self.collection}
            {filter_clause}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """

        # Adjust params for the query
        final_params = [embedding_str]
        if filter_metadata:
            for key, value in filter_metadata.items():
                final_params.append(key)
                final_params.append(str(value))
        final_params.extend([embedding_str, top_k])

        with conn.cursor() as cur:
            cur.execute(query, final_params)
            rows = cur.fetchall()

        results = []
        for row in rows:
            record_id, emb_data, metadata, score = row
            # Parse embedding from pgvector format
            if isinstance(emb_data, str):
                emb_values = [float(x) for x in emb_data.strip("[]").split(",")]
            else:
                emb_values = list(emb_data)

            results.append(
                VectorRecord(
                    id=record_id,
                    embedding=np.array(emb_values),
                    metadata=metadata if isinstance(metadata, dict) else json_module.loads(metadata),
                    score=float(score),
                )
            )

        return results

    def fetch(self, ids: Iterable[str], namespace: Optional[str] = None) -> List[VectorRecord]:
        """Fetch records by ID."""
        import json as json_module

        conn = self._get_connection()
        ids_list = list(ids)
        if not ids_list:
            return []

        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(ids_list))
            cur.execute(
                f"SELECT id, embedding, metadata FROM {self.collection} WHERE id IN ({placeholders})",
                ids_list,
            )
            rows = cur.fetchall()

        results = []
        for row in rows:
            record_id, emb_data, metadata = row
            if isinstance(emb_data, str):
                emb_values = [float(x) for x in emb_data.strip("[]").split(",")]
            else:
                emb_values = list(emb_data)

            results.append(
                VectorRecord(
                    id=record_id,
                    embedding=np.array(emb_values),
                    metadata=metadata if isinstance(metadata, dict) else json_module.loads(metadata),
                )
            )

        return results

    def list(
        self, namespace: Optional[str] = None, filter_metadata: Optional[dict[str, Any]] = None
    ) -> List[VectorRecord]:
        """List all records, optionally filtered."""
        import json as json_module

        conn = self._get_connection()

        filter_clause = ""
        params: List[Any] = []
        if filter_metadata:
            conditions = []
            for key, value in filter_metadata.items():
                conditions.append(f"metadata->>%s = %s")
                params.append(key)
                params.append(str(value))
            if conditions:
                filter_clause = "WHERE " + " AND ".join(conditions)

        with conn.cursor() as cur:
            cur.execute(
                f"SELECT id, embedding, metadata FROM {self.collection} {filter_clause}",
                params,
            )
            rows = cur.fetchall()

        results = []
        for row in rows:
            record_id, emb_data, metadata = row
            if isinstance(emb_data, str):
                emb_values = [float(x) for x in emb_data.strip("[]").split(",")]
            else:
                emb_values = list(emb_data)

            results.append(
                VectorRecord(
                    id=record_id,
                    embedding=np.array(emb_values),
                    metadata=metadata if isinstance(metadata, dict) else json_module.loads(metadata),
                )
            )

        return results


class SqliteVectorStore(BaseVectorStore):
    """SQLite-backed vector store with on-disk persistence."""

    def __init__(self, connection_uri: str, collection: str = "embeddings", dimension: int = 1536):
        self.connection_uri = connection_uri
        self.collection = collection
        self.dimension = dimension
        self._conn: Optional[sqlite3.Connection] = None

    def _get_connection(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn
        self._conn = sqlite3.connect(self.connection_uri)
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.collection} (
                id TEXT PRIMARY KEY,
                embedding TEXT NOT NULL,
                metadata TEXT NOT NULL
            )
            """
        )
        self._conn.commit()
        return self._conn

    def upsert(self, records: Iterable[VectorRecord], namespace: Optional[str] = None) -> None:
        import json as json_module

        conn = self._get_connection()
        records_list = list(records)
        if not records_list:
            return

        payload = [
            (
                record.id,
                json_module.dumps(record.embedding.tolist()),
                json_module.dumps(record.metadata),
            )
            for record in records_list
        ]
        conn.executemany(
            f"""
            INSERT INTO {self.collection} (id, embedding, metadata)
            VALUES (?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                embedding = excluded.embedding,
                metadata = excluded.metadata
            """,
            payload,
        )
        conn.commit()

    def query(
        self,
        embedding: np.ndarray,
        top_k: int = 10,
        namespace: Optional[str] = None,
        filter_metadata: Optional[dict[str, Any]] = None,
    ) -> List[VectorRecord]:
        import json as json_module

        conn = self._get_connection()
        cursor = conn.execute(f"SELECT id, embedding, metadata FROM {self.collection}")
        rows = cursor.fetchall()

        records: list[VectorRecord] = []
        for record_id, emb_data, metadata in rows:
            emb_values = json_module.loads(emb_data)
            meta_obj = json_module.loads(metadata)
            records.append(
                VectorRecord(
                    id=record_id,
                    embedding=np.array(emb_values),
                    metadata=meta_obj,
                )
            )

        def _matches_filter(record: VectorRecord) -> bool:
            if not filter_metadata:
                return True
            return all(record.metadata.get(k) == v for k, v in filter_metadata.items())

        filtered = [rec for rec in records if _matches_filter(rec)]
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
        import json as json_module

        conn = self._get_connection()
        ids_list = list(ids)
        if not ids_list:
            return []

        placeholders = ",".join(["?"] * len(ids_list))
        cursor = conn.execute(
            f"SELECT id, embedding, metadata FROM {self.collection} WHERE id IN ({placeholders})",
            ids_list,
        )
        rows = cursor.fetchall()

        results = []
        for record_id, emb_data, metadata in rows:
            results.append(
                VectorRecord(
                    id=record_id,
                    embedding=np.array(json_module.loads(emb_data)),
                    metadata=json_module.loads(metadata),
                )
            )
        return results

    def list(
        self, namespace: Optional[str] = None, filter_metadata: Optional[dict[str, Any]] = None
    ) -> List[VectorRecord]:
        import json as json_module

        conn = self._get_connection()
        cursor = conn.execute(f"SELECT id, embedding, metadata FROM {self.collection}")
        rows = cursor.fetchall()

        results = []
        for record_id, emb_data, metadata in rows:
            meta_obj = json_module.loads(metadata)
            if filter_metadata and not all(meta_obj.get(k) == v for k, v in filter_metadata.items()):
                continue
            results.append(
                VectorRecord(
                    id=record_id,
                    embedding=np.array(json_module.loads(emb_data)),
                    metadata=meta_obj,
                )
            )
        return results


class VertexAIVectorStore(BaseVectorStore):
    """
    Google Cloud Vertex AI Vector Search based vector store.

    Requires:
    - google-cloud-aiplatform package
    - Valid GCP project with Vertex AI enabled
    - Pre-created Vector Search index and deployed endpoint

    Metadata caching:
    - Metadata is stored locally because Vertex AI does not persist it for vector lookups.
    - Cache entries are maintained using an LRU policy to avoid unbounded growth.
    - When the cache exceeds the configured maximum, least-recently-used entries are evicted.
    - Fetch/list return only currently cached metadata; evicted entries are simply absent.
    """

    def __init__(
        self,
        project_id: str,
        location: str,
        index_endpoint_name: str,
        deployed_index_id: str,
        index_name: Optional[str] = None,
        max_metadata_entries: Optional[int] = 10000,
    ):
        self.project_id = project_id
        self.location = location
        self.index_endpoint_name = index_endpoint_name
        self.deployed_index_id = deployed_index_id
        self.index_name = index_name
        self.max_metadata_entries = max_metadata_entries
        self._endpoint = None
        self._index = None
        self._metadata_store: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()

    def _touch_metadata(self, record_id: str) -> None:
        if record_id in self._metadata_store:
            self._metadata_store.move_to_end(record_id)

    def _cache_metadata(self, record_id: str, metadata: Dict[str, Any]) -> None:
        self._metadata_store[record_id] = metadata
        self._metadata_store.move_to_end(record_id)
        self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        if self.max_metadata_entries is None:
            return
        while len(self._metadata_store) > self.max_metadata_entries:
            self._metadata_store.popitem(last=False)

    def _get_endpoint(self):
        """Get or create endpoint client."""
        if self._endpoint is not None:
            return self._endpoint

        try:
            from google.cloud import aiplatform
        except ImportError:
            raise RuntimeError(
                "google-cloud-aiplatform is required for Vertex AI Vector Search. "
                "Install with: pip install google-cloud-aiplatform"
            )

        aiplatform.init(project=self.project_id, location=self.location)

        self._endpoint = aiplatform.MatchingEngineIndexEndpoint(
            index_endpoint_name=self.index_endpoint_name
        )

        if self.index_name:
            self._index = aiplatform.MatchingEngineIndex(index_name=self.index_name)

        return self._endpoint

    def upsert(self, records: Iterable[VectorRecord], namespace: Optional[str] = None) -> None:
        """
        Upsert records into Vertex AI Vector Search.

        Note: Vertex AI Vector Search requires batch updates to the index.
        This method stores metadata locally and requires index rebuild for new vectors.
        """
        if self._index is None:
            raise RuntimeError("Index name required for upsert operations")

        try:
            from google.cloud import aiplatform
        except ImportError:
            raise RuntimeError("google-cloud-aiplatform is required")

        records_list = list(records)
        if not records_list:
            return

        # Store metadata locally (Vertex AI doesn't store metadata natively)
        for record in records_list:
            self._cache_metadata(record.id, record.metadata)

        # Prepare datapoints for upsert
        datapoints = []
        for record in records_list:
            datapoints.append({
                "datapoint_id": record.id,
                "feature_vector": record.embedding.tolist(),
            })

        # Upsert to index
        self._index.upsert_datapoints(datapoints=datapoints)

    def query(
        self,
        embedding: np.ndarray,
        top_k: int = 10,
        namespace: Optional[str] = None,
        filter_metadata: Optional[dict[str, Any]] = None,
    ) -> List[VectorRecord]:
        """Query similar vectors using Vertex AI Vector Search."""
        endpoint = self._get_endpoint()

        # Perform the query
        response = endpoint.find_neighbors(
            deployed_index_id=self.deployed_index_id,
            queries=[embedding.tolist()],
            num_neighbors=top_k,
        )

        results = []
        if response and len(response) > 0:
            for neighbor in response[0]:
                record_id = neighbor.id
                score = 1.0 - neighbor.distance  # Convert distance to similarity

                # Get metadata from local cache
                metadata = self._metadata_store.get(record_id)

                if metadata is None:
                    if filter_metadata:
                        continue
                    metadata = {}
                else:
                    self._touch_metadata(record_id)

                # Apply metadata filter if provided
                if filter_metadata:
                    matches = all(metadata.get(k) == v for k, v in filter_metadata.items())
                    if not matches:
                        continue

                results.append(
                    VectorRecord(
                        id=record_id,
                        embedding=np.zeros(1),  # Vertex AI doesn't return embeddings
                        metadata=metadata,
                        score=score,
                    )
                )

        return results

    def fetch(self, ids: Iterable[str], namespace: Optional[str] = None) -> List[VectorRecord]:
        """Fetch records by ID (metadata only, embeddings not returned by Vertex AI)."""
        results = []
        for record_id in ids:
            if record_id in self._metadata_store:
                self._touch_metadata(record_id)
                results.append(
                    VectorRecord(
                        id=record_id,
                        embedding=np.zeros(1),
                        metadata=self._metadata_store[record_id],
                    )
                )
        return results

    def list(
        self, namespace: Optional[str] = None, filter_metadata: Optional[dict[str, Any]] = None
    ) -> List[VectorRecord]:
        """List all records from metadata cache."""
        results = []
        accessed_ids = []
        for record_id, metadata in list(self._metadata_store.items()):
            if filter_metadata:
                if not all(metadata.get(k) == v for k, v in filter_metadata.items()):
                    continue

            accessed_ids.append(record_id)
            results.append(
                VectorRecord(
                    id=record_id,
                    embedding=np.zeros(1),
                    metadata=metadata,
                )
            )
        for record_id in accessed_ids:
            self._touch_metadata(record_id)
        return results


class VectorStoreClient:
    """Convenience wrapper that hides provider-specific details."""

    def __init__(self, config: VectorStoreConfig, store: Optional[BaseVectorStore] = None):
        self.config = config
        self.store = store or self._create_store(config)

    def _create_store(self, config: VectorStoreConfig) -> BaseVectorStore:
        provider = (config.provider or "in-memory").lower()
        if provider == "in-memory":
            return InMemoryVectorStore()

        if provider == "sqlite":
            if not config.connection_uri:
                raise ValueError("sqlite provider requires connection_uri in config")
            return SqliteVectorStore(
                connection_uri=config.connection_uri,
                collection=config.collection,
                dimension=config.dimension or 1536,
            )

        if provider == "pgvector":
            if not config.connection_uri:
                raise ValueError("pgvector provider requires connection_uri in config")
            return PgVectorStore(
                connection_uri=config.connection_uri,
                collection=config.collection,
                dimension=config.dimension or 1536,
            )

        if provider in {"vertex", "vertex-ai", "vertexai"}:
            if not config.project_id or not config.location:
                raise ValueError("Vertex AI provider requires project_id and location in config")
            # For Vertex AI, we need additional config
            # These would typically come from environment or extended config
            import os
            index_endpoint = os.getenv("VERTEX_INDEX_ENDPOINT", "")
            deployed_index_id = os.getenv("VERTEX_DEPLOYED_INDEX_ID", "")
            index_name = os.getenv("VERTEX_INDEX_NAME", "")

            if not index_endpoint or not deployed_index_id:
                raise ValueError(
                    "Vertex AI provider requires VERTEX_INDEX_ENDPOINT and "
                    "VERTEX_DEPLOYED_INDEX_ID environment variables"
                )

            return VertexAIVectorStore(
                project_id=config.project_id,
                location=config.location,
                index_endpoint_name=index_endpoint,
                deployed_index_id=deployed_index_id,
                index_name=index_name or None,
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


def _optional_int(value: Optional[str]) -> Optional[int]:
    if value is None or value == "":
        return None
    return int(value)
