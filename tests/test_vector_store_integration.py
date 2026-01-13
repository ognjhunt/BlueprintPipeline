import os

import numpy as np

from tools.asset_catalog.vector_store import VectorRecord, VectorStoreClient, VectorStoreConfig


def test_sqlite_vector_store_persistence_and_query(tmp_path, monkeypatch):
    db_path = tmp_path / "vectors.db"
    monkeypatch.setenv("VECTOR_STORE_PROVIDER", "sqlite")
    monkeypatch.setenv("VECTOR_STORE_COLLECTION", "asset_vectors")
    monkeypatch.setenv("VECTOR_STORE_CONNECTION_URI", str(db_path))
    monkeypatch.setenv("VECTOR_STORE_DIMENSION", "3")

    config = VectorStoreConfig.from_env()
    client = VectorStoreClient(config)

    records = [
        VectorRecord(
            id="asset-1",
            embedding=np.array([1.0, 0.0, 0.0]),
            metadata={"kind": "text", "category": "chair"},
        ),
        VectorRecord(
            id="asset-2",
            embedding=np.array([0.0, 1.0, 0.0]),
            metadata={"kind": "text", "category": "table"},
        ),
    ]
    client.upsert(records)

    fresh_client = VectorStoreClient(VectorStoreConfig.from_env())
    query_embedding = np.array([0.9, 0.1, 0.0])
    matches = fresh_client.query(query_embedding, top_k=1)

    assert matches
    assert matches[0].id == "asset-1"

    filtered = fresh_client.list(filter_metadata={"category": "table"})
    assert len(filtered) == 1
    assert filtered[0].id == "asset-2"

    fetched = fresh_client.fetch(["asset-1"])
    assert len(fetched) == 1
    assert fetched[0].metadata["category"] == "chair"


def test_vector_store_config_env_override(monkeypatch):
    monkeypatch.setenv("VECTOR_STORE_PROVIDER", "sqlite")
    monkeypatch.setenv("VECTOR_STORE_COLLECTION", "override")
    monkeypatch.setenv("VECTOR_STORE_CONNECTION_URI", ":memory:")

    config = VectorStoreConfig.from_env(collection="explicit")

    assert config.provider == "sqlite"
    assert config.collection == "explicit"
    assert config.connection_uri == ":memory:"
