from __future__ import annotations

from typing import Any, Dict, List

import pytest

from tools.firestore import migrations


class FakeDoc:
    def __init__(self, reference: str, data: Dict[str, Any]) -> None:
        self.reference = reference
        self._data = data

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._data)


class FakeBatch:
    def __init__(self) -> None:
        self.set_calls: List[Dict[str, Any]] = []
        self.commit_count = 0

    def set(self, reference: str, data: Dict[str, Any], merge: bool = False) -> None:
        self.set_calls.append(
            {
                "reference": reference,
                "data": dict(data),
                "merge": merge,
            }
        )

    def commit(self) -> None:
        self.commit_count += 1


class FakeCollection:
    def __init__(self, docs: List[FakeDoc]) -> None:
        self._docs = docs

    def stream(self) -> List[FakeDoc]:
        return list(self._docs)


class FakeClient:
    def __init__(self, docs: List[FakeDoc]) -> None:
        self._docs = docs
        self.batches: List[FakeBatch] = []

    def collection(self, name: str) -> FakeCollection:
        return FakeCollection(self._docs)

    def batch(self) -> FakeBatch:
        batch = FakeBatch()
        self.batches.append(batch)
        return batch


@pytest.fixture(autouse=True)
def _reset_migrations() -> None:
    original = dict(migrations.MIGRATIONS)
    migrations.MIGRATIONS.clear()
    yield
    migrations.MIGRATIONS.clear()
    migrations.MIGRATIONS.update(original)


def test_noop_when_versions_match() -> None:
    docs = [FakeDoc("doc-1", {"schema_version": 1, "field": "value"})]
    client = FakeClient(docs)

    updated = migrations.migrate_collection(client, "scenes", 1, 1)

    assert updated == 0
    assert client.batches
    assert client.batches[0].set_calls == []
    assert client.batches[0].commit_count == 0


def test_transform_applied_and_version_updated() -> None:
    def transform(data: Dict[str, Any]) -> Dict[str, Any]:
        data["renamed"] = data.pop("old", None)
        return data

    migrations.MIGRATIONS["scenes"][(1, 2)] = transform

    docs = [FakeDoc("doc-1", {"schema_version": 1, "old": "value"})]
    client = FakeClient(docs)

    updated = migrations.migrate_collection(client, "scenes", 1, 2)

    assert updated == 1
    batch = client.batches[0]
    assert batch.set_calls == [
        {
            "reference": "doc-1",
            "data": {"schema_version": 2, "renamed": "value"},
            "merge": True,
        }
    ]
    assert batch.commit_count == 1


def test_batch_commit_boundaries() -> None:
    def transform(data: Dict[str, Any]) -> Dict[str, Any]:
        data["migrated"] = True
        return data

    migrations.MIGRATIONS["scenes"][(1, 2)] = transform

    docs = [
        FakeDoc("doc-1", {"schema_version": 1}),
        FakeDoc("doc-2", {"schema_version": 1}),
        FakeDoc("doc-3", {"schema_version": 1}),
    ]
    client = FakeClient(docs)

    updated = migrations.migrate_collection(client, "scenes", 1, 2, batch_size=2)

    assert updated == 3
    assert len(client.batches) == 2
    assert client.batches[0].commit_count == 1
    assert client.batches[1].commit_count == 1
    assert client.batches[0].set_calls[0]["data"]["schema_version"] == 2
