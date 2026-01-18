"""Tests for Firestore-dependent services."""

from __future__ import annotations

import datetime

import pytest


class FakeDoc:
    """Simple Firestore document wrapper."""

    def __init__(self, data, exists: bool = True):
        self._data = data
        self.exists = exists

    def to_dict(self):
        return self._data


class FakeDocRef:
    """Document reference with basic CRUD tracking."""

    def __init__(self, collection, doc_id: str):
        self.collection = collection
        self.doc_id = doc_id
        self.set_calls = []
        self.update_calls = []

    def get(self):
        exists = self.doc_id in self.collection._documents
        return FakeDoc(self.collection._documents.get(self.doc_id), exists=exists)

    def set(self, data, merge: bool = False):
        self.set_calls.append({"data": data, "merge": merge})
        if merge and self.doc_id in self.collection._documents:
            merged = dict(self.collection._documents[self.doc_id])
            merged.update(data)
            self.collection._documents[self.doc_id] = merged
        else:
            self.collection._documents[self.doc_id] = dict(data)

    def update(self, data):
        self.update_calls.append(data)
        self.collection._documents.setdefault(self.doc_id, {}).update(data)


class FakeQuery:
    """Query implementation that supports basic filtering."""

    def __init__(self, collection, filters=None):
        self.collection = collection
        self.filters = filters or []
        self.order_by_args = None
        self.limit_count = None

    def where(self, field, op, value):
        self.filters.append((field, op, value))
        return self

    def order_by(self, field, direction=None):
        self.order_by_args = (field, direction)
        return self

    def limit(self, count):
        self.limit_count = count
        return self

    def stream(self):
        docs = list(self.collection._documents.values())
        for field, op, value in self.filters:
            if op == "==":
                docs = [doc for doc in docs if doc.get(field) == value]
            elif op == ">=":
                docs = [doc for doc in docs if doc.get(field) >= value]
        if self.limit_count is not None:
            docs = docs[: self.limit_count]
        return [FakeDoc(doc) for doc in docs]


class FakeCollection:
    """Collection wrapper that produces document refs and queries."""

    def __init__(self, name, documents=None):
        self.name = name
        self._documents = documents or {}
        self._doc_refs = {}
        self.last_query = None

    def document(self, doc_id: str):
        if doc_id not in self._doc_refs:
            self._doc_refs[doc_id] = FakeDocRef(self, doc_id)
        return self._doc_refs[doc_id]

    def where(self, field, op, value):
        self.last_query = FakeQuery(self, [(field, op, value)])
        return self.last_query


class FakeFirestoreClient:
    """Client wrapper that stores collections."""

    def __init__(self, initial=None):
        self._collections = {}
        for name, docs in (initial or {}).items():
            self._collections[name] = FakeCollection(name, documents=docs)

    def collection(self, name: str):
        if name not in self._collections:
            self._collections[name] = FakeCollection(name)
        return self._collections[name]


class FakeFirestoreModule:
    """Fake firestore module providing Client and constants."""

    class Query:
        DESCENDING = "DESCENDING"

    SERVER_TIMESTAMP = object()

    class Increment:
        def __init__(self, value):
            self.value = value

    def __init__(self, client):
        self._client = client

    def Client(self, project=None):
        return self._client


def _load_scene_generation_module(load_job_module):
    return load_job_module("scene_generation", "generate_scene_images.py")


def _load_customer_config_module(load_job_module):
    return load_job_module("upsell_features", "customer_config.py")


def test_generation_history_tracker_queries_firestore(load_job_module, monkeypatch):
    module = _load_scene_generation_module(load_job_module)
    now = datetime.datetime.now(datetime.timezone.utc)
    collection_name = module.FIRESTORE_COLLECTION_HISTORY
    client = FakeFirestoreClient(
        {
            collection_name: {
                "doc1": {
                    "archetype": module.EnvironmentArchetype.KITCHEN.value,
                    "generated_at": now,
                    "variation_tags": ["bright"],
                },
                "doc2": {
                    "archetype": module.EnvironmentArchetype.OFFICE.value,
                    "generated_at": now,
                    "variation_tags": ["quiet"],
                },
            }
        }
    )
    fake_firestore = FakeFirestoreModule(client)
    monkeypatch.setattr(module, "firestore", fake_firestore)
    monkeypatch.setattr(module, "HAVE_CLOUD_DEPS", True)

    tracker = module.GenerationHistoryTracker()
    prompts = tracker.get_recent_prompts(
        module.EnvironmentArchetype.KITCHEN,
        days=1,
        limit=5,
    )

    collection = client.collection(collection_name)
    query = collection.last_query
    assert ("archetype", "==", "kitchen") in query.filters
    assert any(field == "generated_at" and op == ">=" for field, op, _ in query.filters)
    assert query.order_by_args == ("generated_at", fake_firestore.Query.DESCENDING)
    assert query.limit_count == 5
    assert prompts == [collection._documents["doc1"]]


def test_generation_history_tracker_records_generation(load_job_module, monkeypatch):
    module = _load_scene_generation_module(load_job_module)
    collection_name = module.FIRESTORE_COLLECTION_HISTORY
    client = FakeFirestoreClient({collection_name: {}})
    fake_firestore = FakeFirestoreModule(client)
    monkeypatch.setattr(module, "firestore", fake_firestore)
    monkeypatch.setattr(module, "HAVE_CLOUD_DEPS", True)

    tracker = module.GenerationHistoryTracker()
    entry = module.GenerationHistoryEntry(
        scene_id="scene-123",
        archetype="kitchen",
        prompt_hash="hash",
        prompt_summary="summary",
        variation_tags=["tag1"],
        generated_at=datetime.datetime.now(datetime.timezone.utc),
        success=True,
    )

    assert tracker.record_generation(entry) is True
    doc_ref = client.collection(collection_name)._doc_refs[entry.scene_id]
    assert doc_ref.set_calls
    payload = doc_ref.set_calls[0]["data"]
    assert payload["generated_at"] is fake_firestore.SERVER_TIMESTAMP


def test_generation_history_tracker_firestore_unavailable(load_job_module, monkeypatch, capsys):
    module = _load_scene_generation_module(load_job_module)

    class BadFirestore:
        def Client(self, project=None):
            raise RuntimeError("boom")

    monkeypatch.setattr(module, "firestore", BadFirestore())
    monkeypatch.setattr(module, "HAVE_CLOUD_DEPS", True)

    tracker = module.GenerationHistoryTracker()
    captured = capsys.readouterr()
    assert "WARNING: Firestore unavailable" in captured.err
    assert tracker.get_recent_prompts(module.EnvironmentArchetype.KITCHEN) == []


def test_customer_config_firestore_reads_and_writes(load_job_module, monkeypatch):
    module = _load_customer_config_module(load_job_module)
    collection_name = module.FIRESTORE_COLLECTION_CUSTOMERS
    client = FakeFirestoreClient(
        {
            collection_name: {
                "cust-1": {
                    "bundle_tier": "pro",
                    "organization_name": "Acme",
                    "email": "test@example.com",
                }
            }
        }
    )
    fake_firestore = FakeFirestoreModule(client)
    monkeypatch.setattr(module, "firestore", fake_firestore)
    monkeypatch.setattr(module, "HAVE_FIRESTORE", True)
    monkeypatch.setattr(module, "HAVE_GCS", False)
    monkeypatch.setattr(module, "storage", None)

    service = module.CustomerConfigService(verbose=True)
    config = service.get_customer_config("cust-1")
    assert config.bundle_tier == module.BundleTier.PRO
    assert config.organization_name == "Acme"

    new_config = module.CustomerConfig(
        customer_id="cust-1",
        bundle_tier=module.BundleTier.STANDARD,
        organization_name="Updated",
        email="new@example.com",
    )
    assert service.save_customer_config(new_config) is True

    doc_ref = client.collection(collection_name)._doc_refs["cust-1"]
    assert doc_ref.set_calls
    set_call = doc_ref.set_calls[-1]
    assert set_call["merge"] is True
    assert set_call["data"]["updated_at"] is fake_firestore.SERVER_TIMESTAMP


def test_customer_config_firestore_unavailable(load_job_module, monkeypatch, capsys):
    module = _load_customer_config_module(load_job_module)

    class BadFirestore:
        def Client(self, project=None):
            raise RuntimeError("boom")

    monkeypatch.setattr(module, "firestore", BadFirestore())
    monkeypatch.setattr(module, "HAVE_FIRESTORE", True)
    monkeypatch.setattr(module, "HAVE_GCS", False)
    monkeypatch.setattr(module, "storage", None)

    service = module.CustomerConfigService(verbose=True)
    captured = capsys.readouterr()
    assert "WARNING: Firestore unavailable" in captured.out

    config = service.get_customer_config("cust-2")
    assert config.bundle_tier == module.BundleTier.STANDARD
    assert service.save_customer_config(config) is False

    record = module.UsageRecord(
        customer_id="cust-2",
        scene_id="scene-1",
        action="scene_generated",
        timestamp=datetime.datetime.now(datetime.timezone.utc),
    )
    assert service.record_usage(record) is False
