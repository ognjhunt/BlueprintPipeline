import importlib.util
from pathlib import Path

import pytest
pytest.importorskip("flask")


def _load_module(module_name: str, relative_path: str):
    module_path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module: {relative_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeSnapshot:
    def __init__(self, data):
        self._data = data

    @property
    def exists(self):
        return self._data is not None

    def get(self, key, default=None):
        if self._data is None:
            return default
        return self._data.get(key, default)

    def to_dict(self):
        return {} if self._data is None else self._data.copy()


class FakeDocument:
    def __init__(self, store, key, already_exists_exception):
        self._store = store
        self._key = key
        self._already_exists_exception = already_exists_exception

    def get(self, transaction=None):
        return FakeSnapshot(self._store.get(self._key))

    def create(self, data):
        if self._key in self._store:
            raise self._already_exists_exception("already exists")
        self._store[self._key] = data.copy()

    def set(self, data):
        self._store[self._key] = data.copy()

    def update(self, data):
        existing = self._store.get(self._key, {}).copy()
        existing.update(data)
        self._store[self._key] = existing

    def delete(self):
        self._store.pop(self._key, None)


class FakeTransaction:
    def set(self, document, data):
        document.set(data)

    def update(self, document, data):
        document.update(data)


class FakeCollection:
    def __init__(self, store, already_exists_exception):
        self._store = store
        self._already_exists_exception = already_exists_exception

    def document(self, key):
        return FakeDocument(self._store, key, self._already_exists_exception)


class FakeFirestoreClient:
    def __init__(self, store, already_exists_exception):
        self._store = store
        self._already_exists_exception = already_exists_exception

    def collection(self, name):
        return FakeCollection(self._store.setdefault(name, {}), self._already_exists_exception)

    def transaction(self):
        return FakeTransaction()


class FakeExecutionResponse:
    def __init__(self, name):
        self.name = name


class FakeExecutionsClient:
    def create_execution(self, request, retry=None):
        return FakeExecutionResponse("executions/fake")


def _configure_firestore(monkeypatch, module):
    store = {}
    already_exists_exception = module.google_exceptions.AlreadyExists

    def _client_factory(project=None):
        return FakeFirestoreClient(store, already_exists_exception)

    def _transactional(func):
        def wrapper(transaction, *args, **kwargs):
            return func(transaction, *args, **kwargs)

        return wrapper

    monkeypatch.setattr(module.firestore, "Client", _client_factory)
    monkeypatch.setattr(module.firestore, "transactional", _transactional)
    return store


def _configure_common(monkeypatch, module):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setattr(module, "_is_authenticated", lambda body: True)
    monkeypatch.setattr(module.executions_v1, "ExecutionsClient", FakeExecutionsClient)


def test_rate_limit_allows_below_limit(monkeypatch):
    monkeypatch.setenv("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    monkeypatch.setenv("WEBHOOK_RATE_LIMIT_PER_MIN", "2")
    module = _load_module("webhook_rate_limit", "genie-sim-import-webhook/main.py")
    store = _configure_firestore(monkeypatch, module)
    _configure_common(monkeypatch, module)

    client = module.app.test_client()
    response = client.post(
        "/webhooks/geniesim/job-complete",
        json={"job_id": "job-1", "status": "completed", "scene_id": "scene-1"},
        environ_base={"REMOTE_ADDR": "10.0.0.1"},
    )

    assert response.status_code == 202
    assert response.headers.get("X-RateLimit-Remaining") == "1"
    assert store.get("webhook_rate_limits")


def test_rate_limit_blocks_above_limit(monkeypatch):
    monkeypatch.setenv("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    monkeypatch.setenv("WEBHOOK_RATE_LIMIT_PER_MIN", "1")
    module = _load_module("webhook_rate_limit_block", "genie-sim-import-webhook/main.py")
    _configure_firestore(monkeypatch, module)
    _configure_common(monkeypatch, module)

    client = module.app.test_client()
    response_first = client.post(
        "/webhooks/geniesim/job-complete",
        json={"job_id": "job-2", "status": "completed", "scene_id": "scene-2"},
        environ_base={"REMOTE_ADDR": "10.0.0.2"},
    )
    response_second = client.post(
        "/webhooks/geniesim/job-complete",
        json={"job_id": "job-3", "status": "completed", "scene_id": "scene-2"},
        environ_base={"REMOTE_ADDR": "10.0.0.2"},
    )

    assert response_first.status_code == 202
    assert response_second.status_code == 429
    assert response_second.headers.get("Retry-After")
    assert response_second.headers.get("X-RateLimit-Remaining") == "0"


def test_rate_limit_scene_id_falls_back_to_ip(monkeypatch):
    monkeypatch.setenv("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    monkeypatch.setenv("WEBHOOK_RATE_LIMIT_PER_MIN", "1")
    module = _load_module("webhook_rate_limit_ip", "genie-sim-import-webhook/main.py")
    store = _configure_firestore(monkeypatch, module)
    _configure_common(monkeypatch, module)

    client = module.app.test_client()
    response_first = client.post(
        "/webhooks/geniesim/job-complete",
        json={"job_id": "job-4", "status": "completed"},
        environ_base={"REMOTE_ADDR": "10.0.0.3"},
    )
    response_second = client.post(
        "/webhooks/geniesim/job-complete",
        json={"job_id": "job-5", "status": "completed"},
        environ_base={"REMOTE_ADDR": "10.0.0.3"},
    )

    assert response_first.status_code == 202
    assert response_second.status_code == 429
    rate_limit_docs = list(store.get("webhook_rate_limits", {}).keys())
    assert any(key.startswith("ip-only:10.0.0.3:") for key in rate_limit_docs)
