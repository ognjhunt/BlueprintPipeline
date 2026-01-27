from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import importlib.util
import sys
from types import ModuleType


def _load_cleanup_job_module():
    fake_firebase = ModuleType("firebase_admin")
    fake_storage = ModuleType("firebase_admin.storage")
    fake_storage.bucket = lambda: None
    fake_firebase.storage = fake_storage
    sys.modules.setdefault("firebase_admin", fake_firebase)
    sys.modules.setdefault("firebase_admin.storage", fake_storage)
    fake_pkg = ModuleType("tools.firebase_upload")
    fake_pkg.__path__ = []
    fake_uploader = ModuleType("tools.firebase_upload.uploader")
    fake_uploader.cleanup_firebase_paths = lambda *args, **kwargs: {}
    fake_uploader.init_firebase = lambda *args, **kwargs: None
    sys.modules.setdefault("tools.firebase_upload", fake_pkg)
    sys.modules.setdefault("tools.firebase_upload.uploader", fake_uploader)

    module_path = Path(__file__).resolve().parents[1] / "tools" / "firebase_upload" / "cleanup_job.py"
    spec = importlib.util.spec_from_file_location("firebase_cleanup_job", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["firebase_cleanup_job"] = module
    spec.loader.exec_module(module)
    return module


class _FakeBlob:
    def __init__(self, name: str, updated=None, time_created=None):
        self.name = name
        self.updated = updated
        self.time_created = time_created


def test_parse_max_age_hours_valid_invalid():
    cleanup_job = _load_cleanup_job_module()
    assert cleanup_job._parse_max_age_hours(None) is None
    assert cleanup_job._parse_max_age_hours("") is None
    assert cleanup_job._parse_max_age_hours("1.5") == 1.5

    with pytest.raises(ValueError):
        cleanup_job._parse_max_age_hours("abc")

    with pytest.raises(ValueError):
        cleanup_job._parse_max_age_hours("0")


def test_load_manifest_variants(tmp_path: Path):
    cleanup_job = _load_cleanup_job_module()
    json_list = tmp_path / "list.json"
    json_list.write_text(json.dumps(["path/a", "path/b"]))
    paths, prefixes = cleanup_job._load_manifest(str(json_list))
    assert paths == {"path/a", "path/b"}
    assert prefixes == set()

    json_obj = tmp_path / "obj.json"
    json_obj.write_text(json.dumps({"paths": ["path/c"], "prefixes": ["pref/"]}))
    paths, prefixes = cleanup_job._load_manifest(str(json_obj))
    assert paths == {"path/c"}
    assert prefixes == {"pref/"}

    text_manifest = tmp_path / "manifest.txt"
    text_manifest.write_text(
        "# comment\n\npath/d\npath/e\n",
        encoding="utf-8",
    )
    paths, prefixes = cleanup_job._load_manifest(str(text_manifest))
    assert paths == {"path/d", "path/e"}
    assert prefixes == set()


def test_cleanup_orphaned_blobs_summary(monkeypatch, tmp_path: Path):
    cleanup_job = _load_cleanup_job_module()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"paths": ["datasets/keep.txt"], "prefixes": ["datasets/keep_prefix/"]})
    )

    fixed_now = datetime(2024, 1, 1, tzinfo=timezone.utc)

    class _FixedDateTime:
        @classmethod
        def now(cls, tz=None):
            return fixed_now

    monkeypatch.setenv("FIREBASE_CLEANUP_PREFIX", "datasets")
    monkeypatch.setenv("FIREBASE_CLEANUP_MAX_AGE_HOURS", "1")
    monkeypatch.setenv("FIREBASE_CLEANUP_MANIFEST_PATH", str(manifest_path))

    blobs = [
        _FakeBlob("datasets/keep.txt", updated=fixed_now - timedelta(hours=3)),
        _FakeBlob("datasets/keep_prefix/old.txt", updated=fixed_now - timedelta(hours=3)),
        _FakeBlob("datasets/recent.txt", updated=fixed_now - timedelta(minutes=30)),
        _FakeBlob("datasets/old.txt", updated=fixed_now - timedelta(hours=2)),
        _FakeBlob("datasets/no_timestamp.txt", updated=None, time_created=None),
    ]

    class _Bucket:
        def list_blobs(self, prefix: str):
            return blobs

    def _cleanup(paths):
        return {
            "mode": "paths",
            "prefix": "datasets",
            "requested": list(paths),
            "deleted": list(paths),
            "failed": [],
        }

    monkeypatch.setattr(cleanup_job, "datetime", _FixedDateTime)
    monkeypatch.setattr(cleanup_job, "init_firebase", lambda: None)
    monkeypatch.setattr(cleanup_job.storage, "bucket", lambda: _Bucket())
    monkeypatch.setattr(cleanup_job, "cleanup_firebase_paths", _cleanup)

    summary = cleanup_job.cleanup_orphaned_blobs()

    assert summary["considered"] == 5
    assert summary["skipped_known_good"] == 2
    assert summary["skipped_recent"] == 2
    assert summary["deleted"] == 1
