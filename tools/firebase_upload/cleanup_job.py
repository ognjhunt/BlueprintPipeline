"""Cleanup Firebase Storage blobs by age and manifest."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional, Set, Tuple

from firebase_admin import storage

from tools.firebase_upload.uploader import cleanup_firebase_paths, init_firebase
from tools.tracing import init_tracing

logger = logging.getLogger(__name__)


def _log_json(action: str, payload: dict) -> None:
    logger.info(json.dumps({"action": action, **payload}, sort_keys=True))


def _parse_max_age_hours(value: Optional[str]) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        hours = float(value)
    except ValueError as exc:
        raise ValueError(f"Invalid FIREBASE_CLEANUP_MAX_AGE_HOURS: {value}") from exc
    if hours <= 0:
        raise ValueError("FIREBASE_CLEANUP_MAX_AGE_HOURS must be > 0")
    return hours


def _load_manifest(path: Optional[str]) -> Tuple[Set[str], Set[str]]:
    if not path:
        return set(), set()
    manifest_path = Path(path).expanduser()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    raw_text = manifest_path.read_text(encoding="utf-8")
    paths: Set[str] = set()
    prefixes: Set[str] = set()
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        for line in raw_text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            paths.add(stripped)
        return paths, prefixes

    if isinstance(payload, list):
        for entry in payload:
            if isinstance(entry, str) and entry:
                paths.add(entry)
        return paths, prefixes

    if isinstance(payload, dict):
        for entry in payload.get("paths", []) or []:
            if isinstance(entry, str) and entry:
                paths.add(entry)
        for entry in payload.get("prefixes", []) or []:
            if isinstance(entry, str) and entry:
                prefixes.add(entry)
        return paths, prefixes

    raise ValueError("Manifest must be a JSON list or object with paths/prefixes")


def _is_known_good(blob_name: str, paths: Set[str], prefixes: Set[str]) -> bool:
    if blob_name in paths:
        return True
    return any(blob_name.startswith(prefix) for prefix in prefixes)


def _iter_blobs(prefix: str) -> Iterable:
    init_firebase()
    bucket = storage.bucket()
    return bucket.list_blobs(prefix=prefix)


def cleanup_orphaned_blobs() -> dict:
    prefix = os.getenv("FIREBASE_CLEANUP_PREFIX", "datasets")
    if not prefix:
        raise ValueError("FIREBASE_CLEANUP_PREFIX must be set")

    max_age_hours = _parse_max_age_hours(os.getenv("FIREBASE_CLEANUP_MAX_AGE_HOURS"))
    manifest_path = os.getenv("FIREBASE_CLEANUP_MANIFEST_PATH")
    known_paths, known_prefixes = _load_manifest(manifest_path)

    if max_age_hours is None and not known_paths and not known_prefixes:
        raise ValueError(
            "Set FIREBASE_CLEANUP_MAX_AGE_HOURS and/or FIREBASE_CLEANUP_MANIFEST_PATH"
        )

    now = datetime.now(tz=timezone.utc)
    cutoff = None
    if max_age_hours is not None:
        cutoff = now - timedelta(hours=max_age_hours)

    considered = 0
    orphaned = []
    skipped_known_good = 0
    skipped_recent = 0
    for blob in _iter_blobs(prefix):
        considered += 1
        blob_name = blob.name
        if _is_known_good(blob_name, known_paths, known_prefixes):
            skipped_known_good += 1
            continue

        if cutoff is not None:
            updated = blob.updated or blob.time_created
            if updated is None:
                skipped_recent += 1
                continue
            if updated >= cutoff:
                skipped_recent += 1
                continue

        orphaned.append(blob_name)

    cleanup_result = {
        "mode": "paths",
        "prefix": prefix,
        "requested": [],
        "deleted": [],
        "failed": [],
    }
    if orphaned:
        cleanup_result = cleanup_firebase_paths(paths=orphaned)

    summary = {
        "prefix": prefix,
        "considered": considered,
        "requested": len(cleanup_result["requested"]),
        "deleted": len(cleanup_result["deleted"]),
        "failed": len(cleanup_result["failed"]),
        "skipped_known_good": skipped_known_good,
        "skipped_recent": skipped_recent,
        "max_age_hours": max_age_hours,
        "manifest_path": manifest_path,
        "failed_details": cleanup_result["failed"],
    }

    _log_json(
        "cleanup_stats",
        {
            "prefix": prefix,
            "requested": summary["requested"],
            "deleted": summary["deleted"],
            "failed": summary["failed"],
        },
    )
    _log_json("cleanup_summary", summary)
    print(json.dumps(summary, sort_keys=True))
    return summary


def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    init_tracing(service_name=os.getenv("OTEL_SERVICE_NAME", "firebase-cleanup-job"))
    cleanup_orphaned_blobs()


if __name__ == "__main__":
    main()
