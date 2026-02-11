#!/usr/bin/env python3
"""Process async asset replication queue entries and copy files to Backblaze B2.

Queue entries are JSON files written by text-scene-adapter-job under:
  automation/asset_replication/queue/*.json
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

try:
    import boto3
except Exception:  # pragma: no cover - dependency guard
    boto3 = None

GCS_ROOT = Path("/mnt/gcs")


def _is_truthy(raw: str | None, *, default: bool) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _queue_candidates(root: Path, queue_prefix: str, queue_object: str | None, max_items: int) -> List[Tuple[str, Path]]:
    if queue_object:
        rel = queue_object.strip().lstrip("/")
        path = root / rel
        if path.is_file():
            return [(rel, path)]
        return []

    queue_root = root / queue_prefix
    if not queue_root.is_dir():
        return []

    candidates: List[Tuple[str, Path]] = []
    for path in sorted(queue_root.glob("*.json")):
        rel = path.relative_to(root).as_posix()
        candidates.append((rel, path))
        if len(candidates) >= max_items:
            break
    return candidates


def _make_b2_client():
    if boto3 is None:
        raise RuntimeError("boto3 is required for Backblaze replication")

    endpoint = (os.getenv("B2_S3_ENDPOINT") or "").strip()
    key_id = (os.getenv("B2_KEY_ID") or "").strip()
    app_key = (os.getenv("B2_APPLICATION_KEY") or "").strip()
    region = (os.getenv("B2_REGION") or "us-west-000").strip()
    if not endpoint or not key_id or not app_key:
        raise RuntimeError("Missing B2 credentials/env: B2_S3_ENDPOINT, B2_KEY_ID, B2_APPLICATION_KEY")

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=key_id,
        aws_secret_access_key=app_key,
        region_name=region,
    )


def _upload_asset_files(
    *,
    root: Path,
    queue_payload: Dict[str, Any],
    b2_client: Any,
    b2_bucket: str,
    dry_run: bool,
) -> Dict[str, Any]:
    assets = queue_payload.get("assets")
    if not isinstance(assets, list):
        return {"uploaded": 0, "skipped": 0, "errors": ["invalid_assets_payload"]}

    uploaded = 0
    skipped = 0
    errors: List[str] = []

    for asset in assets:
        files = asset.get("files")
        if not isinstance(files, list):
            continue
        for file_entry in files:
            if not isinstance(file_entry, dict):
                continue
            rel = str(file_entry.get("path") or "").strip()
            target_key = str(file_entry.get("target_key") or rel).strip()
            if not rel or not target_key:
                skipped += 1
                continue

            local_path = root / rel
            if not local_path.is_file():
                errors.append(f"missing_file:{rel}")
                continue

            if dry_run:
                uploaded += 1
                continue

            try:
                b2_client.upload_file(str(local_path), b2_bucket, target_key)
                uploaded += 1
            except Exception as exc:  # pragma: no cover - network error
                errors.append(f"upload_failed:{rel}:{exc}")

    return {
        "uploaded": uploaded,
        "skipped": skipped,
        "errors": errors,
    }


def _finalize_queue_item(
    *,
    root: Path,
    queue_object: str,
    queue_path: Path,
    summary: Dict[str, Any],
    ok: bool,
) -> None:
    processed_prefix = (os.getenv("TEXT_ASSET_REPLICATION_PROCESSED_PREFIX") or "automation/asset_replication/processed").strip().strip("/")
    failed_prefix = (os.getenv("TEXT_ASSET_REPLICATION_FAILED_PREFIX") or "automation/asset_replication/failed").strip().strip("/")
    target_prefix = processed_prefix if ok else failed_prefix
    result_object = f"{target_prefix}/{Path(queue_object).name}"
    result_path = root / result_object
    _write_json(result_path, summary)
    try:
        queue_path.unlink()
    except FileNotFoundError:
        pass


def main() -> int:
    bucket = (os.getenv("BUCKET") or "").strip()
    if not bucket:
        raise RuntimeError("BUCKET is required")

    queue_prefix = (os.getenv("TEXT_ASSET_REPLICATION_QUEUE_PREFIX") or "automation/asset_replication/queue").strip().strip("/")
    queue_object = (os.getenv("QUEUE_OBJECT") or "").strip() or None
    max_items = int(os.getenv("TEXT_ASSET_REPLICATION_MAX_ITEMS") or "10")
    dry_run = _is_truthy(os.getenv("TEXT_ASSET_REPLICATION_DRY_RUN"), default=False)
    fail_on_error = _is_truthy(os.getenv("TEXT_ASSET_REPLICATION_FAIL_ON_ERROR"), default=True)
    b2_bucket = (os.getenv("B2_BUCKET") or "").strip()
    if not b2_bucket and not dry_run:
        raise RuntimeError("B2_BUCKET is required unless TEXT_ASSET_REPLICATION_DRY_RUN=true")

    candidates = _queue_candidates(GCS_ROOT, queue_prefix, queue_object, max_items=max(1, max_items))
    if not candidates:
        print("[ASSET-REPL] No replication queue items found")
        return 0

    b2_client = None
    if not dry_run:
        b2_client = _make_b2_client()

    failures = 0
    for rel, path in candidates:
        payload = _read_json(path)
        upload_summary = _upload_asset_files(
            root=GCS_ROOT,
            queue_payload=payload,
            b2_client=b2_client,
            b2_bucket=b2_bucket,
            dry_run=dry_run,
        )

        ok = len(upload_summary["errors"]) == 0
        if not ok:
            failures += 1

        summary = {
            "schema_version": "v1",
            "queue_object": rel,
            "scene_id": payload.get("scene_id"),
            "processed_at": _now(),
            "status": "succeeded" if ok else "failed",
            "dry_run": dry_run,
            "bucket": bucket,
            "replication_target": "backblaze_b2",
            "b2_bucket": b2_bucket if not dry_run else "",
            "result": upload_summary,
        }
        _finalize_queue_item(
            root=GCS_ROOT,
            queue_object=rel,
            queue_path=path,
            summary=summary,
            ok=ok,
        )

    if failures and fail_on_error:
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"[ASSET-REPL] ERROR: {exc}", file=sys.stderr)
        raise

