#!/usr/bin/env python3
"""Backfill embedding queue objects for existing asset catalog documents."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from tools.asset_catalog import AssetCatalogClient


GCS_ROOT = Path("/mnt/gcs")


def _safe_int(value: Optional[str], default: int) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _read_json(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
    if not path.is_file():
        return dict(default)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return dict(default)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2), encoding="utf-8")


def _descriptor_text(payload: Mapping[str, Any]) -> str:
    class_name = str(payload.get("class_name") or payload.get("category") or "object")
    description = str(payload.get("description") or "")
    sim_roles_raw = payload.get("sim_roles")
    sim_roles = [str(item) for item in sim_roles_raw] if isinstance(sim_roles_raw, list) else []
    tags_raw = payload.get("tags")
    tags = [str(item) for item in tags_raw] if isinstance(tags_raw, list) else []
    return " ".join(
        part for part in [class_name, description, " ".join(sim_roles), " ".join(tags)] if part
    )


def _build_asset_entry(payload: Mapping[str, Any], model_name: str) -> Optional[Dict[str, Any]]:
    asset_id = str(payload.get("asset_id") or payload.get("id") or "").strip()
    if not asset_id:
        return None
    descriptor_text = _descriptor_text(payload)
    descriptor_hash = sha256(descriptor_text.encode("utf-8")).hexdigest()
    idempotency_key = sha256(f"{asset_id}|{model_name}|{descriptor_hash}".encode("utf-8")).hexdigest()

    sim_roles_raw = payload.get("sim_roles")
    sim_roles = [str(item) for item in sim_roles_raw] if isinstance(sim_roles_raw, list) else []
    tags_raw = payload.get("tags")
    tags = [str(item) for item in tags_raw] if isinstance(tags_raw, list) else []
    return {
        "asset_id": asset_id,
        "descriptor_text": descriptor_text,
        "descriptor_hash": descriptor_hash,
        "idempotency_key": idempotency_key,
        "class_name": str(payload.get("class_name") or payload.get("category") or "object"),
        "sim_roles": sim_roles,
        "usd_path": str(payload.get("usd_path") or payload.get("asset_path") or ""),
        "gcs_uri": str(payload.get("gcs_uri") or ""),
        "source": str(payload.get("source") or ""),
        "tags": tags,
        "dimensions": payload.get("dimensions"),
    }


def _existing_hash(payload: Mapping[str, Any]) -> str:
    embeddings = payload.get("embeddings")
    if not isinstance(embeddings, Mapping):
        return ""
    text_emb = embeddings.get("text")
    if not isinstance(text_emb, Mapping):
        return ""
    return str(text_emb.get("descriptor_hash") or "")


def _iter_firestore_assets(
    *,
    page_size: int,
    cursor_asset_id: str,
) -> Tuple[List[Dict[str, Any]], str]:
    client = AssetCatalogClient()
    firestore_client = client._ensure_client()  # noqa: SLF001 - internal helper for backfill utility
    if firestore_client is None:
        raise RuntimeError("Firestore client unavailable for asset backfill")

    coll = firestore_client.collection(client.config.assets_collection)
    query = coll.order_by("asset_id").limit(page_size)
    if cursor_asset_id:
        query = query.start_after({"asset_id": cursor_asset_id})

    docs = list(query.stream())
    payloads: List[Dict[str, Any]] = []
    new_cursor = cursor_asset_id
    for doc in docs:
        data = doc.to_dict() or {}
        payloads.append(data)
        new_cursor = str(data.get("asset_id") or new_cursor)
    return payloads, new_cursor


def _write_queue_objects(
    *,
    entries: List[Dict[str, Any]],
    queue_prefix: str,
    embedding_model: str,
    chunk_size: int,
) -> List[str]:
    if not entries:
        return []
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    queue_objects: List[str] = []

    for idx in range(0, len(entries), chunk_size):
        chunk = entries[idx : idx + chunk_size]
        chunk_idx = (idx // chunk_size) + 1
        suffix = sha256(f"{stamp}|{chunk_idx}|{len(chunk)}".encode("utf-8")).hexdigest()[:8]
        object_name = f"{queue_prefix}/{stamp}-backfill-{chunk_idx:04d}-{suffix}.json"
        payload = {
            "schema_version": "v1",
            "scene_id": "asset_catalog_backfill",
            "created_at": now.isoformat(),
            "embedding_model": embedding_model,
            "source_pipeline": "asset-catalog-backfill",
            "assets": chunk,
        }
        _write_json(GCS_ROOT / object_name, payload)
        queue_objects.append(object_name)
    return queue_objects


def main() -> int:
    bucket = (os.getenv("BUCKET") or "").strip()
    if not bucket:
        raise RuntimeError("BUCKET is required")

    queue_prefix = (os.getenv("TEXT_ASSET_EMBEDDING_QUEUE_PREFIX") or "automation/asset_embedding/queue").strip().strip("/")
    state_prefix = (os.getenv("TEXT_ASSET_EMBEDDING_BACKFILL_STATE_PREFIX") or "automation/asset_embedding/backfill").strip().strip("/")
    model_name = (os.getenv("TEXT_ASSET_EMBEDDING_MODEL") or "text-embedding-3-small").strip()
    page_size = max(1, _safe_int(os.getenv("TEXT_ASSET_EMBEDDING_BACKFILL_PAGE_SIZE"), 500))
    max_enqueue = max(1, _safe_int(os.getenv("TEXT_ASSET_EMBEDDING_BACKFILL_MAX_ENQUEUE"), 20000))
    queue_chunk = max(1, _safe_int(os.getenv("TEXT_ASSET_EMBEDDING_BACKFILL_QUEUE_CHUNK"), 500))

    state_path = GCS_ROOT / state_prefix / "state.json"
    state = _read_json(
        state_path,
        {
            "schema_version": "v1",
            "cursor_asset_id": "",
            "enqueued_total": 0,
            "skipped_total": 0,
            "last_run_at": "",
        },
    )
    cursor_asset_id = str(state.get("cursor_asset_id") or "")

    queued_entries: List[Dict[str, Any]] = []
    skipped = 0
    scanned = 0
    current_cursor = cursor_asset_id

    while len(queued_entries) < max_enqueue:
        payloads, next_cursor = _iter_firestore_assets(page_size=page_size, cursor_asset_id=current_cursor)
        if not payloads:
            break

        for payload in payloads:
            scanned += 1
            entry = _build_asset_entry(payload, model_name=model_name)
            if entry is None:
                continue

            existing_hash = _existing_hash(payload)
            if existing_hash and existing_hash == entry["descriptor_hash"]:
                skipped += 1
                continue
            queued_entries.append(entry)
            if len(queued_entries) >= max_enqueue:
                break
        current_cursor = next_cursor
        if not next_cursor:
            break

    queue_objects = _write_queue_objects(
        entries=queued_entries,
        queue_prefix=queue_prefix,
        embedding_model=model_name,
        chunk_size=queue_chunk,
    )

    state["schema_version"] = "v1"
    state["cursor_asset_id"] = current_cursor
    state["enqueued_total"] = int(state.get("enqueued_total") or 0) + len(queued_entries)
    state["skipped_total"] = int(state.get("skipped_total") or 0) + skipped
    state["last_run_at"] = datetime.now(timezone.utc).isoformat()
    state["last_scanned"] = scanned
    state["last_enqueued"] = len(queued_entries)
    state["last_queue_objects"] = queue_objects
    _write_json(state_path, state)

    print(
        "[ASSET-BACKFILL] scanned=%s enqueued=%s skipped=%s queue_objects=%s cursor=%s"
        % (scanned, len(queued_entries), skipped, len(queue_objects), current_cursor)
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"[ASSET-BACKFILL] ERROR: {exc}", file=sys.stderr)
        raise

