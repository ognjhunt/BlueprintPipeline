#!/usr/bin/env python3
"""Process asset embedding queue objects and upsert vectors to ANN storage."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

from tools.asset_catalog import (
    AssetCatalogClient,
    AssetEmbeddings,
    EmbeddingConfig,
    VectorRecord,
    VectorStoreClient,
    VectorStoreConfig,
)


GCS_ROOT = Path("/mnt/gcs")


def _is_truthy(raw: str | None, *, default: bool) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _safe_int(value: str | None, default: int) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2), encoding="utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _deterministic_embedding(text: str, dimension: int) -> np.ndarray:
    digest = sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "big")
    rng = np.random.default_rng(seed)
    return rng.standard_normal(dimension).astype(np.float32)


class EmbeddingRuntime:
    def __init__(self) -> None:
        self.backend = (os.getenv("TEXT_ASSET_EMBEDDING_BACKEND") or "openai").strip().lower()
        self.model_name = (os.getenv("TEXT_ASSET_EMBEDDING_MODEL") or "text-embedding-3-small").strip()
        self.dimension = max(8, _safe_int(os.getenv("VECTOR_STORE_DIMENSION"), 1536))
        self._embedder: Optional[AssetEmbeddings] = None

    def _build_embedder(self) -> AssetEmbeddings:
        if self._embedder is not None:
            return self._embedder
        api_key = (
            os.getenv("OPENAI_API_KEY")
            or os.getenv("TEXT_ASSET_OPENAI_API_KEY")
            or os.getenv("TEXT_GEN_OPENAI_API_KEY")
            or ""
        ).strip()
        config = EmbeddingConfig(
            backend=self.backend,
            model_name=self.model_name,
            api_key=api_key or None,
            dimension=self.dimension,
        )
        self._embedder = AssetEmbeddings(config=config)
        return self._embedder

    def embed_text(self, text: str) -> np.ndarray:
        if self.backend in {"deterministic", "stub"}:
            return _deterministic_embedding(text, self.dimension)
        return self._build_embedder().embed_text(text)


def _vector_store() -> VectorStoreClient:
    provider = (os.getenv("VECTOR_STORE_PROVIDER") or "vertex").strip()
    namespace = (os.getenv("TEXT_ASSET_ANN_NAMESPACE") or os.getenv("VECTOR_STORE_NAMESPACE") or "assets-v1").strip()
    config = VectorStoreConfig.from_env(
        provider=provider,
        collection=namespace,
        namespace=namespace,
        dimension=max(8, _safe_int(os.getenv("VECTOR_STORE_DIMENSION"), 1536)),
    )
    return VectorStoreClient(config)


def _asset_entries(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    assets = payload.get("assets")
    if not isinstance(assets, list):
        return []
    return [item for item in assets if isinstance(item, Mapping)]


def _descriptor_hash(entry: Mapping[str, Any]) -> str:
    descriptor_text = str(entry.get("descriptor_text") or "")
    return sha256(descriptor_text.encode("utf-8")).hexdigest()


def _idempotency_key(asset_id: str, model_name: str, descriptor_hash: str) -> str:
    return sha256(f"{asset_id}|{model_name}|{descriptor_hash}".encode("utf-8")).hexdigest()


def _upsert_firestore_embedding_metadata(
    *,
    asset_id: str,
    vector_id: str,
    descriptor_hash: str,
    model_name: str,
    provider: str,
    dimension: int,
) -> None:
    try:
        client = AssetCatalogClient()
        doc = client.fetch_asset_document(asset_id=asset_id)
        if doc is None:
            return
        extra = dict(doc.extra_metadata or {})
        embeddings = extra.get("embeddings")
        if not isinstance(embeddings, dict):
            embeddings = {}
        embeddings["text"] = {
            "vector_id": vector_id,
            "provider": provider,
            "dimension": dimension,
            "model": model_name,
            "descriptor_hash": descriptor_hash,
            "updated_at": _now(),
        }
        extra["embeddings"] = embeddings
        doc.extra_metadata = extra
        client.upsert_asset_document(doc)
    except Exception:
        return


def _process_queue_item(
    *,
    queue_object: str,
    queue_payload: Mapping[str, Any],
    vector_store: VectorStoreClient,
    runtime: EmbeddingRuntime,
    dry_run: bool,
    batch_size: int,
) -> Dict[str, Any]:
    provider_name = str(vector_store.config.provider or "unknown")
    entries = _asset_entries(queue_payload)
    if not entries:
        return {
            "processed": 0,
            "upserted": 0,
            "skipped": 0,
            "errors": ["invalid_assets_payload"],
        }

    seen_keys: set[str] = set()
    pending_records: List[VectorRecord] = []
    pending_metadata: List[Tuple[str, str, str]] = []
    upserted = 0
    skipped = 0
    errors: List[str] = []

    def _flush() -> None:
        nonlocal upserted
        if not pending_records:
            return
        if dry_run:
            upserted += len(pending_records)
        else:
            vector_store.upsert(pending_records, namespace=vector_store.config.collection)
            upserted += len(pending_records)
            for asset_id, vector_id, descriptor_hash in pending_metadata:
                _upsert_firestore_embedding_metadata(
                    asset_id=asset_id,
                    vector_id=vector_id,
                    descriptor_hash=descriptor_hash,
                    model_name=runtime.model_name,
                    provider=provider_name,
                    dimension=runtime.dimension,
                )
        pending_records.clear()
        pending_metadata.clear()

    for entry in entries:
        asset_id = str(entry.get("asset_id") or "").strip()
        if not asset_id:
            skipped += 1
            continue
        descriptor_text = str(entry.get("descriptor_text") or "").strip()
        if not descriptor_text:
            skipped += 1
            continue

        descriptor_hash = str(entry.get("descriptor_hash") or _descriptor_hash(entry))
        idem_key = str(entry.get("idempotency_key") or _idempotency_key(asset_id, runtime.model_name, descriptor_hash))
        if idem_key in seen_keys:
            skipped += 1
            continue
        seen_keys.add(idem_key)

        vector_id = f"{asset_id}:text"
        metadata = {
            "asset_id": asset_id,
            "class_name": str(entry.get("class_name") or "object"),
            "sim_roles": [str(role) for role in entry.get("sim_roles") or [] if role is not None],
            "source": str(entry.get("source") or ""),
            "usd_path": str(entry.get("usd_path") or ""),
            "gcs_uri": str(entry.get("gcs_uri") or ""),
            "descriptor_hash": descriptor_hash,
            "idempotency_key": idem_key,
            "model": runtime.model_name,
            "kind": "text",
            "queue_object": queue_object,
        }
        try:
            embedding = runtime.embed_text(descriptor_text)
        except Exception as exc:
            errors.append(f"embedding_failed:{asset_id}:{exc}")
            continue

        pending_records.append(
            VectorRecord(
                id=vector_id,
                embedding=embedding,
                metadata=metadata,
            )
        )
        pending_metadata.append((asset_id, vector_id, descriptor_hash))
        if len(pending_records) >= batch_size:
            try:
                _flush()
            except Exception as exc:
                errors.append(f"vector_upsert_failed:{exc}")
                pending_records.clear()
                pending_metadata.clear()

    if pending_records:
        try:
            _flush()
        except Exception as exc:
            errors.append(f"vector_upsert_failed:{exc}")

    return {
        "processed": len(entries),
        "upserted": upserted,
        "skipped": skipped,
        "errors": errors,
        "provider": provider_name,
        "model": runtime.model_name,
    }


def _finalize_queue_item(
    *,
    queue_object: str,
    queue_path: Path,
    summary: Mapping[str, Any],
    ok: bool,
) -> None:
    processed_prefix = (os.getenv("TEXT_ASSET_EMBEDDING_PROCESSED_PREFIX") or "automation/asset_embedding/processed").strip().strip("/")
    failed_prefix = (os.getenv("TEXT_ASSET_EMBEDDING_FAILED_PREFIX") or "automation/asset_embedding/failed").strip().strip("/")
    target_prefix = processed_prefix if ok else failed_prefix
    output_object = f"{target_prefix}/{Path(queue_object).name}"
    _write_json(GCS_ROOT / output_object, dict(summary))
    try:
        queue_path.unlink()
    except FileNotFoundError:
        pass


def main() -> int:
    bucket = (os.getenv("BUCKET") or "").strip()
    if not bucket:
        raise RuntimeError("BUCKET is required")

    queue_prefix = (os.getenv("TEXT_ASSET_EMBEDDING_QUEUE_PREFIX") or "automation/asset_embedding/queue").strip().strip("/")
    queue_object = (os.getenv("QUEUE_OBJECT") or "").strip() or None
    max_items = max(1, _safe_int(os.getenv("TEXT_ASSET_EMBEDDING_MAX_ITEMS"), 20))
    batch_size = max(1, _safe_int(os.getenv("TEXT_ASSET_EMBEDDING_BATCH_SIZE"), 32))
    dry_run = _is_truthy(os.getenv("TEXT_ASSET_EMBEDDING_DRY_RUN"), default=False)
    fail_on_error = _is_truthy(os.getenv("TEXT_ASSET_EMBEDDING_FAIL_ON_ERROR"), default=True)

    candidates = _queue_candidates(GCS_ROOT, queue_prefix, queue_object, max_items)
    if not candidates:
        print("[ASSET-EMBED] No embedding queue items found")
        return 0

    vector_store = _vector_store()
    runtime = EmbeddingRuntime()

    failures = 0
    for rel, path in candidates:
        payload = _read_json(path)
        result = _process_queue_item(
            queue_object=rel,
            queue_payload=payload,
            vector_store=vector_store,
            runtime=runtime,
            dry_run=dry_run,
            batch_size=batch_size,
        )
        ok = len(result.get("errors") or []) == 0
        if not ok:
            failures += 1
        summary = {
            "schema_version": "v1",
            "status": "succeeded" if ok else "failed",
            "bucket": bucket,
            "queue_object": rel,
            "processed_at": _now(),
            "dry_run": dry_run,
            "result": result,
        }
        _finalize_queue_item(
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
        print(f"[ASSET-EMBED] ERROR: {exc}", file=sys.stderr)
        raise

