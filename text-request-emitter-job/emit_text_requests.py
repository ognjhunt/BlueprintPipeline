#!/usr/bin/env python3
"""Emit autonomous text scene_request payloads for source-orchestrator."""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore[assignment]

try:
    from google.api_core.exceptions import Conflict, PreconditionFailed
    from google.cloud import storage
except Exception:  # pragma: no cover
    Conflict = Exception  # type: ignore[assignment]
    PreconditionFailed = Exception  # type: ignore[assignment]
    storage = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from tools.source_pipeline.prompt_engine import (  # noqa: E402
    build_prompt_constraints_metadata,
    generate_prompt,
    load_prompt_matrix,
)

logger = logging.getLogger(__name__)
GCS_ROOT = Path("/mnt/gcs")
JOB_NAME = "text-request-emitter-job"


def _is_truthy(raw: Optional[str], default: bool = False) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _load_json(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
    if not path.is_file():
        return dict(default)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return dict(default)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _resolve_run_date(timezone_name: str, override: str) -> str:
    if override:
        return override
    now = datetime.utcnow()
    if ZoneInfo is not None:
        try:
            now = datetime.now(ZoneInfo(timezone_name))
        except Exception:
            now = datetime.utcnow()
    return now.strftime("%Y-%m-%d")


def _build_scene_id(run_date: str, slot_index: int, prompt_hash: str) -> str:
    date_token = run_date.replace("-", "")
    return f"textauto_{date_token}_{slot_index:03d}_{prompt_hash[:8]}"


def _trim_recent_prompts(recent_prompts: List[Dict[str, Any]], dedupe_window: int) -> List[Dict[str, Any]]:
    if dedupe_window < 1:
        dedupe_window = 1
    return recent_prompts[-dedupe_window:]


def _acquire_lock_via_gcs_api(bucket: str, lock_object: str, payload: Dict[str, Any]) -> Tuple[bool, str]:
    if storage is None:
        raise RuntimeError("google-cloud-storage is unavailable")

    client = storage.Client()
    blob = client.bucket(bucket).blob(lock_object)
    blob.upload_from_string(
        json.dumps(payload),
        if_generation_match=0,
        content_type="application/json",
    )
    return True, "gcs"


def _acquire_lock_via_filesystem(lock_path: Path, payload: Dict[str, Any]) -> Tuple[bool, str]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(lock_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError:
        return False, "filesystem"

    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return True, "filesystem"


def _acquire_daily_lock(
    *,
    bucket: str,
    lock_object: str,
    lock_payload: Dict[str, Any],
) -> Tuple[bool, str]:
    mode = os.getenv("TEXT_AUTONOMY_STORAGE_MODE", "auto").strip().lower()

    if mode in {"auto", "gcs"}:
        try:
            return _acquire_lock_via_gcs_api(bucket, lock_object, lock_payload)
        except (PreconditionFailed, Conflict):
            return False, "gcs"
        except Exception as exc:
            if mode == "gcs":
                raise
            logger.warning("[TEXT-AUTONOMY] GCS lock unavailable; falling back to filesystem lock: %s", exc)

    lock_path = GCS_ROOT / lock_object
    return _acquire_lock_via_filesystem(lock_path, lock_payload)


def main() -> int:
    bucket = os.getenv("BUCKET", "").strip()
    if not bucket:
        raise ValueError("BUCKET is required")
    state_prefix = os.getenv("TEXT_AUTONOMY_STATE_PREFIX", "automation/text_daily").strip("/")
    provider_policy = os.getenv("TEXT_AUTONOMY_PROVIDER_POLICY", "openai_primary").strip() or "openai_primary"
    text_backend = os.getenv("TEXT_AUTONOMY_TEXT_BACKEND", "sage").strip().lower() or "sage"
    if text_backend not in {"internal", "scenesmith", "sage", "hybrid_serial"}:
        raise ValueError(
            "TEXT_AUTONOMY_TEXT_BACKEND must be internal|scenesmith|sage|hybrid_serial, "
            f"got {text_backend!r}"
        )
    quality_tier = os.getenv("TEXT_AUTONOMY_QUALITY_TIER", "premium").strip().lower() or "premium"
    if quality_tier not in {"standard", "premium"}:
        raise ValueError(f"TEXT_AUTONOMY_QUALITY_TIER must be standard|premium, got {quality_tier!r}")

    quota = int(os.getenv("TEXT_DAILY_QUOTA", "1"))
    if quota < 1:
        raise ValueError(f"TEXT_DAILY_QUOTA must be >= 1, got {quota}")
    if quota > 100:
        raise ValueError(f"TEXT_DAILY_QUOTA exceeds safety ceiling of 100, got {quota}")

    seed_count = int(os.getenv("TEXT_AUTONOMY_SEED_COUNT", "1"))
    if seed_count < 1:
        raise ValueError(f"TEXT_AUTONOMY_SEED_COUNT must be >= 1, got {seed_count}")

    allow_image_fallback = _is_truthy(os.getenv("TEXT_AUTONOMY_ALLOW_IMAGE_FALLBACK"), default=False)
    timezone_name = os.getenv("TEXT_AUTONOMY_TIMEZONE", "America/New_York")
    run_date = _resolve_run_date(timezone_name, os.getenv("TEXT_AUTONOMY_RUN_DATE", "").strip())

    pause_object = f"{state_prefix}/.paused"
    state_object = f"{state_prefix}/state.json"
    lock_object = f"{state_prefix}/locks/{run_date}.lock"
    emitted_index_object = f"{state_prefix}/runs/{run_date}/emitted_requests.json"
    emit_manifest_object = f"{state_prefix}/runs/{run_date}/emit_manifest.json"

    pause_path = GCS_ROOT / pause_object
    if pause_path.is_file():
        logger.warning("[TEXT-AUTONOMY] Paused marker present (%s); skipping emission", pause_object)
        return 0

    state_path = GCS_ROOT / state_object
    state = _load_json(
        state_path,
        default={
            "schema_version": "v1",
            "consecutive_failures": 0,
            "paused": False,
            "recent_prompts": [],
        },
    )
    if bool(state.get("paused")):
        logger.warning("[TEXT-AUTONOMY] State is paused; skipping emission")
        return 0

    lock_payload = {
        "schema_version": "v1",
        "job": JOB_NAME,
        "run_date": run_date,
        "bucket": bucket,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    lock_acquired, lock_backend = _acquire_daily_lock(
        bucket=bucket,
        lock_object=lock_object,
        lock_payload=lock_payload,
    )
    if not lock_acquired:
        logger.info("[TEXT-AUTONOMY] Lock already exists for %s (%s); skipping", run_date, lock_backend)
        return 0

    prompt_matrix = load_prompt_matrix()
    dedupe_window = int(prompt_matrix.get("dedupe_window", 60))

    recent_prompts = list(state.get("recent_prompts") or [])
    emitted_entries: List[Dict[str, Any]] = []

    for slot_index in range(1, quota + 1):
        prompt_result = generate_prompt(
            run_date=run_date,
            slot_index=slot_index,
            provider_policy=provider_policy,
            recent_prompts=recent_prompts,
        )

        scene_id = _build_scene_id(run_date, slot_index, prompt_result.prompt_hash)
        request_object = f"scenes/{scene_id}/prompts/scene_request.json"

        constraints: Dict[str, Any] = {
            "room_type": prompt_result.dimensions.get("archetype", "generic_room"),
            "autonomy": {
                "mode": "text_daily",
                "run_date": run_date,
                "slot_index": slot_index,
                "job": JOB_NAME,
            },
        }
        constraints.update(build_prompt_constraints_metadata(prompt_result))

        request_payload: Dict[str, Any] = {
            "schema_version": "v1",
            "scene_id": scene_id,
            "source_mode": "text",
            "text_backend": text_backend,
            "prompt": prompt_result.prompt,
            "quality_tier": quality_tier,
            "seed_count": seed_count,
            "constraints": constraints,
            "provider_policy": provider_policy,
            "fallback": {
                "allow_image_fallback": allow_image_fallback,
            },
        }

        _write_json(GCS_ROOT / request_object, request_payload)

        emitted_entries.append(
            {
                "scene_id": scene_id,
                "request_object": request_object,
                "prompt_hash": prompt_result.prompt_hash,
                "novelty_score": prompt_result.novelty_score,
                "novelty_override": prompt_result.novelty_override,
                "tags": prompt_result.tags,
                "dimensions": prompt_result.dimensions,
                "used_llm": prompt_result.used_llm,
                "llm_attempts": prompt_result.llm_attempts,
                "llm_provider": prompt_result.llm_provider,
                "llm_failure_reason": prompt_result.llm_failure_reason,
            }
        )

        recent_prompts.append(
            {
                "scene_id": scene_id,
                "prompt": prompt_result.prompt,
                "base_prompt": prompt_result.base_prompt,
                "prompt_hash": prompt_result.prompt_hash,
                "novelty_score": prompt_result.novelty_score,
                "tags": prompt_result.tags,
                "dimensions": prompt_result.dimensions,
                "used_llm": prompt_result.used_llm,
                "llm_attempts": prompt_result.llm_attempts,
                "llm_failure_reason": prompt_result.llm_failure_reason,
                "generated_at": datetime.utcnow().isoformat() + "Z",
            }
        )

    recent_prompts = _trim_recent_prompts(recent_prompts, dedupe_window)
    state["schema_version"] = "v1"
    state["recent_prompts"] = recent_prompts
    state["last_emit_run_date"] = run_date
    state["last_emitted_at"] = datetime.utcnow().isoformat() + "Z"
    _write_json(state_path, state)

    emitted_index_payload = {
        "schema_version": "v1",
        "run_date": run_date,
        "bucket": bucket,
        "scene_ids": [entry["scene_id"] for entry in emitted_entries],
        "requests": emitted_entries,
    }
    _write_json(GCS_ROOT / emitted_index_object, emitted_index_payload)

    emit_manifest_payload = {
        "schema_version": "v1",
        "job": JOB_NAME,
        "status": "emitted",
        "run_date": run_date,
        "quota": quota,
        "quality_tier": quality_tier,
        "text_backend": text_backend,
        "provider_policy": provider_policy,
        "state_object": state_object,
        "emitted_index_object": emitted_index_object,
        "lock_object": lock_object,
        "lock_backend": lock_backend,
        "emitted_count": len(emitted_entries),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    _write_json(GCS_ROOT / emit_manifest_object, emit_manifest_payload)

    logger.info(
        "[TEXT-AUTONOMY] emitted_count=%s run_date=%s quota=%s quality_tier=%s text_backend=%s provider_policy=%s index=%s",
        len(emitted_entries),
        run_date,
        quota,
        quality_tier,
        text_backend,
        provider_policy,
        emitted_index_object,
    )
    return 0


if __name__ == "__main__":
    from tools.logging_config import init_logging

    init_logging()
    raise SystemExit(main())
