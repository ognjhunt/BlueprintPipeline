import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from import_manifest_utils import compute_manifest_checksum
from tools.utils.atomic_write import write_json_atomic

from tools.schema_migrations import SchemaMigrationError, migrate_import_manifest_payload
from tools.firebase_upload.uploader import (
    get_firebase_storage_bucket,
    get_firebase_upload_mode,
    resolve_firebase_local_upload_root,
)
from tools.firebase_upload.firebase_upload_orchestrator import build_firebase_upload_prefix

from genie_sim_import.integrity import _sha256_file


def _compute_episode_content_manifest(
    recordings_dir: Path,
    episode_id: str,
) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for path in sorted(recordings_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.stem != episode_id:
            continue
        try:
            rel_path = path.relative_to(recordings_dir).as_posix()
        except ValueError:
            rel_path = path.as_posix()
        entries.append(
            {
                "path": rel_path,
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    entries.sort(key=lambda entry: entry["path"])
    return entries


def _compute_episode_content_hash(entries: List[Dict[str, Any]]) -> str:
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _build_episode_hash_index_path(
    *,
    scene_id: str,
    robot_type: Optional[str],
    prefix: str,
    content_hash: str,
) -> str:
    base_prefix = build_firebase_upload_prefix(
        scene_id,
        robot_type=robot_type,
        prefix=prefix,
    )
    return f"{base_prefix}/episode_hash_index/{content_hash}.json"


def _lookup_episode_hash_index(
    *,
    episode_hashes: Dict[str, str],
    scene_id: str,
    robot_type: Optional[str],
    prefix: str,
    log: logging.LoggerAdapter,
) -> Dict[str, str]:
    if not episode_hashes:
        return {}
    upload_mode = get_firebase_upload_mode()
    existing: Dict[str, str] = {}
    try:
        if upload_mode == "local":
            local_root = resolve_firebase_local_upload_root()
            for episode_id, content_hash in episode_hashes.items():
                index_path = _build_episode_hash_index_path(
                    scene_id=scene_id,
                    robot_type=robot_type,
                    prefix=prefix,
                    content_hash=content_hash,
                )
                if (local_root / index_path).exists():
                    existing[episode_id] = content_hash
            return existing

        bucket = get_firebase_storage_bucket()
        for episode_id, content_hash in episode_hashes.items():
            blob_path = _build_episode_hash_index_path(
                scene_id=scene_id,
                robot_type=robot_type,
                prefix=prefix,
                content_hash=content_hash,
            )
            blob = bucket.blob(blob_path)
            if blob.exists():
                existing[episode_id] = content_hash
    except Exception as exc:
        log.warning("Episode hash index lookup failed: %s", exc)
        return {}
    return existing


def _persist_episode_hash_index(
    *,
    episode_hashes: Dict[str, str],
    scene_id: str,
    robot_type: Optional[str],
    prefix: str,
    job_id: str,
    log: logging.LoggerAdapter,
) -> None:
    if not episode_hashes:
        return
    upload_mode = get_firebase_upload_mode()
    payload_template = {
        "scene_id": scene_id,
        "robot_type": robot_type,
        "job_id": job_id,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    if upload_mode == "local":
        local_root = resolve_firebase_local_upload_root()
        for episode_id, content_hash in episode_hashes.items():
            index_path = _build_episode_hash_index_path(
                scene_id=scene_id,
                robot_type=robot_type,
                prefix=prefix,
                content_hash=content_hash,
            )
            full_path = local_root / index_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                **payload_template,
                "episode_id": episode_id,
                "content_hash": content_hash,
            }
            write_json_atomic(full_path, payload, indent=2)
        return

    try:
        bucket = get_firebase_storage_bucket()
    except Exception as exc:
        log.warning("Episode hash index write skipped: %s", exc)
        return
    for episode_id, content_hash in episode_hashes.items():
        blob_path = _build_episode_hash_index_path(
            scene_id=scene_id,
            robot_type=robot_type,
            prefix=prefix,
            content_hash=content_hash,
        )
        blob = bucket.blob(blob_path)
        payload = {
            **payload_template,
            "episode_id": episode_id,
            "content_hash": content_hash,
        }
        blob.upload_from_string(json.dumps(payload, indent=2), content_type="application/json")


def _load_json_file(path: Path) -> Any:
    try:
        with open(path, "r") as handle:
            return json.load(handle)
    except (FileNotFoundError, PermissionError, json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"Failed to load JSON file {path}: {exc}") from exc


def _load_import_manifest_with_migration(
    path: Path,
    log: logging.Logger,
    *,
    required: bool = True,
) -> Optional[Dict[str, Any]]:
    payload = _load_json_file(path)
    try:
        migration = migrate_import_manifest_payload(payload)
    except SchemaMigrationError as exc:
        message = f"Unsupported import_manifest schema version in {path}: {exc}"
        if required:
            raise ValueError(message) from exc
        log.warning(message)
        return None
    if migration.applied_steps:
        log.info(
            "Applied import manifest migrations for %s: %s",
            path,
            ", ".join(migration.applied_steps),
        )
    return migration.payload


def _load_existing_import_manifest(
    output_dir: Path,
    log: logging.Logger,
) -> Optional[Dict[str, Any]]:
    manifest_path = output_dir / "import_manifest.json"
    if not manifest_path.exists():
        return None
    return _load_import_manifest_with_migration(
        manifest_path,
        log,
        required=True,
    )


def _resolve_manifest_import_status(import_manifest: Dict[str, Any]) -> str:
    status = import_manifest.get("import_status")
    if isinstance(status, str):
        return status.strip().lower()
    success = import_manifest.get("success")
    if isinstance(success, bool):
        return "success" if success else "failed"
    checksums_verification = import_manifest.get("verification", {}).get("checksums", {})
    if isinstance(checksums_verification, dict):
        checksums_success = checksums_verification.get("success")
        if isinstance(checksums_success, bool):
            return "success" if checksums_success else "failed"
        output_bundle = checksums_verification.get("output_bundle")
        if isinstance(output_bundle, dict):
            checksums_success = output_bundle.get("success")
            if isinstance(checksums_success, bool):
                return "success" if checksums_success else "failed"
    return "unknown"


def _update_import_manifest_status(
    manifest_path: Optional[Path],
    status: str,
    success: Optional[bool] = None,
) -> None:
    if manifest_path is None or not manifest_path.exists():
        return
    with open(manifest_path, "r") as handle:
        import_manifest = json.load(handle)
    import_manifest["import_status"] = status
    if success is not None:
        import_manifest["success"] = success
    import_manifest["status_updated_at"] = datetime.utcnow().isoformat() + "Z"
    checksums = import_manifest.setdefault("checksums", {})
    metadata_checksums = checksums.setdefault("metadata", {})
    metadata_checksums.setdefault("import_manifest.json", {})
    metadata_checksums["import_manifest.json"]["sha256"] = compute_manifest_checksum(
        import_manifest
    )
    write_json_atomic(manifest_path, import_manifest, indent=2)


def _update_import_manifest_firebase_summary(
    manifest_path: Optional[Path],
    firebase_summary: Dict[str, Any],
) -> None:
    if manifest_path is None:
        return
    if not manifest_path.exists():
        print(
            "[GENIE-SIM-IMPORT] ⚠️  Import manifest not found; "
            "skipping Firebase summary update."
        )
        return
    with open(manifest_path, "r") as handle:
        import_manifest = json.load(handle)
    import_manifest["firebase_upload"] = firebase_summary
    firebase_verification = firebase_summary.get("firebase_verification")
    if firebase_verification is not None:
        import_manifest.setdefault("verification", {})[
            "firebase_upload"
        ] = firebase_verification
    checksums = import_manifest.setdefault("checksums", {})
    metadata_checksums = checksums.setdefault("metadata", {})
    metadata_checksums.setdefault("import_manifest.json", {})
    metadata_checksums["import_manifest.json"]["sha256"] = compute_manifest_checksum(
        import_manifest
    )
    write_json_atomic(manifest_path, import_manifest, indent=2)


def _update_import_manifest_dedup_summary(
    manifest_path: Optional[Path],
    dedup_summary: Dict[str, Any],
) -> None:
    if manifest_path is None:
        return
    if not manifest_path.exists():
        print(
            "[GENIE-SIM-IMPORT] ⚠️  Import manifest not found; "
            "skipping dedup summary update."
        )
        return
    with open(manifest_path, "r") as handle:
        import_manifest = json.load(handle)
    import_manifest["deduplication"] = dedup_summary
    episodes_summary = import_manifest.setdefault("episodes", {})
    episodes_summary["deduplicated"] = dedup_summary.get("deduplicated_count", 0)
    episodes_summary["deduplicated_ids"] = dedup_summary.get(
        "deduplicated_episode_ids", []
    )
    episodes_summary["deduplicated_filtered_from_outputs"] = dedup_summary.get(
        "filtered_from_outputs",
        False,
    )
    episodes_summary["deduplicated_filtered"] = dedup_summary.get(
        "filtered_count",
        0,
    )
    episodes_summary["deduplicated_filtered_ids"] = dedup_summary.get(
        "filtered_episode_ids",
        [],
    )
    post_dedup_passed = dedup_summary.get("post_dedup_passed_validation")
    if isinstance(post_dedup_passed, int):
        episodes_summary["passed_validation"] = post_dedup_passed
    post_dedup_filtered = dedup_summary.get("post_dedup_filtered")
    if isinstance(post_dedup_filtered, int):
        episodes_summary["filtered"] = post_dedup_filtered
    post_dedup_converted = dedup_summary.get("post_dedup_lerobot_converted_count")
    if isinstance(post_dedup_converted, int):
        lerobot_summary = import_manifest.setdefault("lerobot", {})
        lerobot_summary["converted_count"] = post_dedup_converted
    post_dedup_frames = dedup_summary.get("post_dedup_lerobot_total_frames")
    if isinstance(post_dedup_frames, int):
        lerobot_summary = import_manifest.setdefault("lerobot", {})
        lerobot_summary["total_frames"] = post_dedup_frames
    checksums = import_manifest.setdefault("checksums", {})
    metadata_checksums = checksums.setdefault("metadata", {})
    metadata_checksums.setdefault("import_manifest.json", {})
    metadata_checksums["import_manifest.json"]["sha256"] = compute_manifest_checksum(
        import_manifest
    )
    write_json_atomic(manifest_path, import_manifest, indent=2)
