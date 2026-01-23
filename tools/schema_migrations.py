from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Tuple

DATASET_INFO_SCHEMA_VERSION = "1.0.0"
MANIFEST_SCHEMA_VERSION = "1.2"

SUPPORTED_DATASET_INFO_VERSIONS = {DATASET_INFO_SCHEMA_VERSION}
SUPPORTED_MANIFEST_VERSIONS = {MANIFEST_SCHEMA_VERSION}

logger = logging.getLogger(__name__)


class SchemaMigrationError(ValueError):
    """Raised when a payload cannot be migrated to a supported schema version."""


@dataclass(frozen=True)
class MigrationResult:
    payload: Dict[str, Any]
    applied_steps: List[str]
    original_version: str
    target_version: str


def _normalize_schema_version(version: str) -> Tuple[str, List[str]]:
    if version == "1.0":
        return DATASET_INFO_SCHEMA_VERSION, ["normalize-schema-version-1.0-to-1.0.0"]
    return version, []


def migrate_dataset_info_payload(payload: Mapping[str, Any]) -> MigrationResult:
    if not isinstance(payload, Mapping):
        raise SchemaMigrationError("dataset_info payload must be a mapping")

    payload_copy = copy.deepcopy(dict(payload))
    applied_steps: List[str] = []
    version = payload_copy.get("schema_version")
    if version:
        normalized_version, normalization_steps = _normalize_schema_version(str(version))
        applied_steps.extend(normalization_steps)
        payload_copy["schema_version"] = normalized_version
        version = normalized_version
    elif "version" in payload_copy and "format" in payload_copy:
        payload_copy = _migrate_legacy_dataset_info_payload(payload_copy)
        applied_steps.append("migrate-legacy-dataset-info-version-field")
        version = payload_copy.get("schema_version")
    else:
        raise SchemaMigrationError("dataset_info payload missing schema_version")

    if version not in SUPPORTED_DATASET_INFO_VERSIONS:
        raise SchemaMigrationError(f"unsupported dataset_info schema_version: {version}")

    return MigrationResult(
        payload=payload_copy,
        applied_steps=applied_steps,
        original_version=str(payload.get("schema_version") or payload.get("version") or "unknown"),
        target_version=DATASET_INFO_SCHEMA_VERSION,
    )


def _migrate_legacy_dataset_info_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    legacy_episodes = payload.get("episodes")
    episodes_list = legacy_episodes if isinstance(legacy_episodes, list) else []
    total_episodes = payload.get("total_episodes")
    if total_episodes is None:
        total_episodes = legacy_episodes if isinstance(legacy_episodes, int) else len(episodes_list)

    skipped_episodes = payload.get("skipped_episodes")
    if skipped_episodes is None:
        skipped_episodes = payload.get("skipped", 0) if isinstance(payload.get("skipped"), int) else 0

    return {
        "dataset_type": payload.get("format", "lerobot"),
        "format_version": "1.0",
        "schema_version": DATASET_INFO_SCHEMA_VERSION,
        "scene_id": payload.get("scene_id", "unknown"),
        "job_id": payload.get("job_id"),
        "run_id": payload.get("run_id") or payload.get("job_id") or "legacy",
        "pipeline_commit": payload.get("pipeline_commit", "unknown"),
        "export_schema_version": payload.get("export_schema_version")
        or payload.get("version", "unknown"),
        "episodes": episodes_list,
        "total_frames": int(payload.get("total_frames", 0) or 0),
        "average_quality_score": float(payload.get("average_quality_score", 0.0) or 0.0),
        "source": payload.get("source", "legacy"),
        "converted_at": payload.get("converted_at") or payload.get("exported_at") or "unknown",
        "total_episodes": int(total_episodes or 0),
        "skipped_episodes": int(skipped_episodes or 0),
        "legacy_payload": payload,
    }


def migrate_import_manifest_payload(payload: Mapping[str, Any]) -> MigrationResult:
    if not isinstance(payload, Mapping):
        raise SchemaMigrationError("import_manifest payload must be a mapping")

    payload_copy = copy.deepcopy(dict(payload))
    applied_steps: List[str] = []
    version = payload_copy.get("schema_version")
    if version == MANIFEST_SCHEMA_VERSION:
        return MigrationResult(
            payload=payload_copy,
            applied_steps=applied_steps,
            original_version=str(version),
            target_version=MANIFEST_SCHEMA_VERSION,
        )

    if version == "0.1.0":
        payload_copy["schema_version"] = MANIFEST_SCHEMA_VERSION
        payload_copy.setdefault(
            "schema_definition",
            {
                "version": MANIFEST_SCHEMA_VERSION,
                "description": "Migrated legacy import manifest schema.",
                "fields": {"schema_version": "Schema version string."},
                "notes": ["Migrated from 0.1.0; fields may be incomplete."],
            },
        )
        applied_steps.append("migrate-import-manifest-0.1.0-to-1.2")
        return MigrationResult(
            payload=payload_copy,
            applied_steps=applied_steps,
            original_version="0.1.0",
            target_version=MANIFEST_SCHEMA_VERSION,
        )

    raise SchemaMigrationError(f"unsupported import_manifest schema_version: {version}")
