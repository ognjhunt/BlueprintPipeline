from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

DATASET_INFO_SCHEMA_VERSION = "1.0.0"
MANIFEST_SCHEMA_VERSION = "1.3"

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
        payload_copy.setdefault("run_id", payload_copy.get("job_id"))
        payload_copy.setdefault("scene_id", "unknown")
        payload_copy.setdefault("robot_types", [])
        payload_copy.setdefault("recordings_format", "unknown")
        payload_copy.setdefault("artifact_contract_version", "1.0")
        applied_steps.append("migrate-import-manifest-0.1.0-to-1.3")
        return MigrationResult(
            payload=payload_copy,
            applied_steps=applied_steps,
            original_version="0.1.0",
            target_version=MANIFEST_SCHEMA_VERSION,
        )

    if version == "1.2":
        payload_copy["schema_version"] = MANIFEST_SCHEMA_VERSION
        payload_copy.setdefault(
            "schema_definition",
            {
                "version": MANIFEST_SCHEMA_VERSION,
                "description": "Migrated import manifest schema.",
                "fields": {"schema_version": "Schema version string."},
                "notes": ["Migrated from 1.2 to 1.3 with backfilled metadata fields."],
            },
        )
        payload_copy["scene_id"] = _infer_manifest_scene_id(payload_copy)
        payload_copy["run_id"] = str(
            payload_copy.get("run_id")
            or (payload_copy.get("job_metadata", {}) or {}).get("run_id")
            or payload_copy.get("job_id")
            or "unknown"
        )
        payload_copy["robot_types"] = _infer_manifest_robot_types(payload_copy)
        payload_copy["recordings_format"] = _infer_recordings_format(payload_copy)
        payload_copy["artifact_contract_version"] = str(
            payload_copy.get("artifact_contract_version") or "1.0"
        )
        applied_steps.append("migrate-import-manifest-1.2-to-1.3")
        return MigrationResult(
            payload=payload_copy,
            applied_steps=applied_steps,
            original_version="1.2",
            target_version=MANIFEST_SCHEMA_VERSION,
        )

    raise SchemaMigrationError(f"unsupported import_manifest schema_version: {version}")


def _infer_manifest_scene_id(payload: Mapping[str, Any]) -> str:
    scene_id = payload.get("scene_id")
    if isinstance(scene_id, str) and scene_id.strip():
        return scene_id.strip()

    provenance = payload.get("provenance")
    if isinstance(provenance, Mapping):
        prov_scene = provenance.get("scene_id")
        if isinstance(prov_scene, str) and prov_scene.strip():
            return prov_scene.strip()

    job_metadata = payload.get("job_metadata")
    if isinstance(job_metadata, Mapping):
        meta_scene = job_metadata.get("scene_id")
        if isinstance(meta_scene, str) and meta_scene.strip():
            return meta_scene.strip()

    env = (payload.get("provenance") or {}).get("config_snapshot", {}).get("env")
    if isinstance(env, Mapping):
        env_scene = env.get("SCENE_ID")
        if isinstance(env_scene, str) and env_scene.strip():
            return env_scene.strip()

    output_dir = payload.get("output_dir")
    if isinstance(output_dir, str):
        segments = [seg for seg in output_dir.split("/") if seg]
        for idx, segment in enumerate(segments):
            if segment == "scenes" and idx + 1 < len(segments):
                candidate = segments[idx + 1]
                if candidate:
                    return candidate

    return "unknown"


def _infer_manifest_robot_types(payload: Mapping[str, Any]) -> List[str]:
    robots: List[str] = []
    robot_entries = payload.get("robots")
    if isinstance(robot_entries, list):
        for entry in robot_entries:
            if not isinstance(entry, Mapping):
                continue
            robot_type = entry.get("robot_type")
            if isinstance(robot_type, str) and robot_type.strip():
                robots.append(robot_type.strip())

    if not robots:
        generation_params = (payload.get("job_metadata") or {}).get("generation_params")
        if isinstance(generation_params, Mapping):
            robot_types = generation_params.get("robot_types")
            if isinstance(robot_types, list):
                robots.extend(
                    rt.strip()
                    for rt in robot_types
                    if isinstance(rt, str) and rt.strip()
                )
            robot_type = generation_params.get("robot_type")
            if isinstance(robot_type, str) and robot_type.strip():
                robots.append(robot_type.strip())

    deduped: List[str] = []
    for robot in robots:
        if robot not in deduped:
            deduped.append(robot)
    return deduped


def _infer_recordings_format(payload: Mapping[str, Any]) -> str:
    recordings_format = payload.get("recordings_format")
    if isinstance(recordings_format, str) and recordings_format.strip():
        return recordings_format.strip()

    episodes = payload.get("episodes")
    if isinstance(episodes, Mapping):
        if "recording_format_counts" in episodes:
            counts = episodes.get("recording_format_counts")
            if isinstance(counts, Mapping):
                if "json" in counts and counts.get("json", 0):
                    return "json"
                if "parquet" in counts and counts.get("parquet", 0):
                    return "parquet"

    file_inventory = payload.get("file_inventory")
    if isinstance(file_inventory, list):
        saw_json = False
        saw_parquet = False
        for entry in file_inventory:
            if not isinstance(entry, Mapping):
                continue
            path = entry.get("path")
            if not isinstance(path, str):
                continue
            if path.endswith(".json"):
                saw_json = True
            if path.endswith(".parquet"):
                saw_parquet = True
        if saw_json and saw_parquet:
            return "mixed"
        if saw_json:
            return "json"
        if saw_parquet:
            return "parquet"

    return "unknown"
