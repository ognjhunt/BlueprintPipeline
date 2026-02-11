import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from import_manifest_utils import (
    ENV_SNAPSHOT_KEYS,
    MANIFEST_SCHEMA_DEFINITION,
    MANIFEST_SCHEMA_VERSION,
    build_directory_checksums,
    build_file_inventory,
    collect_provenance,
    get_lerobot_metadata_paths,
    snapshot_env,
)
from tools.quality.quality_config import ResolvedQualitySettings

from genie_sim_import.constants import MIN_EPISODES_REQUIRED
from genie_sim_import.integrity import _sha256_file, _write_checksums_file
from genie_sim_import.manifest import _load_import_manifest_with_migration

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_CONTRACT_VERSION = "1.0"


def _relative_to_bundle(bundle_root: Path, path: Path) -> str:
    try:
        rel_path = path.resolve().relative_to(bundle_root.resolve())
    except ValueError:
        return path.as_posix()
    rel_str = rel_path.as_posix()
    return rel_str if rel_str else "."


def _resolve_asset_provenance_reference(
    bundle_root: Path,
    output_dir: Path,
    job_metadata: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    local_path = output_dir / "legal" / "asset_provenance.json"
    if local_path.exists():
        return _relative_to_bundle(bundle_root, local_path)
    if job_metadata:
        artifacts = job_metadata.get("artifacts", {})
        bundle = job_metadata.get("bundle", {})
        reference = (
            artifacts.get("asset_provenance")
            or artifacts.get("asset_provenance_path")
            or bundle.get("asset_provenance")
            or bundle.get("asset_provenance_path")
        )
        if reference:
            return reference
    return None


def _normalize_gcs_output_path(path_value: Optional[str]) -> Optional[str]:
    if not path_value:
        return None
    if path_value.startswith("gs://"):
        return path_value
    if path_value.startswith("/mnt/gcs/"):
        return "gs://" + path_value[len("/mnt/gcs/"):]
    return path_value


def _resolve_job_idempotency(job_metadata: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not job_metadata:
        return None
    idempotency = job_metadata.get("idempotency")
    return idempotency if isinstance(idempotency, dict) else None


def _aggregate_metrics_summary(robot_metrics: Dict[str, Any]) -> Dict[str, Any]:
    aggregated_stats: Dict[str, Any] = {}
    backends = set()
    for metrics in robot_metrics.values():
        if not isinstance(metrics, dict):
            continue
        backend = metrics.get("backend")
        if backend:
            backends.add(backend)
        stats = metrics.get("stats")
        if isinstance(stats, dict):
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    aggregated_stats[key] = aggregated_stats.get(key, 0) + value
    if not backends:
        backend = "unknown"
    elif len(backends) == 1:
        backend = next(iter(backends))
    else:
        backend = "mixed"
    return {
        "backend": backend,
        "stats": aggregated_stats,
        "robots": robot_metrics,
    }


def _write_combined_import_manifest(
    output_dir: Path,
    job_id: str,
    gcs_output_path: Optional[str],
    job_metadata: Optional[Dict[str, Any]],
    robot_entries: List[Dict[str, Any]],
    quality_settings: ResolvedQualitySettings,
    *,
    logger: Any,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_gcs_output_path = _normalize_gcs_output_path(gcs_output_path)
    total_downloaded = sum(entry["episodes"]["downloaded"] for entry in robot_entries)
    total_passed = sum(entry["episodes"]["passed_validation"] for entry in robot_entries)
    total_filtered = sum(entry["episodes"]["filtered"] for entry in robot_entries)
    total_parse_failed = sum(entry["episodes"]["parse_failed"] for entry in robot_entries)
    idempotency = _resolve_job_idempotency(job_metadata)
    scene_id = None
    run_id = job_id
    robot_types: List[str] = []
    if isinstance(job_metadata, dict):
        scene_candidate = job_metadata.get("scene_id")
        if isinstance(scene_candidate, str) and scene_candidate.strip():
            scene_id = scene_candidate.strip()
        run_candidate = job_metadata.get("run_id")
        if isinstance(run_candidate, str) and run_candidate.strip():
            run_id = run_candidate.strip()
        generation_params = job_metadata.get("generation_params")
        if isinstance(generation_params, dict):
            robot_list = generation_params.get("robot_types")
            if isinstance(robot_list, list):
                robot_types.extend(
                    item.strip()
                    for item in robot_list
                    if isinstance(item, str) and item.strip()
                )
            single_robot = generation_params.get("robot_type")
            if isinstance(single_robot, str) and single_robot.strip():
                robot_types.append(single_robot.strip())
    robot_types.extend(
        entry.get("robot_type", "").strip()
        for entry in robot_entries
        if isinstance(entry.get("robot_type"), str) and entry.get("robot_type", "").strip()
    )
    robot_types = list(dict.fromkeys(robot_types))
    scene_id = scene_id or "unknown"
    recording_format_counts: Dict[str, int] = {"json": 0, "parquet": 0, "unknown": 0}
    for entry in robot_entries:
        episodes_payload = entry.get("episodes", {})
        per_robot_counts = episodes_payload.get("recording_format_counts", {})
        if not isinstance(per_robot_counts, dict):
            continue
        for key in ("json", "parquet", "unknown"):
            value = per_robot_counts.get(key, 0)
            if isinstance(value, (int, float)):
                recording_format_counts[key] += int(value)
    recordings_format = "unknown"
    if recording_format_counts.get("json", 0) > 0 and recording_format_counts.get("parquet", 0) > 0:
        recordings_format = "mixed"
    elif recording_format_counts.get("json", 0) > 0:
        recordings_format = "json"
    elif recording_format_counts.get("parquet", 0) > 0:
        recordings_format = "parquet"

    weighted_quality_sum = 0.0
    quality_weight_total = 0
    min_quality_scores = []
    max_quality_scores = []
    for entry in robot_entries:
        quality = entry["quality"]
        episodes_downloaded = entry["episodes"]["downloaded"]
        weighted_quality_sum += quality["average_score"] * episodes_downloaded
        quality_weight_total += episodes_downloaded
        if episodes_downloaded > 0:
            min_quality_scores.append(quality["min_score"])
            max_quality_scores.append(quality["max_score"])

    average_quality = (
        weighted_quality_sum / quality_weight_total if quality_weight_total else 0.0
    )
    min_quality = min(min_quality_scores) if min_quality_scores else 0.0
    max_quality = max(max_quality_scores) if max_quality_scores else 0.0
    component_failure_counts: Dict[str, int] = {}
    component_failed_total = 0
    for entry in robot_entries:
        quality = entry.get("quality", {})
        component_failed_total += int(quality.get("component_failed_episodes", 0))
        counts = quality.get("component_failure_counts")
        if isinstance(counts, dict):
            for key, value in counts.items():
                if isinstance(value, (int, float)):
                    component_failure_counts[key] = component_failure_counts.get(key, 0) + int(
                        value
                    )

    normalized_robot_entries = []
    robot_metrics: Dict[str, Any] = {}
    robot_provenance: Dict[str, Any] = {}
    lerobot_metadata_paths: Dict[str, List[str]] = {}
    for entry in robot_entries:
        entry_copy = copy.deepcopy(entry)
        entry_copy["gcs_output_path"] = _normalize_gcs_output_path(
            entry_copy.get("gcs_output_path")
        )
        entry_copy["import_manifest_path"] = _normalize_gcs_output_path(
            entry_copy.get("import_manifest_path")
        )
        normalized_robot_entries.append(entry_copy)
        robot_type = entry_copy.get("robot_type", "unknown")
        output_dir_str = entry_copy.get("output_dir")
        if output_dir_str:
            robot_output_dir = Path(output_dir_str)
            metadata_paths = get_lerobot_metadata_paths(robot_output_dir)
            lerobot_metadata_paths[robot_type] = [
                path.relative_to(robot_output_dir).as_posix()
                for path in metadata_paths
            ]
        manifest_path_str = entry.get("import_manifest_local_path")
        if manifest_path_str:
            manifest_path = Path(manifest_path_str)
            if manifest_path.exists():
                try:
                    manifest_payload = _load_import_manifest_with_migration(
                        manifest_path,
                        logger,
                        required=False,
                    )
                except ValueError as exc:
                    print(
                        "[GENIE-SIM-IMPORT] ⚠️  Failed to load robot manifest "
                        f"{manifest_path}: {exc}"
                    )
                    continue
                if not manifest_payload:
                    print(
                        "[GENIE-SIM-IMPORT] ⚠️  Skipping robot manifest with unsupported "
                        f"schema {manifest_path}."
                    )
                    continue
                robot_metrics_payload = manifest_payload.get("metrics_summary")
                if robot_metrics_payload:
                    robot_metrics[robot_type] = robot_metrics_payload
                robot_provenance_payload = manifest_payload.get("provenance")
                if robot_provenance_payload:
                    robot_provenance[robot_type] = robot_provenance_payload

    metrics_summary = _aggregate_metrics_summary(robot_metrics)
    bundle_root = output_dir.resolve()
    manifest_path = output_dir / "import_manifest.json"
    directory_checksums = build_directory_checksums(
        output_dir, exclude_paths=[manifest_path]
    )
    checksums_path, checksums_signature = _write_checksums_file(
        output_dir,
        directory_checksums,
    )
    checksums_rel_path = checksums_path.relative_to(output_dir).as_posix()
    checksums_entry = {
        "sha256": _sha256_file(checksums_path),
        "size_bytes": checksums_path.stat().st_size,
    }
    checksums_payload = {
        "download_manifest": None,
        "episodes": {},
        "filtered_episodes": [],
        "lerobot": {
            "dataset_info": None,
            "episodes_index": None,
            "episodes": [],
        },
        "metadata": dict(directory_checksums),
        "missing_episode_ids": [],
        "missing_metadata_files": [],
        "episode_files": {},
        "bundle_files": dict(directory_checksums),
    }
    if checksums_signature:
        checksums_payload["signature"] = checksums_signature
    checksums_payload["metadata"][checksums_rel_path] = checksums_entry
    checksums_payload["bundle_files"][checksums_rel_path] = checksums_entry
    file_inventory = build_file_inventory(output_dir, exclude_paths=[manifest_path])
    asset_provenance_path = _resolve_asset_provenance_reference(
        bundle_root=bundle_root,
        output_dir=output_dir,
        job_metadata=job_metadata,
    )
    config_snapshot = {
        "env": snapshot_env(ENV_SNAPSHOT_KEYS),
        "config": {
            "job_id": job_id,
            "output_dir": str(output_dir),
            "gcs_output_path": normalized_gcs_output_path,
            "robots": [entry["robot_type"] for entry in normalized_robot_entries],
        },
        "job_metadata": job_metadata or {},
    }
    base_provenance = {
        "source": "genie_sim",
        "job_id": job_id,
        "imported_by": "BlueprintPipeline",
        "importer": "genie-sim-import-job",
        "client_mode": "local",
    }
    provenance = collect_provenance(REPO_ROOT, config_snapshot)
    provenance.update(base_provenance)
    provenance["robots"] = robot_provenance

    manifest_status = "completed" if all(entry["success"] for entry in normalized_robot_entries) else "failed"
    import_manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "schema_definition": MANIFEST_SCHEMA_DEFINITION,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "scene_id": scene_id,
        "run_id": run_id,
        "status": manifest_status,
        "import_status": manifest_status,
        "robot_types": robot_types,
        "recordings_format": recordings_format,
        "artifact_contract_version": ARTIFACT_CONTRACT_VERSION,
        "job_id": job_id,
        "output_dir": str(output_dir),
        "gcs_output_path": normalized_gcs_output_path,
        "job_idempotency": {
            "key": idempotency.get("key") if idempotency else None,
            "first_submitted_at": idempotency.get("first_submitted_at") if idempotency else None,
        },
        "success": all(entry["success"] for entry in normalized_robot_entries),
        "episodes": {
            "downloaded": total_downloaded,
            "passed_validation": total_passed,
            "filtered": total_filtered,
            "download_errors": 0,
            "parse_failed": total_parse_failed,
            "parse_failures": [],
            "recording_format_counts": recording_format_counts,
            "min_required": min(
                (entry["episodes"].get("min_required", MIN_EPISODES_REQUIRED) for entry in robot_entries),
                default=MIN_EPISODES_REQUIRED,
            ),
        },
        "quality": {
            "average_score": average_quality,
            "min_score": min_quality,
            "max_score": max_quality,
            "threshold": min(
                (entry["quality"]["threshold"] for entry in robot_entries),
                default=0.0,
            ),
            "validation_enabled": any(
                entry["quality"]["validation_enabled"] for entry in normalized_robot_entries
            ),
            "component_thresholds": quality_settings.config.dimension_thresholds,
            "component_failed_episodes": component_failed_total,
            "component_failure_counts": component_failure_counts,
        },
        "quality_config": {
            "min_quality_score": quality_settings.min_quality_score,
            "filter_low_quality": quality_settings.filter_low_quality,
            "dimension_thresholds": quality_settings.config.dimension_thresholds,
            "range": {
                "min_allowed": quality_settings.config.min_allowed,
                "max_allowed": quality_settings.config.max_allowed,
            },
            "defaults": {
                "min_quality_score": quality_settings.config.default_min_quality_score,
                "filter_low_quality": quality_settings.config.default_filter_low_quality,
            },
            "source_path": quality_settings.config.source_path,
        },
        "validation": {
            "episodes": {
                "enabled": any(
                    entry["quality"]["validation_enabled"] for entry in normalized_robot_entries
                ),
                "min_required": min(
                    (
                        entry["episodes"].get("min_required", MIN_EPISODES_REQUIRED)
                        for entry in robot_entries
                    ),
                    default=MIN_EPISODES_REQUIRED,
                ),
                "total_downloaded": total_downloaded,
                "total_passed_validation": total_passed,
            },
        },
        "robots": normalized_robot_entries,
        "job_metadata": job_metadata or {},
        "readme_path": None,
        "checksums_path": _relative_to_bundle(bundle_root, checksums_path),
        "asset_provenance_path": asset_provenance_path,
        "lerobot_metadata_paths": lerobot_metadata_paths,
        "metrics_summary": metrics_summary,
        "provenance": provenance,
        "checksums": checksums_payload,
        "file_inventory": file_inventory,
        "verification": {
            "checksums": {},
        },
        "upload_status": "not_configured",
        "upload_failures": [],
        "upload_started_at": None,
        "upload_completed_at": None,
        "upload_summary": {
            "firebase": None,
            "gcs": None,
        },
    }

    with open(manifest_path, "w") as handle:
        json.dump(import_manifest, handle, indent=2)

    return manifest_path
