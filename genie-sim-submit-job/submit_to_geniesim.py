#!/usr/bin/env python3
"""
Genie Sim Submission Job.

Submits a Genie Sim generation job using the export bundle produced by
`genie-sim-export-job/` and persists the resulting job ID to GCS.

Environment Variables:
    BUCKET: GCS bucket name
    SCENE_ID: Scene identifier
    GENIESIM_PREFIX: Prefix where export bundle is stored (default: scenes/<scene>/geniesim)
    JOB_OUTPUT_PATH: GCS path to write job metadata (default: scenes/<scene>/geniesim/job.json)
    ROBOT_TYPE: Robot type (default: franka)
    EPISODES_PER_TASK: Episodes per task (default: tools/config/pipeline_config.json)
    NUM_VARIATIONS: Scene variations (default: 5)
    MIN_QUALITY_SCORE: Minimum quality score (default: 0.85)
"""

import hashlib
import json
import logging
import os
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from google.cloud import storage

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.path.insert(0, str(REPO_ROOT / "tools"))
from geniesim_adapter.local_framework import (
    DataCollectionResult,
    GenieSimConfig,
    GenieSimLocalFramework,
    run_geniesim_preflight_or_exit,
)
from tools.metrics.pipeline_metrics import get_metrics
from tools.gcs_upload import (
    calculate_md5_base64,
    upload_blob_from_filename,
    verify_blob_upload,
)
from tools.config import load_pipeline_config
from tools.validation.entrypoint_checks import validate_required_env_vars

EXPECTED_EXPORT_SCHEMA_VERSION = "1.0.0"
EXPECTED_GENIESIM_SERVER_VERSION = "3.0.0"
REQUIRED_GENIESIM_CAPABILITIES = {
    "data_collection",
    "recording",
    "observation",
    "environment_reset",
}
CONTRACT_SCHEMAS = {
    "scene_graph": "scene_graph.schema.json",
    "asset_index": "asset_index.schema.json",
    "task_config": "task_config.schema.json",
}


@dataclass(frozen=True)
class LocalGenerationParams:
    episodes_per_task: int
    num_variations: int
    robot_type: str
    min_quality_score: float


@dataclass(frozen=True)
class LocalJobMetricsBuilder:
    generation_params: LocalGenerationParams
    task_config: Dict[str, Any]

    def _total_episodes(self) -> int:
        task_count = len(self.task_config.get("tasks", [])) or 1
        return max(1, self.generation_params.episodes_per_task * task_count)

    @staticmethod
    def _duration_seconds(created_at: Optional[str], completed_at: Optional[str]) -> Optional[float]:
        if not created_at or not completed_at:
            return None
        try:
            created_dt = datetime.fromisoformat(created_at.replace("Z", ""))
            completed_dt = datetime.fromisoformat(completed_at.replace("Z", ""))
        except ValueError:
            return None
        return max(0.0, (completed_dt - created_dt).total_seconds())

    @staticmethod
    def _quality_pass_rate(
        episodes_collected: Optional[int],
        episodes_passed: Optional[int],
    ) -> Optional[float]:
        if episodes_collected and episodes_collected > 0 and episodes_passed is not None:
            return episodes_passed / episodes_collected
        return None

    def build(
        self,
        *,
        job_id: str,
        created_at: str,
        completed_at: Optional[str],
        status: str,
        episodes_collected: Optional[int],
        episodes_passed: Optional[int],
        failure_reason: Optional[str],
        failure_details: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        total_episodes = self._total_episodes()
        return {
            "job_id": job_id,
            "status": status,
            "created_at": created_at,
            "completed_at": completed_at,
            "duration_seconds": self._duration_seconds(created_at, completed_at),
            "total_episodes": total_episodes,
            "episodes_collected": episodes_collected,
            "episodes_passed": episodes_passed,
            "quality_pass_rate": self._quality_pass_rate(episodes_collected, episodes_passed),
            "failure_reason": failure_reason,
            "failure_details": failure_details,
        }


def _write_local_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _hash_payload(payload: Dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _build_idempotency_key(scene_id: str, task_config_hash: str, export_manifest_hash: str) -> str:
    payload = {
        "scene_id": scene_id,
        "task_config_hash": task_config_hash,
        "export_manifest_hash": export_manifest_hash,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _read_optional_json_blob(
    client: storage.Client,
    bucket: str,
    blob_name: str,
) -> Optional[Dict[str, Any]]:
    blob = client.bucket(bucket).blob(blob_name)
    if not blob.exists():
        return None
    return json.loads(blob.download_as_text())


def _read_json_blob(client: storage.Client, bucket: str, blob_name: str) -> Dict[str, Any]:
    blob = client.bucket(bucket).blob(blob_name)
    if not blob.exists():
        raise FileNotFoundError(f"Missing required export bundle file: gs://{bucket}/{blob_name}")
    return json.loads(blob.download_as_text())


def _write_json_blob(client: storage.Client, bucket: str, blob_name: str, payload: Dict[str, Any]) -> None:
    blob = client.bucket(bucket).blob(blob_name)
    payload_json = json.dumps(payload, indent=2)
    payload_bytes = payload_json.encode("utf-8")
    blob.upload_from_string(payload_json, content_type="application/json")
    gcs_uri = f"gs://{bucket}/{blob_name}"
    verified, failure_reason = verify_blob_upload(
        blob,
        gcs_uri=gcs_uri,
        expected_size=len(payload_bytes),
        expected_md5=calculate_md5_base64(payload_bytes),
        logger=logging.getLogger("genie-sim-submit-job"),
    )
    if not verified:
        raise RuntimeError(f"GCS upload verification failed for {gcs_uri}: {failure_reason}")


def _write_failure_marker(
    client: storage.Client,
    bucket: str,
    geniesim_prefix: str,
    payload: Dict[str, Any],
) -> None:
    _write_json_blob(client, bucket, f"{geniesim_prefix}/.failed", payload)


def _resolve_episodes_per_task() -> int:
    env_value = os.getenv("EPISODES_PER_TASK")
    if env_value is not None:
        return int(env_value)
    pipeline_config = load_pipeline_config()
    return int(pipeline_config.episode_generation.episodes_per_task)


def _find_scene_usd_path(
    *,
    scene_manifest: Dict[str, Any],
    scene_id: str,
    geniesim_prefix: str,
    gcs_root: Path,
    local_root: Path,
    use_gcs_fuse: bool,
) -> Optional[Path]:
    existing_path = scene_manifest.get("usd_path")
    if existing_path:
        candidate = Path(existing_path)
        if candidate.is_file():
            return candidate
        if not candidate.is_absolute():
            relative_candidate = (gcs_root if use_gcs_fuse else local_root) / candidate
            if relative_candidate.is_file():
                return relative_candidate

    base_root = gcs_root if use_gcs_fuse else local_root
    candidates = []
    for usd_filename in ("scene.usda", "scene.usd", "scene.usdc"):
        candidates.extend(
            [
                base_root / "scenes" / scene_id / "usd" / usd_filename,
                base_root / geniesim_prefix / "usd" / usd_filename,
            ]
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _parse_version(version: str) -> tuple:
    parts = version.split(".")
    padded = (parts + ["0", "0", "0"])[:3]
    parsed = []
    for part in padded:
        if part.lower() in {"x", "*"}:
            parsed.append(None)
        else:
            parsed.append(int(part))
    return tuple(parsed)


def _version_tuple(version: str) -> tuple:
    major, minor, patch = _parse_version(version)
    return (major or 0, minor or 0, patch or 0)


def _is_version_compatible(expected_version: str, min_version: str, max_version: str) -> bool:
    expected = _version_tuple(expected_version)
    min_required = _version_tuple(min_version)
    max_parsed = _parse_version(max_version)
    if expected < min_required:
        return False
    if max_parsed[1] is None or max_parsed[2] is None:
        return expected[0] == (max_parsed[0] or expected[0])
    max_required = _version_tuple(max_version)
    return expected <= max_required


def _validate_export_marker(marker: Dict[str, Any]) -> None:
    export_schema = marker.get("export_schema_version")
    geniesim_schema = marker.get("geniesim_schema_version")
    compatibility = marker.get("schema_compatibility", {})

    if not export_schema:
        raise RuntimeError("Missing export_schema_version in _GENIESIM_EXPORT_COMPLETE.")
    if export_schema != EXPECTED_EXPORT_SCHEMA_VERSION:
        raise RuntimeError(
            "Export schema mismatch: expected "
            f"{EXPECTED_EXPORT_SCHEMA_VERSION}, found {export_schema}."
        )

    if not geniesim_schema:
        raise RuntimeError("Missing geniesim_schema_version in _GENIESIM_EXPORT_COMPLETE.")
    if _parse_version(geniesim_schema)[0] != _parse_version(EXPECTED_GENIESIM_SERVER_VERSION)[0]:
        raise RuntimeError(
            "Genie Sim server version incompatibility: "
            f"expected major {EXPECTED_GENIESIM_SERVER_VERSION}, found {geniesim_schema}."
        )

    min_version = compatibility.get("min_geniesim_version")
    max_version = compatibility.get("max_geniesim_version")
    if not min_version or not max_version:
        raise RuntimeError(
            "Missing schema_compatibility ranges in _GENIESIM_EXPORT_COMPLETE."
        )
    if not _is_version_compatible(EXPECTED_GENIESIM_SERVER_VERSION, min_version, max_version):
        raise RuntimeError(
            "Genie Sim server version incompatibility: expected "
            f"{EXPECTED_GENIESIM_SERVER_VERSION} to be within [{min_version}, {max_version}]."
        )


def _load_contract_schema(schema_name: str) -> Dict[str, Any]:
    schema_path = REPO_ROOT / "fixtures" / "contracts" / schema_name
    if not schema_path.exists():
        raise RuntimeError(f"Missing contract schema file: {schema_path}")
    return json.loads(schema_path.read_text())


def _validate_minimal_schema(payload: Any, schema: Dict[str, Any], path: str) -> None:
    schema_type = schema.get("type")
    if schema_type == "object":
        if not isinstance(payload, dict):
            raise ValueError(f"{path}: expected object")
        for key in schema.get("required", []):
            if key not in payload:
                raise ValueError(f"{path}: missing required field '{key}'")
        for key, prop_schema in schema.get("properties", {}).items():
            if key in payload:
                _validate_minimal_schema(payload[key], prop_schema, f"{path}.{key}")
    elif schema_type == "array":
        if not isinstance(payload, list):
            raise ValueError(f"{path}: expected array")
        min_items = schema.get("minItems")
        max_items = schema.get("maxItems")
        if min_items is not None and len(payload) < min_items:
            raise ValueError(f"{path}: expected at least {min_items} items")
        if max_items is not None and len(payload) > max_items:
            raise ValueError(f"{path}: expected at most {max_items} items")
        items_schema = schema.get("items")
        if items_schema:
            for idx, item in enumerate(payload):
                _validate_minimal_schema(item, items_schema, f"{path}[{idx}]")
    elif schema_type == "string":
        if not isinstance(payload, str):
            raise ValueError(f"{path}: expected string")
        enum = schema.get("enum")
        if enum and payload not in enum:
            raise ValueError(f"{path}: value '{payload}' not in enum {enum}")
    elif schema_type == "integer":
        if not isinstance(payload, int):
            raise ValueError(f"{path}: expected integer")
    elif schema_type == "number":
        if not isinstance(payload, (int, float)):
            raise ValueError(f"{path}: expected number")
    elif schema_type == "boolean":
        if not isinstance(payload, bool):
            raise ValueError(f"{path}: expected boolean")
    elif isinstance(schema_type, list):
        if not any(_type_matches(payload, schema_type_option) for schema_type_option in schema_type):
            raise ValueError(f"{path}: expected one of types {schema_type}")


def _type_matches(payload: Any, schema_type: str) -> bool:
    if schema_type == "object":
        return isinstance(payload, dict)
    if schema_type == "array":
        return isinstance(payload, list)
    if schema_type == "string":
        return isinstance(payload, str)
    if schema_type == "integer":
        return isinstance(payload, int)
    if schema_type == "number":
        return isinstance(payload, (int, float))
    if schema_type == "boolean":
        return isinstance(payload, bool)
    if schema_type == "null":
        return payload is None
    return False


def _validate_json_schema(payload: Any, schema: Dict[str, Any]) -> None:
    try:
        import jsonschema  # type: ignore
    except ImportError:
        _validate_minimal_schema(payload, schema, path="$")
    else:
        jsonschema.validate(instance=payload, schema=schema)


def _validate_bundle_schemas(payloads: Dict[str, Any]) -> None:
    errors = []
    for label, payload in payloads.items():
        schema_name = CONTRACT_SCHEMAS.get(label)
        if not schema_name:
            continue
        schema = _load_contract_schema(schema_name)
        try:
            _validate_json_schema(payload, schema)
        except Exception as exc:
            errors.append(f"{label}.json: {exc}")
    if errors:
        error_text = "\n".join(f"- {error}" for error in errors)
        raise RuntimeError(
            "Export bundle schema validation failed:\n"
            f"{error_text}"
        )


def _handshake_geniesim_server(
    framework: GenieSimLocalFramework,
    *,
    expected_server_version: str,
    required_capabilities: set[str],
    connection_hint: str,
    startup_hint: str,
) -> Dict[str, Any]:
    try:
        server_info = framework.verify_server_capabilities(
            expected_server_version=expected_server_version,
            required_capabilities=sorted(required_capabilities),
        )
    except RuntimeError as exc:
        required_list = ", ".join(sorted(required_capabilities))
        raise RuntimeError(
            "Genie Sim server capability handshake failed. "
            f"Required capabilities: {required_list}. "
            "Ensure the server exposes the GET_CHECKER_STATUS endpoint and "
            f"supports Genie Sim API {expected_server_version}. "
            f"Connection: {connection_hint}. "
            f"Start the server with: {startup_hint}. "
            f"Details: {exc}"
        ) from exc
    return server_info


def _run_local_data_collection_with_handshake(
    *,
    scene_manifest: Dict[str, Any],
    task_config: Dict[str, Any],
    output_dir: Path,
    robot_type: str,
    episodes_per_task: int,
    expected_server_version: str,
    required_capabilities: set[str],
    verbose: bool = True,
) -> DataCollectionResult:
    config = GenieSimConfig.from_env()
    config.robot_type = robot_type
    config.episodes_per_task = episodes_per_task
    config.recording_dir = output_dir / "recordings"
    framework = GenieSimLocalFramework(config, verbose=verbose)
    connection_hint = f"{config.host}:{config.port}"
    startup_hint = (
        f"{config.isaac_sim_path}/python.sh "
        f"{config.geniesim_root}/source/data_collection/scripts/data_collector_server.py "
        f"--headless --port {config.port}"
    )
    scene_usd = scene_manifest.get("usd_path")
    server_info: Dict[str, Any] = {}

    def _handshake_failure(message: str) -> DataCollectionResult:
        result = DataCollectionResult(
            success=False,
            task_name=task_config.get("name", "unknown"),
        )
        result.errors.append(message)
        return result

    if framework.is_server_running():
        if not framework.connect():
            return _handshake_failure(
                "Unable to connect to the Genie Sim server. "
                f"Verify the server is reachable at {connection_hint}."
            )
        try:
            server_info = _handshake_geniesim_server(
                framework,
                expected_server_version=expected_server_version,
                required_capabilities=required_capabilities,
                connection_hint=connection_hint,
                startup_hint=startup_hint,
            )
            result = framework.run_data_collection(
                task_config,
                scene_manifest,
            )
        except RuntimeError as exc:
            result = _handshake_failure(str(exc))
        finally:
            framework.disconnect()
    else:
        try:
            with framework.server_context(Path(scene_usd) if scene_usd else None) as fw:
                server_info = _handshake_geniesim_server(
                    fw,
                    expected_server_version=expected_server_version,
                    required_capabilities=required_capabilities,
                    connection_hint=connection_hint,
                    startup_hint=startup_hint,
                )
                result = fw.run_data_collection(
                    task_config,
                    scene_manifest,
                )
        except RuntimeError as exc:
            result = _handshake_failure(str(exc))

    if not result.server_info and server_info:
        result.server_info = server_info

    if result.success and result.recording_dir:
        lerobot_dir = output_dir / "lerobot"
        framework.export_to_lerobot(result.recording_dir, lerobot_dir)

    return result


def main() -> int:
    validate_required_env_vars(
        {
            "BUCKET": "GCS bucket name",
            "SCENE_ID": "Scene identifier",
        },
        label="[GENIESIM-SUBMIT]",
    )
    bucket = os.environ["BUCKET"]
    scene_id = os.environ["SCENE_ID"]

    preflight_report = run_geniesim_preflight_or_exit(
        "genie-sim-submit-job",
        require_server=False,
    )

    geniesim_prefix = os.getenv("GENIESIM_PREFIX", f"scenes/{scene_id}/geniesim")
    job_output_path = os.getenv("JOB_OUTPUT_PATH", f"{geniesim_prefix}/job.json")

    robot_type = os.getenv("ROBOT_TYPE", "franka")
    episodes_per_task = _resolve_episodes_per_task()
    num_variations = int(os.getenv("NUM_VARIATIONS", "5"))
    min_quality_score = float(os.getenv("MIN_QUALITY_SCORE", "0.85"))

    storage_client = storage.Client()

    scene_graph = _read_json_blob(storage_client, bucket, f"{geniesim_prefix}/scene_graph.json")
    asset_index = _read_json_blob(storage_client, bucket, f"{geniesim_prefix}/asset_index.json")
    task_config = _read_json_blob(storage_client, bucket, f"{geniesim_prefix}/task_config.json")
    export_manifest = _read_json_blob(
        storage_client,
        bucket,
        f"{geniesim_prefix}/export_manifest.json",
    )
    export_marker = _read_json_blob(
        storage_client,
        bucket,
        f"{geniesim_prefix}/_GENIESIM_EXPORT_COMPLETE",
    )
    _validate_bundle_schemas(
        {
            "scene_graph": scene_graph,
            "asset_index": asset_index,
            "task_config": task_config,
        }
    )
    _validate_export_marker(export_marker)

    generation_params = LocalGenerationParams(
        episodes_per_task=episodes_per_task,
        num_variations=num_variations,
        robot_type=robot_type,
        min_quality_score=min_quality_score,
    )

    task_config_hash = _hash_payload(task_config)
    export_manifest_hash = _hash_payload(export_manifest)
    idempotency_key = _build_idempotency_key(scene_id, task_config_hash, export_manifest_hash)
    job_idempotency_path = f"{geniesim_prefix}/job_idempotency.json"

    existing_job_payload = _read_optional_json_blob(storage_client, bucket, job_output_path)
    if (
        existing_job_payload
        and existing_job_payload.get("idempotency", {}).get("key") == idempotency_key
    ):
        print(
            "[GENIESIM-SUBMIT] Duplicate submission detected; "
            f"existing job metadata at gs://{bucket}/{job_output_path}."
        )
        return 0

    existing_idempotency = _read_optional_json_blob(storage_client, bucket, job_idempotency_path)
    if existing_idempotency and existing_idempotency.get("key") == idempotency_key:
        existing_metadata_path = existing_idempotency.get("job_metadata_path")
        location_hint = (
            existing_metadata_path
            if existing_metadata_path
            else f"gs://{bucket}/{job_output_path}"
        )
        print(
            "[GENIESIM-SUBMIT] Duplicate submission detected; "
            f"existing idempotency record at gs://{bucket}/{job_idempotency_path} "
            f"for {location_hint}."
        )
        return 0

    job_id = f"local-{uuid.uuid4()}"
    submission_message = "Local Genie Sim execution started."
    local_run_result = None
    failure_reason = None
    failure_details: Dict[str, Any] = {}
    episodes_output_prefix = os.getenv("OUTPUT_PREFIX", f"scenes/{scene_id}/episodes")
    submitted_at = datetime.utcnow().isoformat() + "Z"
    original_submitted_at = submitted_at
    local_run_end = None

    job_status = "submitted"
    server_info: Dict[str, Any] = {}
    output_dir: Optional[Path] = None
    use_gcs_fuse = False
    local_root = None
    try:
        scene_manifest = _read_json_blob(
            storage_client,
            bucket,
            f"{geniesim_prefix}/merged_scene_manifest.json",
        )
    except FileNotFoundError:
        scene_manifest = {"scene_graph": scene_graph}
    task_config_local = task_config

    gcs_root = Path("/mnt/gcs") / bucket
    use_gcs_fuse = gcs_root.exists()
    local_root = gcs_root if use_gcs_fuse else Path("/tmp") / "geniesim-local"
    output_dir = local_root / episodes_output_prefix / f"geniesim_{job_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    config_dir = output_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    scene_manifest_path = config_dir / "scene_manifest.json"
    task_config_path = config_dir / "task_config.json"
    scene_usd_path = _find_scene_usd_path(
        scene_manifest=scene_manifest,
        scene_id=scene_id,
        geniesim_prefix=geniesim_prefix,
        gcs_root=gcs_root,
        local_root=local_root,
        use_gcs_fuse=use_gcs_fuse,
    )
    if scene_usd_path:
        scene_manifest["usd_path"] = str(scene_usd_path)
        print(f"[GENIESIM-SUBMIT-JOB] Using USD scene path: {scene_usd_path}")
    _write_local_json(scene_manifest_path, scene_manifest)
    _write_local_json(task_config_path, task_config_local)

    local_run_result = _run_local_data_collection_with_handshake(
        scene_manifest=scene_manifest,
        task_config=task_config_local,
        output_dir=output_dir,
        robot_type=robot_type,
        episodes_per_task=episodes_per_task,
        verbose=True,
        expected_server_version=EXPECTED_GENIESIM_SERVER_VERSION,
        required_capabilities=REQUIRED_GENIESIM_CAPABILITIES,
    )
    local_run_end = datetime.utcnow()
    server_info = getattr(local_run_result, "server_info", {})
    if server_info:
        print(
            "[GENIESIM-SUBMIT-JOB] Genie Sim server info: "
            f"version={server_info.get('version')}, "
            f"capabilities={server_info.get('capabilities')}"
        )
    if local_run_result and local_run_result.success:
        submission_message = "Local Genie Sim execution completed."
        job_status = "completed"
    else:
        submission_message = "Local Genie Sim execution failed."
        failure_reason = "Local Genie Sim execution failed"
        failure_details = {
            "episodes_collected": getattr(local_run_result, "episodes_collected", 0)
            if local_run_result
            else 0,
            "episodes_passed": getattr(local_run_result, "episodes_passed", 0)
            if local_run_result
            else 0,
            "errors": getattr(local_run_result, "errors", []) if local_run_result else [],
        }
        job_status = "failed"

    if output_dir and local_root and not use_gcs_fuse:
        upload_failures: list[dict[str, str]] = []
        logger = logging.getLogger("genie-sim-submit-job")
        for file_path in output_dir.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(local_root)
                blob = storage_client.bucket(bucket).blob(str(relative_path))
                gcs_uri = f"gs://{bucket}/{relative_path}"
                result = upload_blob_from_filename(
                    blob,
                    file_path,
                    gcs_uri,
                    logger=logger,
                    verify_upload=True,
                )
                if not result.success:
                    upload_failures.append(
                        {
                            "path": str(relative_path),
                            "error": result.error or "unknown error",
                        }
                    )
        if upload_failures:
            failure_details = {
                **failure_details,
                "upload_failures": upload_failures,
            }
            if job_status != "failed":
                submission_message = "Local Genie Sim execution completed with upload failures."
                job_status = "failed"
            failure_reason = failure_reason or "GCS upload failed"

    metrics = get_metrics()
    metrics_summary = {
        "backend": metrics.backend.value,
        "stats": metrics.get_stats(),
    }
    if job_status == "failed" and not failure_reason:
        failure_reason = "Genie Sim submission failed"

    job_payload = {
        "job_id": job_id,
        "scene_id": scene_id,
        "status": job_status,
        "submitted_at": submitted_at,
        "idempotency": {
            "key": idempotency_key,
            "task_config_hash": task_config_hash,
            "export_manifest_hash": export_manifest_hash,
            "first_submitted_at": original_submitted_at,
        },
        "message": submission_message,
        "bundle": {
            "scene_graph": f"gs://{bucket}/{geniesim_prefix}/scene_graph.json",
            "asset_index": f"gs://{bucket}/{geniesim_prefix}/asset_index.json",
            "task_config": f"gs://{bucket}/{geniesim_prefix}/task_config.json",
            "asset_provenance": f"gs://{bucket}/{geniesim_prefix}/legal/asset_provenance.json",
        },
        "generation_params": {
            "robot_type": robot_type,
            "episodes_per_task": episodes_per_task,
            "num_variations": num_variations,
            "min_quality_score": min_quality_score,
        },
        "export_schema": {
            "export_schema_version": export_marker.get("export_schema_version"),
            "geniesim_schema_version": export_marker.get("geniesim_schema_version"),
            "blueprintpipeline_version": export_marker.get("blueprintpipeline_version"),
            "export_timestamp": export_marker.get("export_timestamp"),
            "schema_compatibility": export_marker.get("schema_compatibility", {}),
        },
        "metrics_summary": metrics_summary,
        "local_execution": {
            "preflight": preflight_report,
        },
    }
    try:
        metrics_builder = LocalJobMetricsBuilder(
            generation_params=generation_params,
            task_config=task_config,
        )
        job_payload["job_metrics"] = metrics_builder.build(
            job_id=job_id,
            created_at=submitted_at,
            completed_at=(local_run_end.isoformat() + "Z") if local_run_end else None,
            status="completed" if job_status == "completed" else "failed",
            episodes_collected=(
                getattr(local_run_result, "episodes_collected", 0)
                if local_run_result
                else 0
            ),
            episodes_passed=(
                getattr(local_run_result, "episodes_passed", 0) if local_run_result else 0
            ),
            failure_reason=failure_reason,
            failure_details=failure_details if failure_details else None,
        )
    except Exception as exc:
        job_payload["job_metrics_error"] = str(exc)
    if job_status == "failed":
        job_payload["failure_reason"] = failure_reason
        job_payload["failure_details"] = failure_details
    job_payload["artifacts"] = {
        "episodes_prefix": f"gs://{bucket}/{episodes_output_prefix}/geniesim_{job_id}",
        "lerobot_prefix": (
            f"gs://{bucket}/{episodes_output_prefix}/geniesim_{job_id}/lerobot"
        ),
    }
    job_payload["local_execution"] = {
        "success": bool(local_run_result and local_run_result.success),
        "episodes_collected": getattr(local_run_result, "episodes_collected", 0) if local_run_result else 0,
        "episodes_passed": getattr(local_run_result, "episodes_passed", 0) if local_run_result else 0,
        "preflight": preflight_report,
        "server_info": server_info if server_info else None,
    }

    _write_json_blob(storage_client, bucket, job_output_path, job_payload)
    _write_json_blob(
        storage_client,
        bucket,
        job_idempotency_path,
        {
            "key": idempotency_key,
            "scene_id": scene_id,
            "task_config_hash": task_config_hash,
            "export_manifest_hash": export_manifest_hash,
            "submitted_at": original_submitted_at,
            "job_id": job_id,
            "job_metadata_path": f"gs://{bucket}/{job_output_path}",
        },
    )
    if job_status == "failed":
        _write_failure_marker(
            storage_client,
            bucket,
            geniesim_prefix,
            {
                "scene_id": scene_id,
                "job_id": job_id,
                "status": job_status,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "reason": failure_reason,
                "details": failure_details,
                "job_metadata_path": f"gs://{bucket}/{job_output_path}",
            },
        )
    print(f"[GENIESIM-SUBMIT] Stored job metadata at gs://{bucket}/{job_output_path}")
    return 1 if job_status == "failed" else 0


if __name__ == "__main__":
    from tools.startup_validation import validate_and_fail_fast

    validate_and_fail_fast(job_name="GENIE-SIM-SUBMIT", validate_gcs=True)
    metrics = get_metrics()
    scene_id = os.environ.get("SCENE_ID", "unknown")
    with metrics.track_job("genie-sim-submit-job", scene_id):
        raise SystemExit(main())
