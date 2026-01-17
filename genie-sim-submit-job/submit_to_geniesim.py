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
    EPISODES_PER_TASK: Episodes per task (default: 10)
    NUM_VARIATIONS: Scene variations (default: 5)
    MIN_QUALITY_SCORE: Minimum quality score (default: 0.85)
"""

import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from google.cloud import storage

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.path.insert(0, str(REPO_ROOT / "genie-sim-export-job"))
from geniesim_client import GenerationParams, GenieSimClient, JobStatus

sys.path.insert(0, str(REPO_ROOT / "tools"))
from geniesim_adapter.local_framework import (
    run_geniesim_preflight_or_exit,
    run_local_data_collection,
)
from tools.metrics.pipeline_metrics import get_metrics
from tools.validation.entrypoint_checks import validate_required_env_vars

EXPECTED_EXPORT_SCHEMA_VERSION = "1.0.0"
EXPECTED_GENIESIM_API_VERSION = "3.0.0"
CONTRACT_SCHEMAS = {
    "scene_graph": "scene_graph.schema.json",
    "asset_index": "asset_index.schema.json",
    "task_config": "task_config.schema.json",
}


def _write_local_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _read_json_blob(client: storage.Client, bucket: str, blob_name: str) -> Dict[str, Any]:
    blob = client.bucket(bucket).blob(blob_name)
    if not blob.exists():
        raise FileNotFoundError(f"Missing required export bundle file: gs://{bucket}/{blob_name}")
    return json.loads(blob.download_as_text())


def _write_json_blob(client: storage.Client, bucket: str, blob_name: str, payload: Dict[str, Any]) -> None:
    blob = client.bucket(bucket).blob(blob_name)
    blob.upload_from_string(json.dumps(payload, indent=2), content_type="application/json")


def _write_failure_marker(
    client: storage.Client,
    bucket: str,
    geniesim_prefix: str,
    payload: Dict[str, Any],
) -> None:
    _write_json_blob(client, bucket, f"{geniesim_prefix}/.failed", payload)


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
    if _parse_version(geniesim_schema)[0] != _parse_version(EXPECTED_GENIESIM_API_VERSION)[0]:
        raise RuntimeError(
            "Genie Sim API version incompatibility: "
            f"expected major {EXPECTED_GENIESIM_API_VERSION}, found {geniesim_schema}."
        )

    min_version = compatibility.get("min_geniesim_version")
    max_version = compatibility.get("max_geniesim_version")
    if not min_version or not max_version:
        raise RuntimeError(
            "Missing schema_compatibility ranges in _GENIESIM_EXPORT_COMPLETE."
        )
    if not _is_version_compatible(EXPECTED_GENIESIM_API_VERSION, min_version, max_version):
        raise RuntimeError(
            "Genie Sim API version incompatibility: expected "
            f"{EXPECTED_GENIESIM_API_VERSION} to be within [{min_version}, {max_version}]."
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
    episodes_per_task = int(os.getenv("EPISODES_PER_TASK", "10"))
    num_variations = int(os.getenv("NUM_VARIATIONS", "5"))
    min_quality_score = float(os.getenv("MIN_QUALITY_SCORE", "0.85"))

    storage_client = storage.Client()

    scene_graph = _read_json_blob(storage_client, bucket, f"{geniesim_prefix}/scene_graph.json")
    asset_index = _read_json_blob(storage_client, bucket, f"{geniesim_prefix}/asset_index.json")
    task_config = _read_json_blob(storage_client, bucket, f"{geniesim_prefix}/task_config.json")
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

    generation_params = GenerationParams(
        episodes_per_task=episodes_per_task,
        num_variations=num_variations,
        robot_type=robot_type,
        min_quality_score=min_quality_score,
    )

    submission_mode = "local"
    job_id = None
    submission_message = None
    local_run_result = None
    failure_reason = None
    failure_details: Dict[str, Any] = {}
    episodes_output_prefix = os.getenv("OUTPUT_PREFIX", f"scenes/{scene_id}/episodes")
    submitted_at = datetime.utcnow().isoformat() + "Z"
    local_run_end = None

    job_id = f"local-{uuid.uuid4()}"
    submission_message = "Local Genie Sim execution started."
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

    local_run_result = run_local_data_collection(
        scene_manifest_path=scene_manifest_path,
        task_config_path=task_config_path,
        output_dir=output_dir,
        robot_type=robot_type,
        episodes_per_task=episodes_per_task,
        verbose=True,
    )
    local_run_end = datetime.utcnow()
    if local_run_result and local_run_result.success:
        submission_message = "Local Genie Sim execution completed."
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
        }

    if not use_gcs_fuse:
        for file_path in output_dir.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(local_root)
                blob = storage_client.bucket(bucket).blob(str(relative_path))
                blob.upload_from_filename(str(file_path))

    metrics = get_metrics()
    metrics_summary = {
        "backend": metrics.backend.value,
        "stats": metrics.get_stats(),
    }
    job_status = (
        "completed"
        if local_run_result and local_run_result.success
        else ("failed" if local_run_result else "submitted")
    )
    if job_status == "failed" and not failure_reason:
        failure_reason = "Genie Sim submission failed"

    job_payload = {
        "job_id": job_id,
        "scene_id": scene_id,
        "status": job_status,
        "submission_mode": submission_mode,
        "submitted_at": submitted_at,
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
    metrics_client = GenieSimClient(mock_mode=True, validate_on_init=False)
    metrics_client.register_mock_job_metrics(
        job_id=job_id,
        generation_params=generation_params,
        task_config=task_config,
        created_at=submitted_at,
        completed_at=(local_run_end.isoformat() + "Z") if local_run_end else None,
        status=JobStatus.COMPLETED if job_status == "completed" else JobStatus.FAILED,
        episodes_collected=(
            getattr(local_run_result, "episodes_collected", 0) if local_run_result else 0
        ),
        episodes_passed=(
            getattr(local_run_result, "episodes_passed", 0) if local_run_result else 0
        ),
        failure_reason=failure_reason,
        failure_details=failure_details if failure_details else None,
    )
    try:
        job_payload["job_metrics"] = metrics_client.get_job_metrics(job_id)
    except Exception as exc:
        job_payload["job_metrics_error"] = str(exc)
    if job_status == "failed":
        job_payload["failure_reason"] = failure_reason
        job_payload["failure_details"] = failure_details
    if submission_mode == "local":
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
        }

    _write_json_blob(storage_client, bucket, job_output_path, job_payload)
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
    metrics = get_metrics()
    scene_id = os.environ.get("SCENE_ID", "unknown")
    with metrics.track_job("genie-sim-submit-job", scene_id):
        raise SystemExit(main())
