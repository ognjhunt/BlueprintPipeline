#!/usr/bin/env python3
"""
Dataset delivery job.

Reads a Genie Sim import manifest from GCS, verifies the bundle checksum,
generates a dataset card, uploads artifacts to lab delivery buckets, and
sends webhook notifications.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
import requests
from google.cloud import storage

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.error_handling.retry import NonRetryableError, RetryableError, retry_with_backoff
from tools.quality_gates.quality_gate import (
    QualityGate,
    QualityGateCheckpoint,
    QualityGateRegistry,
    QualityGateResult,
    QualityGateSeverity,
)
from tools.quality_reports.asset_provenance_generator import COMMERCIAL_OK_LICENSES, LicenseType
from tools.tracing.correlation import ensure_request_id
from tools.validation.entrypoint_checks import validate_required_env_vars

GCS_ROOT = Path("/mnt/gcs")
JOB_NAME = "dataset-delivery-job"
logger = logging.getLogger(__name__)


def _gate_report_path(root: Path, scene_id: str) -> Path:
    report_path = root / f"scenes/{scene_id}/{JOB_NAME}/quality_gate_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    return report_path


def _emit_delivery_quality_gate(
    scene_id: str,
    context: Dict[str, Any],
    report_path: Path,
) -> None:
    checkpoint = QualityGateCheckpoint.DELIVERED
    registry = QualityGateRegistry(verbose=True)

    def _check_delivery(ctx: Dict[str, Any]) -> QualityGateResult:
        passed = ctx["success"]
        severity = QualityGateSeverity.INFO if passed else QualityGateSeverity.ERROR
        message = (
            "Dataset delivery completed successfully"
            if passed
            else "Dataset delivery completed with errors"
        )
        details = {
            "scene_id": ctx["scene_id"],
            "delivery_targets": ctx["delivery_targets"],
            "artifact_counts": ctx["artifact_counts"],
            "dataset_card_paths": ctx["dataset_card_paths"],
            "warnings": ctx["warnings"],
            "errors": ctx["errors"],
        }
        return QualityGateResult(
            gate_id="dataset_delivered",
            checkpoint=checkpoint,
            passed=passed,
            severity=severity,
            message=message,
            details=details,
        )

    registry.register(QualityGate(
        id="dataset_delivered",
        name="Dataset Delivered",
        checkpoint=checkpoint,
        severity=QualityGateSeverity.INFO,
        description="Emit a completion gate for dataset delivery.",
        check_fn=_check_delivery,
    ))

    registry.run_checkpoint(checkpoint, context)
    registry.save_report(scene_id, report_path)


def parse_mapping(value: str | None) -> Dict[str, str]:
    """Parse JSON or comma-separated key=value mapping."""
    if not value:
        return {}
    payload = value.strip()
    if not payload:
        return {}
    if payload.startswith("{"):
        mapping = json.loads(payload)
        if not isinstance(mapping, dict):
            raise ValueError("Mapping JSON must be an object")
        return {str(key): str(val) for key, val in mapping.items()}
    mapping: Dict[str, str] = {}
    for item in payload.split(","):
        item = item.strip()
        if not item:
            continue
        key, sep, val = item.partition("=")
        if not sep:
            raise ValueError(f"Invalid mapping entry: {item}")
        mapping[key.strip()] = val.strip()
    return mapping


def parse_gs_uri(uri: str) -> Tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got {uri}")
    path = uri[len("gs://") :]
    bucket, sep, obj = path.partition("/")
    if not bucket or not sep:
        raise ValueError(f"Invalid gs:// URI: {uri}")
    return bucket, obj


def join_gs_uri(base_uri: str, rel_path: str) -> str:
    if rel_path.startswith("gs://"):
        return rel_path
    bucket, prefix = parse_gs_uri(base_uri)
    combined = "/".join(segment for segment in [prefix.rstrip("/"), rel_path.lstrip("/")] if segment)
    return f"gs://{bucket}/{combined}"


def read_json_from_gcs(client: storage.Client, uri: str) -> Dict[str, Any]:
    bucket_name, object_name = parse_gs_uri(uri)
    blob = client.bucket(bucket_name).blob(object_name)
    return json.loads(blob.download_as_text())


def download_blob_to_path(client: storage.Client, uri: str, path: str) -> int:
    bucket_name, object_name = parse_gs_uri(uri)
    blob = client.bucket(bucket_name).blob(object_name)
    blob.download_to_filename(path)
    return blob.size or 0


def compute_sha256(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def resolve_bundle_base(import_manifest: Dict[str, Any], manifest_uri: str) -> str:
    gcs_output_path = import_manifest.get("gcs_output_path")
    if isinstance(gcs_output_path, str) and gcs_output_path.startswith("gs://"):
        return gcs_output_path.rstrip("/")
    bucket_name, object_name = parse_gs_uri(manifest_uri)
    prefix = object_name.rsplit("/", 1)[0]
    return f"gs://{bucket_name}/{prefix}"


def get_checksum_entry(checksums: Dict[str, Any], rel_path: str) -> Dict[str, Any]:
    for key in ("bundle_files", "files"):
        section = checksums.get(key)
        if isinstance(section, dict) and rel_path in section:
            entry = section[rel_path]
            if isinstance(entry, dict):
                return entry
            if isinstance(entry, str):
                return {"sha256": entry}
    raise KeyError(f"No checksum entry for {rel_path}")


def verify_package_checksum(
    client: storage.Client,
    package_uri: str,
    checksums_uri: str,
    package_rel_path: str,
) -> None:
    checksums = read_json_from_gcs(client, checksums_uri)
    entry = get_checksum_entry(checksums, package_rel_path)
    expected_sha = entry.get("sha256")
    expected_size = entry.get("size_bytes")
    if not expected_sha:
        raise ValueError(f"Missing sha256 for {package_rel_path} in checksums.json")

    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = os.path.join(tmpdir, os.path.basename(package_rel_path))
        actual_size = download_blob_to_path(client, package_uri, local_path)
        actual_sha = compute_sha256(local_path)

    if actual_sha != expected_sha:
        raise ValueError(
            "Package checksum mismatch; "
            f"expected {expected_sha}, got {actual_sha}"
        )
    if expected_size is not None and actual_size != expected_size:
        raise ValueError(
            "Package size mismatch; "
            f"expected {expected_size}, got {actual_size}"
        )


def verify_blob_checksum(
    client: storage.Client,
    blob_uri: str,
    checksums_uri: str,
    blob_rel_path: str,
) -> None:
    verify_package_checksum(client, blob_uri, checksums_uri, blob_rel_path)


def _normalize_license(license_value: Any) -> LicenseType:
    if isinstance(license_value, LicenseType):
        return license_value
    if isinstance(license_value, str):
        try:
            return LicenseType(license_value)
        except ValueError:
            return LicenseType.UNKNOWN
    return LicenseType.UNKNOWN


def _offending_asset_licenses(asset_provenance: Dict[str, Any]) -> List[str]:
    offenders: List[str] = []
    assets = asset_provenance.get("assets")
    if isinstance(assets, list) and assets:
        for asset in assets:
            if not isinstance(asset, dict):
                offenders.append("unknown:unknown")
                continue
            asset_id = (
                asset.get("asset_id")
                or asset.get("id")
                or asset.get("asset_path")
                or "unknown"
            )
            license_info = asset.get("license")
            if isinstance(license_info, dict):
                license_value = license_info.get("type") or license_info.get("license")
            elif isinstance(license_info, str):
                license_value = license_info
            else:
                license_value = None
            license_type = _normalize_license(license_value)
            if license_type not in COMMERCIAL_OK_LICENSES:
                offenders.append(f"{asset_id}:{license_value or license_type.value}")
        return offenders

    license_info = asset_provenance.get("license")
    if isinstance(license_info, dict):
        license_value = license_info.get("type") or license_info.get("license")
    elif isinstance(license_info, str):
        license_value = license_info
    else:
        license_value = None
    license_type = _normalize_license(license_value)
    if license_type not in COMMERCIAL_OK_LICENSES:
        offenders.append(f"scene:{license_value or license_type.value}")
    return offenders


def _resolve_asset_provenance_uri(import_manifest: Dict[str, Any], bundle_base: str) -> str:
    asset_provenance_path = (
        import_manifest.get("asset_provenance_path")
        or import_manifest.get("asset_provenance")
        or "legal/asset_provenance.json"
    )
    if not asset_provenance_path:
        raise ValueError("Import manifest missing asset_provenance_path")
    return join_gs_uri(bundle_base, asset_provenance_path)


def build_dataset_card(import_manifest: Dict[str, Any], scene_id: str, job_id: str) -> Dict[str, Any]:
    return {
        "scene_id": scene_id,
        "job_id": job_id,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "provenance": import_manifest.get("provenance", {}),
        "quality": import_manifest.get("quality", {}),
        "episodes": import_manifest.get("episodes", {}),
        "package": import_manifest.get("package", {}),
    }


def _log_webhook_retry(attempt: int, error: Exception, delay: float) -> None:
    logger.warning(
        "Retrying webhook attempt %s after %.2fs: %s",
        attempt,
        delay,
        error,
    )


def _log_webhook_failure(attempt: int, error: Exception) -> None:
    logger.error(
        "Webhook failed after %s attempts: %s",
        attempt,
        error,
    )


def _classify_webhook_response(webhook_url: str, response: requests.Response) -> None:
    status_code = response.status_code
    if status_code == 429 or status_code >= 500:
        raise RetryableError(
            f"Webhook {webhook_url} failed with {status_code}: {response.text}"
        )
    if status_code >= 400:
        raise NonRetryableError(
            f"Webhook {webhook_url} failed with {status_code}: {response.text}"
        )


@retry_with_backoff(
    max_retries=3,
    base_delay=1.0,
    max_delay=30.0,
    backoff_factor=2.0,
    jitter=True,
    on_retry=_log_webhook_retry,
    on_failure=_log_webhook_failure,
)
def send_webhook(webhook_url: str, payload: Dict[str, Any]) -> None:
    try:
        response = requests.post(webhook_url, json=payload, timeout=15)
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
        raise RetryableError(f"Webhook {webhook_url} failed: {exc}") from exc
    _classify_webhook_response(webhook_url, response)


def write_failure_marker(
    client: storage.Client,
    bucket_name: str,
    scene_id: str,
    job_id: str,
    error: str,
) -> None:
    marker = {
        "scene_id": scene_id,
        "job_id": job_id,
        "error": error,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    blob = client.bucket(bucket_name).blob(f"scenes/{scene_id}/geniesim/.dataset_delivery_failed")
    blob.upload_from_string(json.dumps(marker, indent=2), content_type="application/json")


def _seed_scene_id_from_manifest_path(import_manifest_path: str) -> None:
    if os.environ.get("SCENE_ID"):
        return
    match = re.search(r"/scenes/([^/]+)/", import_manifest_path)
    if match:
        os.environ["SCENE_ID"] = match.group(1)
        return
    os.environ["SCENE_ID"] = "unknown"
    logger.warning("Unable to derive SCENE_ID from IMPORT_MANIFEST_PATH; using 'unknown'.")


def main() -> int:
    os.environ["REQUEST_ID"] = ensure_request_id()
    import_manifest_path = os.environ.get("IMPORT_MANIFEST_PATH")
    if import_manifest_path:
        _seed_scene_id_from_manifest_path(import_manifest_path)
        required_env_vars = {
            "BUCKET": "Bucket name for pipeline artifacts",
            "IMPORT_MANIFEST_PATH": "GCS URI for the import manifest",
        }
    else:
        required_env_vars = {
            "BUCKET": "Bucket name for pipeline artifacts",
            "SCENE_ID": "Scene identifier",
        }
    validate_required_env_vars(required_env_vars, label="[DATASET-DELIVERY]")

    bucket_name = os.environ.get("BUCKET")
    scene_id = os.environ.get("SCENE_ID") or ""
    job_id = os.environ.get("JOB_ID") or ""
    delivery_prefix_template = os.environ.get("DELIVERY_PREFIX", "deliveries/{scene_id}/{job_id}")
    if not import_manifest_path:
        import_manifest_path = f"gs://{bucket_name}/scenes/{scene_id}/geniesim/import_manifest.json"

    client = storage.Client()

    try:
        import_manifest = read_json_from_gcs(client, import_manifest_path)
        manifest_scene_id = import_manifest.get("provenance", {}).get("scene_id") or import_manifest.get("scene_id")
        manifest_job_id = import_manifest.get("job_id")
        scene_id = scene_id or manifest_scene_id or "unknown"
        job_id = job_id or manifest_job_id or "unknown"

        bundle_base = resolve_bundle_base(import_manifest, import_manifest_path)
        package_rel_path = import_manifest.get("package", {}).get("path")
        checksums_rel_path = import_manifest.get("checksums_path")
        if not package_rel_path or not checksums_rel_path:
            raise ValueError("Import manifest missing package.path or checksums_path")

        package_uri = join_gs_uri(bundle_base, package_rel_path)
        checksums_uri = join_gs_uri(bundle_base, checksums_rel_path)
        verify_package_checksum(client, package_uri, checksums_uri, package_rel_path)

        asset_provenance_uri = _resolve_asset_provenance_uri(import_manifest, bundle_base)
        asset_provenance = read_json_from_gcs(client, asset_provenance_uri)
        offending_licenses = _offending_asset_licenses(asset_provenance)
        if offending_licenses:
            raise RuntimeError(
                "Compliance error: "
                f"scene_id={scene_id} has non-commercial or unknown licenses: "
                f"{', '.join(offending_licenses)}"
            )

        dataset_card = build_dataset_card(import_manifest, scene_id, job_id)

        lab_delivery_buckets = parse_mapping(os.environ.get("LAB_DELIVERY_BUCKETS"))
        lab_webhook_urls = parse_mapping(os.environ.get("LAB_WEBHOOK_URLS"))
        default_bucket = os.environ.get("DEFAULT_DELIVERY_BUCKET")
        if default_bucket and "default" not in lab_delivery_buckets:
            lab_delivery_buckets["default"] = default_bucket

        if not lab_delivery_buckets:
            raise ValueError("No delivery buckets configured; set LAB_DELIVERY_BUCKETS or DEFAULT_DELIVERY_BUCKET")

        delivery_targets = []
        delivery_warnings = []
        delivery_errors = []
        dataset_card_paths: Dict[str, Any] = {"local": None, "gcs": {}}
        artifact_counts = {"total_artifacts": 0, "per_lab": {}}

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_card_path = os.path.join(tmpdir, "dataset_card.json")
            with open(dataset_card_path, "w", encoding="utf-8") as file_handle:
                json.dump(dataset_card, file_handle, indent=2)
            dataset_card_paths["local"] = dataset_card_path

            source_bucket_name, package_object = parse_gs_uri(package_uri)
            source_bucket = client.bucket(source_bucket_name)
            source_blob = source_bucket.blob(package_object)

            for lab, dest_bucket_name in lab_delivery_buckets.items():
                delivery_prefix = delivery_prefix_template.format(scene_id=scene_id, job_id=job_id, lab=lab).strip("/")
                package_object_name = f"{delivery_prefix}/{package_rel_path}"
                dataset_card_object_name = f"{delivery_prefix}/dataset_card.json"
                staging_prefix = f"{delivery_prefix}/_staging/{job_id or 'unknown'}"
                staging_object_name = f"{staging_prefix}/{package_rel_path}"

                dest_bucket = client.bucket(dest_bucket_name)
                dataset_card_url = f"gs://{dest_bucket_name}/{dataset_card_object_name}"
                bundle_url = f"gs://{dest_bucket_name}/{package_object_name}"
                staging_url = f"gs://{dest_bucket_name}/{staging_object_name}"
                checksum_verified = False
                checksum_error = None
                delivery_success = False
                staging_blob = dest_bucket.blob(staging_object_name)
                try:
                    source_bucket.copy_blob(source_blob, dest_bucket, new_name=staging_object_name)
                    verify_blob_checksum(client, staging_url, checksums_uri, package_rel_path)
                    checksum_verified = True
                    dest_bucket.copy_blob(staging_blob, dest_bucket, new_name=package_object_name)
                    staging_blob.delete()
                    dest_bucket.blob(dataset_card_object_name).upload_from_filename(dataset_card_path)
                    delivery_success = True
                except Exception as exc:
                    checksum_error = str(exc)
                    delivery_errors.append(
                        "Delivery verification failed for "
                        f"{staging_url} -> {bundle_url}: {checksum_error}"
                    )
                    try:
                        if staging_blob.exists():
                            staging_blob.delete()
                    except Exception as cleanup_exc:
                        delivery_errors.append(
                            f"Failed to cleanup staging blob {staging_url}: {cleanup_exc}"
                        )

                delivery_targets.append(
                    {
                        "lab": lab,
                        "bucket": dest_bucket_name,
                        "delivery_prefix": delivery_prefix,
                        "package_object": package_object_name,
                        "dataset_card_object": dataset_card_object_name,
                        "bundle_url": bundle_url if delivery_success else None,
                        "dataset_card_url": dataset_card_url if delivery_success else None,
                        "staging_bundle_url": staging_url,
                        "checksum_verified": checksum_verified,
                        "checksum_error": checksum_error,
                        "delivery_success": delivery_success,
                    }
                )
                if delivery_success:
                    dataset_card_paths["gcs"][lab] = dataset_card_url
                    artifact_counts["per_lab"][lab] = 2
                    artifact_counts["total_artifacts"] += 2

                webhook_url = lab_webhook_urls.get(lab)
                if webhook_url and delivery_success:
                    send_webhook(
                        webhook_url,
                        {
                            "scene_id": scene_id,
                            "job_id": job_id,
                            "lab": lab,
                            "dataset_card_url": dataset_card_url,
                            "bundle_url": bundle_url,
                        },
                    )

        report_path = _gate_report_path(GCS_ROOT, scene_id)
        success = not delivery_errors
        _emit_delivery_quality_gate(
            scene_id,
            {
                "scene_id": scene_id,
                "success": success,
                "delivery_targets": delivery_targets,
                "artifact_counts": artifact_counts,
                "dataset_card_paths": dataset_card_paths,
                "warnings": delivery_warnings,
                "errors": delivery_errors,
            },
            report_path,
        )

        if not success:
            error_details = "Checksum verification failed for delivered artifacts."
            if delivery_errors:
                error_details = "\n".join(delivery_errors)
            write_failure_marker(client, bucket_name, scene_id, job_id or "unknown", error_details)
            print("Dataset delivery completed with errors", file=sys.stderr)
            return 1

        print("Dataset delivery completed")
        return 0
    except Exception as exc:
        error_details = f"{exc}\n{traceback.format_exc()}"
        marker_bucket = bucket_name
        if not marker_bucket and import_manifest_path:
            marker_bucket, _ = parse_gs_uri(import_manifest_path)
        if marker_bucket and scene_id:
            write_failure_marker(client, marker_bucket, scene_id, job_id or "unknown", error_details)
        print(f"Dataset delivery failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
