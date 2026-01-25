#!/usr/bin/env python3
"""
Dataset delivery integrity audit job.

Enumerates delivered bundles in a delivery bucket, validates bundle files
against checksums.json, and uploads an audit report to GCS.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from google.cloud import storage

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
GENIE_SIM_IMPORT_PATH = REPO_ROOT / "genie-sim-import-job"
if str(GENIE_SIM_IMPORT_PATH) not in sys.path:
    sys.path.insert(0, str(GENIE_SIM_IMPORT_PATH))

from tools.config.production_mode import resolve_production_mode
from tools.metrics.pipeline_metrics import get_metrics
from import_manifest_utils import verify_checksums_signature

GCS_ROOT = Path("/mnt/gcs")
JOB_NAME = "dataset-delivery-integrity-audit-job"
LOGGER = logging.getLogger(JOB_NAME)


def _parse_int(value: Optional[str], default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Invalid integer value '{value}'") from exc


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _format_timestamp(dt: datetime) -> str:
    return dt.strftime("%Y%m%dT%H%M%SZ")


def _resolve_bundle_prefix(checksums_object: str) -> str:
    prefix = checksums_object.rsplit("/", 1)[0]
    if prefix.endswith("/meta"):
        prefix = prefix.rsplit("/", 1)[0]
    return prefix.rstrip("/")


def _extract_checksums_entries(payload: Dict[str, Any]) -> Dict[str, Any]:
    for key in ("bundle_files", "files"):
        entries = payload.get(key)
        if isinstance(entries, dict):
            return entries
    raise ValueError("checksums.json missing expected 'files' or 'bundle_files' mapping")


def _iter_checksum_blobs(
    bucket: storage.Bucket,
    prefix: str,
    filename: str,
    max_bundles: int,
) -> Iterable[storage.Blob]:
    count = 0
    for blob in bucket.list_blobs(prefix=prefix):
        if not blob.name.endswith(filename):
            continue
        yield blob
        count += 1
        if max_bundles > 0 and count >= max_bundles:
            break


def _download_and_hash(blob: storage.Blob) -> Tuple[str, int]:
    hasher = hashlib.sha256()
    size = 0
    with blob.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
            size += len(chunk)
    return hasher.hexdigest(), size


def _verify_checksums_signature(
    payload: Dict[str, Any],
    production_mode: bool,
    hmac_key: Optional[str],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "success": True,
        "errors": [],
        "expected": None,
        "actual": None,
        "required": production_mode,
    }
    signature = payload.get("signature")
    if signature is None:
        if production_mode:
            result["success"] = False
            result["errors"].append("checksums.json signature is required in production")
        return result

    verify_result = verify_checksums_signature(payload, hmac_key)
    result["expected"] = verify_result.get("expected")
    result["actual"] = verify_result.get("actual")
    if not verify_result.get("success"):
        result["success"] = False
        result["errors"].extend(verify_result.get("errors", []))
    return result


def _audit_bundle(
    bucket: storage.Bucket,
    bundle_prefix: str,
    checksums_payload: Dict[str, Any],
    max_files: int,
    max_failure_details: int,
    production_mode: bool,
    hmac_key: Optional[str],
) -> Dict[str, Any]:
    entries = _extract_checksums_entries(checksums_payload)
    missing_files: List[str] = []
    checksum_mismatches: List[Dict[str, Any]] = []
    size_mismatches: List[Dict[str, Any]] = []
    invalid_entries: List[str] = []
    signature_result = _verify_checksums_signature(
        checksums_payload,
        production_mode,
        hmac_key,
    )
    checked_files = 0
    skipped_files = 0

    for rel_path, entry in entries.items():
        if max_files > 0 and checked_files >= max_files:
            skipped_files += 1
            continue
        checked_files += 1
        expected_sha: Optional[str] = None
        expected_size: Optional[int] = None

        if isinstance(entry, dict):
            expected_sha = entry.get("sha256")
            expected_size = entry.get("size_bytes")
        elif isinstance(entry, str):
            expected_sha = entry
        else:
            invalid_entries.append(rel_path)
            continue

        if not expected_sha:
            invalid_entries.append(rel_path)
            continue

        blob_name = f"{bundle_prefix}/{rel_path}" if bundle_prefix else rel_path
        blob = bucket.blob(blob_name)
        if not blob.exists():
            if len(missing_files) < max_failure_details:
                missing_files.append(rel_path)
            continue

        actual_sha, actual_size = _download_and_hash(blob)
        if expected_sha != actual_sha:
            if len(checksum_mismatches) < max_failure_details:
                checksum_mismatches.append(
                    {
                        "path": rel_path,
                        "expected": expected_sha,
                        "actual": actual_sha,
                    }
                )
        if expected_size is not None and actual_size != expected_size:
            if len(size_mismatches) < max_failure_details:
                size_mismatches.append(
                    {
                        "path": rel_path,
                        "expected": expected_size,
                        "actual": actual_size,
                    }
                )

    success = (
        not missing_files
        and not checksum_mismatches
        and not size_mismatches
        and not invalid_entries
        and signature_result["success"]
    )

    return {
        "bundle_prefix": bundle_prefix,
        "checked_files": checked_files,
        "skipped_files": skipped_files,
        "missing_files": missing_files,
        "checksum_mismatches": checksum_mismatches,
        "size_mismatches": size_mismatches,
        "invalid_entries": invalid_entries,
        "signature": signature_result,
        "success": success,
    }


def _emit_log(event: str, payload: Dict[str, Any]) -> None:
    entry = {
        "bp_metric": "delivery_integrity_audit",
        "event": event,
        **payload,
    }
    LOGGER.info("AuditEvent %s", json.dumps(entry, sort_keys=True))


def main() -> int:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

    bucket_name = os.getenv("DELIVERY_BUCKET") or os.getenv("BUCKET")
    if not bucket_name:
        raise ValueError("DELIVERY_BUCKET or BUCKET is required")

    prefix = os.getenv("DELIVERY_PREFIX", "deliveries/")
    checksums_filename = os.getenv("CHECKSUMS_FILENAME", "checksums.json")
    max_bundles = _parse_int(os.getenv("MAX_BUNDLES"), 100)
    max_files = _parse_int(os.getenv("MAX_FILES_PER_BUNDLE"), 0)
    max_failure_details = _parse_int(os.getenv("MAX_FAILURE_DETAILS"), 200)
    production_mode = resolve_production_mode()
    hmac_key = os.getenv("CHECKSUMS_HMAC_KEY")

    report_bucket_name = os.getenv("AUDIT_REPORT_BUCKET", bucket_name)
    report_prefix = os.getenv("AUDIT_REPORT_PREFIX", "audit-reports/delivery-integrity")

    start_time = _utc_now()
    _emit_log(
        "start",
        {
            "job": JOB_NAME,
            "bucket": bucket_name,
            "prefix": prefix,
            "max_bundles": max_bundles,
            "max_files_per_bundle": max_files,
            "report_bucket": report_bucket_name,
            "report_prefix": report_prefix,
            "timestamp": start_time.isoformat(),
        },
    )

    metrics = None
    try:
        metrics = get_metrics()
    except Exception as exc:
        LOGGER.warning("Metrics unavailable: %s", exc)

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    report_bucket = client.bucket(report_bucket_name)

    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    pass_count = 0
    fail_count = 0

    for blob in _iter_checksum_blobs(bucket, prefix, checksums_filename, max_bundles):
        bundle_prefix = _resolve_bundle_prefix(blob.name)
        bundle_id = bundle_prefix.replace(prefix.rstrip("/"), "").lstrip("/")
        try:
            payload = json.loads(blob.download_as_text())
            bundle_result = _audit_bundle(
                bucket,
                bundle_prefix,
                payload,
                max_files,
                max_failure_details,
                production_mode,
                hmac_key,
            )
            bundle_result["checksums_object"] = blob.name
            bundle_result["bundle_id"] = bundle_id
            results.append(bundle_result)

            status = "success" if bundle_result["success"] else "failure"
            if bundle_result["success"]:
                pass_count += 1
            else:
                fail_count += 1

            if metrics:
                metrics.pipeline_runs_total.inc(
                    labels={"job": JOB_NAME, "status": status, "scene_id": bundle_id or "unknown"}
                )
                if status == "failure":
                    metrics.errors_total.inc(
                        labels={
                            "job": JOB_NAME,
                            "scene_id": bundle_id or "unknown",
                            "error_type": "integrity_audit_failed",
                        }
                    )

            if status == "failure":
                _emit_log(
                    "bundle_failed",
                    {
                        "job": JOB_NAME,
                        "bundle": bundle_prefix,
                        "bundle_id": bundle_id,
                        "missing_files": len(bundle_result["missing_files"]),
                        "checksum_mismatches": len(bundle_result["checksum_mismatches"]),
                        "size_mismatches": len(bundle_result["size_mismatches"]),
                        "invalid_entries": len(bundle_result["invalid_entries"]),
                    },
                )
        except Exception as exc:
            fail_count += 1
            error_payload = {
                "bundle": bundle_prefix,
                "checksums_object": blob.name,
                "error": str(exc),
            }
            errors.append(error_payload)
            _emit_log("bundle_error", {"job": JOB_NAME, **error_payload})
            if metrics:
                metrics.errors_total.inc(
                    labels={
                        "job": JOB_NAME,
                        "scene_id": bundle_id or "unknown",
                        "error_type": type(exc).__name__,
                    }
                )

    end_time = _utc_now()
    report = {
        "job": JOB_NAME,
        "bucket": bucket_name,
        "prefix": prefix,
        "max_bundles": max_bundles,
        "max_files_per_bundle": max_files,
        "checksums_filename": checksums_filename,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "pass_count": pass_count,
        "fail_count": fail_count,
        "bundle_count": len(results),
        "errors": errors,
        "bundles": results,
    }

    report_name = f"{report_prefix.rstrip('/')}/{_format_timestamp(end_time)}_integrity_audit.json"
    report_blob = report_bucket.blob(report_name)
    report_payload = json.dumps(report, indent=2, sort_keys=True)
    report_blob.upload_from_string(report_payload, content_type="application/json")

    if metrics:
        metrics.storage_bytes_written.inc(
            len(report_payload.encode("utf-8")),
            labels={"job": JOB_NAME, "scene_id": "audit_report"},
        )

    _emit_log(
        "complete",
        {
            "job": JOB_NAME,
            "bucket": bucket_name,
            "prefix": prefix,
            "bundle_count": len(results),
            "pass_count": pass_count,
            "fail_count": fail_count,
            "report_uri": f"gs://{report_bucket_name}/{report_name}",
            "timestamp": end_time.isoformat(),
        },
    )

    if fail_count > 0 or errors:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
