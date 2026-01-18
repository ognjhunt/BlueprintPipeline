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
import os
import sys
import tempfile
import traceback
from datetime import datetime
from typing import Any, Dict, Tuple

import requests
from google.cloud import storage


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


def send_webhook(webhook_url: str, payload: Dict[str, Any]) -> None:
    response = requests.post(webhook_url, json=payload, timeout=15)
    if response.status_code >= 400:
        raise RuntimeError(
            f"Webhook {webhook_url} failed with {response.status_code}: {response.text}"
        )


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


def main() -> int:
    bucket_name = os.environ.get("BUCKET")
    scene_id = os.environ.get("SCENE_ID") or ""
    job_id = os.environ.get("JOB_ID") or ""
    import_manifest_path = os.environ.get("IMPORT_MANIFEST_PATH")
    delivery_prefix_template = os.environ.get("DELIVERY_PREFIX", "deliveries/{scene_id}/{job_id}")

    if not bucket_name:
        print("BUCKET is required", file=sys.stderr)
        return 1

    if not import_manifest_path:
        if not scene_id:
            print("SCENE_ID is required when IMPORT_MANIFEST_PATH is not set", file=sys.stderr)
            return 1
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

        dataset_card = build_dataset_card(import_manifest, scene_id, job_id)

        lab_delivery_buckets = parse_mapping(os.environ.get("LAB_DELIVERY_BUCKETS"))
        lab_webhook_urls = parse_mapping(os.environ.get("LAB_WEBHOOK_URLS"))
        default_bucket = os.environ.get("DEFAULT_DELIVERY_BUCKET")
        if default_bucket and "default" not in lab_delivery_buckets:
            lab_delivery_buckets["default"] = default_bucket

        if not lab_delivery_buckets:
            raise ValueError("No delivery buckets configured; set LAB_DELIVERY_BUCKETS or DEFAULT_DELIVERY_BUCKET")

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_card_path = os.path.join(tmpdir, "dataset_card.json")
            with open(dataset_card_path, "w", encoding="utf-8") as file_handle:
                json.dump(dataset_card, file_handle, indent=2)

            source_bucket_name, package_object = parse_gs_uri(package_uri)
            source_bucket = client.bucket(source_bucket_name)
            source_blob = source_bucket.blob(package_object)

            for lab, dest_bucket_name in lab_delivery_buckets.items():
                delivery_prefix = delivery_prefix_template.format(scene_id=scene_id, job_id=job_id, lab=lab).strip("/")
                package_object_name = f"{delivery_prefix}/{package_rel_path}"
                dataset_card_object_name = f"{delivery_prefix}/dataset_card.json"

                dest_bucket = client.bucket(dest_bucket_name)
                source_bucket.copy_blob(source_blob, dest_bucket, new_name=package_object_name)
                dest_bucket.blob(dataset_card_object_name).upload_from_filename(dataset_card_path)

                dataset_card_url = f"gs://{dest_bucket_name}/{dataset_card_object_name}"
                bundle_url = f"gs://{dest_bucket_name}/{package_object_name}"

                webhook_url = lab_webhook_urls.get(lab)
                if webhook_url:
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
