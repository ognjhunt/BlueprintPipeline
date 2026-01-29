#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from import_manifest_utils import (
    MANIFEST_SCHEMA_VERSION,
    compute_manifest_checksum,
    compute_sha256,
)


def _verify_file_inventory(output_dir: Path, inventory: List[Dict[str, Any]]) -> List[str]:
    errors: List[str] = []
    for entry in inventory:
        rel_path = entry.get("path")
        if not rel_path:
            errors.append("Inventory entry missing path.")
            continue
        path = output_dir / rel_path
        if not path.exists():
            errors.append(f"Missing inventory file: {rel_path}")
            continue
        expected_size = entry.get("size_bytes")
        if expected_size is not None and path.stat().st_size != expected_size:
            errors.append(
                f"Size mismatch for {rel_path}: expected {expected_size}, got {path.stat().st_size}"
            )
    return errors


def _verify_checksum_map(
    output_dir: Path,
    checksum_map: Any,
    manifest: Dict[str, Any],
) -> List[str]:
    errors: List[str] = []
    if isinstance(checksum_map, list):
        items = ((entry.get("file_name", ""), entry) for entry in checksum_map)
    elif isinstance(checksum_map, dict):
        items = checksum_map.items()
    else:
        return errors
    for rel_path, checksum_entry in items:
        expected_sha = checksum_entry.get("sha256")
        if not expected_sha:
            continue
        if rel_path == "import_manifest.json":
            actual_sha = compute_manifest_checksum(manifest)
        else:
            path = output_dir / rel_path
            if not path.exists():
                errors.append(f"Missing checksum file: {rel_path}")
                continue
            actual_sha = compute_sha256(path)
            expected_size = checksum_entry.get("size_bytes")
            if expected_size is not None and path.stat().st_size != expected_size:
                errors.append(
                    f"Size mismatch for {rel_path}: expected {expected_size}, got {path.stat().st_size}"
                )
        if actual_sha != expected_sha:
            errors.append(f"Checksum mismatch for {rel_path}: expected {expected_sha}, got {actual_sha}")
    return errors


def verify_manifest(manifest_path: Path) -> int:
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    errors: List[str] = []
    output_dir = Path(manifest.get("output_dir", ""))
    if not output_dir.exists():
        errors.append(f"Output directory does not exist: {output_dir}")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        errors.append(
            f"Unsupported schema_version: {manifest.get('schema_version')} (expected {MANIFEST_SCHEMA_VERSION})"
        )

    inventory = manifest.get("file_inventory", [])
    if output_dir.exists():
        errors.extend(_verify_file_inventory(output_dir, inventory))

    checksums = manifest.get("checksums", {})
    metadata_checksums = checksums.get("metadata", {})
    episode_checksums = checksums.get("episodes", {})
    missing_episode_ids = checksums.get("missing_episode_ids", [])
    missing_metadata_files = checksums.get("missing_metadata_files", [])
    if missing_episode_ids:
        errors.append(f"Missing episode files recorded in manifest: {missing_episode_ids}")
    if missing_metadata_files:
        errors.append(f"Missing metadata files recorded in manifest: {missing_metadata_files}")
    if output_dir.exists():
        errors.extend(_verify_checksum_map(output_dir, episode_checksums, manifest))
        errors.extend(_verify_checksum_map(output_dir, metadata_checksums, manifest))

    if errors:
        print("[VERIFY] ❌ Manifest verification failed:")
        for error in errors:
            print(f"[VERIFY]   - {error}")
        return 1

    print("[VERIFY] ✅ Manifest verification succeeded")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify import_manifest.json checksums and inventory.")
    parser.add_argument("manifest_path", type=Path, help="Path to import_manifest.json")
    args = parser.parse_args()

    sys.exit(verify_manifest(args.manifest_path))


if __name__ == "__main__":
    main()
