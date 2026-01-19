import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

SCHEMA_VERSION = "1.0"


def compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def iter_files_sorted(root: Path) -> List[Path]:
    files = [path for path in root.rglob("*") if path.is_file()]
    return sorted(files, key=lambda path: path.as_posix())


def build_checksums_map(root: Path, paths: Iterable[Path]) -> Dict[str, Dict[str, Any]]:
    checksums: Dict[str, Dict[str, Any]] = {}
    for path in sorted(paths, key=lambda path: path.as_posix()):
        rel_path = path.relative_to(root).as_posix()
        checksums[rel_path] = {
            "sha256": compute_sha256(path),
            "size_bytes": path.stat().st_size,
        }
    return checksums


def write_source_checksums(
    checksum_path: Path,
    root: Path,
    paths: Iterable[Path],
) -> Dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "root": root.as_posix(),
        "files": build_checksums_map(root, paths),
    }
    checksum_path.write_text(json.dumps(payload, indent=2))
    return payload


def load_source_checksums(checksum_path: Path) -> Dict[str, Any]:
    return json.loads(checksum_path.read_text())


def verify_source_checksums(checksum_path: Path, root: Path) -> Dict[str, Any]:
    result = {
        "success": False,
        "errors": [],
        "missing_files": [],
        "checksum_mismatches": [],
        "size_mismatches": [],
    }
    try:
        payload = load_source_checksums(checksum_path)
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        result["errors"].append(f"Failed to load {checksum_path}: {exc}")
        return result

    files = payload.get("files")
    if not isinstance(files, dict):
        result["errors"].append("source_checksums.json missing 'files' mapping")
        return result

    for rel_path, entry in files.items():
        file_path = root / rel_path
        if not file_path.exists():
            result["missing_files"].append(rel_path)
            continue
        expected_sha = entry.get("sha256") if isinstance(entry, dict) else None
        expected_size = entry.get("size_bytes") if isinstance(entry, dict) else None
        actual_sha = compute_sha256(file_path)
        if expected_sha and actual_sha != expected_sha:
            result["checksum_mismatches"].append(
                {
                    "path": rel_path,
                    "expected": expected_sha,
                    "actual": actual_sha,
                }
            )
        if expected_size is not None:
            actual_size = file_path.stat().st_size
            if actual_size != expected_size:
                result["size_mismatches"].append(
                    {
                        "path": rel_path,
                        "expected": expected_size,
                        "actual": actual_size,
                    }
                )

    success = (
        not result["errors"]
        and not result["missing_files"]
        and not result["checksum_mismatches"]
        and not result["size_mismatches"]
    )
    result["success"] = success
    return result
