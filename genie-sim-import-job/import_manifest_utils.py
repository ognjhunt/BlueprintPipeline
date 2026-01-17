import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

MANIFEST_SCHEMA_VERSION = "1.2"
MANIFEST_SCHEMA_DEFINITION = {
    "version": MANIFEST_SCHEMA_VERSION,
    "description": "Genie Sim import manifest schema for downstream workflow integration.",
    "fields": {
        "schema_version": "Schema version string.",
        "generated_at": "UTC timestamp for manifest generation.",
        "output_dir": "Output directory relative to the bundle root (or '.' for the root).",
        "gcs_output_path": "Optional GCS URI pointing at the bundle root containing output_dir.",
        "readme_path": "Relative path to the bundle README describing LeRobot usage.",
        "checksums_path": "Relative path to the checksums JSON file for bundle artifacts.",
        "asset_provenance_path": "Relative path or URI to asset provenance JSON for legal review.",
        "package": "Packaged bundle archive details (path, checksum, size, format).",
        "episodes": "Episode download summary metrics.",
        "quality": "Quality scoring summary and thresholds.",
        "lerobot": "LeRobot conversion status and outputs.",
        "metrics_summary": "Pipeline metrics snapshot.",
        "file_inventory": "List of all output files (path + size).",
        "checksums": "SHA256 checksums for episodes, metadata, and full bundle files.",
        "provenance": "Source control, pipeline version, config snapshot, and tool versions.",
    },
    "notes": [
        "file_inventory is scoped to output_dir and excludes import_manifest.json to avoid self-reference.",
        "checksums.metadata['import_manifest.json'] is computed from a canonical JSON representation of the "
        "manifest with that checksum entry removed.",
    ],
}

ENV_SNAPSHOT_KEYS = [
    "BUCKET",
    "GENIE_SIM_JOB_ID",
    "GENIE_SIM_POLL_INTERVAL",
    "OUTPUT_PREFIX",
    "MIN_QUALITY_SCORE",
    "ENABLE_VALIDATION",
    "FILTER_LOW_QUALITY",
    "REQUIRE_LEROBOT",
    "WAIT_FOR_COMPLETION",
    "FAIL_ON_PARTIAL_ERROR",
    "SCENE_ID",
    "GENIESIM_SUBMISSION_MODE",
    "JOB_METADATA_PATH",
    "LOCAL_EPISODES_PREFIX",
]

PIPELINE_VERSION_KEYS = [
    "PIPELINE_VERSION",
    "IMAGE_VERSION",
    "BUILD_VERSION",
]

UPSTREAM_VERSION_KEYS = {
    "genie_sim": ["GENIE_SIM_VERSION", "GENIE_SIM_BUILD"],
    "regen3d": ["REGEN3D_VERSION", "THREED_REGEN_VERSION", "REGEN_VERSION"],
}


def compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def iter_files_sorted(root: Path) -> List[Path]:
    files = [path for path in root.rglob("*") if path.is_file()]
    return sorted(files, key=lambda path: path.as_posix())


def build_file_inventory(root: Path, exclude_paths: Optional[Iterable[Path]] = None) -> List[Dict[str, Any]]:
    exclude_set = {path.resolve() for path in exclude_paths or []}
    inventory = []
    for path in iter_files_sorted(root):
        if path.resolve() in exclude_set:
            continue
        inventory.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
            }
        )
    return inventory


def build_checksums_map(root: Path, paths: Iterable[Path]) -> Dict[str, Dict[str, Any]]:
    checksums: Dict[str, Dict[str, Any]] = {}
    for path in sorted(paths, key=lambda path: path.as_posix()):
        if not path.exists():
            continue
        checksums[path.relative_to(root).as_posix()] = {
            "sha256": compute_sha256(path),
            "size_bytes": path.stat().st_size,
        }
    return checksums


def build_directory_checksums(
    root: Path,
    exclude_paths: Optional[Iterable[Path]] = None,
) -> Dict[str, Dict[str, Any]]:
    exclude_set = {path.resolve() for path in exclude_paths or []}
    checksums: Dict[str, Dict[str, Any]] = {}
    for path in iter_files_sorted(root):
        if path.resolve() in exclude_set:
            continue
        rel_path = path.relative_to(root).as_posix()
        checksums[rel_path] = {
            "sha256": compute_sha256(path),
            "size_bytes": path.stat().st_size,
        }
    return checksums


def get_episode_file_paths(
    output_dir: Path,
    episode_ids: Iterable[str],
) -> Tuple[List[Path], List[str]]:
    paths: List[Path] = []
    missing: List[str] = []
    for episode_id in sorted(set(episode_ids)):
        primary = output_dir / f"{episode_id}.parquet"
        filtered = output_dir / "filtered" / f"{episode_id}.parquet"
        if primary.exists():
            paths.append(primary)
        elif filtered.exists():
            paths.append(filtered)
        else:
            missing.append(episode_id)
    return paths, missing


def get_lerobot_metadata_paths(output_dir: Path) -> List[Path]:
    candidates = [
        output_dir / "lerobot" / "meta" / "info.json",
        output_dir / "lerobot" / "dataset_info.json",
        output_dir / "lerobot" / "episodes.jsonl",
    ]
    return [path for path in candidates if path.exists()]


def compute_manifest_checksum(manifest: Dict[str, Any]) -> str:
    manifest_copy = json.loads(json.dumps(manifest))
    checksums = manifest_copy.get("checksums", {})
    metadata_checksums = checksums.get("metadata", {})
    metadata_checksums.pop("import_manifest.json", None)
    if "metadata" in checksums:
        checksums["metadata"] = metadata_checksums
    manifest_copy["checksums"] = checksums
    payload = json.dumps(manifest_copy, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def snapshot_env(keys: Iterable[str]) -> Dict[str, Optional[str]]:
    return {key: os.getenv(key) for key in keys}


def get_git_sha(repo_root: Path) -> Optional[str]:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root)
            .decode("utf-8")
            .strip()
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def get_pipeline_version() -> Optional[str]:
    for key in PIPELINE_VERSION_KEYS:
        value = os.getenv(key)
        if value:
            return value
    return None


def get_upstream_versions() -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {}
    for tool, keys in UPSTREAM_VERSION_KEYS.items():
        versions[tool] = next((os.getenv(key) for key in keys if os.getenv(key)), None)
    return versions


def collect_provenance(repo_root: Path, config_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "git_sha": get_git_sha(repo_root),
        "pipeline_version": get_pipeline_version(),
        "config_snapshot": config_snapshot,
        "upstream_versions": get_upstream_versions(),
    }
