"""Compute regression metrics for dataset bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


QUALITY_BUCKETS = [
    (0.0, 0.5, "0.0-0.5"),
    (0.5, 0.8, "0.5-0.8"),
    (0.8, 0.9, "0.8-0.9"),
    (0.9, 1.01, "0.9-1.0"),
]


@dataclass
class ChecksumReport:
    total_files: int
    missing_files: List[str]
    mismatched_files: List[str]

    @property
    def success(self) -> bool:
        return not self.missing_files and not self.mismatched_files

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_files": self.total_files,
            "missing_files": self.missing_files,
            "mismatched_files": self.mismatched_files,
            "success": self.success,
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_episode_payloads(recordings_dir: Path) -> List[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []
    for episode_file in sorted(recordings_dir.rglob("*.json")):
        try:
            payloads.append(json.loads(episode_file.read_text()))
        except json.JSONDecodeError:
            payloads.append({"episode_id": episode_file.stem})
    return payloads


def _bucket_quality(score: float) -> str:
    for lower, upper, label in QUALITY_BUCKETS:
        if lower <= score < upper:
            return label
    return QUALITY_BUCKETS[-1][2]


def _compute_quality_distribution(scores: Iterable[float]) -> Dict[str, int]:
    distribution = {label: 0 for _, _, label in QUALITY_BUCKETS}
    for score in scores:
        distribution[_bucket_quality(score)] += 1
    return distribution


def _derive_duration_seconds(payload: Dict[str, Any]) -> float:
    if isinstance(payload.get("duration_seconds"), (int, float)):
        return max(float(payload["duration_seconds"]), 0.0)
    frames = payload.get("frames", [])
    if frames:
        last_timestamp = frames[-1].get("timestamp")
        if isinstance(last_timestamp, (int, float)):
            return max(float(last_timestamp), 0.0)
    frame_count = payload.get("frame_count")
    if isinstance(frame_count, (int, float)) and frame_count > 0:
        return max(float(frame_count) / 30.0, 0.0)
    return 0.0


def _resolve_dataset_root(
    dataset_dir: Optional[Path], import_manifest_path: Optional[Path]
) -> Path:
    if import_manifest_path:
        return import_manifest_path.parent
    if dataset_dir is None:
        raise ValueError("dataset_dir is required when import_manifest_path is not provided.")
    return dataset_dir


def _resolve_checksums_path(dataset_root: Path, import_manifest_path: Optional[Path]) -> Optional[Path]:
    if import_manifest_path and import_manifest_path.exists():
        try:
            manifest = json.loads(import_manifest_path.read_text())
            checksums_rel = manifest.get("checksums_path")
            if checksums_rel:
                candidate = dataset_root / checksums_rel
                if candidate.exists():
                    return candidate
        except json.JSONDecodeError:
            return None
    fallback = dataset_root / "checksums.json"
    if fallback.exists():
        return fallback
    return None


def _verify_checksums(checksums_path: Optional[Path], dataset_root: Path) -> ChecksumReport:
    if not checksums_path or not checksums_path.exists():
        return ChecksumReport(total_files=0, missing_files=[], mismatched_files=[])
    payload = json.loads(checksums_path.read_text())
    files = payload.get("files", {})
    missing: List[str] = []
    mismatched: List[str] = []
    for rel_path, entry in files.items():
        expected_sha = entry.get("sha256")
        target = dataset_root / rel_path
        if not target.exists():
            missing.append(rel_path)
            continue
        if expected_sha and _sha256_file(target) != expected_sha:
            mismatched.append(rel_path)
    return ChecksumReport(
        total_files=len(files),
        missing_files=missing,
        mismatched_files=mismatched,
    )


def compute_regression_metrics(
    dataset_dir: Optional[Path] = None,
    *,
    import_manifest_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Compute regression metrics for a dataset directory or import manifest."""
    dataset_root = _resolve_dataset_root(dataset_dir, import_manifest_path)
    recordings_dir = dataset_root / "recordings"
    payloads = _load_episode_payloads(recordings_dir) if recordings_dir.exists() else []
    scores = [float(payload.get("quality_score", 0.0)) for payload in payloads]
    collisions = [int(payload.get("collision_count", 0)) for payload in payloads]
    durations = [_derive_duration_seconds(payload) for payload in payloads]
    duration_summary = {
        "average_seconds": float(sum(durations) / len(durations)) if durations else 0.0,
        "min_seconds": float(min(durations)) if durations else 0.0,
        "max_seconds": float(max(durations)) if durations else 0.0,
    }
    collision_rate = (
        float(sum(1 for count in collisions if count > 0) / len(collisions))
        if collisions
        else 0.0
    )
    checksums_path = _resolve_checksums_path(dataset_root, import_manifest_path)
    checksum_report = _verify_checksums(checksums_path, dataset_root)
    return {
        "episode_count": len(payloads),
        "quality_distribution": _compute_quality_distribution(scores),
        "quality_summary": {
            "average_score": float(sum(scores) / len(scores)) if scores else 0.0,
            "min_score": float(min(scores)) if scores else 0.0,
            "max_score": float(max(scores)) if scores else 0.0,
        },
        "collision_rate": collision_rate,
        "duration_seconds": duration_summary,
        "checksums": checksum_report.to_dict(),
        "checksums_path": checksums_path.as_posix() if checksums_path else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute dataset regression metrics.")
    parser.add_argument("--dataset-dir", type=Path, help="Path to dataset directory.")
    parser.add_argument(
        "--import-manifest",
        type=Path,
        help="Path to import_manifest.json (optional).",
    )
    args = parser.parse_args()
    metrics = compute_regression_metrics(args.dataset_dir, import_manifest_path=args.import_manifest)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
