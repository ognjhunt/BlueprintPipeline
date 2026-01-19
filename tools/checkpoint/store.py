"""Checkpoint storage helpers for local pipeline runs.

Audit note: this module only writes local filesystem checkpoints today, so there
is no GCS-backed checkpoint store in this repository to apply optimistic
locking against.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class CheckpointRecord:
    """Serialized checkpoint information for a pipeline step."""

    step: str
    status: str
    started_at: str
    completed_at: str
    outputs: Dict[str, Any]
    output_paths: List[str]
    output_hashes: Dict[str, str]
    scene_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert the record to a JSON-serializable dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "CheckpointRecord":
        """Create a checkpoint record from a dict payload."""
        return cls(
            step=payload.get("step", ""),
            status=payload.get("status", ""),
            started_at=payload.get("started_at", ""),
            completed_at=payload.get("completed_at", ""),
            outputs=payload.get("outputs", {}) or {},
            output_paths=payload.get("output_paths", []) or [],
            output_hashes=payload.get("output_hashes", {}) or {},
            scene_id=payload.get("scene_id"),
        )


def checkpoint_dir(scene_dir: Path) -> Path:
    """Return the checkpoint directory for a scene."""
    return Path(scene_dir) / ".checkpoints"


def checkpoint_path(scene_dir: Path, step: str) -> Path:
    """Return the checkpoint file path for a step."""
    return checkpoint_dir(scene_dir) / f"{step}.json"


def write_checkpoint(
    scene_dir: Path,
    step: str,
    *,
    status: str,
    started_at: str,
    completed_at: str,
    outputs: Optional[Dict[str, Any]] = None,
    output_paths: Optional[List[Path]] = None,
    scene_id: Optional[str] = None,
    store_output_hashes: bool = False,
) -> Path:
    """Write a checkpoint record to disk."""
    target_dir = checkpoint_dir(scene_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    serialized_paths = [str(path) for path in (output_paths or [])]
    output_hashes: Dict[str, str] = {}
    if store_output_hashes and serialized_paths:
        output_hashes = _compute_output_hashes(serialized_paths)
    record = CheckpointRecord(
        step=step,
        status=status,
        started_at=started_at,
        completed_at=completed_at,
        outputs=outputs or {},
        output_paths=serialized_paths,
        output_hashes=output_hashes,
        scene_id=scene_id,
    )
    path = checkpoint_path(scene_dir, step)
    path.write_text(json.dumps(record.to_dict(), indent=2))
    return path


def load_checkpoint(scene_dir: Path, step: str) -> Optional[CheckpointRecord]:
    """Load a checkpoint record if it exists."""
    path = checkpoint_path(scene_dir, step)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    return CheckpointRecord.from_dict(payload)


def _outputs_exist(output_paths: List[str]) -> bool:
    if not output_paths:
        return False
    return all(Path(path).exists() for path in output_paths)


def _output_is_nonempty(path: Path) -> bool:
    if path.is_dir():
        return any(path.iterdir())
    if not path.exists():
        return False
    return path.stat().st_size > 0


def _sidecar_metadata_paths(path: Path) -> List[Path]:
    return [
        Path(f"{path}.metadata.json"),
        Path(f"{path}.meta.json"),
    ]


def _parse_iso8601(timestamp: str) -> Optional[datetime]:
    if not timestamp:
        return None
    normalized = timestamp.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _output_is_fresh(path: Path, completed_at: Optional[datetime]) -> bool:
    if completed_at is None or not path.exists():
        return True
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return mtime >= completed_at


def _calculate_file_hash(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compute_output_hashes(output_paths: List[str]) -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    for output_path in output_paths:
        path = Path(output_path)
        output_hash = _calculate_file_hash(path)
        if output_hash:
            hashes[output_path] = output_hash
    return hashes


def _outputs_match_hashes(output_hashes: Dict[str, str]) -> bool:
    for output_path, expected_hash in output_hashes.items():
        current_hash = _calculate_file_hash(Path(output_path))
        if current_hash != expected_hash:
            return False
    return True


def should_skip_step(
    scene_dir: Path,
    step: str,
    *,
    expected_outputs: Optional[List[Path]] = None,
    require_nonempty: bool = False,
    require_fresh_outputs: bool = False,
    validate_sidecar_metadata: bool = False,
    validate_output_hashes: bool = True,
) -> bool:
    """Return True if a step checkpoint exists and outputs are present."""
    checkpoint = load_checkpoint(scene_dir, step)
    if not checkpoint or checkpoint.status != "completed":
        return False
    output_paths = checkpoint.output_paths
    if expected_outputs:
        output_paths = [str(path) for path in expected_outputs]
    if not _outputs_exist(output_paths):
        return False
    if validate_output_hashes and checkpoint.output_hashes:
        relevant_hashes = {
            path: digest
            for path, digest in checkpoint.output_hashes.items()
            if path in output_paths
        }
        if relevant_hashes and not _outputs_match_hashes(relevant_hashes):
            return False
    completed_at = _parse_iso8601(checkpoint.completed_at)
    for output_path in output_paths:
        path = Path(output_path)
        if require_nonempty and not _output_is_nonempty(path):
            return False
        if require_fresh_outputs and not _output_is_fresh(path, completed_at):
            return False
        if validate_sidecar_metadata:
            for sidecar in _sidecar_metadata_paths(path):
                if sidecar.exists() and not _output_is_nonempty(sidecar):
                    return False
    return True
