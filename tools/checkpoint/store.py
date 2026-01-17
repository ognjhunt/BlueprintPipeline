"""Checkpoint storage helpers for local pipeline runs.

Audit note: this module only writes local filesystem checkpoints today, so there
is no GCS-backed checkpoint store in this repository to apply optimistic
locking against.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
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
) -> Path:
    """Write a checkpoint record to disk."""
    target_dir = checkpoint_dir(scene_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    record = CheckpointRecord(
        step=step,
        status=status,
        started_at=started_at,
        completed_at=completed_at,
        outputs=outputs or {},
        output_paths=[str(path) for path in (output_paths or [])],
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


def should_skip_step(
    scene_dir: Path,
    step: str,
    *,
    expected_outputs: Optional[List[Path]] = None,
) -> bool:
    """Return True if a step checkpoint exists and outputs are present."""
    checkpoint = load_checkpoint(scene_dir, step)
    if not checkpoint or checkpoint.status != "completed":
        return False
    output_paths = checkpoint.output_paths
    if expected_outputs:
        output_paths = [str(path) for path in expected_outputs]
    return _outputs_exist(output_paths)
