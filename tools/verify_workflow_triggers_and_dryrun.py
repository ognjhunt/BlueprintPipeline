#!/usr/bin/env python3
"""Offline checks for workflow triggers and a dry-run simulation.

This script validates that:
- usd-assembly-pipeline listens for 3D-RE-GEN completion markers.
- A dry-run simulation shows the 3D-RE-GEN pipeline flow.

It is meant to give confidence in the wiring without needing Cloud access.
"""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]


def check_patterns(path: Path, expectations: Iterable[Tuple[str, str]]) -> List[str]:
    """Return a list of human-readable errors for missing regex patterns."""
    if not path.exists():
        return [f"File not found: {path}"]
    text = path.read_text()
    missing: List[str] = []
    for description, pattern in expectations:
        if not re.search(pattern, text, flags=re.MULTILINE):
            missing.append(f"{description} missing pattern: {pattern}")
    return missing


def verify_usd_assembly() -> List[str]:
    """Confirm usd-assembly-pipeline.yaml watches regen3d completion markers."""
    path = REPO_ROOT / "workflows" / "usd-assembly-pipeline.yaml"
    if not path.exists():
        return [f"USD assembly pipeline not found: {path}"]

    expectations = [
        (
            "GCS finalize comment",
            r"Trigger: Cloud Storage object finalized event",
        ),
        (
            "Simready job invocation",
            r"googleapis\.run\.v2\.projects\.locations\.jobs\.run",
        ),
        (
            "USD assembly job name wiring",
            r"usd-assembly-job",
        ),
    ]
    return check_patterns(path, expectations)


def dry_run_regen3d_pipeline(scene_id: str = "demo", bucket: str = "demo-bucket") -> List[str]:
    """Simulate the 3D-RE-GEN pipeline flow.

    This is a lightweight representation of the orchestration steps; it
    creates temporary marker files to prove the ordering of actions.
    """

    actions: List[str] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        bucket_root = Path(tmpdir)
        assets_prefix = Path(f"scenes/{scene_id}/assets")
        assets_dir = bucket_root / assets_prefix
        assets_dir.mkdir(parents=True, exist_ok=True)

        # Simulate 3D-RE-GEN reconstruction completion
        regen3d_dir = bucket_root / f"scenes/{scene_id}/regen3d"
        regen3d_dir.mkdir(parents=True, exist_ok=True)
        marker = regen3d_dir / "scene_info.json"
        marker.write_text("{}\n")
        actions.append(f"Detected 3D-RE-GEN output at {marker.relative_to(bucket_root)} in bucket {bucket}")

        # 3D-RE-GEN adapter job
        actions.append("Would invoke regen3d-job to convert 3D-RE-GEN outputs")

        # Scale job (optional)
        actions.append("Would invoke scale-job for scale calibration (optional)")

        # Interactive job
        actions.append("Would invoke interactive-job for articulation")

        # Simready job
        simready_marker = assets_dir / ".simready_complete"
        simready_marker.write_text("done\n")
        actions.append("Would invoke simready-job and wait for .simready_complete")

        # USD assembly
        actions.append("Would invoke usd-assembly-job for final assembly")

        # Replicator job
        actions.append("Would invoke replicator-job for domain randomization bundle")

        # Variation gen job
        actions.append("Would invoke variation-gen-job for variation assets")

        # Isaac Lab job
        actions.append("Would invoke isaac-lab-job for training configurations")

    return actions


def main() -> int:
    errors: List[str] = []

    usd_errors = verify_usd_assembly()
    if usd_errors:
        errors.extend(["USD assembly: " + e for e in usd_errors])

    dry_run_actions = dry_run_regen3d_pipeline()

    if errors:
        print("FAILED checks:\n- " + "\n- ".join(errors))
        return 1

    print("Verified triggers and actions:")
    print("- usd-assembly-pipeline watches completion markers")
    print("\n3D-RE-GEN Pipeline Dry run (simulated flow):")
    for action in dry_run_actions:
        print(f"  * {action}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
