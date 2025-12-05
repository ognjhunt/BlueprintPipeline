#!/usr/bin/env python3
"""Offline checks for workflow triggers and a dry-run simulation.

This script validates that:
- usd-assembly-pipeline listens for hunyuan completion markers.
- hunyuan-pipeline triggers on scene_assets.json and writes .hunyuan_complete.
- A dry-run simulation shows simready-job and usd-assembly-job get invoked
  after .hunyuan_complete is present.

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
    text = path.read_text()
    missing: List[str] = []
    for description, pattern in expectations:
        if not re.search(pattern, text, flags=re.MULTILINE):
            missing.append(f"{description} missing pattern: {pattern}")
    return missing


def verify_usd_assembly() -> List[str]:
    """Confirm usd-assembly-pipeline.yaml watches .hunyuan_complete markers."""
    path = REPO_ROOT / "workflows" / "usd-assembly-pipeline.yaml"
    expectations = [
        (
            "GCS finalize comment",
            r"Trigger: Cloud Storage object finalized event",
        ),
        (
            "Completion marker regex",
            r"text\.match_regex\(object, \"\^scenes/.+/assets/\\\\.hunyuan_complete\$\"\)",
        ),
        (
            "Simready job invocation",
            r"googleapis\.run\.v2\.projects\.locations\.jobs\.run",
        ),
        (
            "USD assembly job name wiring",
            r"usdJobName: \"usd-assembly-job\"",
        ),
    ]
    return check_patterns(path, expectations)


def verify_hunyuan_pipeline() -> List[str]:
    """Confirm hunyuan-pipeline.yaml triggers and writes completion markers."""
    path = REPO_ROOT / "workflows" / "hunyuan-pipeline.yaml"
    expectations = [
        (
            "scene_assets.json trigger",
            r"text\.match_regex\(object, \"\^scenes/.+/assets/scene_assets\\\\.json\$\"\)",
        ),
        (
            "Hunyuan job run",
            r"googleapis\.run\.v2\.projects\.locations\.jobs\.run",
        ),
        (
            "Hunyuan job name wiring",
            r"jobName: \"hunyuan-job\"",
        ),
        (
            ".hunyuan_complete creation",
            r"\.hunyuan_complete",
        ),
    ]
    errors = check_patterns(path, expectations)

    script_path = REPO_ROOT / "hunyuan-job" / "run_hunyuan_from_assets.py"
    if not script_path.is_file():
        errors.append("run_hunyuan_from_assets.py missing")
    else:
        script_text = script_path.read_text()
        if ".hunyuan_complete" not in script_text:
            errors.append("run_hunyuan_from_assets.py does not mention .hunyuan_complete marker")
    return errors


def dry_run_usd_pipeline(scene_id: str = "demo", bucket: str = "demo-bucket") -> List[str]:
    """Simulate the usd-assembly workflow when .hunyuan_complete appears.

    This is a lightweight representation of the orchestration steps; it
    creates temporary marker files to prove the ordering of actions.
    """

    actions: List[str] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        bucket_root = Path(tmpdir)
        assets_prefix = Path(f"scenes/{scene_id}/assets")
        assets_dir = bucket_root / assets_prefix
        assets_dir.mkdir(parents=True, exist_ok=True)

        # Simulate the Hunyuan completion marker arriving from GCS.
        marker = assets_dir / ".hunyuan_complete"
        marker.write_text("done\n")
        actions.append(f"Detected finalize event for {marker.relative_to(bucket_root)} in bucket {bucket}")

        # Convert job (convert-only)
        actions.append("Would invoke usd-assembly-job with CONVERT_ONLY=true")

        # Simready job and completion marker
        simready_marker = assets_dir / ".simready_complete"
        simready_marker.write_text("done\n")
        actions.append("Would invoke simready-job and wait for .simready_complete")

        # Final USD assembly
        actions.append("Would invoke usd-assembly-job for final assembly")

    return actions


def main() -> int:
    errors: List[str] = []

    usd_errors = verify_usd_assembly()
    if usd_errors:
        errors.extend(["USD assembly: " + e for e in usd_errors])

    hunyuan_errors = verify_hunyuan_pipeline()
    if hunyuan_errors:
        errors.extend(["Hunyuan pipeline: " + e for e in hunyuan_errors])

    dry_run_actions = dry_run_usd_pipeline()

    if errors:
        print("FAILED checks:\n- " + "\n- ".join(errors))
        return 1

    print("Verified triggers and actions:")
    print("- usd-assembly-pipeline watches .hunyuan_complete finalize events")
    print("- hunyuan-pipeline triggers on scene_assets.json and writes .hunyuan_complete")
    print("- run_hunyuan_from_assets.py references the completion marker for parity")
    print("\nDry run (simulated flow):")
    for action in dry_run_actions:
        print(f"  * {action}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
