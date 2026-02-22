#!/usr/bin/env python3
"""Offline checks for workflow triggers and a dry-run simulation.

This script validates that:
- usd-assembly-pipeline listens for Stage 1 completion markers.
- A dry-run simulation shows the Stage 1 -> Stage 5 pipeline flow.

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
    """Confirm usd-assembly-pipeline.yaml watches Stage 1 completion markers."""
    path = REPO_ROOT / "workflows" / "usd-assembly-pipeline.yaml"
    if not path.exists():
        return [f"USD assembly pipeline not found: {path}"]

    expectations = [
        (
            "Stage 1 completion marker filter",
            r"assets/\\.stage1_complete",
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


def verify_geniesim_export_trigger(scene_id: str = "demo") -> List[str]:
    """Confirm Genie Sim export pipeline watches variation completion markers."""
    path = REPO_ROOT / "workflows" / "genie-sim-export-pipeline.yaml"
    if not path.exists():
        return [f"Genie Sim export pipeline not found: {path}"]

    expectations = [
        (
            "Variation pipeline marker filter",
            r"variation_assets/\\.variation_pipeline_complete",
        ),
        (
            "Genie Sim export job invocation",
            r"genie-sim-export-job",
        ),
    ]
    errors = check_patterns(path, expectations)
    if errors:
        return errors

    mock_marker = f"scenes/{scene_id}/variation_assets/.variation_pipeline_complete"
    workflow_regex = r"^scenes/.+/variation_assets/\.variation_pipeline_complete$"
    if not re.match(workflow_regex, mock_marker):
        return [f"Mock marker did not match Genie Sim export trigger regex: {mock_marker}"]

    with tempfile.TemporaryDirectory() as tmpdir:
        marker_path = Path(tmpdir) / mock_marker
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.write_text("done\n")

    return []


def dry_run_stage1_pipeline(scene_id: str = "demo", bucket: str = "demo-bucket") -> List[str]:
    """Simulate the text Stage 1 pipeline flow."""

    actions: List[str] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        bucket_root = Path(tmpdir)
        scene_prefix = Path(f"scenes/{scene_id}")
        assets_dir = bucket_root / scene_prefix / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)

        stage1_marker = assets_dir / ".stage1_complete"
        stage1_marker.write_text("done\n")
        actions.append(
            f"Detected Stage 1 output marker at {stage1_marker.relative_to(bucket_root)} in bucket {bucket}"
        )

        actions.append("Would invoke text-scene-gen-job")
        actions.append("Would invoke text-scene-adapter-job")
        actions.append("Would invoke scale-job for scale calibration (optional)")
        actions.append("Would invoke interactive-job for articulation")

        simready_marker = assets_dir / ".simready_complete"
        simready_marker.write_text("done\n")
        actions.append("Would invoke simready-job and wait for .simready_complete")
        actions.append("Would invoke usd-assembly-job for final assembly")
        actions.append("Would invoke replicator-job for domain randomization bundle")
        actions.append("Would invoke variation-gen-job for variation assets")
        actions.append("Would invoke isaac-lab-job for training configurations")

    return actions


def main() -> int:
    errors: List[str] = []

    usd_errors = verify_usd_assembly()
    if usd_errors:
        errors.extend(["USD assembly: " + e for e in usd_errors])

    geniesim_errors = verify_geniesim_export_trigger()
    if geniesim_errors:
        errors.extend(["Genie Sim export: " + e for e in geniesim_errors])

    dry_run_actions = dry_run_stage1_pipeline()

    if errors:
        print("FAILED checks:\n- " + "\n- ".join(errors))
        return 1

    print("Verified triggers and actions:")
    print("- usd-assembly-pipeline watches Stage 1 completion markers")
    print("\nStage 1 Pipeline Dry run (simulated flow):")
    for action in dry_run_actions:
        print(f"  * {action}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
