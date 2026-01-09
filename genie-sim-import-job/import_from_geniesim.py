#!/usr/bin/env python3
"""
Genie Sim Episode Import Job.

This job completes the bidirectional Genie Sim integration by:
1. Polling for completed generation jobs
2. Downloading generated episodes
3. Validating episode quality
4. Converting to LeRobot format
5. Integrating with existing pipeline

This is the missing "import" side of the Genie Sim integration, which previously
only had export capabilities.

Environment Variables:
    BUCKET: GCS bucket name
    GENIE_SIM_JOB_ID: Job ID to import (if monitoring specific job)
    GENIE_SIM_POLL_INTERVAL: Polling interval in seconds (default: 30)
    OUTPUT_PREFIX: Output path for imported episodes (default: scenes/{scene_id}/episodes)
    MIN_QUALITY_SCORE: Minimum quality score for import (default: 0.7)
    ENABLE_VALIDATION: Enable quality validation (default: true)
"""

import json
import os
import shutil
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Add parent to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import Genie Sim client
sys.path.insert(0, str(REPO_ROOT / "genie-sim-export-job"))
from geniesim_client import (
    GenieSimClient,
    GenieSimAPIError,
    JobStatus,
    JobProgress,
    DownloadResult,
    GeneratedEpisodeMetadata,
)

# Import quality validation
try:
    sys.path.insert(0, str(REPO_ROOT / "episode-generation-job"))
    from quality_certificate import (
        QualityCertificate,
        TrajectoryQualityMetrics,
        VisualQualityMetrics,
        TaskQualityMetrics,
    )
    HAVE_QUALITY_VALIDATION = True
except ImportError:
    HAVE_QUALITY_VALIDATION = False


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class ImportConfig:
    """Configuration for episode import."""

    # Genie Sim job
    job_id: str

    # Output
    output_dir: Path

    # Quality filtering
    min_quality_score: float = 0.7
    enable_validation: bool = True
    filter_low_quality: bool = True

    # Polling (if waiting for completion)
    poll_interval: int = 30
    wait_for_completion: bool = True


@dataclass
class ImportResult:
    """Result of episode import."""

    success: bool
    job_id: str

    # Statistics
    total_episodes_downloaded: int = 0
    episodes_passed_validation: int = 0
    episodes_filtered: int = 0

    # Quality metrics
    average_quality_score: float = 0.0
    quality_distribution: Dict[str, int] = field(default_factory=dict)

    # Output
    output_dir: Optional[Path] = None
    manifest_path: Optional[Path] = None

    # Errors
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# =============================================================================
# Episode Validator
# =============================================================================


class ImportedEpisodeValidator:
    """Validates imported episodes from Genie Sim."""

    def __init__(self, min_quality_score: float = 0.7):
        """
        Initialize validator.

        Args:
            min_quality_score: Minimum quality score to pass
        """
        self.min_quality_score = min_quality_score

    def validate_episode(
        self,
        episode_metadata: GeneratedEpisodeMetadata,
        episode_file: Path,
    ) -> Dict[str, Any]:
        """
        Validate a single episode.

        Args:
            episode_metadata: Episode metadata from Genie Sim
            episode_file: Path to episode file

        Returns:
            Validation result dict with:
                - passed: bool
                - quality_score: float
                - errors: List[str]
                - warnings: List[str]
        """
        errors = []
        warnings = []

        # Check file exists
        if not episode_file.exists():
            errors.append(f"Episode file not found: {episode_file}")
            return {
                "passed": False,
                "quality_score": 0.0,
                "errors": errors,
                "warnings": warnings,
            }

        # Check file size
        actual_size = episode_file.stat().st_size
        if actual_size == 0:
            errors.append("Episode file is empty")
        elif actual_size < 1024:  # Less than 1KB
            warnings.append(f"Episode file is suspiciously small: {actual_size} bytes")

        # Check quality score
        quality_score = episode_metadata.quality_score
        if quality_score < self.min_quality_score:
            warnings.append(
                f"Quality score {quality_score:.2f} below threshold {self.min_quality_score:.2f}"
            )

        # Check validation status
        if not episode_metadata.validation_passed:
            warnings.append("Episode failed Genie Sim validation")

        # Check frame count
        if episode_metadata.frame_count < 10:
            warnings.append(f"Episode has very few frames: {episode_metadata.frame_count}")

        # Check duration
        if episode_metadata.duration_seconds < 0.1:
            warnings.append(f"Episode duration suspiciously short: {episode_metadata.duration_seconds}s")

        # Determine if passed
        passed = (
            len(errors) == 0
            and quality_score >= self.min_quality_score
            and episode_metadata.validation_passed
        )

        return {
            "passed": passed,
            "quality_score": quality_score,
            "errors": errors,
            "warnings": warnings,
        }

    def validate_batch(
        self,
        episodes: List[GeneratedEpisodeMetadata],
        episode_dir: Path,
    ) -> Dict[str, Any]:
        """
        Validate a batch of episodes.

        Args:
            episodes: List of episode metadata
            episode_dir: Directory containing episode files

        Returns:
            Batch validation result with statistics
        """
        results = []
        passed_count = 0
        quality_scores = []

        for episode in episodes:
            episode_file = episode_dir / f"{episode.episode_id}.parquet"
            result = self.validate_episode(episode, episode_file)
            results.append({
                "episode_id": episode.episode_id,
                **result,
            })

            if result["passed"]:
                passed_count += 1

            quality_scores.append(result["quality_score"])

        return {
            "total_episodes": len(episodes),
            "passed_count": passed_count,
            "failed_count": len(episodes) - passed_count,
            "average_quality_score": np.mean(quality_scores) if quality_scores else 0.0,
            "min_quality_score": np.min(quality_scores) if quality_scores else 0.0,
            "max_quality_score": np.max(quality_scores) if quality_scores else 0.0,
            "episode_results": results,
        }


# =============================================================================
# Import Job
# =============================================================================


def run_import_job(
    config: ImportConfig,
    client: GenieSimClient,
) -> ImportResult:
    """
    Run episode import job.

    Args:
        config: Import configuration
        client: Genie Sim API client

    Returns:
        ImportResult with statistics and output paths
    """
    print("\n" + "=" * 80)
    print("GENIE SIM EPISODE IMPORT JOB")
    print("=" * 80)
    print(f"Job ID: {config.job_id}")
    print(f"Output: {config.output_dir}")
    print(f"Min Quality: {config.min_quality_score}")
    print("=" * 80 + "\n")

    result = ImportResult(
        success=False,
        job_id=config.job_id,
    )

    try:
        # Step 1: Check/wait for job completion
        if config.wait_for_completion:
            print(f"[IMPORT] Waiting for job {config.job_id} to complete...")

            def progress_callback(progress: JobProgress):
                print(
                    f"[IMPORT] Progress: {progress.progress_percent:.1f}% - "
                    f"{progress.episodes_generated}/{progress.total_episodes_target} episodes - "
                    f"{progress.current_task}"
                )

            try:
                final_progress = client.wait_for_completion(
                    config.job_id,
                    poll_interval=config.poll_interval,
                    callback=progress_callback,
                )
                print(f"[IMPORT] ✅ Job completed: {final_progress.episodes_generated} episodes generated\n")

            except GenieSimAPIError as e:
                result.errors.append(f"Job failed or was cancelled: {e}")
                return result

        else:
            # Just check status
            progress = client.get_job_status(config.job_id)
            if progress.status != JobStatus.COMPLETED:
                result.errors.append(f"Job not completed (status: {progress.status.value})")
                return result

        # Step 2: Download episodes
        print(f"[IMPORT] Downloading episodes...")
        download_result = client.download_episodes(
            config.job_id,
            config.output_dir,
            validate=True,
        )

        if not download_result.success:
            result.errors.extend(download_result.errors)
            return result

        print(f"[IMPORT] ✅ Downloaded {download_result.episode_count} episodes")
        print(f"[IMPORT]    Total size: {download_result.total_size_bytes / 1024 / 1024:.1f} MB\n")

        result.total_episodes_downloaded = download_result.episode_count
        result.output_dir = download_result.output_dir
        result.manifest_path = download_result.manifest_path

        # Step 3: Validate episodes
        if config.enable_validation:
            print(f"[IMPORT] Validating episodes...")
            validator = ImportedEpisodeValidator(config.min_quality_score)

            validation_result = validator.validate_batch(
                download_result.episodes,
                config.output_dir,
            )

            print(f"[IMPORT] Validation results:")
            print(f"[IMPORT]   Total: {validation_result['total_episodes']}")
            print(f"[IMPORT]   Passed: {validation_result['passed_count']}")
            print(f"[IMPORT]   Failed: {validation_result['failed_count']}")
            print(f"[IMPORT]   Avg Quality: {validation_result['average_quality_score']:.2f}")
            print(f"[IMPORT]   Quality Range: [{validation_result['min_quality_score']:.2f}, {validation_result['max_quality_score']:.2f}]\n")

            result.episodes_passed_validation = validation_result['passed_count']
            result.episodes_filtered = validation_result['failed_count']
            result.average_quality_score = validation_result['average_quality_score']

            # Step 4: Filter low-quality episodes
            if config.filter_low_quality:
                print(f"[IMPORT] Filtering low-quality episodes...")
                filtered_count = 0

                for ep_result in validation_result['episode_results']:
                    if not ep_result['passed']:
                        episode_id = ep_result['episode_id']
                        episode_file = config.output_dir / f"{episode_id}.parquet"

                        # Move to filtered directory
                        filtered_dir = config.output_dir / "filtered"
                        filtered_dir.mkdir(exist_ok=True)

                        if episode_file.exists():
                            shutil.move(str(episode_file), str(filtered_dir / episode_file.name))
                            filtered_count += 1

                            # Log reason
                            reason_file = filtered_dir / f"{episode_id}.reason.txt"
                            with open(reason_file, "w") as f:
                                f.write(f"Quality Score: {ep_result['quality_score']:.2f}\n")
                                f.write(f"Threshold: {config.min_quality_score:.2f}\n")
                                f.write("\nErrors:\n")
                                for error in ep_result['errors']:
                                    f.write(f"  - {error}\n")
                                f.write("\nWarnings:\n")
                                for warning in ep_result['warnings']:
                                    f.write(f"  - {warning}\n")

                print(f"[IMPORT]   Filtered {filtered_count} low-quality episodes\n")

        else:
            result.episodes_passed_validation = result.total_episodes_downloaded
            result.average_quality_score = np.mean([
                ep.quality_score for ep in download_result.episodes
            ]) if download_result.episodes else 0.0

        # Step 5: Update manifest with import metadata
        if result.manifest_path and result.manifest_path.exists():
            with open(result.manifest_path, "r") as f:
                manifest = json.load(f)

            manifest["import_metadata"] = {
                "imported_at": datetime.utcnow().isoformat() + "Z",
                "imported_by": "BlueprintPipeline",
                "total_downloaded": result.total_episodes_downloaded,
                "passed_validation": result.episodes_passed_validation,
                "filtered": result.episodes_filtered,
                "average_quality_score": result.average_quality_score,
            }

            with open(result.manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)

        # Success
        result.success = True

        print("=" * 80)
        print("IMPORT COMPLETE")
        print("=" * 80)
        print(f"✅ Successfully imported {result.episodes_passed_validation} episodes")
        print(f"Output directory: {result.output_dir}")
        print(f"Manifest: {result.manifest_path}")
        print("=" * 80 + "\n")

        return result

    except Exception as e:
        print(f"\n❌ ERROR during import: {e}")
        traceback.print_exc()
        result.errors.append(str(e))
        return result


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Main entry point for import job."""
    print("\n[GENIE-SIM-IMPORT] Starting import job...")

    # Get configuration from environment
    job_id = os.getenv("GENIE_SIM_JOB_ID")
    if not job_id:
        print("[GENIE-SIM-IMPORT] ERROR: GENIE_SIM_JOB_ID is required")
        sys.exit(1)

    # Output configuration
    bucket = os.getenv("BUCKET", "")
    scene_id = os.getenv("SCENE_ID", "unknown")
    output_prefix = os.getenv("OUTPUT_PREFIX", f"scenes/{scene_id}/episodes")

    # Quality configuration
    min_quality_score = float(os.getenv("MIN_QUALITY_SCORE", "0.7"))
    enable_validation = os.getenv("ENABLE_VALIDATION", "true").lower() == "true"
    filter_low_quality = os.getenv("FILTER_LOW_QUALITY", "true").lower() == "true"

    # Polling configuration
    poll_interval = int(os.getenv("GENIE_SIM_POLL_INTERVAL", "30"))
    wait_for_completion = os.getenv("WAIT_FOR_COMPLETION", "true").lower() == "true"

    print(f"[GENIE-SIM-IMPORT] Configuration:")
    print(f"[GENIE-SIM-IMPORT]   Job ID: {job_id}")
    print(f"[GENIE-SIM-IMPORT]   Output Prefix: {output_prefix}")
    print(f"[GENIE-SIM-IMPORT]   Min Quality: {min_quality_score}")
    print(f"[GENIE-SIM-IMPORT]   Enable Validation: {enable_validation}")
    print(f"[GENIE-SIM-IMPORT]   Wait for Completion: {wait_for_completion}\n")

    # Setup paths
    GCS_ROOT = Path("/mnt/gcs")
    output_dir = GCS_ROOT / bucket / output_prefix / f"geniesim_{job_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create configuration
    config = ImportConfig(
        job_id=job_id,
        output_dir=output_dir,
        min_quality_score=min_quality_score,
        enable_validation=enable_validation,
        filter_low_quality=filter_low_quality,
        poll_interval=poll_interval,
        wait_for_completion=wait_for_completion,
    )

    # Create client
    try:
        client = GenieSimClient()
    except Exception as e:
        print(f"[GENIE-SIM-IMPORT] ERROR: Failed to create Genie Sim client: {e}")
        print("[GENIE-SIM-IMPORT] Make sure GENIE_SIM_API_KEY is set")
        sys.exit(1)

    # Run import
    try:
        result = run_import_job(config, client)

        if result.success:
            print(f"[GENIE-SIM-IMPORT] ✅ Import succeeded")
            print(f"[GENIE-SIM-IMPORT] Episodes imported: {result.episodes_passed_validation}")
            print(f"[GENIE-SIM-IMPORT] Average quality: {result.average_quality_score:.2f}")
            sys.exit(0)
        else:
            print(f"[GENIE-SIM-IMPORT] ❌ Import failed")
            for error in result.errors:
                print(f"[GENIE-SIM-IMPORT]   - {error}")
            sys.exit(1)

    finally:
        client.close()


if __name__ == "__main__":
    main()
