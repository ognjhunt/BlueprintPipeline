#!/usr/bin/env python3
"""
Genie Sim 3.0 API Client.

This module provides a comprehensive client for interacting with the Genie Sim 3.0
service API, enabling bidirectional communication for:
- Submitting data generation jobs
- Monitoring generation progress
- Downloading generated episodes
- Quality validation
- Error handling and retries

This completes the missing "import" side of the Genie Sim integration, which
previously only had export capabilities.

Environment Variables:
    GENIE_SIM_API_URL: Genie Sim API endpoint (default: https://api.agibot.com/geniesim/v3)
    GENIE_SIM_API_KEY: API authentication key (required)
    GENIE_SIM_TIMEOUT: Request timeout in seconds (default: 300)
    GENIE_SIM_MAX_RETRIES: Maximum retries for failed requests (default: 3)
"""

import asyncio
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================


class JobStatus(str, Enum):
    """Status of a Genie Sim generation job."""

    PENDING = "pending"  # Job submitted, waiting to start
    RUNNING = "running"  # Job in progress
    COMPLETED = "completed"  # Job finished successfully
    FAILED = "failed"  # Job failed
    CANCELLED = "cancelled"  # Job was cancelled


class GenerationBackend(str, Enum):
    """Backend used for generation."""

    ISAAC_SIM = "isaac_sim"  # NVIDIA Isaac Sim
    CUROBO = "curobo"  # cuRobo motion planning
    HYBRID = "hybrid"  # Both Isaac Sim + cuRobo


@dataclass
class GenerationParams:
    """Parameters for episode generation."""

    # Generation quantities
    episodes_per_task: int = 10
    num_variations: int = 5  # Scene variations

    # Backend selection
    use_curobo: bool = True  # Use cuRobo motion planning
    use_isaac_sim: bool = True  # Use Isaac Sim rendering
    backend: str = GenerationBackend.HYBRID.value

    # Quality settings
    min_quality_score: float = 0.7
    enable_validation: bool = True
    filter_failed_episodes: bool = True

    # Robot configuration
    robot_type: str = "franka"  # franka, ur10, fetch, etc.
    control_frequency_hz: float = 30.0

    # Visual observations
    num_cameras: int = 3  # wrist, overhead, side
    image_resolution: Tuple[int, int] = (640, 480)
    capture_depth: bool = True
    capture_segmentation: bool = True

    # Data format
    output_format: str = "lerobot"  # lerobot, rlds, hdf5
    compression: str = "parquet"  # parquet, zarr, none


@dataclass
class JobSubmissionResult:
    """Result of job submission."""

    success: bool
    job_id: Optional[str] = None
    message: str = ""
    estimated_completion_time: Optional[str] = None  # ISO 8601 timestamp
    estimated_cost_usd: Optional[float] = None


@dataclass
class JobProgress:
    """Progress information for a running job."""

    job_id: str
    status: JobStatus
    progress_percent: float  # 0-100
    current_task: str = ""
    episodes_generated: int = 0
    total_episodes_target: int = 0
    started_at: Optional[str] = None
    estimated_completion: Optional[str] = None
    logs: List[str] = field(default_factory=list)


@dataclass
class GeneratedEpisodeMetadata:
    """Metadata for a generated episode."""

    episode_id: str
    task_name: str
    quality_score: float
    frame_count: int
    duration_seconds: float
    validation_passed: bool
    file_size_bytes: int


@dataclass
class DownloadResult:
    """Result of episode download."""

    success: bool
    output_dir: Path
    episode_count: int = 0
    total_size_bytes: int = 0
    manifest_path: Optional[Path] = None
    episodes: List[GeneratedEpisodeMetadata] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


# =============================================================================
# API Client
# =============================================================================


class GenieSimAPIError(Exception):
    """Base exception for Genie Sim API errors."""

    pass


class GenieSimAuthenticationError(GenieSimAPIError):
    """Authentication failed."""

    pass


class GenieSimJobNotFoundError(GenieSimAPIError):
    """Job not found."""

    pass


class GenieSimClient:
    """
    Client for Genie Sim 3.0 API.

    Provides both synchronous and asynchronous methods for:
    - Job submission
    - Status polling
    - Episode download
    - Quality validation
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        timeout: int = 300,
        max_retries: int = 3,
    ):
        """
        Initialize Genie Sim client.

        Args:
            api_key: API authentication key (or from GENIE_SIM_API_KEY env var)
            endpoint: API endpoint URL (or from GENIE_SIM_API_URL env var)
            timeout: Request timeout in seconds
            max_retries: Maximum retries for failed requests
        """
        self.api_key = api_key or os.getenv("GENIE_SIM_API_KEY")
        if not self.api_key:
            raise GenieSimAuthenticationError(
                "API key required. Set GENIE_SIM_API_KEY environment variable or pass api_key parameter."
            )

        self.endpoint = endpoint or os.getenv(
            "GENIE_SIM_API_URL", "https://api.agibot.com/geniesim/v3"
        )
        self.timeout = timeout
        self.max_retries = max_retries

        self._session: Optional[requests.Session] = None
        self._async_session: Optional[aiohttp.ClientSession] = None

    @property
    def session(self) -> requests.Session:
        """Get or create synchronous session."""
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "BlueprintPipeline/1.0",
            })
        return self._session

    async def _get_async_session(self) -> aiohttp.ClientSession:
        """Get or create asynchronous session."""
        if self._async_session is None or self._async_session.closed:
            self._async_session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": "BlueprintPipeline/1.0",
                }
            )
        return self._async_session

    def close(self):
        """Close synchronous session."""
        if self._session:
            self._session.close()
            self._session = None

    async def close_async(self):
        """Close asynchronous session."""
        if self._async_session and not self._async_session.closed:
            await self._async_session.close()
            self._async_session = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # =========================================================================
    # Job Submission
    # =========================================================================

    def submit_generation_job(
        self,
        scene_graph: Dict[str, Any],
        asset_index: Dict[str, Any],
        task_config: Dict[str, Any],
        generation_params: GenerationParams,
        job_name: Optional[str] = None,
    ) -> JobSubmissionResult:
        """
        Submit an episode generation job to Genie Sim.

        Args:
            scene_graph: Genie Sim scene graph (from export job)
            asset_index: Asset index with RAG embeddings (from export job)
            task_config: Task configuration (from export job)
            generation_params: Generation parameters
            job_name: Optional human-readable job name

        Returns:
            JobSubmissionResult with job ID if successful

        Raises:
            GenieSimAPIError: If submission fails
        """
        payload = {
            "job_name": job_name or f"blueprintpipeline_{int(time.time())}",
            "scene_graph": scene_graph,
            "asset_index": asset_index,
            "task_config": task_config,
            "params": asdict(generation_params),
            "created_at": datetime.utcnow().isoformat() + "Z",
        }

        try:
            response = self._make_request_with_retry(
                "POST",
                f"{self.endpoint}/jobs",
                json=payload,
            )

            if response.status_code == 201:
                data = response.json()
                return JobSubmissionResult(
                    success=True,
                    job_id=data.get("job_id"),
                    message=data.get("message", "Job submitted successfully"),
                    estimated_completion_time=data.get("estimated_completion_time"),
                    estimated_cost_usd=data.get("estimated_cost_usd"),
                )
            else:
                error_msg = response.json().get("error", "Unknown error")
                return JobSubmissionResult(
                    success=False,
                    message=f"Submission failed: {error_msg}",
                )

        except Exception as e:
            logger.error(f"Job submission failed: {e}")
            return JobSubmissionResult(
                success=False,
                message=f"Exception during submission: {str(e)}",
            )

    async def submit_generation_job_async(
        self,
        scene_graph: Dict[str, Any],
        asset_index: Dict[str, Any],
        task_config: Dict[str, Any],
        generation_params: GenerationParams,
        job_name: Optional[str] = None,
    ) -> JobSubmissionResult:
        """Async version of submit_generation_job."""
        payload = {
            "job_name": job_name or f"blueprintpipeline_{int(time.time())}",
            "scene_graph": scene_graph,
            "asset_index": asset_index,
            "task_config": task_config,
            "params": asdict(generation_params),
            "created_at": datetime.utcnow().isoformat() + "Z",
        }

        try:
            session = await self._get_async_session()
            async with session.post(
                f"{self.endpoint}/jobs",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as response:
                if response.status == 201:
                    data = await response.json()
                    return JobSubmissionResult(
                        success=True,
                        job_id=data.get("job_id"),
                        message=data.get("message", "Job submitted successfully"),
                        estimated_completion_time=data.get("estimated_completion_time"),
                        estimated_cost_usd=data.get("estimated_cost_usd"),
                    )
                else:
                    data = await response.json()
                    error_msg = data.get("error", "Unknown error")
                    return JobSubmissionResult(
                        success=False,
                        message=f"Submission failed: {error_msg}",
                    )

        except Exception as e:
            logger.error(f"Async job submission failed: {e}")
            return JobSubmissionResult(
                success=False,
                message=f"Exception during submission: {str(e)}",
            )

    # =========================================================================
    # Status Polling
    # =========================================================================

    def get_job_status(self, job_id: str) -> JobProgress:
        """
        Get current status of a job.

        Args:
            job_id: Job identifier

        Returns:
            JobProgress with current status

        Raises:
            GenieSimJobNotFoundError: If job not found
        """
        try:
            response = self._make_request_with_retry(
                "GET",
                f"{self.endpoint}/jobs/{job_id}/status",
            )

            if response.status_code == 404:
                raise GenieSimJobNotFoundError(f"Job {job_id} not found")

            data = response.json()
            return JobProgress(
                job_id=job_id,
                status=JobStatus(data.get("status", "pending")),
                progress_percent=data.get("progress_percent", 0.0),
                current_task=data.get("current_task", ""),
                episodes_generated=data.get("episodes_generated", 0),
                total_episodes_target=data.get("total_episodes_target", 0),
                started_at=data.get("started_at"),
                estimated_completion=data.get("estimated_completion"),
                logs=data.get("logs", []),
            )

        except GenieSimJobNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to get job status: {e}")
            raise GenieSimAPIError(f"Status check failed: {e}")

    async def get_job_status_async(self, job_id: str) -> JobProgress:
        """Async version of get_job_status."""
        try:
            session = await self._get_async_session()
            async with session.get(
                f"{self.endpoint}/jobs/{job_id}/status",
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as response:
                if response.status == 404:
                    raise GenieSimJobNotFoundError(f"Job {job_id} not found")

                data = await response.json()
                return JobProgress(
                    job_id=job_id,
                    status=JobStatus(data.get("status", "pending")),
                    progress_percent=data.get("progress_percent", 0.0),
                    current_task=data.get("current_task", ""),
                    episodes_generated=data.get("episodes_generated", 0),
                    total_episodes_target=data.get("total_episodes_target", 0),
                    started_at=data.get("started_at"),
                    estimated_completion=data.get("estimated_completion"),
                    logs=data.get("logs", []),
                )

        except GenieSimJobNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to get job status (async): {e}")
            raise GenieSimAPIError(f"Status check failed: {e}")

    def wait_for_completion(
        self,
        job_id: str,
        poll_interval: int = 30,
        callback: Optional[callable] = None,
    ) -> JobProgress:
        """
        Wait for job to complete (blocking).

        Args:
            job_id: Job identifier
            poll_interval: Seconds between status checks
            callback: Optional callback(progress) called on each poll

        Returns:
            Final JobProgress

        Raises:
            GenieSimAPIError: If job fails or is cancelled
        """
        logger.info(f"Waiting for job {job_id} to complete...")

        while True:
            progress = self.get_job_status(job_id)

            if callback:
                callback(progress)
            else:
                logger.info(
                    f"Job {job_id}: {progress.status.value} "
                    f"({progress.progress_percent:.1f}%) - {progress.current_task}"
                )

            if progress.status == JobStatus.COMPLETED:
                logger.info(f"Job {job_id} completed successfully!")
                return progress
            elif progress.status == JobStatus.FAILED:
                raise GenieSimAPIError(f"Job {job_id} failed: {progress.current_task}")
            elif progress.status == JobStatus.CANCELLED:
                raise GenieSimAPIError(f"Job {job_id} was cancelled")

            time.sleep(poll_interval)

    # =========================================================================
    # Episode Download
    # =========================================================================

    def download_episodes(
        self,
        job_id: str,
        output_dir: Path,
        validate: bool = True,
    ) -> DownloadResult:
        """
        Download generated episodes from completed job.

        Args:
            job_id: Job identifier
            output_dir: Output directory for episodes
            validate: Whether to validate downloaded data

        Returns:
            DownloadResult with downloaded episode information

        Raises:
            GenieSimAPIError: If download fails
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Check job status
            progress = self.get_job_status(job_id)
            if progress.status != JobStatus.COMPLETED:
                raise GenieSimAPIError(
                    f"Job {job_id} is not completed (status: {progress.status.value})"
                )

            # Download episode archive
            logger.info(f"Downloading episodes for job {job_id}...")
            archive_path = output_dir / f"{job_id}_episodes.tar.gz"

            response = self._make_request_with_retry(
                "GET",
                f"{self.endpoint}/jobs/{job_id}/episodes",
                stream=True,
            )

            total_size = 0
            with open(archive_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        total_size += len(chunk)

            logger.info(f"Downloaded {total_size / 1024 / 1024:.1f} MB")

            # Extract archive
            logger.info("Extracting episodes...")
            import tarfile

            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(output_dir)

            # Load manifest
            manifest_path = output_dir / "manifest.json"
            if manifest_path.exists():
                with open(manifest_path) as f:
                    manifest = json.load(f)

                episodes = [
                    GeneratedEpisodeMetadata(**ep)
                    for ep in manifest.get("episodes", [])
                ]
            else:
                episodes = []

            # Validate if requested
            errors = []
            if validate:
                errors = self._validate_downloaded_episodes(output_dir, episodes)

            # Clean up archive
            archive_path.unlink()

            return DownloadResult(
                success=len(errors) == 0,
                output_dir=output_dir,
                episode_count=len(episodes),
                total_size_bytes=total_size,
                manifest_path=manifest_path if manifest_path.exists() else None,
                episodes=episodes,
                errors=errors,
            )

        except Exception as e:
            logger.error(f"Episode download failed: {e}")
            return DownloadResult(
                success=False,
                output_dir=output_dir,
                errors=[str(e)],
            )

    def _validate_downloaded_episodes(
        self,
        output_dir: Path,
        episodes: List[GeneratedEpisodeMetadata],
    ) -> List[str]:
        """Validate downloaded episodes."""
        errors = []

        for episode in episodes:
            # Check if episode file exists
            episode_file = output_dir / f"{episode.episode_id}.parquet"
            if not episode_file.exists():
                errors.append(f"Episode file missing: {episode.episode_id}")
                continue

            # Check file size
            actual_size = episode_file.stat().st_size
            if abs(actual_size - episode.file_size_bytes) > 1024:  # 1KB tolerance
                errors.append(
                    f"Episode {episode.episode_id} size mismatch: "
                    f"expected {episode.file_size_bytes}, got {actual_size}"
                )

        return errors

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _make_request_with_retry(
        self,
        method: str,
        url: str,
        **kwargs,
    ) -> requests.Response:
        """Make HTTP request with exponential backoff retry."""
        for attempt in range(self.max_retries):
            try:
                response = self.session.request(
                    method,
                    url,
                    timeout=self.timeout,
                    **kwargs,
                )
                response.raise_for_status()
                return response

            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries - 1:
                    raise GenieSimAPIError(f"Request failed after {self.max_retries} retries: {e}")

                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(f"Request failed (attempt {attempt + 1}/{self.max_retries}), retrying in {wait_time}s...")
                time.sleep(wait_time)

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a running job.

        Args:
            job_id: Job identifier

        Returns:
            True if successfully cancelled
        """
        try:
            response = self._make_request_with_retry(
                "POST",
                f"{self.endpoint}/jobs/{job_id}/cancel",
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to cancel job: {e}")
            return False


# =============================================================================
# CLI Interface
# =============================================================================


def main():
    """CLI for Genie Sim client."""
    import argparse

    parser = argparse.ArgumentParser(description="Genie Sim 3.0 API Client")
    parser.add_argument("command", choices=["submit", "status", "wait", "download", "cancel"])
    parser.add_argument("--job-id", help="Job ID (for status/wait/download/cancel)")
    parser.add_argument("--scene-graph", help="Path to scene_graph.json (for submit)")
    parser.add_argument("--asset-index", help="Path to asset_index.json (for submit)")
    parser.add_argument("--task-config", help="Path to task_config.json (for submit)")
    parser.add_argument("--output-dir", help="Output directory (for download)")
    parser.add_argument("--episodes-per-task", type=int, default=10)
    parser.add_argument("--robot-type", default="franka")

    args = parser.parse_args()

    client = GenieSimClient()

    try:
        if args.command == "submit":
            # Load files
            with open(args.scene_graph) as f:
                scene_graph = json.load(f)
            with open(args.asset_index) as f:
                asset_index = json.load(f)
            with open(args.task_config) as f:
                task_config = json.load(f)

            params = GenerationParams(
                episodes_per_task=args.episodes_per_task,
                robot_type=args.robot_type,
            )

            result = client.submit_generation_job(
                scene_graph, asset_index, task_config, params
            )

            if result.success:
                print(f"✅ Job submitted: {result.job_id}")
                print(f"Estimated completion: {result.estimated_completion_time}")
                print(f"Estimated cost: ${result.estimated_cost_usd:.2f}")
            else:
                print(f"❌ Submission failed: {result.message}")
                sys.exit(1)

        elif args.command == "status":
            progress = client.get_job_status(args.job_id)
            print(f"Job {args.job_id}:")
            print(f"  Status: {progress.status.value}")
            print(f"  Progress: {progress.progress_percent:.1f}%")
            print(f"  Episodes: {progress.episodes_generated}/{progress.total_episodes_target}")
            print(f"  Current task: {progress.current_task}")

        elif args.command == "wait":
            progress = client.wait_for_completion(args.job_id)
            print(f"✅ Job completed: {progress.episodes_generated} episodes generated")

        elif args.command == "download":
            result = client.download_episodes(args.job_id, Path(args.output_dir))
            if result.success:
                print(f"✅ Downloaded {result.episode_count} episodes to {result.output_dir}")
            else:
                print(f"❌ Download failed: {result.errors}")
                sys.exit(1)

        elif args.command == "cancel":
            if client.cancel_job(args.job_id):
                print(f"✅ Job {args.job_id} cancelled")
            else:
                print(f"❌ Failed to cancel job {args.job_id}")
                sys.exit(1)

    finally:
        client.close()


if __name__ == "__main__":
    main()
