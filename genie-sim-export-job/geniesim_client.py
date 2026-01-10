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

# Add repo root to path for imports
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import resilience infrastructure
try:
    from tools.secrets import get_secret_or_env, SecretIds
    from tools.external_services import ServiceClient, ServiceClientConfig, RateLimiter
    from tools.error_handling import CircuitBreaker
    HAVE_RESILIENCE_TOOLS = True
except ImportError:
    HAVE_RESILIENCE_TOOLS = False
    logger.warning("Resilience tools not available - using basic retry logic")

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
        # GAP-SEC-001 FIX: Use Secret Manager for API key (with fallback to env var)
        if HAVE_RESILIENCE_TOOLS:
            try:
                self.api_key = api_key or get_secret_or_env(
                    SecretIds.GENIE_SIM_API_KEY,
                    env_var="GENIE_SIM_API_KEY"
                )
            except Exception as e:
                logger.warning(f"Secret Manager unavailable, falling back to env var: {e}")
                self.api_key = api_key or os.getenv("GENIE_SIM_API_KEY")
        else:
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

        # GAP-EH-002 FIX: Add circuit breaker to prevent cascading failures
        if HAVE_RESILIENCE_TOOLS:
            self._circuit_breaker = CircuitBreaker(
                failure_threshold=5,
                recovery_timeout=60.0,
                expected_exception=GenieSimAPIError
            )
            # GAP-SEC-003 FIX: Add rate limiting (10 requests/second)
            self._rate_limiter = RateLimiter(calls_per_second=10.0)
        else:
            self._circuit_breaker = None
            self._rate_limiter = None

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
        """Async version of submit_generation_job with circuit breaker and retry logic."""
        payload = {
            "job_name": job_name or f"blueprintpipeline_{int(time.time())}",
            "scene_graph": scene_graph,
            "asset_index": asset_index,
            "task_config": task_config,
            "params": asdict(generation_params),
            "created_at": datetime.utcnow().isoformat() + "Z",
        }

        # GAP-ASYNC-001 FIX: Apply rate limiting before request
        if self._rate_limiter:
            self._rate_limiter.acquire()

        # GAP-ASYNC-002 FIX: Wrap in circuit breaker if available
        async def _make_async_request():
            for attempt in range(self.max_retries):
                try:
                    session = await self._get_async_session()
                    async with session.post(
                        f"{self.endpoint}/jobs",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=self.timeout),
                    ) as response:
                        # GAP-STATUS-001 FIX: Validate status codes before parsing
                        if response.status == 201:
                            # GAP-JSON-001 FIX: Validate JSON response before parsing
                            try:
                                data = await response.json()
                            except (json.JSONDecodeError, aiohttp.ContentTypeError) as e:
                                raise GenieSimAPIError(f"Invalid JSON response: {e}")

                            return JobSubmissionResult(
                                success=True,
                                job_id=data.get("job_id"),
                                message=data.get("message", "Job submitted successfully"),
                                estimated_completion_time=data.get("estimated_completion_time"),
                                estimated_cost_usd=data.get("estimated_cost_usd"),
                            )
                        elif response.status == 429:
                            # GAP-RATELIMIT-001 FIX: Handle rate limiting with exponential backoff
                            retry_after = int(response.headers.get("Retry-After", "60"))
                            wait_time = min(retry_after, 300)  # Cap at 5 minutes
                            logger.warning(f"Rate limited (429), waiting {wait_time}s before retry")
                            await asyncio.sleep(wait_time)
                            continue
                        elif response.status == 401:
                            raise GenieSimAuthenticationError("Authentication failed (401)")
                        else:
                            try:
                                data = await response.json()
                                error_msg = data.get("error", f"HTTP {response.status}")
                            except:
                                error_msg = f"HTTP {response.status}"

                            if attempt == self.max_retries - 1:
                                return JobSubmissionResult(
                                    success=False,
                                    message=f"Submission failed: {error_msg}",
                                )

                            # Retry on server errors (5xx)
                            if response.status >= 500:
                                base_wait = 2 ** attempt
                                jitter = base_wait * 0.1 * (2 * time.time() % 1 - 0.5)
                                wait_time = min(base_wait + jitter, 60.0)
                                logger.warning(f"Server error {response.status}, retrying in {wait_time:.1f}s...")
                                await asyncio.sleep(wait_time)
                                continue
                            else:
                                # Client errors (4xx) don't retry
                                return JobSubmissionResult(
                                    success=False,
                                    message=f"Submission failed: {error_msg}",
                                )

                except aiohttp.ClientError as e:
                    if attempt == self.max_retries - 1:
                        raise GenieSimAPIError(f"Request failed after {self.max_retries} retries: {e}")

                    base_wait = 2 ** attempt
                    jitter = base_wait * 0.1 * (2 * time.time() % 1 - 0.5)
                    wait_time = min(base_wait + jitter, 60.0)
                    logger.warning(f"Request failed (attempt {attempt + 1}/{self.max_retries}), retrying in {wait_time:.1f}s... Error: {e}")
                    await asyncio.sleep(wait_time)

            raise GenieSimAPIError(f"Request failed after {self.max_retries} retries")

        try:
            # Use async request with circuit breaker protection
            # Circuit breaker doesn't support async directly, so we check state before proceeding
            if self._circuit_breaker and self._circuit_breaker.is_open():
                raise GenieSimAPIError("Circuit breaker is open - service temporarily unavailable")

            result = await _make_async_request()

            # Mark circuit breaker as healthy on success
            if self._circuit_breaker and result.success:
                self._circuit_breaker.mark_success()

            return result

        except GenieSimAPIError as e:
            # Mark circuit breaker failure
            if self._circuit_breaker:
                self._circuit_breaker.mark_failure()
            logger.error(f"Async job submission failed: {e}")
            return JobSubmissionResult(
                success=False,
                message=f"API error during submission: {str(e)}",
            )
        except Exception as e:
            # Mark circuit breaker failure for unexpected errors
            if self._circuit_breaker:
                self._circuit_breaker.mark_failure()
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
        """Async version of get_job_status with retry logic and validation."""
        # Apply rate limiting
        if self._rate_limiter:
            self._rate_limiter.acquire()

        async def _make_async_request():
            for attempt in range(self.max_retries):
                try:
                    session = await self._get_async_session()
                    async with session.get(
                        f"{self.endpoint}/jobs/{job_id}/status",
                        timeout=aiohttp.ClientTimeout(total=self.timeout),
                    ) as response:
                        # Validate status code
                        if response.status == 404:
                            raise GenieSimJobNotFoundError(f"Job {job_id} not found")
                        elif response.status == 401:
                            raise GenieSimAuthenticationError("Authentication failed")
                        elif response.status == 429:
                            retry_after = int(response.headers.get("Retry-After", "60"))
                            wait_time = min(retry_after, 300)
                            logger.warning(f"Rate limited, waiting {wait_time}s")
                            await asyncio.sleep(wait_time)
                            continue
                        elif response.status != 200:
                            if attempt == self.max_retries - 1:
                                raise GenieSimAPIError(f"HTTP {response.status}")
                            if response.status >= 500:
                                base_wait = 2 ** attempt
                                await asyncio.sleep(min(base_wait, 30.0))
                                continue
                            else:
                                raise GenieSimAPIError(f"HTTP {response.status}")

                        # Validate JSON response
                        try:
                            data = await response.json()
                        except (json.JSONDecodeError, aiohttp.ContentTypeError) as e:
                            raise GenieSimAPIError(f"Invalid JSON response: {e}")

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

                except (GenieSimJobNotFoundError, GenieSimAuthenticationError):
                    raise
                except aiohttp.ClientError as e:
                    if attempt == self.max_retries - 1:
                        raise GenieSimAPIError(f"Request failed after {self.max_retries} retries: {e}")
                    base_wait = 2 ** attempt
                    await asyncio.sleep(min(base_wait, 30.0))

            raise GenieSimAPIError(f"Request failed after {self.max_retries} retries")

        try:
            # Check circuit breaker before making async request
            if self._circuit_breaker and self._circuit_breaker.is_open():
                raise GenieSimAPIError("Circuit breaker is open - service temporarily unavailable")

            result = await _make_async_request()

            # Mark circuit breaker as healthy on success
            if self._circuit_breaker:
                self._circuit_breaker.mark_success()

            return result

        except (GenieSimJobNotFoundError, GenieSimAuthenticationError) as e:
            # Don't mark circuit breaker for client errors
            raise
        except GenieSimAPIError as e:
            # Mark circuit breaker failure
            if self._circuit_breaker:
                self._circuit_breaker.mark_failure()
            logger.error(f"Failed to get job status (async): {e}")
            raise
        except Exception as e:
            # Mark circuit breaker failure for unexpected errors
            if self._circuit_breaker:
                self._circuit_breaker.mark_failure()
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

            # GAP-STREAM-001 FIX: Validate streaming response before consuming
            if response.status_code != 200:
                try:
                    error_data = response.json()
                    error_msg = error_data.get("error", f"HTTP {response.status_code}")
                except:
                    error_msg = f"HTTP {response.status_code}"
                raise GenieSimAPIError(f"Failed to download episodes: {error_msg}")

            # Validate content type (should be application/gzip or application/x-tar+gzip)
            content_type = response.headers.get("Content-Type", "")
            if content_type and "gzip" not in content_type.lower() and "tar" not in content_type.lower():
                logger.warning(f"Unexpected content type: {content_type}")

            total_size = 0
            chunk_count = 0
            max_chunks = 100000  # Sanity limit: ~100GB at 1MB chunks
            with open(archive_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        total_size += len(chunk)
                        chunk_count += 1

                        # Prevent infinite loops or absurdly large downloads
                        if chunk_count > max_chunks:
                            raise GenieSimAPIError(
                                f"Download exceeded safety limit of {max_chunks} chunks"
                            )

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
        """
        Make HTTP request with exponential backoff retry, circuit breaker, and rate limiting.

        GAP-EH-001 FIX: Enhanced retry logic with jitter
        GAP-EH-002 FIX: Circuit breaker to prevent cascading failures
        GAP-SEC-003 FIX: Rate limiting to prevent quota exhaustion
        """
        # Apply rate limiting before making request
        if self._rate_limiter:
            self._rate_limiter.acquire()

        # Define the actual request function
        def make_request():
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

                    # Enhanced exponential backoff with jitter
                    base_wait = 2 ** attempt
                    jitter = base_wait * 0.1 * (2 * time.time() % 1 - 0.5)  # ±5% jitter
                    wait_time = min(base_wait + jitter, 60.0)  # Cap at 60s

                    logger.warning(
                        f"Request failed (attempt {attempt + 1}/{self.max_retries}), "
                        f"retrying in {wait_time:.1f}s... Error: {e}"
                    )
                    time.sleep(wait_time)

        # Use circuit breaker if available
        if self._circuit_breaker:
            return self._circuit_breaker.call(make_request)
        else:
            return make_request()

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

    # =========================================================================
    # Missing API Endpoints - GAP-API-001 through GAP-API-006
    # =========================================================================

    def list_jobs(
        self,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        sort_by: str = "created_at",
        sort_order: str = "desc"
    ) -> Dict[str, Any]:
        """
        List all jobs with optional filtering.

        GAP-API-001 FIX: Implement job listing endpoint

        Args:
            status: Filter by job status (pending/running/completed/failed/cancelled)
            limit: Maximum number of jobs to return
            offset: Offset for pagination
            sort_by: Field to sort by (created_at, updated_at, status)
            sort_order: Sort order (asc/desc)

        Returns:
            Dict with 'jobs' list and pagination info
        """
        params = {
            "limit": limit,
            "offset": offset,
            "sort_by": sort_by,
            "sort_order": sort_order,
        }
        if status:
            params["status"] = status

        try:
            response = self._make_request_with_retry(
                "GET",
                f"{self.endpoint}/jobs",
                params=params,
            )

            if response.status_code != 200:
                raise GenieSimAPIError(f"Failed to list jobs: HTTP {response.status_code}")

            try:
                data = response.json()
            except json.JSONDecodeError as e:
                raise GenieSimAPIError(f"Invalid JSON response: {e}")

            return data

        except Exception as e:
            logger.error(f"Failed to list jobs: {e}")
            raise GenieSimAPIError(f"List jobs failed: {e}")

    def get_job_metrics(self, job_id: str) -> Dict[str, Any]:
        """
        Get performance metrics for a job.

        GAP-API-002 FIX: Implement metrics retrieval endpoint

        Args:
            job_id: Job identifier

        Returns:
            Dict with performance metrics

        Raises:
            GenieSimJobNotFoundError: If job not found
        """
        try:
            response = self._make_request_with_retry(
                "GET",
                f"{self.endpoint}/jobs/{job_id}/metrics",
            )

            if response.status_code == 404:
                raise GenieSimJobNotFoundError(f"Job {job_id} not found")
            elif response.status_code != 200:
                raise GenieSimAPIError(f"Failed to get metrics: HTTP {response.status_code}")

            try:
                data = response.json()
            except json.JSONDecodeError as e:
                raise GenieSimAPIError(f"Invalid JSON response: {e}")

            return data

        except GenieSimJobNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to get job metrics: {e}")
            raise GenieSimAPIError(f"Get metrics failed: {e}")

    def update_job(self, job_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update job parameters.

        GAP-API-003 FIX: Implement job update endpoint (PATCH)

        Args:
            job_id: Job identifier
            updates: Dict of parameters to update

        Returns:
            True if successfully updated

        Raises:
            GenieSimJobNotFoundError: If job not found
        """
        try:
            response = self._make_request_with_retry(
                "PATCH",
                f"{self.endpoint}/jobs/{job_id}",
                json=updates,
            )

            if response.status_code == 404:
                raise GenieSimJobNotFoundError(f"Job {job_id} not found")
            elif response.status_code != 200:
                raise GenieSimAPIError(f"Failed to update job: HTTP {response.status_code}")

            return True

        except GenieSimJobNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to update job: {e}")
            return False

    def delete_job(self, job_id: str, force: bool = False) -> bool:
        """
        Delete a job and its associated data.

        GAP-API-004 FIX: Implement job deletion endpoint

        Args:
            job_id: Job identifier
            force: Force delete even if job is running

        Returns:
            True if successfully deleted

        Raises:
            GenieSimJobNotFoundError: If job not found
        """
        params = {"force": "true" if force else "false"}

        try:
            response = self._make_request_with_retry(
                "DELETE",
                f"{self.endpoint}/jobs/{job_id}",
                params=params,
            )

            if response.status_code == 404:
                raise GenieSimJobNotFoundError(f"Job {job_id} not found")
            elif response.status_code != 200:
                raise GenieSimAPIError(f"Failed to delete job: HTTP {response.status_code}")

            return True

        except GenieSimJobNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to delete job: {e}")
            return False

    def register_webhook(self, webhook_url: str, events: List[str]) -> Dict[str, Any]:
        """
        Register a webhook for job notifications.

        GAP-API-005 FIX: Implement webhook callback registration

        Args:
            webhook_url: URL to receive webhook callbacks
            events: List of events to subscribe to
                   (e.g., ["job.completed", "job.failed", "job.progress"])

        Returns:
            Dict with webhook_id and confirmation

        Raises:
            GenieSimAPIError: If registration fails
        """
        payload = {
            "webhook_url": webhook_url,
            "events": events,
        }

        try:
            response = self._make_request_with_retry(
                "POST",
                f"{self.endpoint}/webhooks",
                json=payload,
            )

            if response.status_code != 201:
                raise GenieSimAPIError(f"Failed to register webhook: HTTP {response.status_code}")

            try:
                data = response.json()
            except json.JSONDecodeError as e:
                raise GenieSimAPIError(f"Invalid JSON response: {e}")

            return data

        except Exception as e:
            logger.error(f"Failed to register webhook: {e}")
            raise GenieSimAPIError(f"Webhook registration failed: {e}")

    def delete_webhook(self, webhook_id: str) -> bool:
        """
        Delete a registered webhook.

        Args:
            webhook_id: Webhook identifier

        Returns:
            True if successfully deleted
        """
        try:
            response = self._make_request_with_retry(
                "DELETE",
                f"{self.endpoint}/webhooks/{webhook_id}",
            )

            return response.status_code == 200

        except Exception as e:
            logger.error(f"Failed to delete webhook: {e}")
            return False

    def submit_batch_jobs(
        self,
        jobs: List[Dict[str, Any]],
        batch_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Submit multiple jobs in a single batch request.

        GAP-API-006 FIX: Implement batch job submission

        Args:
            jobs: List of job configurations, each with:
                  - scene_graph, asset_index, task_config, generation_params
            batch_name: Optional name for the batch

        Returns:
            Dict with batch_id and list of created job_ids

        Raises:
            GenieSimAPIError: If submission fails
        """
        payload = {
            "batch_name": batch_name or f"batch_{int(time.time())}",
            "jobs": jobs,
            "created_at": datetime.utcnow().isoformat() + "Z",
        }

        try:
            response = self._make_request_with_retry(
                "POST",
                f"{self.endpoint}/jobs/batch",
                json=payload,
            )

            if response.status_code != 201:
                try:
                    error_data = response.json()
                    error_msg = error_data.get("error", f"HTTP {response.status_code}")
                except:
                    error_msg = f"HTTP {response.status_code}"
                raise GenieSimAPIError(f"Batch submission failed: {error_msg}")

            try:
                data = response.json()
            except json.JSONDecodeError as e:
                raise GenieSimAPIError(f"Invalid JSON response: {e}")

            logger.info(f"Batch submitted: {data.get('batch_id')} ({len(data.get('job_ids', []))} jobs)")
            return data

        except Exception as e:
            logger.error(f"Failed to submit batch: {e}")
            raise GenieSimAPIError(f"Batch submission failed: {e}")


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
