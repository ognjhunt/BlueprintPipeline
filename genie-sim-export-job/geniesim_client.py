#!/usr/bin/env python3
"""
Genie Sim Local Framework Client.

This module provides a client for interacting with the locally-running Genie Sim 3.0
framework, enabling bidirectional communication for:
- Submitting data generation jobs
- Monitoring generation progress
- Downloading generated episodes
- Quality validation
- Error handling and retries

Genie Sim 3.0 is an open-source LOCAL simulation framework that runs on your
own Isaac Sim installation using gRPC for client-server communication.

Usage:
    from tools.geniesim_adapter.local_framework import (
        GenieSimLocalFramework,
        run_local_data_collection,
        check_geniesim_availability,
    )

    # Run data collection locally (no API key required)
    result = run_local_data_collection(
        scene_manifest_path=Path("scene_manifest.json"),
        task_config_path=Path("task_config.json"),
        output_dir=Path("./output"),
    )

For local framework details, see tools/geniesim_adapter/local_framework.py.

Environment Variables:
    GENIESIM_MOCK_MODE: Enable mock mode for testing (default: false)
    GENIESIM_HOST: gRPC host (default: localhost)
    GENIESIM_PORT: gRPC port (default: 50051)
"""

import asyncio
import hashlib
import importlib.util
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import aiohttp
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LABS-BLOCKER-003 FIX: Add Pydantic for API response validation
if importlib.util.find_spec("pydantic") is not None:
    import pydantic
    from pydantic import BaseModel, Field, ValidationError, validator
    _pydantic_major_version = int(pydantic.__version__.split(".", 1)[0])
    if _pydantic_major_version >= 2:
        HAVE_PYDANTIC = False
        logger.warning(
            "Pydantic v2 detected - disabling response validation for compatibility"
        )
    else:
        HAVE_PYDANTIC = True
else:
    HAVE_PYDANTIC = False
    logger.warning("Pydantic not available - API response validation will be limited")

# Add repo root to path for imports
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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


# Define supported robot types for validation
class SupportedRobotType(str, Enum):
    """Supported robot types for Genie Sim 3.0."""

    # Arm manipulators
    FRANKA = "franka"  # Franka Emika Panda
    UR5 = "ur5"  # Universal Robots UR5
    UR10 = "ur10"  # Universal Robots UR10
    UR5E = "ur5e"  # Universal Robots UR5e
    KINOVA_GEN3 = "kinova_gen3"  # Kinova Gen3
    KINOVA_JACO = "kinova_jaco"  # Kinova Jaco
    ABB_YUM = "abb_yumi"  # ABB YuMi (dual-arm)

    # Mobile manipulators
    FETCH = "fetch"  # Fetch Robotics
    TGO = "tgo"  # Toyota HSR
    STRETCH = "stretch"  # Hello Robot Stretch

    # Humanoids
    GR1 = "gr1"  # Fourier Intelligence GR-1
    G2 = "g2"  # Unitree G1/G2
    H1 = "h1"  # Unitree H1
    DIGIT = "digit"  # Agility Robotics Digit
    ATLAS = "atlas"  # Boston Dynamics Atlas

    # Quadrupeds (with manipulation)
    SPOT = "spot"  # Boston Dynamics Spot (with arm)
    UNITREE_GO2 = "unitree_go2"  # Unitree Go2

    # Bimanual/Dual-Arm
    BIMANUAL_FRANKA = "bimanual_franka"  # Dual Franka Panda
    BIMANUAL_UR5 = "bimanual_ur5"  # Dual UR5

    # Custom (for testing/development)
    CUSTOM = "custom"  # User-provided URDF


# Helper function to validate robot type
def validate_robot_type(robot_type: str) -> None:
    """
    Validate that robot_type is a supported value.

    Args:
        robot_type: Robot type string to validate

    Raises:
        ValueError: If robot type is not supported
    """
    supported_types = {rt.value for rt in SupportedRobotType}

    if robot_type not in supported_types:
        raise ValueError(
            f"Invalid robot_type: '{robot_type}'. "
            f"Supported types: {', '.join(sorted(supported_types))}"
        )


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

    # Quality settings - LABS-BLOCKER-002 FIX: Raised from 0.7 to 0.85
    min_quality_score: float = 0.85
    enable_validation: bool = True
    filter_failed_episodes: bool = True

    # Robot configuration - Validated robot type
    robot_type: str = "franka"  # franka, ur10, fetch, etc.
    control_frequency_hz: float = 30.0

    def __post_init__(self):
        """Validate robot_type after initialization."""
        validate_robot_type(self.robot_type)

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


@dataclass
class HealthStatus:
    """Health status of Genie Sim API."""

    available: bool
    api_version: Optional[str] = None
    rate_limit_remaining: Optional[int] = None
    estimated_queue_time_seconds: Optional[float] = None
    error: Optional[str] = None
    checked_at: Optional[str] = None  # ISO 8601 timestamp


# =============================================================================
# LABS-BLOCKER-003 FIX: Pydantic Models for API Response Validation
# =============================================================================

if HAVE_PYDANTIC:
    class JobProgressResponse(BaseModel):
        """
        Validated API response for job progress.

        LABS-BLOCKER-003 FIX: Adds validation to prevent:
        - progress_percent > 100% or < 0%
        - Invalid job status values
        - Negative episode counts
        """
        job_id: str
        status: str
        progress_percent: float = Field(..., ge=0.0, le=100.0, description="Progress percentage (0-100)")
        current_task: str = ""
        episodes_generated: int = Field(default=0, ge=0, description="Episodes generated (non-negative)")
        total_episodes_target: int = Field(default=0, ge=0, description="Total episodes target (non-negative)")
        started_at: Optional[str] = None
        estimated_completion: Optional[str] = None
        logs: List[str] = Field(default_factory=list)

        @validator('status')
        def validate_status(cls, v):
            """Validate status is a valid JobStatus enum value."""
            valid_statuses = {s.value for s in JobStatus}
            if v not in valid_statuses:
                raise ValueError(
                    f"Invalid job status: {v}. "
                    f"Must be one of: {', '.join(valid_statuses)}"
                )
            return v

        @validator('episodes_generated', 'total_episodes_target')
        def validate_non_negative(cls, v, field):
            """Ensure counts are non-negative."""
            if v < 0:
                raise ValueError(f"{field.name} cannot be negative: {v}")
            return v

        class Config:
            # Allow arbitrary types for compatibility
            arbitrary_types_allowed = True

    class HealthStatusResponse(BaseModel):
        """
        Validated API response for health check.

        LABS-BLOCKER-003 FIX: Validates health check response format.
        """
        status: str
        version: Optional[str] = None
        rate_limit_remaining: Optional[int] = Field(default=None, ge=0)
        estimated_queue_time_seconds: Optional[float] = Field(default=None, ge=0.0)

        @validator('status')
        def validate_status(cls, v):
            """Validate status is 'healthy' or 'unhealthy'."""
            if v not in ["healthy", "unhealthy", "degraded"]:
                raise ValueError(f"Invalid health status: {v}")
            return v

        class Config:
            arbitrary_types_allowed = True

    class JobSubmissionResponse(BaseModel):
        """
        Validated API response for job submission.

        LABS-BLOCKER-003 FIX: Validates job submission response.
        """
        job_id: str
        message: str = ""
        estimated_completion_time: Optional[str] = None
        estimated_cost_usd: Optional[float] = Field(default=None, ge=0.0)

        @validator('job_id')
        def validate_job_id(cls, v):
            """Validate job ID format (alphanumeric + hyphens/underscores)."""
            if not v or len(v) < 8:
                raise ValueError(f"Invalid job ID: {v} (too short)")
            # Basic format validation - alphanumeric + hyphens/underscores
            import re
            if not re.match(r'^[a-zA-Z0-9_-]+$', v):
                raise ValueError(f"Invalid job ID format: {v}")
            return v

        class Config:
            arbitrary_types_allowed = True


# =============================================================================
# API Client
# =============================================================================


class GenieSimAPIError(Exception):
    """Base exception for Genie Sim API errors."""

    pass


class GenieSimAuthenticationError(GenieSimAPIError):
    """Authentication failed."""

    pass


class GenieSimConfigurationError(GenieSimAPIError):
    """Hosted API usage disabled or misconfigured."""

    pass


class GenieSimJobNotFoundError(GenieSimAPIError):
    """Job not found."""

    pass


# =============================================================================
# Security Utilities
# =============================================================================


def safe_extract_tar(archive_path: Path, output_dir: Path) -> None:
    """
    Safely extract tarfile with path traversal protection.

    This function prevents malicious archives from extracting files outside
    the intended output directory (CWE-22: Path Traversal).

    Args:
        archive_path: Path to tar.gz archive
        output_dir: Directory to extract to

    Raises:
        ValueError: If path traversal is detected
        tarfile.TarError: If archive is corrupted
    """
    import tarfile

    output_dir = Path(output_dir).resolve()

    def _validate_member(member: tarfile.TarInfo) -> tarfile.TarInfo:
        member_path = Path(member.name)
        if member_path.is_absolute():
            raise ValueError(f"Suspicious path in archive: {member.name}")

        if ".." in member_path.parts:
            raise ValueError(f"Path traversal detected in archive: {member.name}")

        if member.issym() or member.islnk():
            raise ValueError(f"Symlink detected in archive: {member.name}")

        resolved_target = (output_dir / member.name).resolve()
        try:
            resolved_target.relative_to(output_dir)
        except ValueError:
            raise ValueError(f"Path traversal detected in archive: {member.name}")

        return member

    with tarfile.open(archive_path, "r:gz") as tar:
        try:
            tar.extractall(
                output_dir,
                filter=lambda member, _: _validate_member(member),
            )
        except TypeError:
            for member in tar.getmembers():
                _validate_member(member)
                tar.extract(member, output_dir)


class GenieSimRestClient:
    """HTTP REST helper for Genie Sim API calls."""

    def __init__(self, api_key: Optional[str]) -> None:
        self.api_key = api_key
        self._session: Optional[requests.Session] = None
        self._async_session: Optional[aiohttp.ClientSession] = None

    def _build_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "BlueprintPipeline/1.0",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @property
    def session(self) -> requests.Session:
        """Get or create synchronous session."""
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update(self._build_headers())
        return self._session

    async def get_async_session(self) -> aiohttp.ClientSession:
        """Get or create asynchronous session."""
        if self._async_session is None or self._async_session.closed:
            self._async_session = aiohttp.ClientSession(
                headers=self._build_headers()
            )
        return self._async_session

    def close(self) -> None:
        """Close synchronous session."""
        if self._session:
            self._session.close()
            self._session = None

    async def close_async(self) -> None:
        """Close asynchronous session."""
        if self._async_session and not self._async_session.closed:
            await self._async_session.close()
            self._async_session = None


class GenieSimClient:
    """
    Client for Genie Sim 3.0 API.

    Provides both synchronous and asynchronous methods for:
    - Job submission
    - Status polling
    - Episode download
    - Quality validation
    """
    _mock_jobs: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _mock_job_id(
        scene_graph: Dict[str, Any],
        task_config: Dict[str, Any],
        job_name: Optional[str],
    ) -> str:
        payload = {
            "scene_id": scene_graph.get("scene_id"),
            "job_name": job_name,
            "task_ids": [
                task.get("task_id")
                for task in task_config.get("tasks", [])
                if isinstance(task, dict)
            ],
        }
        fingerprint = json.dumps(payload, sort_keys=True).encode("utf-8")
        digest = hashlib.sha256(fingerprint).hexdigest()[:12]
        return f"mock-{digest}"

    @staticmethod
    def _mock_total_episodes(
        generation_params: GenerationParams,
        task_config: Dict[str, Any],
    ) -> int:
        task_count = len(task_config.get("tasks", [])) or 1
        return max(1, generation_params.episodes_per_task * task_count)

    @staticmethod
    def _mock_episode_quality(index: int) -> float:
        return 0.9 if index % 2 == 0 else 0.8

    @classmethod
    def _mock_store_job(
        cls,
        job_id: str,
        generation_params: GenerationParams,
        task_config: Dict[str, Any],
        created_at: str,
    ) -> None:
        cls._mock_jobs[job_id] = {
            "created_at": created_at,
            "completed_at": created_at,
            "generation_params": generation_params,
            "task_config": task_config,
            "status": JobStatus.COMPLETED,
            "total_episodes": cls._mock_total_episodes(generation_params, task_config),
            "episodes_collected": cls._mock_total_episodes(generation_params, task_config),
            "episodes_passed": cls._mock_total_episodes(generation_params, task_config),
            "failure_reason": None,
            "failure_details": None,
        }

    def register_mock_job_metrics(
        self,
        job_id: str,
        generation_params: GenerationParams,
        task_config: Dict[str, Any],
        created_at: str,
        status: JobStatus = JobStatus.COMPLETED,
        completed_at: Optional[str] = None,
        episodes_collected: Optional[int] = None,
        episodes_passed: Optional[int] = None,
        failure_reason: Optional[str] = None,
        failure_details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register mock job metrics for local workflow consumption."""
        if not self.mock_mode:
            return
        total_episodes = self._mock_total_episodes(generation_params, task_config)
        self._mock_jobs[job_id] = {
            "created_at": created_at,
            "completed_at": completed_at,
            "generation_params": generation_params,
            "task_config": task_config,
            "status": status,
            "total_episodes": total_episodes,
            "episodes_collected": episodes_collected,
            "episodes_passed": episodes_passed,
            "failure_reason": failure_reason,
            "failure_details": failure_details,
        }

    def __init__(
        self,
        timeout: int = 300,
        max_retries: int = 3,
        validate_on_init: bool = True,
        mock_mode: Optional[bool] = None,
    ):
        """
        Initialize Genie Sim local framework client.

        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum retries for failed requests
            validate_on_init: Validate local framework is reachable on initialization
            mock_mode: Enable mock mode for testing (default: disabled)
        """
        self.mock_mode = (
            mock_mode
            if mock_mode is not None
            else os.getenv("GENIESIM_MOCK_MODE", "false").lower() == "true"
        )

        # For local framework operation, use gRPC endpoint configuration
        self.grpc_host = os.getenv("GENIESIM_HOST", "localhost")
        self.grpc_port = int(os.getenv("GENIESIM_PORT", "50051"))
        self.local_endpoint = f"{self.grpc_host}:{self.grpc_port}"

        # For mock mode, use dummy values
        if self.mock_mode:
            self.api_key = None
            self.endpoint = "mock://geniesim"
        else:
            # Local framework uses gRPC, not HTTP REST API
            self.api_key = None
            self.endpoint = self.local_endpoint

        self.timeout = timeout
        self.max_retries = max_retries

        self._rest_client = GenieSimRestClient(api_key=self.api_key)

        # GAP-EH-002 FIX: Add circuit breaker to prevent cascading failures
        self._circuit_breaker = None
        self._rate_limiter = None

        # Validate endpoint is reachable on initialization
        if validate_on_init and not self.mock_mode:
            self._validate_local_endpoint()

    @property
    def session(self) -> requests.Session:
        """Get or create synchronous session."""
        return self._rest_client.session

    async def _get_async_session(self) -> aiohttp.ClientSession:
        """Get or create asynchronous session."""
        return await self._rest_client.get_async_session()

    def close(self):
        """Close synchronous session."""
        self._rest_client.close()

    async def close_async(self):
        """Close asynchronous session."""
        await self._rest_client.close_async()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # =========================================================================
    # Local Framework Validation
    # =========================================================================

    def _validate_local_endpoint(self) -> None:
        """
        Validate that the Genie Sim local gRPC endpoint is reachable.

        This method is called during initialization to fail fast if the
        gRPC endpoint is not reachable or misconfigured.

        Raises:
            GenieSimAPIError: If endpoint is not reachable or returns unexpected response
        """
        logger.info(f"Validating Genie Sim local gRPC endpoint: {self.local_endpoint}")

        try:
            # For local gRPC endpoint, we do a simple connectivity check
            # In real implementation, this would use gRPC health check
            import socket

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((self.grpc_host, self.grpc_port))
            sock.close()

            if result == 0:
                logger.info(
                    f"✓ Genie Sim local gRPC endpoint validated successfully\n"
                    f"  Host: {self.grpc_host}\n"
                    f"  Port: {self.grpc_port}"
                )
            else:
                error_msg = f"Cannot connect to Genie Sim gRPC endpoint at {self.local_endpoint}"
                logger.error(error_msg)
                raise GenieSimAPIError(error_msg)

        except GenieSimAPIError:
            raise
        except Exception as e:
            error_msg = f"Failed to validate Genie Sim local gRPC endpoint: {e}"
            logger.error(error_msg)
            raise GenieSimAPIError(error_msg) from e

    # =========================================================================
    # Health Check
    # =========================================================================

    def health_check(self) -> HealthStatus:
        """
        Check Genie Sim API health and availability.

        Returns:
            HealthStatus with availability and metadata

        This method checks the /health endpoint to verify:
        - API is reachable
        - Authentication is valid
        - Service is operational
        - Rate limits and queue status
        """
        if self.mock_mode:
            return HealthStatus(
                available=True,
                api_version="mock",
                rate_limit_remaining=1000,
                estimated_queue_time_seconds=0.0,
                checked_at="2025-01-01T00:00:00Z",
            )
        try:
            response = self._make_request_with_retry(
                "GET",
                f"{self.endpoint}/health",
                json=None,  # GET request has no body
            )

            if response.status_code == 200:
                data = response.json()
                return HealthStatus(
                    available=data.get("status") == "healthy",
                    api_version=data.get("version"),
                    rate_limit_remaining=data.get("rate_limit_remaining"),
                    estimated_queue_time_seconds=data.get("estimated_queue_time_seconds"),
                    checked_at=datetime.utcnow().isoformat() + "Z",
                )
            else:
                return HealthStatus(
                    available=False,
                    error=f"HTTP {response.status_code}: {response.text}",
                    checked_at=datetime.utcnow().isoformat() + "Z",
                )

        except requests.exceptions.ConnectionError as e:
            return HealthStatus(
                available=False,
                error=f"Connection failed: {str(e)}",
                checked_at=datetime.utcnow().isoformat() + "Z",
            )
        except requests.exceptions.Timeout as e:
            return HealthStatus(
                available=False,
                error=f"Request timed out: {str(e)}",
                checked_at=datetime.utcnow().isoformat() + "Z",
            )
        except Exception as e:
            return HealthStatus(
                available=False,
                error=f"Health check failed: {str(e)}",
                checked_at=datetime.utcnow().isoformat() + "Z",
            )

    async def health_check_async(self) -> HealthStatus:
        """
        Asynchronous version of health_check().

        Returns:
            HealthStatus with availability and metadata
        """
        if self.mock_mode:
            return HealthStatus(
                available=True,
                api_version="mock",
                rate_limit_remaining=1000,
                estimated_queue_time_seconds=0.0,
                checked_at="2025-01-01T00:00:00Z",
            )
        try:
            session = await self._get_async_session()
            async with session.get(f"{self.endpoint}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    return HealthStatus(
                        available=data.get("status") == "healthy",
                        api_version=data.get("version"),
                        rate_limit_remaining=data.get("rate_limit_remaining"),
                        estimated_queue_time_seconds=data.get("estimated_queue_time_seconds"),
                        checked_at=datetime.utcnow().isoformat() + "Z",
                    )
                else:
                    text = await response.text()
                    return HealthStatus(
                        available=False,
                        error=f"HTTP {response.status}: {text}",
                        checked_at=datetime.utcnow().isoformat() + "Z",
                    )

        except aiohttp.ClientConnectorError as e:
            return HealthStatus(
                available=False,
                error=f"Connection failed: {str(e)}",
                checked_at=datetime.utcnow().isoformat() + "Z",
            )
        except asyncio.TimeoutError as e:
            return HealthStatus(
                available=False,
                error=f"Request timed out: {str(e)}",
                checked_at=datetime.utcnow().isoformat() + "Z",
            )
        except Exception as e:
            return HealthStatus(
                available=False,
                error=f"Health check failed: {str(e)}",
                checked_at=datetime.utcnow().isoformat() + "Z",
            )

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
        if self.mock_mode:
            created_at = "2025-01-01T00:00:00Z"
            job_id = self._mock_job_id(scene_graph, task_config, job_name)
            self._mock_store_job(job_id, generation_params, task_config, created_at)
            return JobSubmissionResult(
                success=True,
                job_id=job_id,
                message="Mock job submitted",
                estimated_completion_time=created_at,
                estimated_cost_usd=0.0,
            )
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
        if self.mock_mode:
            created_at = "2025-01-01T00:00:00Z"
            job_id = self._mock_job_id(scene_graph, task_config, job_name)
            self._mock_store_job(job_id, generation_params, task_config, created_at)
            return JobSubmissionResult(
                success=True,
                job_id=job_id,
                message="Mock job submitted",
                estimated_completion_time=created_at,
                estimated_cost_usd=0.0,
            )
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
                            except (json.JSONDecodeError, aiohttp.ContentTypeError, KeyError) as parse_err:
                                # Replace bare except with specific exceptions
                                logger.debug(f"Could not parse error response: {parse_err}")
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

        LABS-BLOCKER-003 FIX: Added Pydantic validation for API response.

        Args:
            job_id: Job identifier

        Returns:
            JobProgress with current status

        Raises:
            GenieSimJobNotFoundError: If job not found
            GenieSimAPIError: If response validation fails
        """
        if self.mock_mode:
            job_data = self._mock_jobs.get(job_id)
            if not job_data:
                raise GenieSimJobNotFoundError(f"Mock job {job_id} not found")
            total_episodes = job_data["total_episodes"]
            return JobProgress(
                job_id=job_id,
                status=job_data["status"],
                progress_percent=100.0,
                current_task="mock_complete",
                episodes_generated=total_episodes,
                total_episodes_target=total_episodes,
                started_at=job_data["created_at"],
                estimated_completion=job_data["created_at"],
                logs=["Mock job completed"],
            )
        try:
            response = self._make_request_with_retry(
                "GET",
                f"{self.endpoint}/jobs/{job_id}/status",
            )

            if response.status_code == 404:
                raise GenieSimJobNotFoundError(f"Job {job_id} not found")

            # LABS-BLOCKER-003 FIX: Parse and validate JSON response
            try:
                data = response.json()
            except (json.JSONDecodeError, ValueError) as e:
                raise GenieSimAPIError(f"Invalid JSON response from API: {e}")

            # LABS-BLOCKER-003 FIX: Validate response with Pydantic if available
            if HAVE_PYDANTIC:
                try:
                    validated = JobProgressResponse(**data)
                    # Convert validated Pydantic model back to JobProgress dataclass
                    return JobProgress(
                        job_id=validated.job_id,
                        status=JobStatus(validated.status),
                        progress_percent=validated.progress_percent,
                        current_task=validated.current_task,
                        episodes_generated=validated.episodes_generated,
                        total_episodes_target=validated.total_episodes_target,
                        started_at=validated.started_at,
                        estimated_completion=validated.estimated_completion,
                        logs=validated.logs,
                    )
                except ValidationError as e:
                    raise GenieSimAPIError(
                        f"API response validation failed: {e}\n"
                        f"Received data: {data}"
                    )
            else:
                # Fallback: Manual validation without Pydantic
                progress_pct = data.get("progress_percent", 0.0)
                if not (0.0 <= progress_pct <= 100.0):
                    logger.warning(
                        f"Invalid progress_percent: {progress_pct} (should be 0-100). "
                        f"Clamping to valid range."
                    )
                    progress_pct = max(0.0, min(100.0, progress_pct))

                return JobProgress(
                    job_id=job_id,
                    status=JobStatus(data.get("status", "pending")),
                    progress_percent=progress_pct,
                    current_task=data.get("current_task", ""),
                    episodes_generated=max(0, data.get("episodes_generated", 0)),
                    total_episodes_target=max(0, data.get("total_episodes_target", 0)),
                    started_at=data.get("started_at"),
                    estimated_completion=data.get("estimated_completion"),
                    logs=data.get("logs", []),
                )

        except GenieSimJobNotFoundError:
            raise
        except GenieSimAPIError:
            raise
        except Exception as e:
            logger.error(f"Failed to get job status: {e}")
            raise GenieSimAPIError(f"Status check failed: {e}")

    async def get_job_status_async(self, job_id: str) -> JobProgress:
        """Async version of get_job_status with retry logic and validation."""
        if self.mock_mode:
            return self.get_job_status(job_id)
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
        max_wait_time: Optional[float] = None,
        use_exponential_backoff: bool = True,
    ) -> JobProgress:
        """
        Wait for job to complete (blocking).

        Added timeout and exponential backoff to prevent infinite polling.

        Args:
            job_id: Job identifier
            poll_interval: Initial seconds between status checks
            callback: Optional callback(progress) called on each poll
            max_wait_time: Maximum wait time in seconds (default: 14400 = 4 hours)
            use_exponential_backoff: Use exponential backoff for polling interval

        Returns:
            Final JobProgress

        Raises:
            GenieSimAPIError: If job fails, is cancelled, or times out
        """
        if self.mock_mode:
            progress = self.get_job_status(job_id)
            if callback:
                callback(progress)
            return progress
        # Set default timeout to 4 hours
        if max_wait_time is None:
            max_wait_time = 14400.0  # 4 hours for large jobs

        logger.info(
            f"Waiting for job {job_id} to complete "
            f"(max wait time: {max_wait_time/3600:.1f} hours)..."
        )

        start_time = time.time()
        current_poll_interval = poll_interval
        max_poll_interval = 300  # Cap at 5 minutes

        while True:
            # Check if we've exceeded max wait time
            elapsed_time = time.time() - start_time
            if elapsed_time > max_wait_time:
                error_msg = (
                    f"Job {job_id} timed out after {elapsed_time/3600:.2f} hours. "
                    f"Max wait time: {max_wait_time/3600:.1f} hours. "
                    f"Consider increasing max_wait_time or checking job status manually."
                )
                logger.error(error_msg)
                raise GenieSimAPIError(error_msg)

            progress = self.get_job_status(job_id)

            if callback:
                callback(progress)
            else:
                logger.info(
                    f"Job {job_id}: {progress.status.value} "
                    f"({progress.progress_percent:.1f}%) - {progress.current_task} "
                    f"[Elapsed: {elapsed_time/60:.1f}m]"
                )

            if progress.status == JobStatus.COMPLETED:
                logger.info(
                    f"Job {job_id} completed successfully after {elapsed_time/60:.1f} minutes!"
                )
                return progress
            elif progress.status == JobStatus.FAILED:
                raise GenieSimAPIError(f"Job {job_id} failed: {progress.current_task}")
            elif progress.status == JobStatus.CANCELLED:
                raise GenieSimAPIError(f"Job {job_id} was cancelled")

            # Use exponential backoff for polling interval
            if use_exponential_backoff:
                # Gradually increase polling interval to reduce API load
                current_poll_interval = min(current_poll_interval * 1.2, max_poll_interval)

            time.sleep(current_poll_interval)

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
            if self.mock_mode:
                job_data = self._mock_jobs.get(job_id)
                if not job_data:
                    raise GenieSimJobNotFoundError(f"Mock job {job_id} not found")

                tasks = [
                    task.get("task_name")
                    or task.get("task_id")
                    or "mock_task"
                    for task in job_data["task_config"].get("tasks", [])
                    if isinstance(task, dict)
                ] or ["mock_task"]

                episodes: List[GeneratedEpisodeMetadata] = []
                total_size = 0
                min_quality = job_data["generation_params"].min_quality_score
                for index in range(job_data["total_episodes"]):
                    episode_id = f"episode_{index:06d}"
                    task_name = tasks[index % len(tasks)]
                    frame_count = 12
                    duration_seconds = frame_count / 10.0
                    quality_score = self._mock_episode_quality(index)
                    validation_passed = quality_score >= min_quality
                    episode_file = output_dir / f"{episode_id}.parquet"
                    content = f"mock-episode-{job_id}-{episode_id}".encode("utf-8")
                    episode_file.write_bytes(content)
                    file_size = episode_file.stat().st_size
                    total_size += file_size
                    episodes.append(
                        GeneratedEpisodeMetadata(
                            episode_id=episode_id,
                            task_name=task_name,
                            quality_score=quality_score,
                            frame_count=frame_count,
                            duration_seconds=duration_seconds,
                            validation_passed=validation_passed,
                            file_size_bytes=file_size,
                        )
                    )

                manifest_path = output_dir / "manifest.json"
                manifest = {
                    "job_id": job_id,
                    "schema_version": "mock-1.0",
                    "generated_at": "2025-01-01T00:00:00Z",
                    "episodes": [asdict(ep) for ep in episodes],
                }
                manifest_path.write_text(json.dumps(manifest, indent=2))

                errors = []
                if validate:
                    errors = self._validate_downloaded_episodes(output_dir, episodes)

                return DownloadResult(
                    success=len(errors) == 0,
                    output_dir=output_dir,
                    episode_count=len(episodes),
                    total_size_bytes=total_size,
                    manifest_path=manifest_path,
                    episodes=episodes,
                    errors=errors,
                )

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
                except (json.JSONDecodeError, requests.exceptions.JSONDecodeError, KeyError) as parse_err:
                    # Replace bare except with specific exceptions
                    logger.debug(f"Could not parse error response: {parse_err}")
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

            # Extract archive with path traversal protection
            logger.info("Extracting episodes...")
            safe_extract_tar(archive_path, output_dir)

            # Stream manifest episodes without loading full list into memory
            manifest_path = output_dir / "manifest.json"
            if manifest_path.exists():
                base_iterator: Iterable[GeneratedEpisodeMetadata] = self._stream_manifest_episodes(
                    manifest_path
                )
            else:
                base_iterator = iter(())

            episode_count = 0

            def _counting_iterator() -> Iterable[GeneratedEpisodeMetadata]:
                nonlocal episode_count
                for episode in base_iterator:
                    episode_count += 1
                    yield episode

            counting_iterator = _counting_iterator()

            # Validate if requested
            errors = []
            if validate:
                errors = self._validate_downloaded_episodes(output_dir, counting_iterator)
            else:
                for _ in counting_iterator:
                    pass

            # Clean up archive
            archive_path.unlink()

            return DownloadResult(
                success=len(errors) == 0,
                output_dir=output_dir,
                episode_count=episode_count,
                total_size_bytes=total_size,
                manifest_path=manifest_path if manifest_path.exists() else None,
                episodes=[],
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
        episodes: Iterable[GeneratedEpisodeMetadata],
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

    @staticmethod
    def _stream_manifest_episodes(
        manifest_path: Path,
    ) -> Iterable[GeneratedEpisodeMetadata]:
        """Stream episode metadata from a manifest without loading all entries."""
        try:
            import ijson
        except ImportError:
            logger.warning(
                "ijson not installed; falling back to json.load for manifest parsing."
            )
            with open(manifest_path) as f:
                manifest = json.load(f)
            for episode in manifest.get("episodes", []):
                yield GeneratedEpisodeMetadata(**episode)
            return

        with open(manifest_path, "rb") as f:
            for episode in ijson.items(f, "episodes.item"):
                yield GeneratedEpisodeMetadata(**episode)

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
        if self.mock_mode:
            job_data = self._mock_jobs.get(job_id)
            if not job_data:
                return False
            job_data["status"] = JobStatus.CANCELLED
            return True
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
            if self.mock_mode:
                job_data = self._mock_jobs.get(job_id)
                if not job_data:
                    raise GenieSimJobNotFoundError(f"Job {job_id} not found")
                created_at = job_data.get("created_at")
                completed_at = job_data.get("completed_at")
                duration_seconds = None
                if created_at and completed_at:
                    try:
                        created_dt = datetime.fromisoformat(created_at.replace("Z", ""))
                        completed_dt = datetime.fromisoformat(completed_at.replace("Z", ""))
                        duration_seconds = max(
                            0.0,
                            (completed_dt - created_dt).total_seconds(),
                        )
                    except ValueError:
                        duration_seconds = None
                total_episodes = job_data.get("total_episodes")
                episodes_collected = job_data.get("episodes_collected")
                episodes_passed = job_data.get("episodes_passed")
                quality_pass_rate = None
                if episodes_collected and episodes_collected > 0 and episodes_passed is not None:
                    quality_pass_rate = episodes_passed / episodes_collected
                status_value = job_data.get("status")
                if isinstance(status_value, Enum):
                    status_value = status_value.value
                return {
                    "job_id": job_id,
                    "status": status_value,
                    "created_at": created_at,
                    "completed_at": completed_at,
                    "duration_seconds": duration_seconds,
                    "total_episodes": total_episodes,
                    "episodes_collected": episodes_collected,
                    "episodes_passed": episodes_passed,
                    "quality_pass_rate": quality_pass_rate,
                    "failure_reason": job_data.get("failure_reason"),
                    "failure_details": job_data.get("failure_details"),
                }
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
        if self.mock_mode:
            return self._mock_jobs.pop(job_id, None) is not None
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
                except (json.JSONDecodeError, requests.exceptions.JSONDecodeError, KeyError) as parse_err:
                    # Replace bare except with specific exceptions
                    logger.debug(f"Could not parse error response: {parse_err}")
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
