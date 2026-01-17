#!/usr/bin/env python3
"""
[DEPRECATED] Genie Sim Client - Use local_framework.py instead

This file is deprecated and kept only for backwards compatibility with data classes.
All REST API functionality has been removed as Genie Sim 3.0 is a LOCAL-ONLY framework.

⚠️  DO NOT USE: Use tools.geniesim_adapter.local_framework instead

Genie Sim 3.0 Architecture:
- LOCAL framework running inside Isaac Sim
- Communication via gRPC (NOT HTTP REST API)
- No API keys required
- Repository: https://github.com/AgibotTech/genie_sim

For local data collection, use:
    from tools.geniesim_adapter.local_framework import (
        GenieSimLocalFramework,
        run_local_data_collection,
        check_geniesim_availability,
    )

This file now only provides data classes for backwards compatibility:
- GenerationParams
- JobStatus
- JobProgress
- DownloadResult
- GeneratedEpisodeMetadata
- GenieSimAPIError (base exception)

All API operations (health check, job submission, polling, download) have been
removed. Use GenieSimLocalFramework instead.
"""

import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Show deprecation warning on import
warnings.warn(
    "geniesim_client.py is deprecated. Use tools.geniesim_adapter.local_framework instead. "
    "Genie Sim 3.0 is a local-only framework using gRPC, not a hosted API. "
    "This file now only provides data classes for backwards compatibility.",
    DeprecationWarning,
    stacklevel=2,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Data Models (Kept for Backwards Compatibility)
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

    # Quality settings
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
    """Health status of Genie Sim."""

    available: bool
    api_version: Optional[str] = None
    rate_limit_remaining: Optional[int] = None
    estimated_queue_time_seconds: Optional[float] = None
    error: Optional[str] = None
    checked_at: Optional[str] = None  # ISO 8601 timestamp


# =============================================================================
# Exceptions (Kept for Backwards Compatibility)
# =============================================================================


class GenieSimAPIError(Exception):
    """Base exception for Genie Sim errors."""

    pass


class GenieSimJobNotFoundError(GenieSimAPIError):
    """Job not found."""

    pass


class GenieSimClientError(GenieSimAPIError):
    """Client-side error (4xx equivalent)."""

    pass


class GenieSimConfigurationError(GenieSimAPIError):
    """Configuration error."""

    pass


# =============================================================================
# Deprecated Client Class (Kept for Import Compatibility Only)
# =============================================================================


class GenieSimClient:
    """
    [DEPRECATED] Genie Sim Client - Use GenieSimLocalFramework instead

    This class is deprecated and should not be used. It previously implemented
    a REST API client that does not exist - Genie Sim 3.0 is local-only.

    Use instead:
        from tools.geniesim_adapter.local_framework import GenieSimLocalFramework

    Example:
        framework = GenieSimLocalFramework()
        if framework.connect():
            result = framework.run_data_collection(task_config, scene_config)
    """

    def __init__(self, *args, **kwargs):
        """Initialize deprecated client with warning."""
        warnings.warn(
            "GenieSimClient is deprecated. Use GenieSimLocalFramework from "
            "tools.geniesim_adapter.local_framework instead. "
            "Genie Sim 3.0 is local-only and does not have a REST API.",
            DeprecationWarning,
            stacklevel=2,
        )
        logger.error(
            "CRITICAL: GenieSimClient is deprecated and non-functional. "
            "Use GenieSimLocalFramework instead. "
            "See tools/geniesim_adapter/local_framework.py"
        )
        raise NotImplementedError(
            "GenieSimClient is deprecated. Genie Sim 3.0 is a local-only framework. "
            "Use GenieSimLocalFramework from tools.geniesim_adapter.local_framework instead. "
            "See: https://github.com/AgibotTech/genie_sim"
        )


# =============================================================================
# Migration Guide
# =============================================================================

def _show_migration_guide():
    """Show migration guide when this module is imported."""
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                        DEPRECATION WARNING                                 ║
║                                                                            ║
║  geniesim_client.py is DEPRECATED                                         ║
║                                                                            ║
║  Genie Sim 3.0 is a LOCAL-ONLY framework using gRPC (not REST API).      ║
║  This file previously implemented a non-existent REST API.                 ║
║                                                                            ║
║  ✅ USE INSTEAD:                                                          ║
║     from tools.geniesim_adapter.local_framework import (                  ║
║         GenieSimLocalFramework,                                           ║
║         run_local_data_collection,                                        ║
║         check_geniesim_availability,                                      ║
║     )                                                                      ║
║                                                                            ║
║  📚 Reference: https://github.com/AgibotTech/genie_sim                   ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)


# Show migration guide on import
_show_migration_guide()
