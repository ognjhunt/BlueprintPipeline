#!/usr/bin/env python3
"""
Isaac Sim Enforcement for Production Episode Generation.

This module enforces Isaac Sim availability for production-quality episode generation.
It prevents the silent generation of unusable mock data by requiring actual physics
simulation and sensor rendering.

Key Principles:
1. Production data MUST use Isaac Sim (PhysX + Replicator)
2. Development/testing can use mock mode (explicit opt-in)
3. Environment clearly identifies data quality level
4. No silent degradation - fail fast with clear error messages

Environment Variables:
    PRODUCTION_MODE: Set to "true" for production enforcement (default: false)
    ALLOW_MOCK_CAPTURE: Set to "true" to allow mock sensor data (default: false)
    ISAAC_SIM_REQUIRED: Force Isaac Sim requirement (default: false)
    DATA_QUALITY_LEVEL: Expected quality level (production|development|test)
"""

import logging
import os
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Add parent to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.config.production_mode import resolve_production_mode

# Import Isaac Sim integration
try:
    from isaac_sim_integration import (
        is_isaac_sim_available,
        is_physx_available,
        is_replicator_available,
        print_availability_report,
        get_availability_status,
    )
    _HAVE_INTEGRATION_MODULE = True
except ImportError:
    _HAVE_INTEGRATION_MODULE = False

    def is_isaac_sim_available() -> bool:
        return False

    def is_physx_available() -> bool:
        return False

    def is_replicator_available() -> bool:
        return False


# =============================================================================
# Data Quality Levels
# =============================================================================


class DataQualityLevel(str, Enum):
    """Data quality levels for generated episodes."""

    PRODUCTION = "production"  # Isaac Sim required - sellable data
    DEVELOPMENT = "development"  # Mock allowed - testing only
    TEST = "test"  # Mock allowed - unit tests


class SensorSource(str, Enum):
    """Source of sensor data in generated episodes."""

    ISAAC_SIM_REPLICATOR = "isaac_sim_replicator"  # Real rendering
    MOCK = "mock"  # Placeholder data
    MUJOCO_FALLBACK = "mujoco_fallback"  # CPU rendering (future)
    BLENDER_FALLBACK = "blender_fallback"  # Offline rendering (future)


class PhysicsValidationBackend(str, Enum):
    """Backend used for physics validation."""

    PHYSX = "physx"  # NVIDIA PhysX (production)
    HEURISTIC = "heuristic"  # Geometric approximations (development)
    MUJOCO = "mujoco"  # Mujoco physics (future fallback)


@dataclass
class EnvironmentCapabilities:
    """Capabilities available in current environment."""

    isaac_sim_available: bool
    physx_available: bool
    replicator_available: bool
    gpu_available: bool
    production_mode: bool
    allow_mock_capture: bool

    @property
    def can_generate_production_data(self) -> bool:
        """Check if environment can generate production-quality data."""
        return (
            self.isaac_sim_available
            and self.physx_available
            and self.replicator_available
        )

    @property
    def sensor_source(self) -> SensorSource:
        """Determine sensor data source."""
        if self.replicator_available:
            return SensorSource.ISAAC_SIM_REPLICATOR
        return SensorSource.MOCK

    @property
    def physics_backend(self) -> PhysicsValidationBackend:
        """Determine physics validation backend."""
        if self.physx_available:
            return PhysicsValidationBackend.PHYSX
        return PhysicsValidationBackend.HEURISTIC

    @property
    def training_suitability(self) -> str:
        """Determine if data is suitable for training."""
        if self.can_generate_production_data:
            return "production"
        return "development_only"


# =============================================================================
# Enforcement Logic
# =============================================================================


class IsaacSimRequirementError(RuntimeError):
    """Raised when Isaac Sim is required but not available."""

    pass


class ProductionDataQualityError(RuntimeError):
    """Raised when production data quality cannot be met."""

    pass


def get_environment_capabilities() -> EnvironmentCapabilities:
    """Detect current environment capabilities."""
    # Check Isaac Sim availability
    isaac_sim_available = is_isaac_sim_available()
    physx_available = is_physx_available()
    replicator_available = is_replicator_available()

    # Check GPU availability (simple check - can be enhanced)
    gpu_available = os.path.exists("/dev/nvidia0") or os.path.exists("/proc/driver/nvidia")

    # Check environment configuration
    production_mode = resolve_production_mode()
    allow_mock_capture = os.environ.get("ALLOW_MOCK_CAPTURE", "false").lower() == "true"

    return EnvironmentCapabilities(
        isaac_sim_available=isaac_sim_available,
        physx_available=physx_available,
        replicator_available=replicator_available,
        gpu_available=gpu_available,
        production_mode=production_mode,
        allow_mock_capture=allow_mock_capture,
    )


def enforce_isaac_sim_for_production(
    required_quality: DataQualityLevel = DataQualityLevel.PRODUCTION,
) -> EnvironmentCapabilities:
    """
    Enforce Isaac Sim availability for production data generation.

    Args:
        required_quality: Required data quality level

    Returns:
        EnvironmentCapabilities object

    Raises:
        IsaacSimRequirementError: If Isaac Sim required but not available
        ProductionDataQualityError: If production quality cannot be met
    """
    capabilities = get_environment_capabilities()

    # Check if Isaac Sim is explicitly required
    isaac_sim_required = os.environ.get("ISAAC_SIM_REQUIRED", "false").lower() == "true"

    # Production mode always requires Isaac Sim
    if capabilities.production_mode or isaac_sim_required:
        if not capabilities.isaac_sim_available:
            raise IsaacSimRequirementError(
                "Isaac Sim is required for production episode generation but is not available.\n"
                "\n"
                "Options:\n"
                "1. Run in Isaac Sim environment (use Dockerfile.isaacsim)\n"
                "2. Set PRODUCTION_MODE=false for development/testing\n"
                "3. Use pre-generated episodes from Genie Sim 3.0\n"
                "\n"
                "Current environment:\n"
                f"  - Isaac Sim: {capabilities.isaac_sim_available}\n"
                f"  - PhysX: {capabilities.physx_available}\n"
                f"  - Replicator: {capabilities.replicator_available}\n"
                f"  - GPU: {capabilities.gpu_available}\n"
            )

        if not capabilities.physx_available:
            raise ProductionDataQualityError(
                "PhysX is required for production physics validation but is not available."
            )

        if not capabilities.replicator_available:
            raise ProductionDataQualityError(
                "Replicator is required for production sensor data but is not available."
            )

    # Check quality level requirements
    if required_quality == DataQualityLevel.PRODUCTION:
        if not capabilities.can_generate_production_data:
            if capabilities.allow_mock_capture:
                logger.warning(
                    "⚠️  WARNING: Production quality requested but using mock data"
                )
                logger.warning("⚠️  This data is NOT suitable for training!")
            else:
                raise ProductionDataQualityError(
                    f"Production quality data requires Isaac Sim.\n"
                    f"Current capabilities:\n"
                    f"  - Training suitability: {capabilities.training_suitability}\n"
                    f"  - Sensor source: {capabilities.sensor_source.value}\n"
                    f"  - Physics backend: {capabilities.physics_backend.value}\n"
                )

    return capabilities


def print_environment_report(capabilities: EnvironmentCapabilities):
    """Print detailed environment capabilities report."""
    logger.info("%s", "=" * 80)
    logger.info("ISAAC SIM ENVIRONMENT REPORT")
    logger.info("%s", "=" * 80)
    logger.info("Runtime Capabilities:")
    logger.info(
        "  • Isaac Sim Available:    %s",
        "✅ YES" if capabilities.isaac_sim_available else "❌ NO",
    )
    logger.info(
        "  • PhysX Available:        %s",
        "✅ YES" if capabilities.physx_available else "❌ NO",
    )
    logger.info(
        "  • Replicator Available:   %s",
        "✅ YES" if capabilities.replicator_available else "❌ NO",
    )
    logger.info(
        "  • GPU Available:          %s",
        "✅ YES" if capabilities.gpu_available else "❌ NO",
    )
    logger.info("Configuration:")
    logger.info(
        "  • Production Mode:        %s",
        "✅ ENABLED" if capabilities.production_mode else "⚠️  DISABLED",
    )
    logger.info(
        "  • Allow Mock Capture:     %s",
        "⚠️  YES" if capabilities.allow_mock_capture else "✅ NO",
    )
    logger.info("Data Quality:")
    logger.info(
        "  • Can Generate Production: %s",
        "✅ YES" if capabilities.can_generate_production_data else "❌ NO",
    )
    logger.info("  • Sensor Source:          %s", capabilities.sensor_source.value)
    logger.info("  • Physics Backend:        %s", capabilities.physics_backend.value)
    logger.info("  • Training Suitability:   %s", capabilities.training_suitability)

    if not capabilities.can_generate_production_data:
        logger.warning(
            "⚠️  WARNING: Current environment CANNOT generate production-quality data!"
        )
        logger.warning(
            "Generated episodes will use mock sensor data and heuristic physics."
        )
        logger.warning("This data is NOT suitable for training robots.")

    if capabilities.production_mode and not capabilities.can_generate_production_data:
        logger.error(
            "❌ ERROR: Production mode is enabled but Isaac Sim is not available!"
        )

    logger.info("%s", "=" * 80)


# =============================================================================
# Convenience Functions
# =============================================================================


def require_isaac_sim():
    """Shorthand to require Isaac Sim with fail-fast behavior."""
    return enforce_isaac_sim_for_production(DataQualityLevel.PRODUCTION)


def check_production_ready() -> bool:
    """Check if environment is production-ready (non-throwing)."""
    try:
        capabilities = get_environment_capabilities()
        return capabilities.can_generate_production_data
    except Exception:
        return False


def get_data_quality_level() -> DataQualityLevel:
    """Get current data quality level from environment."""
    env_level = os.environ.get("DATA_QUALITY_LEVEL", "development").lower()
    try:
        return DataQualityLevel(env_level)
    except ValueError:
        return DataQualityLevel.DEVELOPMENT


if __name__ == "__main__":
    # Print environment report when run directly
    from tools.logging_config import init_logging

    init_logging()
    capabilities = get_environment_capabilities()
    print_environment_report(capabilities)

    # Test enforcement
    try:
        required_quality = get_data_quality_level()
        logger.info(
            "Testing enforcement for quality level: %s", required_quality.value
        )
        enforce_isaac_sim_for_production(required_quality)
        logger.info("✅ Enforcement check passed!")
    except (IsaacSimRequirementError, ProductionDataQualityError) as e:
        logger.error("❌ Enforcement check failed: %s", e)
        sys.exit(1)
