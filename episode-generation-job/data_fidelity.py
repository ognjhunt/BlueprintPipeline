"""Data fidelity enforcement for episode generation.

This module provides fail-fast gates that ensure episode data meets realism
requirements. Gates are enabled by default and can be disabled for development
via environment variables.

Usage:
    from data_fidelity import (
        DataFidelityError,
        require_real_contacts,
        require_real_efforts,
        require_valid_rgb,
        require_object_motion,
        require_intrinsics,
        is_production_mode,
    )

Environment Variables:
    DATA_FIDELITY_MODE: "production" or "dev"
        When unset, this module infers production mode from PIPELINE_ENV
        ("production"/"prod"/"staging" => production, else dev).
    STRICT_REALISM: "true" or "false"
        When unset, defaults to true in production mode and false otherwise.
        When true, all realism gates are enforced in every environment.
    REQUIRE_REAL_CONTACTS: "true" (default) or "false"
    REQUIRE_REAL_EFFORTS: "true" (default) or "false"
    REQUIRE_VALID_RGB: "true" (default) or "false"
    REQUIRE_OBJECT_MOTION: "true" (default) or "false"
    REQUIRE_INTRINSICS: "true" (default) or "false"
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class DataFidelityError(Exception):
    """Raised when episode data doesn't meet realism requirements.

    This error indicates that the data pipeline produced output that fails
    to meet production quality standards. The error message includes:
    - What check failed
    - Current values vs required thresholds
    - Environment variable to disable the check (for dev only)
    """

    def __init__(self, message: str, gate_name: str = "", diagnostics: Optional[Dict[str, Any]] = None):
        self.gate_name = gate_name
        self.diagnostics = diagnostics or {}
        super().__init__(message)


class ConfigurationError(Exception):
    """Raised when required configuration is missing or invalid."""
    pass


# =============================================================================
# Environment Variable Helpers
# =============================================================================

def _get_bool_env(name: str, default: bool = True) -> bool:
    """Get boolean environment variable with default."""
    val = os.getenv(name, "").lower()
    if val in ("true", "1", "yes"):
        return True
    if val in ("false", "0", "no"):
        return False
    return default


def is_production_mode() -> bool:
    """Check if running in production mode (fail-fast enabled)."""
    mode = os.getenv("DATA_FIDELITY_MODE")
    if mode:
        return mode.lower() != "dev"
    pipeline_env = (os.getenv("PIPELINE_ENV") or "").strip().lower()
    return pipeline_env in {"production", "prod", "staging"}


def is_strict_realism() -> bool:
    """Check if strict realism is enabled (enforce gates everywhere)."""
    return _get_bool_env("STRICT_REALISM", default=is_production_mode())


def require_real_contacts() -> bool:
    """Check if real PhysX contacts are required."""
    if not is_production_mode() and not is_strict_realism():
        return False
    return _get_bool_env("REQUIRE_REAL_CONTACTS", default=True)


def require_real_efforts() -> bool:
    """Check if real joint efforts (not backfilled) are required."""
    if not is_production_mode() and not is_strict_realism():
        return False
    return _get_bool_env("REQUIRE_REAL_EFFORTS", default=True)


def require_valid_rgb() -> bool:
    """Check if RGB frame quality validation is required."""
    if not is_production_mode() and not is_strict_realism():
        return False
    return _get_bool_env("REQUIRE_VALID_RGB", default=True)


def require_object_motion() -> bool:
    """Check if object motion validation is required for manipulation tasks."""
    if not is_production_mode() and not is_strict_realism():
        return False
    return _get_bool_env("REQUIRE_OBJECT_MOTION", default=True)


def require_intrinsics() -> bool:
    """Check if camera intrinsics are required."""
    if not is_production_mode() and not is_strict_realism():
        return False
    return _get_bool_env("REQUIRE_INTRINSICS", default=True)


def validate_scene_state_provenance(
    scene_state_provenance: str,
    real_scene_state_frames: int,
    total_frames: int,
    *,
    required_source: str = "physx_server",
    min_real_ratio: float = 1.0,
) -> Tuple[bool, Dict[str, Any]]:
    """Validate that scene_state is real and from PhysX for all frames.

    Args:
        scene_state_provenance: Provenance label for scene_state.
        real_scene_state_frames: Number of frames with real scene_state.
        total_frames: Total frame count.
        required_source: Required provenance label.
        min_real_ratio: Minimum real frame ratio (default 1.0).

    Returns:
        (is_valid, diagnostics_dict)

    Raises:
        DataFidelityError: If strict realism and scene_state not fully real.
    """
    ratio = real_scene_state_frames / total_frames if total_frames > 0 else 0.0
    diagnostics = {
        "scene_state_provenance": scene_state_provenance,
        "real_scene_state_frames": real_scene_state_frames,
        "total_frames": total_frames,
        "real_ratio": ratio,
        "required_source": required_source,
        "min_real_ratio": min_real_ratio,
    }

    is_valid = (
        scene_state_provenance == required_source
        and ratio >= min_real_ratio
    )

    if not is_valid and (require_object_motion() or is_strict_realism()):
        raise DataFidelityError(
            f"Scene state provenance invalid. "
            f"source={scene_state_provenance}, real_ratio={ratio:.2f} "
            f"(required={required_source}, min_ratio={min_real_ratio:.2f}). "
            f"Set STRICT_REALISM=false or REQUIRE_OBJECT_MOTION=false to allow fallback.",
            gate_name="scene_state_provenance",
            diagnostics=diagnostics,
        )

    return is_valid, diagnostics


# =============================================================================
# Validation Gate Functions
# =============================================================================

def validate_contact_data(
    contacts_available: bool,
    contact_source: str,
    contact_count: int = 0,
) -> Tuple[bool, Dict[str, Any]]:
    """Validate that contact data is from PhysX, not heuristics.

    Args:
        contacts_available: Whether contact capture succeeded
        contact_source: Source of contact data (e.g., "physx_contact_report", "heuristic")
        contact_count: Number of contacts captured

    Returns:
        (is_valid, diagnostics_dict)

    Raises:
        DataFidelityError: If contacts required but unavailable in production mode
    """
    diagnostics = {
        "contacts_available": contacts_available,
        "contact_source": contact_source,
        "contact_count": contact_count,
        "is_physx": contact_source.startswith("physx") if contact_source else False,
    }

    is_valid = contacts_available and diagnostics["is_physx"]

    if not is_valid and require_real_contacts():
        raise DataFidelityError(
            f"Contact data unavailable or not from PhysX. "
            f"Source: {contact_source}, Available: {contacts_available}. "
            f"Set REQUIRE_REAL_CONTACTS=false or DATA_FIDELITY_MODE=dev to allow heuristic fallback.",
            gate_name="contact_data",
            diagnostics=diagnostics,
        )

    return is_valid, diagnostics


def validate_joint_efforts(
    efforts_source: str,
    real_effort_count: int,
    total_frames: int,
) -> Tuple[bool, Dict[str, Any]]:
    """Validate that joint efforts are from PhysX, not inverse dynamics.

    Args:
        efforts_source: Source of effort data ("physx", "estimated_inverse_dynamics", etc.)
        real_effort_count: Number of frames with real PhysX efforts
        total_frames: Total number of frames

    Returns:
        (is_valid, diagnostics_dict)

    Raises:
        DataFidelityError: If efforts backfilled in production mode
    """
    diagnostics = {
        "efforts_source": efforts_source,
        "real_effort_count": real_effort_count,
        "total_frames": total_frames,
        "real_effort_ratio": real_effort_count / total_frames if total_frames > 0 else 0.0,
        "is_physx": efforts_source == "physx",
    }

    is_valid = efforts_source == "physx"

    if not is_valid and require_real_efforts():
        raise DataFidelityError(
            f"Joint efforts are backfilled via {efforts_source}. "
            f"Real efforts: {real_effort_count}/{total_frames} frames. "
            f"Set REQUIRE_REAL_EFFORTS=false or DATA_FIDELITY_MODE=dev to allow backfilled efforts.",
            gate_name="joint_efforts",
            diagnostics=diagnostics,
        )

    return is_valid, diagnostics


def validate_object_motion(
    any_moved: bool,
    max_displacement: float,
    task_requires_motion: bool,
    min_displacement_threshold: float = 0.001,  # 1mm
) -> Tuple[bool, Dict[str, Any]]:
    """Validate that objects actually moved for manipulation tasks.

    Args:
        any_moved: Whether any object moved above threshold
        max_displacement: Maximum displacement observed (meters)
        task_requires_motion: Whether the task type requires object motion
        min_displacement_threshold: Minimum displacement to consider valid (meters)

    Returns:
        (is_valid, diagnostics_dict)

    Raises:
        DataFidelityError: If motion required but not detected in production mode
    """
    diagnostics = {
        "any_moved": any_moved,
        "max_displacement_m": max_displacement,
        "task_requires_motion": task_requires_motion,
        "min_threshold_m": min_displacement_threshold,
        "above_threshold": max_displacement >= min_displacement_threshold,
    }

    # Valid if: task doesn't require motion, or motion was detected
    is_valid = not task_requires_motion or (any_moved and max_displacement >= min_displacement_threshold)

    if not is_valid and require_object_motion():
        raise DataFidelityError(
            f"Object motion required but max_displacement={max_displacement:.6f}m "
            f"(threshold: {min_displacement_threshold}m). "
            f"Objects may be reading USD time=0 instead of live physics state. "
            f"Set REQUIRE_OBJECT_MOTION=false or DATA_FIDELITY_MODE=dev to allow static scenes.",
            gate_name="object_motion",
            diagnostics=diagnostics,
        )

    return is_valid, diagnostics


def validate_camera_intrinsics(
    fx: Optional[float],
    fy: Optional[float],
    ppx: Optional[float],
    ppy: Optional[float],
    camera_id: str = "",
) -> Tuple[bool, Dict[str, Any]]:
    """Validate that camera intrinsics are present.

    Args:
        fx: Focal length x (pixels)
        fy: Focal length y (pixels)
        ppx: Principal point x (pixels)
        ppy: Principal point y (pixels)
        camera_id: Camera identifier for error messages

    Returns:
        (is_valid, diagnostics_dict)

    Raises:
        DataFidelityError: If intrinsics missing in production mode
    """
    diagnostics = {
        "camera_id": camera_id,
        "fx": fx,
        "fy": fy,
        "ppx": ppx,
        "ppy": ppy,
        "has_focal_length": fx is not None and fy is not None,
        "has_principal_point": ppx is not None and ppy is not None,
    }

    is_valid = all([fx is not None, fy is not None, ppx is not None, ppy is not None])

    if not is_valid and require_intrinsics():
        missing = []
        if fx is None:
            missing.append("fx")
        if fy is None:
            missing.append("fy")
        if ppx is None:
            missing.append("ppx")
        if ppy is None:
            missing.append("ppy")
        raise DataFidelityError(
            f"Camera {camera_id} missing intrinsics: {', '.join(missing)}. "
            f"Update camera handler patch to extract from USD camera attributes. "
            f"Set REQUIRE_INTRINSICS=false or DATA_FIDELITY_MODE=dev to allow missing intrinsics.",
            gate_name="camera_intrinsics",
            diagnostics=diagnostics,
        )

    return is_valid, diagnostics


def log_gate_status() -> None:
    """Log current status of all data fidelity gates."""
    mode = "production" if is_production_mode() else "dev"
    logger.info(
        "Data fidelity mode: %s | Gates: contacts=%s, efforts=%s, rgb=%s, motion=%s, intrinsics=%s",
        mode,
        require_real_contacts(),
        require_real_efforts(),
        require_valid_rgb(),
        require_object_motion(),
        require_intrinsics(),
    )
