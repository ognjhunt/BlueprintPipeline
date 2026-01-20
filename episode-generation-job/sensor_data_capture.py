#!/usr/bin/env python3
"""
Sensor Data Capture for Episode Generation.

Captures synthetic sensor data and ground-truth labels during simulation,
enabling rich visual observations for robot learning datasets.

Supports three data pack tiers:
- Core: RGB (1-N cams) + robot state + actions + episode metadata + success/QC
- Plus: Core + depth + segmentation + 2D/3D bboxes
- Full: Plus + object poses + contacts + normals + privileged state

Isaac Sim Replicator Integration:
- Uses omni.replicator for synthetic data generation
- Configurable camera setups (wrist, overhead, side views)
- All annotations generated per-frame during trajectory execution

DEPLOYMENT ARCHITECTURE:
========================
PRODUCTION (Cloud Run / Docker with GPU):
  - Isaac Sim IS available and IS the expected runtime
  - Uses Dockerfile.isaacsim (extends NVIDIA Isaac Sim 4.2.0)
  - docker-compose.isaacsim.yaml explicitly sets USE_MOCK_CAPTURE="false"
  - Real physics, real sensor data, production-quality episodes

LOCAL DEVELOPMENT (without GPU/Isaac Sim):
  - MockSensorCapture generates placeholder data for testing
  - Set USE_MOCK_CAPTURE="true" for development iteration
  - NOT for production training data

The production deployment DOES use Isaac Sim. Mock mode is development-only.

Compatible with:
- LeRobot v2.0 format (images as separate video files)
- RLDS format (observation dict with image keys)
- HuggingFace datasets (with image features)
"""

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Add parent to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.config.env import parse_bool_env
from tools.config.production_mode import resolve_production_mode

# Import Isaac Sim integration for availability checking
try:
    from isaac_sim_integration import (
        is_isaac_sim_available,
        is_replicator_available,
        print_availability_report,
        get_availability_status,
    )
    _HAVE_INTEGRATION_MODULE = True
except ImportError:
    _HAVE_INTEGRATION_MODULE = False
    def is_isaac_sim_available() -> bool:
        return False
    def is_replicator_available() -> bool:
        return False


# =============================================================================
# Data Pack Configuration (imported from data_pack_config)
# =============================================================================
# Import DataPackTier from primary definition to avoid duplication
from data_pack_config import DataPackTier
from usd_scene_scan import discover_camera_prim_specs, get_usd_stage, resolve_robot_prim_paths


class SensorDataCaptureMode(Enum):
    """
    Sensor data capture mode - controls Isaac Sim enforcement.

    This enum makes it EXPLICIT whether we're capturing real data or mock data,
    preventing silent degradation to unusable training data.
    """

    ISAAC_SIM = "isaac_sim"           # Full quality (production) - requires Isaac Sim
    MOCK_DEV = "mock_dev"             # Explicit dev mode - allows mock data with warnings
    FAIL_CLOSED = "fail_closed"       # Error if no Isaac Sim (default for production)


def _is_production_run() -> bool:
    """Detect production runs where fail-closed should be enforced."""
    return resolve_production_mode()


def _mock_capture_disallowed() -> bool:
    """Block mock capture in production or staging environments."""
    return _is_production_run()


def _mock_capture_block_reason() -> str:
    """Return the first production flag that disallows mock capture."""
    pipeline_env = (os.getenv("PIPELINE_ENV", "") or "").strip().lower()
    if pipeline_env in {"prod", "production"}:
        return f"PIPELINE_ENV={pipeline_env}"
    geniesim_env = (os.getenv("GENIESIM_ENV", "") or "").strip().lower()
    if geniesim_env in {"prod", "production"}:
        return f"GENIESIM_ENV={geniesim_env}"
    bp_env = (os.getenv("BP_ENV", "") or "").strip().lower()
    if bp_env in {"prod", "production"}:
        return f"BP_ENV={bp_env}"
    if parse_bool_env(os.getenv("REQUIRE_REAL_PHYSICS"), default=False):
        return "REQUIRE_REAL_PHYSICS=true"
    if os.getenv("DATA_QUALITY_LEVEL", "").lower() == "production":
        return "DATA_QUALITY_LEVEL=production"
    if parse_bool_env(os.getenv("PRODUCTION_MODE"), default=False):
        return "PRODUCTION_MODE=true"
    if parse_bool_env(os.getenv("ISAAC_SIM_REQUIRED"), default=False):
        return "ISAAC_SIM_REQUIRED=true"
    if parse_bool_env(os.getenv("PRODUCTION"), default=False):
        return "PRODUCTION=true"
    if os.getenv("LABS_STAGING", "").lower() in {"1", "true", "yes", "y"}:
        return "LABS_STAGING=true"
    return "production environment detected"


def _log_mock_capture_blocked(
    reason: str,
    capture_mode: Optional[SensorDataCaptureMode],
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """Emit a structured log entry when mock capture is blocked in production."""
    payload: Dict[str, Any] = {
        "event": "mock_capture_blocked",
        "reason": reason,
        "capture_mode": capture_mode.value if capture_mode else None,
        "use_mock": parse_bool_env(os.getenv("USE_MOCK_CAPTURE"), default=False),
        "allow_mock_capture": parse_bool_env(os.getenv("ALLOW_MOCK_CAPTURE"), default=False),
        "allow_mock_data": parse_bool_env(os.getenv("ALLOW_MOCK_DATA"), default=False),
        "sensor_capture_mode": os.getenv("SENSOR_CAPTURE_MODE", ""),
        "data_quality_level": os.getenv("DATA_QUALITY_LEVEL", ""),
        "labs_staging": os.getenv("LABS_STAGING", ""),
        "isaac_sim_required": os.getenv("ISAAC_SIM_REQUIRED", ""),
    }
    if details:
        payload.update(details)
    logger.error("%s", json.dumps(payload))


@dataclass
class CameraCalibration:
    """
    Full camera calibration data (DROID-style).

    This is what robotics labs expect for proper sim-to-real transfer
    and multi-view geometry applications.
    """

    # Intrinsic matrix (3x3) - converts 3D camera coords to 2D pixel coords
    # [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    intrinsic_matrix: Optional[np.ndarray] = None

    # Distortion coefficients (OpenCV format: k1, k2, p1, p2, k3, ...)
    distortion_coeffs: Optional[np.ndarray] = None

    # Extrinsic matrix (4x4) - camera-to-world transform
    # Transforms points from camera frame to world frame
    extrinsic_matrix: Optional[np.ndarray] = None

    # Camera-to-robot-base transform (4x4) - useful for eye-in-hand
    camera_to_robot_base: Optional[np.ndarray] = None

    # Timestamp of calibration (for tracking drift)
    calibration_timestamp: Optional[str] = None

    def compute_intrinsic_from_params(
        self,
        focal_length_mm: float,
        sensor_width_mm: float,
        sensor_height_mm: float,
        resolution: Tuple[int, int],
    ) -> np.ndarray:
        """Compute intrinsic matrix from physical camera parameters."""
        width, height = resolution

        # Focal length in pixels
        fx = focal_length_mm * width / sensor_width_mm
        fy = focal_length_mm * height / sensor_height_mm

        # Principal point (assume center)
        cx = width / 2.0
        cy = height / 2.0

        self.intrinsic_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float64)

        return self.intrinsic_matrix

    def to_dict(self) -> Dict[str, Any]:
        """Serialize calibration data."""
        result = {}
        if self.intrinsic_matrix is not None:
            result["intrinsic_matrix"] = self.intrinsic_matrix.tolist()
        if self.distortion_coeffs is not None:
            result["distortion_coeffs"] = self.distortion_coeffs.tolist()
        if self.extrinsic_matrix is not None:
            result["extrinsic_matrix"] = self.extrinsic_matrix.tolist()
        if self.camera_to_robot_base is not None:
            result["camera_to_robot_base"] = self.camera_to_robot_base.tolist()
        if self.calibration_timestamp is not None:
            result["calibration_timestamp"] = self.calibration_timestamp
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CameraCalibration":
        """Deserialize calibration data."""
        calib = cls()
        if "intrinsic_matrix" in data:
            calib.intrinsic_matrix = np.array(data["intrinsic_matrix"], dtype=np.float64)
        if "distortion_coeffs" in data:
            calib.distortion_coeffs = np.array(data["distortion_coeffs"], dtype=np.float64)
        if "extrinsic_matrix" in data:
            calib.extrinsic_matrix = np.array(data["extrinsic_matrix"], dtype=np.float64)
        if "camera_to_robot_base" in data:
            calib.camera_to_robot_base = np.array(data["camera_to_robot_base"], dtype=np.float64)
        if "calibration_timestamp" in data:
            calib.calibration_timestamp = data["calibration_timestamp"]
        return calib


@dataclass
class CameraConfig:
    """Configuration for a single camera."""

    camera_id: str
    prim_path: str  # USD prim path (e.g., "/World/Cameras/wrist_camera")
    resolution: Tuple[int, int] = (640, 480)
    focal_length: float = 24.0  # mm
    sensor_width: float = 36.0  # mm
    sensor_height: float = 24.0  # mm (default 3:2 aspect ratio)
    near_clip: float = 0.01  # meters
    far_clip: float = 100.0  # meters

    # Camera type hints for LeRobot naming
    camera_type: str = "rgb"  # "rgb", "wrist", "overhead", "side"

    # Which annotations to capture (based on data pack)
    capture_rgb: bool = True
    capture_depth: bool = False
    capture_segmentation: bool = False
    capture_instance_segmentation: bool = False
    capture_bbox_2d: bool = False
    capture_bbox_3d: bool = False
    capture_normals: bool = False

    # Full camera calibration (DROID-style)
    calibration: Optional[CameraCalibration] = None

    def get_lerobot_key(self) -> str:
        """Get the LeRobot observation key for this camera."""
        return f"observation.images.{self.camera_type}"

    def get_or_compute_calibration(self) -> CameraCalibration:
        """Get calibration, computing intrinsics from params if not set."""
        if self.calibration is None:
            self.calibration = CameraCalibration()

        if self.calibration.intrinsic_matrix is None:
            self.calibration.compute_intrinsic_from_params(
                focal_length_mm=self.focal_length,
                sensor_width_mm=self.sensor_width,
                sensor_height_mm=self.sensor_height,
                resolution=self.resolution,
            )
            # Default distortion to zero (simulation has no distortion)
            self.calibration.distortion_coeffs = np.zeros(5, dtype=np.float64)

        return self.calibration

    def set_extrinsic_from_transform(
        self,
        position: np.ndarray,
        rotation_quat: np.ndarray,
    ) -> None:
        """Set extrinsic matrix from position and quaternion."""
        if self.calibration is None:
            self.calibration = CameraCalibration()

        # Convert quaternion to rotation matrix
        # Quaternion format: [w, x, y, z]
        w, x, y, z = rotation_quat

        rotation_matrix = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ], dtype=np.float64)

        # Build 4x4 extrinsic matrix (camera-to-world)
        self.calibration.extrinsic_matrix = np.eye(4, dtype=np.float64)
        self.calibration.extrinsic_matrix[:3, :3] = rotation_matrix
        self.calibration.extrinsic_matrix[:3, 3] = position


@dataclass
class SensorDataConfig:
    """Configuration for sensor data capture during episode generation."""

    # Data pack tier
    data_pack: DataPackTier = DataPackTier.CORE

    # Camera configurations
    cameras: List[CameraConfig] = field(default_factory=list)

    # Image settings
    image_format: str = "png"  # "png", "jpg", "npy"
    depth_format: str = "npy"  # "npy", "exr", "png" (normalized)
    video_codec: str = "h264"  # For LeRobot video export

    # Resolution for all cameras (if not specified per-camera)
    default_resolution: Tuple[int, int] = (640, 480)

    # FPS for video encoding
    fps: float = 30.0

    # Ground-truth settings
    include_semantic_labels: bool = False
    include_instance_ids: bool = False
    include_object_poses: bool = False
    include_contact_info: bool = False
    include_privileged_state: bool = False

    # Performance settings
    render_offscreen: bool = True
    use_gpu_compression: bool = True
    max_concurrent_captures: int = 4
    # If set, raise when more than this many cameras fail in a frame (0 == require all)
    max_camera_failures: Optional[int] = None

    # Scene + robot configuration
    scene_usd_path: Optional[str] = None
    robot_prim_paths: Optional[List[str]] = None

    @classmethod
    def from_data_pack(
        cls,
        tier: DataPackTier,
        num_cameras: int = 1,
        resolution: Tuple[int, int] = (640, 480),
        fps: float = 30.0,
        camera_specs: Optional[List[Dict[str, str]]] = None,
        scene_usd_path: Optional[str] = None,
    ) -> "SensorDataConfig":
        """Create configuration from data pack tier."""
        config = cls(
            data_pack=tier,
            default_resolution=resolution,
            fps=fps,
        )

        # Configure based on tier
        if tier == DataPackTier.CORE:
            # Core: RGB only
            config.cameras = cls._create_default_cameras(
                num_cameras,
                resolution,
                camera_specs=camera_specs,
                scene_usd_path=scene_usd_path,
                capture_depth=False,
            )

        elif tier == DataPackTier.PLUS:
            # Plus: RGB + depth + segmentation + bboxes
            config.cameras = cls._create_default_cameras(
                num_cameras,
                resolution,
                camera_specs=camera_specs,
                scene_usd_path=scene_usd_path,
                capture_depth=True,
                capture_segmentation=True,
                capture_bbox_2d=True,
                capture_bbox_3d=True,
            )
            config.include_semantic_labels = True
            config.include_instance_ids = True

        elif tier == DataPackTier.FULL:
            # Full: everything
            config.cameras = cls._create_default_cameras(
                num_cameras,
                resolution,
                camera_specs=camera_specs,
                scene_usd_path=scene_usd_path,
                capture_depth=True,
                capture_segmentation=True,
                capture_bbox_2d=True,
                capture_bbox_3d=True,
                capture_normals=True,
            )
            config.include_semantic_labels = True
            config.include_instance_ids = True
            config.include_object_poses = True
            config.include_contact_info = True
            config.include_privileged_state = True

        return config

    @staticmethod
    def _create_default_cameras(
        num_cameras: int,
        resolution: Tuple[int, int],
        camera_specs: Optional[List[Dict[str, str]]] = None,
        scene_usd_path: Optional[str] = None,
        capture_depth: bool = False,
        capture_segmentation: bool = False,
        capture_bbox_2d: bool = False,
        capture_bbox_3d: bool = False,
        capture_normals: bool = False,
    ) -> List[CameraConfig]:
        """Create default camera configurations."""
        cameras: List[CameraConfig] = []

        normalized_specs = []
        for spec in camera_specs or []:
            if not isinstance(spec, dict):
                continue
            prim_path = spec.get("prim_path") or spec.get("path")
            if not prim_path:
                continue
            camera_type = spec.get("camera_type") or spec.get("type") or "rgb"
            camera_id = spec.get("camera_id") or spec.get("id") or camera_type
            normalized_specs.append(
                {
                    "prim_path": prim_path,
                    "camera_type": camera_type,
                    "camera_id": camera_id,
                }
            )

        if not normalized_specs:
            normalized_specs = discover_camera_prim_specs(scene_usd_path=scene_usd_path)

        if not normalized_specs:
            raise RuntimeError(
                "No camera configuration provided and no cameras discovered in the USD. "
                "Add cameras to the scene config or ensure the USD contains camera prims."
            )

        stage = get_usd_stage(scene_usd_path) if scene_usd_path else None
        missing_prims: List[str] = []
        validated_specs = []

        for spec in normalized_specs:
            prim_path = spec["prim_path"]
            if stage is not None:
                prim = stage.GetPrimAtPath(prim_path)
                if not prim.IsValid():
                    missing_prims.append(prim_path)
                    continue
                try:
                    from pxr import UsdGeom
                    is_camera = prim.IsA(UsdGeom.Camera)
                except Exception:
                    is_camera = False
                if not is_camera:
                    missing_prims.append(prim_path)
                    continue
            validated_specs.append(spec)

        if missing_prims:
            missing_list = ", ".join(sorted(set(missing_prims)))
            raise RuntimeError(
                f"Required camera prims are missing or invalid in the USD: {missing_list}"
            )

        if len(validated_specs) < num_cameras:
            raise RuntimeError(
                f"Requested {num_cameras} cameras but only {len(validated_specs)} available."
            )

        for i in range(num_cameras):
            spec = validated_specs[i]
            camera_id = spec.get("camera_id") or f"camera_{i + 1}"
            prim_path = spec["prim_path"]
            camera_type = spec.get("camera_type") or "rgb"
            cameras.append(
                CameraConfig(
                    camera_id=camera_id,
                    prim_path=prim_path,
                    resolution=resolution,
                    camera_type=camera_type,
                    capture_rgb=True,
                    capture_depth=capture_depth,
                    capture_segmentation=capture_segmentation,
                    capture_instance_segmentation=capture_segmentation,
                    capture_bbox_2d=capture_bbox_2d,
                    capture_bbox_3d=capture_bbox_3d,
                    capture_normals=capture_normals,
                )
            )

        return cameras


# =============================================================================
# Sensor Data Models
# =============================================================================


@dataclass
class ContactData:
    """
    Formal ContactData schema.

    Represents a single contact event from physics simulation.
    Includes all information needed for contact-aware learning.
    """

    # Bodies in contact
    body_a: str                           # First body name (e.g., 'gripper', 'table')
    body_b: str                           # Second body name

    # Contact geometry
    position: Tuple[float, float, float]  # Contact point in world coordinates (3D)
    normal: Tuple[float, float, float]    # Contact normal vector (unit vector)

    # Forces and properties
    force_magnitude: float                # Magnitude of impulse/force (N)
    separation: float                     # Separation distance (negative = penetrating)

    # Optional fields for advanced analysis
    contact_area: Optional[float] = None  # Contact area (if available)
    relative_velocity: Optional[Tuple[float, float, float]] = None  # Relative velocity at contact
    friction_coefficient: Optional[float] = None  # Friction coefficient

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "body_a": self.body_a,
            "body_b": self.body_b,
            "position": list(self.position),
            "normal": list(self.normal),
            "force_magnitude": float(self.force_magnitude),
            "separation": float(self.separation),
            "contact_area": float(self.contact_area) if self.contact_area is not None else None,
            "relative_velocity": list(self.relative_velocity) if self.relative_velocity is not None else None,
            "friction_coefficient": float(self.friction_coefficient) if self.friction_coefficient is not None else None,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'ContactData':
        """Create from dictionary."""
        return ContactData(
            body_a=str(data.get("body_a", "")),
            body_b=str(data.get("body_b", "")),
            position=tuple(data.get("position", [0, 0, 0])),
            normal=tuple(data.get("normal", [0, 0, 1])),
            force_magnitude=float(data.get("force_magnitude", 0)),
            separation=float(data.get("separation", 0)),
            contact_area=float(data["contact_area"]) if data.get("contact_area") is not None else None,
            relative_velocity=tuple(data["relative_velocity"]) if data.get("relative_velocity") is not None else None,
            friction_coefficient=float(data["friction_coefficient"]) if data.get("friction_coefficient") is not None else None,
        )


@dataclass
class FrameSensorData:
    """
    Formal FrameSensorData schema with proper type hints.

    Sensor data captured for a single frame during trajectory execution.
    Includes visual observations, ground-truth annotations, and physics state.
    """

    frame_index: int                                    # Frame number in episode
    timestamp: float                                    # Timestamp in seconds

    # =========================================================================
    # Visual Observations (per-camera)
    # =========================================================================

    # RGB images: {camera_id: ndarray(H, W, 3) uint8 [0-255]}
    rgb_images: Dict[str, np.ndarray] = field(default_factory=dict)

    # Depth maps: {camera_id: ndarray(H, W) float32 [meters]}
    depth_maps: Dict[str, np.ndarray] = field(default_factory=dict)

    # =========================================================================
    # Segmentation (per-camera)
    # =========================================================================

    # Semantic segmentation: {camera_id: ndarray(H, W) uint8 [class_id]}
    semantic_masks: Dict[str, np.ndarray] = field(default_factory=dict)

    # Instance segmentation: {camera_id: ndarray(H, W) uint16 [instance_id]}
    instance_masks: Dict[str, np.ndarray] = field(default_factory=dict)

    # =========================================================================
    # Bounding Boxes (per-camera)
    # =========================================================================

    # 2D bounding boxes: {camera_id: List[Dict]} (COCO format: [x, y, w, h])
    bboxes_2d: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    # 3D bounding boxes: {camera_id: List[Dict]} (center + size + orientation)
    bboxes_3d: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    # =========================================================================
    # Geometry (per-camera)
    # =========================================================================

    # Surface normals: {camera_id: ndarray(H, W, 3) float32}
    normals: Dict[str, np.ndarray] = field(default_factory=dict)

    # =========================================================================
    # Ground-Truth Annotations
    # =========================================================================

    # Object poses (world frame): {object_id: {'position': [x,y,z], 'orientation': [qx,qy,qz,qw], ...}}
    object_poses: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Contact information: List[ContactData] - contact events at this frame
    contacts: List[Dict[str, Any]] = field(default_factory=list)
    contacts_available: Optional[bool] = None

    # =========================================================================
    # Privileged State (full physics state, not observable by robot)
    # =========================================================================

    # Privileged state: {'object_velocities': {...}, 'contact_flags': {...}, ...}
    privileged_state: Optional[Dict[str, Any]] = None

    # =========================================================================
    # Metadata
    # =========================================================================

    # Flag indicating this frame is from mock sensor capture (not real simulation)
    is_mock: bool = False

    def validate_rgb_frames(self) -> List[str]:
        """
        Validate RGB frames meet quality standards.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        for camera_id, rgb in self.rgb_images.items():
            # Check shape
            if len(rgb.shape) != 3 or rgb.shape[2] != 3:
                errors.append(f"{camera_id} RGB: Expected shape (H, W, 3), got {rgb.shape}")
            # Check dtype
            if rgb.dtype != np.uint8:
                errors.append(f"{camera_id} RGB: Expected dtype uint8, got {rgb.dtype}")
            # Check value range
            if rgb.min() < 0 or rgb.max() > 255:
                errors.append(f"{camera_id} RGB: Values outside [0, 255] range")
        return errors

    def validate_depth_frames(self) -> List[str]:
        """
        Validate depth maps meet quality standards.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        for camera_id, depth in self.depth_maps.items():
            # Check shape
            if len(depth.shape) != 2:
                errors.append(f"{camera_id} depth: Expected shape (H, W), got {depth.shape}")
            # Check dtype
            if depth.dtype != np.float32:
                errors.append(f"{camera_id} depth: Expected dtype float32, got {depth.dtype}")
            # Check for NaN/Inf
            if np.any(np.isnan(depth)) or np.any(np.isinf(depth)):
                errors.append(f"{camera_id} depth: Contains NaN or Inf values")
            # Check reasonable range (0.01m to 200m)
            valid_depth = depth[(depth > 0) & np.isfinite(depth)]
            if len(valid_depth) > 0:
                if valid_depth.min() < 0.01 or valid_depth.max() > 200.0:
                    errors.append(
                        f"{camera_id} depth: Values outside [0.01m, 200m] range "
                        f"(min={valid_depth.min():.3f}, max={valid_depth.max():.1f})"
                    )
        return errors

    def validate_segmentation_frames(self) -> List[str]:
        """
        Validate segmentation masks meet quality standards.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Check semantic masks
        for camera_id, sem_mask in self.semantic_masks.items():
            if len(sem_mask.shape) != 2:
                errors.append(f"{camera_id} semantic: Expected shape (H, W), got {sem_mask.shape}")
            if sem_mask.dtype not in (np.uint8, np.uint16):
                errors.append(f"{camera_id} semantic: Expected dtype uint8/uint16, got {sem_mask.dtype}")

        # Check instance masks
        for camera_id, inst_mask in self.instance_masks.items():
            if len(inst_mask.shape) != 2:
                errors.append(f"{camera_id} instance: Expected shape (H, W), got {inst_mask.shape}")
            if inst_mask.dtype != np.uint16:
                errors.append(f"{camera_id} instance: Expected dtype uint16, got {inst_mask.dtype}")

        return errors


@dataclass
class EpisodeSensorData:
    """All sensor data captured for an episode."""

    episode_id: str
    config: SensorDataConfig

    # Per-frame data
    frames: List[FrameSensorData] = field(default_factory=list)

    # Semantic label mapping {label_id: label_name}
    semantic_labels: Dict[int, str] = field(default_factory=dict)

    # Instance ID to object mapping {instance_id: object_id}
    instance_to_object: Dict[int, str] = field(default_factory=dict)

    # Object metadata {object_id: {class, dimensions, etc.}}
    object_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Capture diagnostics
    camera_capture_warnings: List[str] = field(default_factory=list)
    camera_error_counts: Dict[str, int] = field(default_factory=dict)

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    @property
    def has_rgb(self) -> bool:
        return len(self.frames) > 0 and len(self.frames[0].rgb_images) > 0

    @property
    def has_depth(self) -> bool:
        return len(self.frames) > 0 and len(self.frames[0].depth_maps) > 0

    @property
    def has_segmentation(self) -> bool:
        return len(self.frames) > 0 and len(self.frames[0].semantic_masks) > 0

    @property
    def camera_ids(self) -> List[str]:
        if len(self.frames) == 0:
            return []
        return list(self.frames[0].rgb_images.keys())

    def validate_all_frames(self) -> List[str]:
        """
        Validate all frames in episode.

        Calls validation methods on each frame and collects errors.

        Returns:
            List of validation error messages
        """
        all_errors = []

        for frame in self.frames:
            frame_errors = []
            frame_errors.extend(frame.validate_rgb_frames())
            frame_errors.extend(frame.validate_depth_frames())
            frame_errors.extend(frame.validate_segmentation_frames())

            # Prefix with frame number
            for error in frame_errors:
                all_errors.append(f"Frame {frame.frame_index}: {error}")

        return all_errors


# =============================================================================
# Isaac Sim Sensor Data Capture
# =============================================================================


class IsaacSimSensorCapture:
    """
    Captures sensor data during Isaac Sim trajectory execution.

    Integrates with omni.replicator for efficient synthetic data generation.
    Supports all annotation types in Isaac Sim Replicator.

    IMPORTANT: This class requires Isaac Sim to be running for real sensor capture.
    When running outside Isaac Sim, initialization will fail.

    Usage (inside Isaac Sim):
        capture = IsaacSimSensorCapture(config)
        if capture.initialize():
            frame_data = capture.capture_frame(0, 0.0, scene_objects)
        else:
            raise RuntimeError("Isaac Sim not available; use dev-only mock capture explicitly.")
    """

    def __init__(
        self,
        config: SensorDataConfig,
        verbose: bool = True,
    ):
        self.config = config
        self.verbose = verbose
        self.initialized = False
        self._using_mock = False

        # Replicator handles (set during initialization)
        self._render_products: Dict[str, Any] = {}
        self._annotators: Dict[str, Dict[str, Any]] = {}
        self._writer = None
        self._camera_error_counts: Dict[str, int] = {}

        # Isaac Sim module references
        self._rep = None
        self._omni = None
        self._physx = None

        # Check Isaac Sim availability upfront
        self._isaac_sim_available = is_isaac_sim_available()
        self._replicator_available = is_replicator_available()
        self._require_contacts = parse_bool_env(os.getenv("REQUIRE_CONTACTS"), default=False)
        self._require_sim_object_poses = parse_bool_env(
            os.getenv("REQUIRE_SIM_OBJECT_POSES"), default=False
        )

    def log(self, msg: str, level: str = "INFO") -> None:
        if self.verbose:
            level_map = {
                "DEBUG": logger.debug,
                "INFO": logger.info,
                "WARNING": logger.warning,
                "ERROR": logger.error,
            }
            log_fn = level_map.get(level.upper(), logger.info)
            log_fn("[SENSOR-CAPTURE] [%s] %s", level, msg)

    def initialize(self, scene_path: Optional[str] = None) -> bool:
        """
        Initialize sensor capture in Isaac Sim.

        Args:
            scene_path: Optional USD scene path to load

        Returns:
            True if initialization successful (with real Isaac Sim),
            False if Isaac Sim is not available (caller should use MockSensorCapture)
        """
        # First, check if we're in Isaac Sim environment
        if not self._isaac_sim_available:
            self.log(
                "ERROR: Not running inside Isaac Sim environment!\n"
                "       Real sensor capture requires Isaac Sim.\n"
                "       Run with: /isaac-sim/python.sh your_script.py\n"
                "       Or use MockSensorCapture for testing.",
                "ERROR"
            )
            return False

        # Try to import and initialize Replicator
        try:
            import omni.replicator.core as rep
            import omni.usd
            self._rep = rep
            self.log("Isaac Sim Replicator initialized successfully")
        except ImportError as e:
            self.log(
                f"Failed to import Replicator: {e}\n"
                "       Ensure Isaac Sim 2023.1+ with Replicator extension enabled.",
                "ERROR"
            )
            return False

        # Load scene if provided
        if scene_path:
            try:
                import omni.usd
                self.log(f"Loading scene: {scene_path}")
                omni.usd.get_context().open_stage(scene_path)
            except Exception as e:
                self.log(f"Failed to load scene: {e}", "WARNING")

        # Set up cameras and annotators
        try:
            for camera_config in self.config.cameras:
                self._setup_camera(camera_config)

            self.initialized = True
            self.log(f"Initialized {len(self.config.cameras)} cameras for real capture")
            return True

        except Exception as e:
            self.log(f"Camera initialization failed: {e}", "ERROR")
            return False

    def is_using_real_capture(self) -> bool:
        """Check if using real Isaac Sim capture (not mock)."""
        return self.initialized and self._rep is not None

    def is_mock(self) -> bool:
        """
        Check if this is a mock sensor capture instance.

        Returns:
            False for real Isaac Sim sensor capture
        """
        return False

    def _setup_camera(self, camera_config: CameraConfig) -> None:
        """Set up a camera with its annotators."""
        if self._rep is None:
            return

        rep = self._rep

        # Create render product for this camera
        try:
            render_product = rep.create.render_product(
                camera_config.prim_path,
                resolution=camera_config.resolution,
            )
            self._render_products[camera_config.camera_id] = render_product
        except Exception as e:
            self.log(f"Failed to create render product for {camera_config.camera_id}: {e}", "WARNING")
            return

        # Create annotators based on configuration
        annotators = {}

        if camera_config.capture_rgb:
            annotators["rgb"] = rep.AnnotatorRegistry.get_annotator("rgb")
            annotators["rgb"].attach([render_product])

        if camera_config.capture_depth:
            annotators["depth"] = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
            annotators["depth"].attach([render_product])

        if camera_config.capture_segmentation:
            annotators["semantic"] = rep.AnnotatorRegistry.get_annotator("semantic_segmentation")
            annotators["semantic"].attach([render_product])

        if camera_config.capture_instance_segmentation:
            annotators["instance"] = rep.AnnotatorRegistry.get_annotator("instance_segmentation")
            annotators["instance"].attach([render_product])

        if camera_config.capture_bbox_2d:
            annotators["bbox_2d"] = rep.AnnotatorRegistry.get_annotator("bounding_box_2d_tight")
            annotators["bbox_2d"].attach([render_product])

        if camera_config.capture_bbox_3d:
            annotators["bbox_3d"] = rep.AnnotatorRegistry.get_annotator("bounding_box_3d")
            annotators["bbox_3d"].attach([render_product])

        if camera_config.capture_normals:
            annotators["normals"] = rep.AnnotatorRegistry.get_annotator("normals")
            annotators["normals"].attach([render_product])

        self._annotators[camera_config.camera_id] = annotators

    def capture_frame(
        self,
        frame_index: int,
        timestamp: float,
        scene_objects: Optional[List[Dict[str, Any]]] = None,
    ) -> FrameSensorData:
        """
        Capture sensor data for a single frame.

        Args:
            frame_index: Frame index in episode
            timestamp: Timestamp in seconds
            scene_objects: List of scene objects for pose capture

        Returns:
            FrameSensorData with all captured data
        """
        frame_data = FrameSensorData(
            frame_index=frame_index,
            timestamp=timestamp,
        )

        if not self.initialized:
            return frame_data

        # Capture from each camera
        failed_cameras = []
        for camera_id, annotators in self._annotators.items():
            try:
                # RGB
                if "rgb" in annotators:
                    rgb_data = annotators["rgb"].get_data()
                    if rgb_data is not None:
                        frame_data.rgb_images[camera_id] = rgb_data["data"]

                # Depth
                if "depth" in annotators:
                    depth_data = annotators["depth"].get_data()
                    if depth_data is not None:
                        frame_data.depth_maps[camera_id] = depth_data["data"]

                # Semantic segmentation
                if "semantic" in annotators:
                    sem_data = annotators["semantic"].get_data()
                    if sem_data is not None:
                        frame_data.semantic_masks[camera_id] = sem_data["data"]

                # Instance segmentation
                if "instance" in annotators:
                    inst_data = annotators["instance"].get_data()
                    if inst_data is not None:
                        frame_data.instance_masks[camera_id] = inst_data["data"]

                # 2D bounding boxes
                if "bbox_2d" in annotators:
                    bbox2d_data = annotators["bbox_2d"].get_data()
                    if bbox2d_data is not None:
                        frame_data.bboxes_2d[camera_id] = self._parse_bbox2d(bbox2d_data)

                # 3D bounding boxes
                if "bbox_3d" in annotators:
                    bbox3d_data = annotators["bbox_3d"].get_data()
                    if bbox3d_data is not None:
                        frame_data.bboxes_3d[camera_id] = self._parse_bbox3d(bbox3d_data)

                # Normals
                if "normals" in annotators:
                    normals_data = annotators["normals"].get_data()
                    if normals_data is not None:
                        frame_data.normals[camera_id] = normals_data["data"]

            except Exception as e:
                import traceback
                failed_cameras.append(camera_id)
                self._camera_error_counts[camera_id] = self._camera_error_counts.get(camera_id, 0) + 1
                error_payload = {
                    "event": "camera_capture_error",
                    "camera_id": camera_id,
                    "frame_index": frame_index,
                    "error_count": self._camera_error_counts[camera_id],
                    "error": str(e),
                }
                self.log(json.dumps(error_payload), "ERROR")
                if self.verbose:
                    self.log(traceback.format_exc(), "DEBUG")

        # Check failure thresholds
        if failed_cameras:
            total_cameras = len(self._annotators)
            max_failures = self.config.max_camera_failures
            if max_failures is not None and len(failed_cameras) > max_failures:
                error_payload = {
                    "event": "camera_failure_threshold_exceeded",
                    "frame_index": frame_index,
                    "failed_cameras": failed_cameras,
                    "failed_count": len(failed_cameras),
                    "total_cameras": total_cameras,
                    "max_camera_failures": max_failures,
                }
                self.log(json.dumps(error_payload), "ERROR")
                raise RuntimeError(
                    "Camera failure threshold exceeded for frame "
                    f"{frame_index}: failed {len(failed_cameras)}/{total_cameras} "
                    f"cameras (max allowed={max_failures})."
                )
            if len(failed_cameras) == total_cameras:
                error_payload = {
                    "event": "all_cameras_failed",
                    "frame_index": frame_index,
                    "failed_cameras": failed_cameras,
                    "failed_count": len(failed_cameras),
                    "total_cameras": total_cameras,
                }
                self.log(json.dumps(error_payload), "ERROR")
                raise RuntimeError(f"All {len(failed_cameras)} cameras failed to capture frame {frame_index}")
            self.log(
                f"Frame {frame_index}: {len(failed_cameras)}/{total_cameras} cameras failed",
                "WARNING",
            )

        # Object poses (if enabled)
        if self.config.include_object_poses and scene_objects:
            frame_data.object_poses = self._capture_object_poses(scene_objects)

        # Contact information (if enabled)
        if self.config.include_contact_info:
            contacts, contacts_available = self._capture_contacts()
            frame_data.contacts = contacts
            frame_data.contacts_available = contacts_available

        # Privileged state (if enabled)
        if self.config.include_privileged_state:
            frame_data.privileged_state = self._capture_privileged_state(scene_objects)

        return frame_data

    def _parse_bbox2d(self, bbox_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse 2D bounding box data to COCO format."""
        bboxes = []

        if "data" not in bbox_data:
            return bboxes

        data = bbox_data["data"]
        id_to_labels = bbox_data.get("info", {}).get("idToLabels", {})

        for i, bbox in enumerate(data):
            x_min, y_min, x_max, y_max = bbox[:4]
            semantic_id = int(bbox[4]) if len(bbox) > 4 else 0

            bboxes.append({
                "bbox": [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)],
                "category_id": semantic_id,
                "category_name": id_to_labels.get(str(semantic_id), "unknown"),
                "area": float((x_max - x_min) * (y_max - y_min)),
            })

        return bboxes

    def _parse_bbox3d(self, bbox_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse 3D bounding box data."""
        bboxes = []

        if "data" not in bbox_data:
            return bboxes

        data = bbox_data["data"]
        id_to_labels = bbox_data.get("info", {}).get("idToLabels", {})

        for bbox in data:
            # Extract 3D bbox parameters
            center = bbox.get("center", [0, 0, 0])
            dimensions = bbox.get("dimensions", [0, 0, 0])
            rotation = bbox.get("rotation", [1, 0, 0, 0])  # quaternion
            semantic_id = bbox.get("semanticId", 0)

            bboxes.append({
                "center": [float(x) for x in center],
                "dimensions": [float(x) for x in dimensions],
                "rotation_quat": [float(x) for x in rotation],
                "category_id": semantic_id,
                "category_name": id_to_labels.get(str(semantic_id), "unknown"),
            })

        return bboxes

    def _capture_object_poses(
        self, scene_objects: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Capture object poses from simulation.

        Args:
            scene_objects: List of scene objects with metadata

        Returns:
            Dict mapping object_id to pose data (position, rotation, velocities)
        """
        poses = {}
        fallback_used = False
        sim_query_count = 0

        try:
            # Try to get poses from Isaac Sim
            if self._omni is not None:
                from pxr import UsdGeom, UsdPhysics
                import omni.isaac.core.utils.stage as stage_utils
                from omni.isaac.core.utils.stage import get_current_stage

                stage = get_current_stage()
                if stage is None:
                    self.log("No stage available for object pose capture", "WARNING")
                    fallback_used = True
                else:
                    for obj in scene_objects:
                        obj_id = obj.get("id", obj.get("name", ""))
                        prim_path = obj.get("prim_path", f"/World/Objects/{obj_id}")

                        try:
                            prim = stage.GetPrimAtPath(prim_path)
                            if prim.IsValid():
                                xformable = UsdGeom.Xformable(prim)
                                world_transform = xformable.ComputeLocalToWorldTransform(0)

                                # Extract position
                                translation = world_transform.ExtractTranslation()
                                position = [
                                    float(translation[0]),
                                    float(translation[1]),
                                    float(translation[2])
                                ]

                                # Extract rotation quaternion
                                rotation = world_transform.ExtractRotationQuat()
                                rotation_quat = [
                                    float(rotation.GetReal()),
                                    float(rotation.GetImaginary()[0]),
                                    float(rotation.GetImaginary()[1]),
                                    float(rotation.GetImaginary()[2])
                                ]

                                # Try to get velocities from rigid body
                                linear_velocity = [0.0, 0.0, 0.0]
                                angular_velocity = [0.0, 0.0, 0.0]

                                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                                    try:
                                        from omni.isaac.core.prims import RigidPrim
                                        rigid = RigidPrim(prim_path)
                                        vel = rigid.get_linear_velocity()
                                        ang_vel = rigid.get_angular_velocity()
                                        if vel is not None:
                                            linear_velocity = [float(v) for v in vel]
                                        if ang_vel is not None:
                                            angular_velocity = [float(v) for v in ang_vel]
                                    except Exception as exc:
                                        self.log(
                                            f"Failed to read velocities for {obj_id} ({prim_path}): {exc}",
                                            "DEBUG",
                                        )

                                poses[obj_id] = {
                                    "position": position,
                                    "rotation_quat": rotation_quat,
                                    "linear_velocity": linear_velocity,
                                    "angular_velocity": angular_velocity,
                                    "prim_path": prim_path,
                                    "source": "simulation",
                                }
                                sim_query_count += 1

                        except Exception as e:
                            self.log(f"Failed to get pose for {obj_id}: {e}", "DEBUG")

        except Exception as e:
            self.log(f"Isaac Sim object pose query failed: {e}", "WARNING")
            fallback_used = True

        # Fallback for objects not found in simulation
        for obj in scene_objects:
            obj_id = obj.get("id", obj.get("name", ""))
            if obj_id not in poses:
                fallback_used = True
                position = obj.get("position", [0, 0, 0])
                rotation = obj.get("rotation", [1, 0, 0, 0])

                poses[obj_id] = {
                    "position": [float(x) for x in position],
                    "rotation_quat": [float(x) for x in rotation],
                    "linear_velocity": [0.0, 0.0, 0.0],
                    "angular_velocity": [0.0, 0.0, 0.0],
                    "source": "input_fallback",
                }

        # Log summary
        if fallback_used and sim_query_count > 0:
            self.log(
                f"Object poses: {sim_query_count} from sim, "
                f"{len(poses) - sim_query_count} from fallback",
                "WARNING"
            )
        elif fallback_used:
            self.log(
                "All object poses from input fallback (not from simulation)",
                "WARNING"
            )

        if fallback_used and (_is_production_run() or self._require_sim_object_poses):
            raise RuntimeError(
                "Object pose capture used input fallback data. "
                "Production runs require simulation-derived object poses. "
                "Ensure Isaac Sim provides object poses or set REQUIRE_SIM_OBJECT_POSES=false "
                "outside production."
            )

        return poses

    def _handle_missing_contacts(self, reason: str) -> bool:
        if _is_production_run():
            raise RuntimeError(
                "Contact capture unavailable in production mode: "
                f"{reason}. Ensure PhysX contact reporting is available."
            )
        if self._require_contacts:
            raise RuntimeError(
                "Contact capture required but unavailable: "
                f"{reason}. Set REQUIRE_CONTACTS=false to allow missing contacts."
            )
        self.log(f"Contact capture unavailable: {reason}", "WARNING")
        return False

    def _capture_contacts(self) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Capture contact information from physics simulation.

        Returns:
            Tuple of (contacts, contacts_available). When contacts_available is False,
            contacts were not captured due to missing PhysX support.
        """
        contacts = []

        if self._omni is None:
            return contacts, self._handle_missing_contacts("omni not available")

        try:
            from omni.physx import get_physx_interface

            physx_interface = get_physx_interface()
            if physx_interface is None:
                return contacts, self._handle_missing_contacts(
                    "PhysX interface not available for contact capture"
                )

            # Get contact report
            contact_data = physx_interface.get_contact_report()

            if contact_data is None:
                self.log("Contact report returned None", "DEBUG")
                return contacts, True

            for contact in contact_data:
                try:
                    contacts.append({
                        "body_a": str(contact.get("actor0", "")),
                        "body_b": str(contact.get("actor1", "")),
                        "position": [
                            float(x) for x in contact.get("position", [0, 0, 0])
                        ],
                        "normal": [
                            float(x) for x in contact.get("normal", [0, 0, 1])
                        ],
                        "force_magnitude": float(contact.get("impulse", 0)),
                        "separation": float(contact.get("separation", 0)),
                    })
                except (TypeError, ValueError) as e:
                    self.log(f"Failed to parse contact data: {e}", "DEBUG")

            self.log(f"Captured {len(contacts)} contacts", "DEBUG")

        except ImportError as e:
            return contacts, self._handle_missing_contacts(f"PhysX import failed: {e}")
        except AttributeError as e:
            return contacts, self._handle_missing_contacts(f"PhysX API error: {e}")
        except Exception as e:
            self.log(f"Unexpected error capturing contacts: {e}", "ERROR")
            import traceback
            self.log(traceback.format_exc(), "DEBUG")
            return contacts, self._handle_missing_contacts(
                f"Unexpected error capturing contacts: {e}"
            )

        return contacts, True

    def _capture_privileged_state(
        self, scene_objects: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Capture full privileged physics state from simulation.

        This includes ground-truth information not available to the robot:
        - Object poses and velocities
        - Robot internal state
        - Contact information
        - Scene metadata

        Args:
            scene_objects: List of scene objects for pose lookup

        Returns:
            Dict with object_states, robot_state, scene_state, contacts
        """
        state = {
            "object_states": {},
            "robot_state": {},
            "scene_state": {},
            "contacts": [],
            "contacts_available": None,
            "data_source": "simulation" if self._omni is not None else "input_fallback",
            "is_mock": self.is_mock(),  # Explicit flag for mock data
        }

        # Capture contacts first
        contacts, contacts_available = self._capture_contacts()
        state["contacts"] = contacts
        state["contacts_available"] = contacts_available

        # Check for active grasps based on contacts
        active_grasps = set()
        gripper_keywords = ["hand", "finger", "gripper", "panda_leftfinger", "panda_rightfinger"]
        for contact in state["contacts"]:
            body_a = contact.get("body_a", "").lower()
            body_b = contact.get("body_b", "").lower()
            # If gripper is in contact with something
            for kw in gripper_keywords:
                if kw in body_a:
                    active_grasps.add(body_b.split("/")[-1])
                if kw in body_b:
                    active_grasps.add(body_a.split("/")[-1])

        # Capture object states
        if scene_objects:
            # First try to get poses from simulation
            sim_poses = self._capture_object_poses(scene_objects)

            for obj in scene_objects:
                obj_id = obj.get("id", obj.get("name", ""))

                # Use simulation data if available
                if obj_id in sim_poses:
                    pose_data = sim_poses[obj_id]
                    state["object_states"][obj_id] = {
                        "position": pose_data.get("position", [0, 0, 0]),
                        "rotation": pose_data.get("rotation_quat", [1, 0, 0, 0]),
                        "linear_velocity": pose_data.get("linear_velocity", [0, 0, 0]),
                        "angular_velocity": pose_data.get("angular_velocity", [0, 0, 0]),
                        "is_grasped": obj_id.lower() in active_grasps,
                        "in_contact": any(
                            obj_id.lower() in c.get("body_a", "").lower() or
                            obj_id.lower() in c.get("body_b", "").lower()
                            for c in state["contacts"]
                        ),
                        "data_source": pose_data.get("source", "unknown"),
                    }
                else:
                    # Fallback to input data
                    state["object_states"][obj_id] = {
                        "position": obj.get("position", [0, 0, 0]),
                        "rotation": obj.get("rotation", [1, 0, 0, 0]),
                        "linear_velocity": [0.0, 0.0, 0.0],
                        "angular_velocity": [0.0, 0.0, 0.0],
                        "is_grasped": False,
                        "in_contact": False,
                        "data_source": "input_fallback",
                    }

        # Capture robot state
        if self._omni is not None:
            try:
                from omni.isaac.core.articulations import Articulation
                from omni.isaac.core.utils.stage import get_current_stage

                stage = get_current_stage()
                if stage is not None:
                    robot_paths = resolve_robot_prim_paths(
                        self.config.robot_prim_paths,
                        scene_usd_path=self.config.scene_usd_path,
                        stage=stage,
                    )
                    if not robot_paths:
                        raise RuntimeError(
                            "No robot prims found for sensor capture. "
                            "Provide robot_prim_paths in scenes/<scene_id>/config.json "
                            "or ensure the USD scene has an ArticulationRoot prim."
                        )

                    for robot_path in robot_paths:
                        robot_prim = stage.GetPrimAtPath(robot_path)
                        if not robot_prim.IsValid():
                            continue
                        try:
                            robot = Articulation(robot_path)
                            if not robot.initialized:
                                robot.initialize()

                            joint_pos = robot.get_joint_positions()
                            joint_vel = robot.get_joint_velocities()

                            state["robot_state"] = {
                                "prim_path": robot_path,
                                "joint_positions": [
                                    float(x) for x in (joint_pos if joint_pos is not None else [])
                                ],
                                "joint_velocities": [
                                    float(x) for x in (joint_vel if joint_vel is not None else [])
                                ],
                                "num_dof": robot.num_dof if hasattr(robot, "num_dof") else 0,
                                "data_source": "simulation",
                            }
                            break
                        except Exception as e:
                            self.log(f"Failed to get robot state from {robot_path}: {e}", "DEBUG")

                    if not state["robot_state"]:
                        raise RuntimeError(
                            "Robot prim paths were configured/discovered but none were valid in the USD stage. "
                            "Update scenes/<scene_id>/config.json with the correct robot_prim_paths."
                        )

            except Exception as e:
                self.log(f"Failed to capture robot state: {e}", "WARNING")

        # Scene state metadata
        state["scene_state"] = {
            "num_objects": len(scene_objects) if scene_objects else 0,
            "num_contacts": len(state["contacts"]),
            "num_active_grasps": len(active_grasps),
            "physics_enabled": self._omni is not None,
        }

        return state

    def _summarize_camera_capture_warnings(
        self,
        episode_data: EpisodeSensorData,
    ) -> Tuple[List[str], Dict[str, List[int]]]:
        expected_cameras = [
            cam.camera_id for cam in self.config.cameras if cam.capture_rgb
        ]
        if not expected_cameras or episode_data.num_frames == 0:
            return [], {}

        missing_frames_by_camera = {camera_id: [] for camera_id in expected_cameras}
        for frame in episode_data.frames:
            present_cameras = set(frame.rgb_images.keys())
            for camera_id in expected_cameras:
                if camera_id not in present_cameras:
                    missing_frames_by_camera[camera_id].append(frame.frame_index)

        warnings = []
        for camera_id, missing_frames in missing_frames_by_camera.items():
            if not missing_frames:
                continue
            preview = missing_frames[:5]
            warnings.append(
                "Missing RGB frames for camera "
                f"'{camera_id}' in {len(missing_frames)}/{episode_data.num_frames} frames "
                f"(first missing frames: {preview})"
            )

        return warnings, missing_frames_by_camera

    def capture_episode(
        self,
        episode_id: str,
        trajectory_states: List[Any],
        scene_objects: Optional[List[Dict[str, Any]]] = None,
    ) -> EpisodeSensorData:
        """
        Capture sensor data for an entire episode.

        Args:
            episode_id: Unique episode identifier
            trajectory_states: List of trajectory states with timestamps
            scene_objects: List of scene objects

        Returns:
            EpisodeSensorData with all captured frames
        """
        episode_data = EpisodeSensorData(
            episode_id=episode_id,
            config=self.config,
        )
        self._camera_error_counts = {
            cam.camera_id: 0 for cam in self.config.cameras
        }

        self.log(f"Capturing sensor data for episode {episode_id}")

        for i, state in enumerate(trajectory_states):
            timestamp = getattr(state, "timestamp", i / self.config.fps)
            frame_data = self.capture_frame(i, timestamp, scene_objects)
            episode_data.frames.append(frame_data)

        self.log(f"Captured {len(episode_data.frames)} frames")

        warnings, missing_frames = self._summarize_camera_capture_warnings(episode_data)
        if warnings:
            episode_data.camera_capture_warnings.extend(warnings)
            for camera_id, frames in missing_frames.items():
                if frames:
                    self.log(
                        f"Episode {episode_id}: camera '{camera_id}' missing {len(frames)} frame(s)",
                        "WARNING",
                    )
        episode_data.camera_error_counts = dict(self._camera_error_counts)

        # Collect semantic label mapping
        if self.config.include_semantic_labels and scene_objects:
            for obj in scene_objects:
                obj_id = obj.get("id", obj.get("name", ""))
                category = obj.get("category", "object")
                semantic_id = hash(category) % 1000  # Simple hash for ID
                episode_data.semantic_labels[semantic_id] = category
                episode_data.object_metadata[obj_id] = {
                    "category": category,
                    "dimensions": obj.get("dimensions", [0.1, 0.1, 0.1]),
                    "semantic_id": semantic_id,
                }

        return episode_data

    def cleanup(self) -> None:
        """Clean up resources."""
        self._render_products.clear()
        self._annotators.clear()
        self.initialized = False


# =============================================================================
# Mock Sensor Capture (for testing without Isaac Sim)
# =============================================================================


class MockSensorCapture(IsaacSimSensorCapture):
    """
    Mock sensor capture for testing without Isaac Sim.

    Generates placeholder data with correct shapes and formats.

    WARNING: Mock data is NOT suitable for production training!
    - RGB images are random noise
    - Depth maps are synthetic random values
    - Segmentation masks are random labels
    - Physics are not validated

    For real training data, run with: /isaac-sim/python.sh
    """

    # Class-level warning tracker to avoid spam
    _warning_shown = False

    def __init__(self, config: SensorDataConfig, verbose: bool = True):
        # Direct instantiation is blocked whenever production is detected.
        if _mock_capture_disallowed():
            reason = f"MockSensorCapture initialization blocked by {_mock_capture_block_reason()}"
            _log_mock_capture_blocked(reason, None)
            raise RuntimeError(
                "MockSensorCapture is not allowed when DATA_QUALITY_LEVEL=production, "
                "LABS_STAGING is enabled, or ISAAC_SIM_REQUIRED=true. "
                "Use create_sensor_capture() to enforce production-safe behavior."
            )
        super().__init__(config=config, verbose=verbose)

    def is_mock(self) -> bool:
        """
        Check if this is a mock sensor capture instance.

        Returns:
            True for mock sensor capture (not suitable for production)
        """
        return True

    def initialize(self, scene_path: Optional[str] = None) -> bool:
        self.initialized = True
        self._using_mock = True

        # Show prominent warning on first use
        if not MockSensorCapture._warning_shown:
            MockSensorCapture._warning_shown = True
            self._show_mock_warning()

        self.log("Using mock sensor capture (no Isaac Sim)")
        return True

    def _show_mock_warning(self) -> None:
        """Show a prominent warning about mock data."""
        warning = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                        ⚠️  MOCK DATA WARNING  ⚠️                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  You are running WITHOUT Isaac Sim. Sensor data will be MOCK (random noise). ║
║                                                                              ║
║  Mock data is NOT suitable for:                                              ║
║    • Training robot policies                                                 ║
║    • Production datasets                                                     ║
║    • Selling episode data                                                    ║
║                                                                              ║
║  For real sensor data and physics validation, run with:                      ║
║    /isaac-sim/python.sh your_script.py                                       ║
║                                                                              ║
║  Or use the Isaac Sim Docker container:                                      ║
║    docker run --gpus all nvcr.io/nvidia/isaac-sim:4.2.0 python.sh script.py  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
        logger.warning("%s", warning)

    def capture_frame(
        self,
        frame_index: int,
        timestamp: float,
        scene_objects: Optional[List[Dict[str, Any]]] = None,
    ) -> FrameSensorData:
        """Generate mock frame data."""
        frame_data = FrameSensorData(
            frame_index=frame_index,
            timestamp=timestamp,
        )

        # Generate mock data for each configured camera
        for camera_config in self.config.cameras:
            camera_id = camera_config.camera_id
            h, w = camera_config.resolution[1], camera_config.resolution[0]

            # Mock RGB
            if camera_config.capture_rgb:
                frame_data.rgb_images[camera_id] = np.random.randint(
                    0, 255, (h, w, 3), dtype=np.uint8
                )

            # Mock depth
            if camera_config.capture_depth:
                frame_data.depth_maps[camera_id] = np.random.uniform(
                    0.1, 10.0, (h, w)
                ).astype(np.float32)

            # Mock segmentation
            if camera_config.capture_segmentation:
                frame_data.semantic_masks[camera_id] = np.random.randint(
                    0, 10, (h, w), dtype=np.uint8
                )

            if camera_config.capture_instance_segmentation:
                frame_data.instance_masks[camera_id] = np.random.randint(
                    0, 20, (h, w), dtype=np.uint16
                )

            # Mock bboxes
            if camera_config.capture_bbox_2d:
                frame_data.bboxes_2d[camera_id] = [
                    {
                        "bbox": [100.0, 100.0, 50.0, 50.0],
                        "category_id": 1,
                        "category_name": "object",
                        "area": 2500.0,
                    }
                ]

            if camera_config.capture_bbox_3d:
                frame_data.bboxes_3d[camera_id] = [
                    {
                        "center": [0.5, 0.0, 0.5],
                        "dimensions": [0.1, 0.1, 0.1],
                        "rotation_quat": [1.0, 0.0, 0.0, 0.0],
                        "category_id": 1,
                        "category_name": "object",
                    }
                ]

            # Mock normals
            if camera_config.capture_normals:
                frame_data.normals[camera_id] = np.random.uniform(
                    -1, 1, (h, w, 3)
                ).astype(np.float32)

        # Mock object poses
        if self.config.include_object_poses and scene_objects:
            for obj in scene_objects:
                obj_id = obj.get("id", obj.get("name", ""))
                frame_data.object_poses[obj_id] = {
                    "position": obj.get("position", [0.5, 0.0, 0.5]),
                    "rotation_quat": [1.0, 0.0, 0.0, 0.0],
                }

        # Mock contacts
        if self.config.include_contact_info:
            frame_data.contacts = [
                {
                    "body_a": "robot_gripper",
                    "body_b": "target_object",
                    "position": [0.5, 0.0, 0.5],
                    "normal": [0.0, 0.0, 1.0],
                    "force_magnitude": 5.0,
                }
            ]
            frame_data.contacts_available = True

        # Mock privileged state
        if self.config.include_privileged_state:
            frame_data.privileged_state = {
                "object_states": {},
                "robot_state": {"gripper_force": 10.0},
                "scene_state": {"gravity": [0, 0, -9.81]},
                "contacts_available": frame_data.contacts_available,
            }

        return frame_data


# =============================================================================
# Sensor Data Exporter
# =============================================================================


class SensorDataExporter:
    """
    Exports captured sensor data to various formats.

    Supports:
    - Individual frame images (PNG/NPY)
    - Video files (H.264/MP4) for LeRobot
    - Annotation JSON files
    - Metadata files
    """

    def __init__(self, output_dir: Path, verbose: bool = True):
        self.output_dir = Path(output_dir)
        self.verbose = verbose

    def log(self, msg: str, level: str = "INFO") -> None:
        if self.verbose:
            level_map = {
                "DEBUG": logger.debug,
                "INFO": logger.info,
                "WARNING": logger.warning,
                "ERROR": logger.error,
            }
            log_fn = level_map.get(level.upper(), logger.info)
            log_fn("[SENSOR-EXPORTER] [%s] %s", level, msg)

    def export_episode(
        self,
        episode_data: EpisodeSensorData,
        episode_index: int,
        chunk_index: int = 0,
    ) -> Dict[str, Path]:
        """
        Export episode sensor data to disk.

        Args:
            episode_data: Captured sensor data
            episode_index: Episode index in dataset
            chunk_index: Chunk index for LeRobot format

        Returns:
            Dict mapping data type to output path
        """
        output_paths = {}

        # Create output directories
        chunk_dir = self.output_dir / f"chunk-{chunk_index:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        config = episode_data.config

        # Export images per camera
        for camera_id in episode_data.camera_ids:
            camera_config = self._get_camera_config(config, camera_id)

            # RGB frames -> video
            if episode_data.has_rgb:
                video_path = self._export_rgb_video(
                    episode_data, camera_id, episode_index, chunk_dir
                )
                output_paths[f"video_{camera_id}"] = video_path

            # Depth frames
            if episode_data.has_depth:
                depth_path = self._export_depth(
                    episode_data, camera_id, episode_index, chunk_dir
                )
                output_paths[f"depth_{camera_id}"] = depth_path

            # Segmentation
            if episode_data.has_segmentation:
                seg_path = self._export_segmentation(
                    episode_data, camera_id, episode_index, chunk_dir
                )
                output_paths[f"segmentation_{camera_id}"] = seg_path

        # Export ground-truth annotations
        if config.include_object_poses or config.include_contact_info:
            gt_path = self._export_ground_truth(episode_data, episode_index, chunk_dir)
            output_paths["ground_truth"] = gt_path

        # Export metadata
        meta_path = self._export_metadata(episode_data, episode_index, chunk_dir)
        output_paths["metadata"] = meta_path

        self.log(f"Exported episode {episode_index}: {len(output_paths)} files")

        return output_paths

    def _get_camera_config(
        self, config: SensorDataConfig, camera_id: str
    ) -> Optional[CameraConfig]:
        """Get camera config by ID."""
        for cam in config.cameras:
            if cam.camera_id == camera_id:
                return cam
        return None

    def _export_rgb_video(
        self,
        episode_data: EpisodeSensorData,
        camera_id: str,
        episode_index: int,
        chunk_dir: Path,
    ) -> Path:
        """Export RGB frames as video."""
        # Create video directory structure
        video_dir = chunk_dir / f"observation.images.{camera_id}"
        video_dir.mkdir(parents=True, exist_ok=True)

        video_path = video_dir / f"episode_{episode_index:06d}.mp4"

        # Collect frames
        frames = []
        for frame in episode_data.frames:
            if camera_id in frame.rgb_images:
                frames.append(frame.rgb_images[camera_id])

        if not frames:
            return video_path

        # Write video using OpenCV or imageio
        try:
            import imageio

            fps = episode_data.config.fps
            writer = imageio.get_writer(str(video_path), fps=fps, codec="libx264")
            for frame in frames:
                # Ensure RGB order
                if frame.shape[-1] == 3:
                    writer.append_data(frame)
            writer.close()

        except ImportError:
            # Fallback: save individual frames
            frames_dir = video_dir / f"episode_{episode_index:06d}_frames"
            frames_dir.mkdir(exist_ok=True)
            for i, frame in enumerate(frames):
                frame_path = frames_dir / f"frame_{i:06d}.png"
                try:
                    from PIL import Image

                    Image.fromarray(frame).save(frame_path)
                except ImportError:
                    np.save(frame_path.with_suffix(".npy"), frame)

        return video_path

    def _export_depth(
        self,
        episode_data: EpisodeSensorData,
        camera_id: str,
        episode_index: int,
        chunk_dir: Path,
    ) -> Path:
        """Export depth maps."""
        depth_dir = chunk_dir / f"observation.depth.{camera_id}"
        depth_dir.mkdir(parents=True, exist_ok=True)

        depth_file = depth_dir / f"episode_{episode_index:06d}.npz"

        # Collect depth frames
        depth_frames = []
        for frame in episode_data.frames:
            if camera_id in frame.depth_maps:
                depth_frames.append(frame.depth_maps[camera_id])

        if depth_frames:
            np.savez_compressed(
                depth_file, depth=np.stack(depth_frames), fps=episode_data.config.fps
            )

        return depth_file

    def _export_segmentation(
        self,
        episode_data: EpisodeSensorData,
        camera_id: str,
        episode_index: int,
        chunk_dir: Path,
    ) -> Path:
        """Export segmentation masks."""
        seg_dir = chunk_dir / f"observation.segmentation.{camera_id}"
        seg_dir.mkdir(parents=True, exist_ok=True)

        seg_file = seg_dir / f"episode_{episode_index:06d}.npz"

        # Collect segmentation frames
        semantic_frames = []
        instance_frames = []

        for frame in episode_data.frames:
            if camera_id in frame.semantic_masks:
                semantic_frames.append(frame.semantic_masks[camera_id])
            if camera_id in frame.instance_masks:
                instance_frames.append(frame.instance_masks[camera_id])

        data = {}
        if semantic_frames:
            data["semantic"] = np.stack(semantic_frames)
        if instance_frames:
            data["instance"] = np.stack(instance_frames)
        if data:
            data["label_mapping"] = json.dumps(episode_data.semantic_labels)
            np.savez_compressed(seg_file, **data)

        return seg_file

    def _export_ground_truth(
        self,
        episode_data: EpisodeSensorData,
        episode_index: int,
        chunk_dir: Path,
    ) -> Path:
        """Export ground-truth annotations."""
        gt_dir = chunk_dir / "ground_truth"
        gt_dir.mkdir(parents=True, exist_ok=True)

        gt_file = gt_dir / f"episode_{episode_index:06d}.json"

        # Collect per-frame ground truth
        frames_gt = []
        for frame in episode_data.frames:
            frame_gt = {
                "frame_index": frame.frame_index,
                "timestamp": frame.timestamp,
            }

            if frame.object_poses:
                frame_gt["object_poses"] = frame.object_poses

            if frame.contacts:
                frame_gt["contacts"] = frame.contacts
            if frame.contacts_available is not None:
                frame_gt["contacts_available"] = frame.contacts_available

            if frame.privileged_state:
                frame_gt["privileged_state"] = frame.privileged_state

            if frame.bboxes_2d:
                frame_gt["bboxes_2d"] = frame.bboxes_2d

            if frame.bboxes_3d:
                frame_gt["bboxes_3d"] = frame.bboxes_3d

            frames_gt.append(frame_gt)

        gt_data = {
            "episode_id": episode_data.episode_id,
            "num_frames": len(frames_gt),
            "object_metadata": episode_data.object_metadata,
            "semantic_labels": {str(k): v for k, v in episode_data.semantic_labels.items()},
            "frames": frames_gt,
        }

        with open(gt_file, "w") as f:
            json.dump(gt_data, f, indent=2, default=self._json_serializer)

        return gt_file

    def _export_metadata(
        self,
        episode_data: EpisodeSensorData,
        episode_index: int,
        chunk_dir: Path,
    ) -> Path:
        """Export episode metadata."""
        meta_file = chunk_dir / f"episode_{episode_index:06d}_sensor_meta.json"

        config = episode_data.config

        meta = {
            "episode_id": episode_data.episode_id,
            "episode_index": episode_index,
            "num_frames": episode_data.num_frames,
            "data_pack": config.data_pack.value,
            "cameras": [
                {
                    "camera_id": cam.camera_id,
                    "camera_type": cam.camera_type,
                    "resolution": list(cam.resolution),
                    "has_rgb": cam.capture_rgb,
                    "has_depth": cam.capture_depth,
                    "has_segmentation": cam.capture_segmentation,
                    "has_bbox_2d": cam.capture_bbox_2d,
                    "has_bbox_3d": cam.capture_bbox_3d,
                }
                for cam in config.cameras
            ],
            "ground_truth": {
                "has_object_poses": config.include_object_poses,
                "has_contacts": config.include_contact_info,
                "has_privileged_state": config.include_privileged_state,
            },
            "fps": config.fps,
        }

        with open(meta_file, "w") as f:
            json.dump(meta, f, indent=2)

        return meta_file

    def _json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer for numpy types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# =============================================================================
# Factory Functions
# =============================================================================


def get_capture_mode_from_env() -> SensorDataCaptureMode:
    """
    Get sensor capture mode from environment variable.

    Returns fail-closed by default for production safety.

    Environment variable: SENSOR_CAPTURE_MODE
    Values: isaac_sim | mock_dev | fail_closed (default)
    """
    mode_str = os.getenv("SENSOR_CAPTURE_MODE", "fail_closed").lower()

    try:
        return SensorDataCaptureMode(mode_str)
    except ValueError:
        logger.warning(
            "[WARNING] Invalid SENSOR_CAPTURE_MODE='%s', defaulting to fail_closed",
            mode_str,
        )
        return SensorDataCaptureMode.FAIL_CLOSED


def create_sensor_capture(
    data_pack: Union[str, DataPackTier] = DataPackTier.CORE,
    num_cameras: int = 1,
    resolution: Tuple[int, int] = (640, 480),
    fps: float = 30.0,
    camera_specs: Optional[List[Dict[str, str]]] = None,
    robot_prim_paths: Optional[List[str]] = None,
    scene_usd_path: Optional[str] = None,
    use_mock: bool = False,
    require_real: bool = False,
    capture_mode: Optional[SensorDataCaptureMode] = None,
    allow_mock_capture: bool = False,
    verbose: bool = True,
) -> IsaacSimSensorCapture:
    """
    Create a sensor capture instance with explicit mode control.

    This factory function creates the appropriate sensor capture based on
    the specified capture mode:
    - ISAAC_SIM: Requires real Isaac Sim (fails if unavailable)
    - MOCK_DEV: Allows mock capture with prominent warnings
    - FAIL_CLOSED: Default - fails if Isaac Sim unavailable (production safe)

    Args:
        data_pack: Data pack tier ("core", "plus", "full") or DataPackTier enum
        num_cameras: Number of cameras to configure
        resolution: Image resolution (width, height)
        fps: Frames per second
        camera_specs: Optional camera specifications from scene config
        robot_prim_paths: Optional list of robot prim paths to track
        scene_usd_path: Optional USD scene path for auto-discovery
        use_mock: [DEPRECATED] Force mock capture (use capture_mode=MOCK_DEV instead)
        require_real: [DEPRECATED] Require real (use capture_mode=ISAAC_SIM instead)
        capture_mode: Explicit capture mode (recommended over use_mock/require_real)
        allow_mock_capture: Allow mock capture when capture_mode=MOCK_DEV
        verbose: Print progress

    Returns:
        Configured sensor capture instance (real or mock)

    Raises:
        RuntimeError: If Isaac Sim required but not available
    """
    if isinstance(data_pack, str):
        data_pack = DataPackTier(data_pack.lower())

    config = SensorDataConfig.from_data_pack(
        tier=data_pack,
        num_cameras=num_cameras,
        resolution=resolution,
        fps=fps,
        camera_specs=camera_specs,
        scene_usd_path=scene_usd_path,
    )
    config.robot_prim_paths = robot_prim_paths
    config.scene_usd_path = scene_usd_path
    max_camera_failures_env = os.getenv("MAX_CAMERA_FAILURES")
    if max_camera_failures_env:
        try:
            config.max_camera_failures = int(max_camera_failures_env)
        except ValueError:
            logger.warning(
                "Invalid MAX_CAMERA_FAILURES value %r; expected integer. Using default.",
                max_camera_failures_env,
            )
    elif parse_bool_env(os.getenv("REQUIRE_ALL_CAMERAS"), default=False):
        config.max_camera_failures = 0

    # Determine capture mode (prioritize new explicit mode)
    if capture_mode is None:
        # Legacy parameter support
        if use_mock:
            capture_mode = SensorDataCaptureMode.MOCK_DEV
        elif require_real:
            capture_mode = SensorDataCaptureMode.ISAAC_SIM
        else:
            # Default: fail-closed (production safe)
            capture_mode = get_capture_mode_from_env()

    allow_mock_env = (
        parse_bool_env(os.getenv("ALLOW_MOCK_DATA"), default=False)
        or parse_bool_env(os.getenv("ALLOW_MOCK_CAPTURE"), default=False)
    )

    if _is_production_run():
        mock_request_reasons = []
        if allow_mock_env:
            mock_request_reasons.append("ALLOW_MOCK_DATA/ALLOW_MOCK_CAPTURE set")
        if use_mock:
            mock_request_reasons.append("use_mock=True")
        if allow_mock_capture:
            mock_request_reasons.append("allow_mock_capture=True")
        if capture_mode == SensorDataCaptureMode.MOCK_DEV:
            mock_request_reasons.append("capture_mode=MOCK_DEV")

        if mock_request_reasons:
            _log_mock_capture_blocked(
                "; ".join(mock_request_reasons),
                capture_mode,
                details={
                    "use_mock_param": use_mock,
                    "allow_mock_capture_param": allow_mock_capture,
                },
            )
            raise RuntimeError(
                "Mock sensor capture is not permitted in production or labs-staging mode. "
                "Remove mock flags (ALLOW_MOCK_DATA/ALLOW_MOCK_CAPTURE, use_mock, "
                "capture_mode=MOCK_DEV) and run with Isaac Sim available "
                "(capture_mode=ISAAC_SIM or FAIL_CLOSED)."
            )
    elif capture_mode == SensorDataCaptureMode.MOCK_DEV and not (allow_mock_capture and allow_mock_env):
        raise RuntimeError(
            "Mock sensor capture requested without explicit allow. "
            "Set ALLOW_MOCK_CAPTURE=true and allow_mock_capture=True in a non-production run."
        )

    # Check if Isaac Sim is available
    if capture_mode in (SensorDataCaptureMode.ISAAC_SIM, SensorDataCaptureMode.FAIL_CLOSED):
        if not _HAVE_INTEGRATION_MODULE:
            raise RuntimeError(
                "Isaac Sim integration module is not available in this environment. "
                "Run inside the Isaac Sim Python environment "
                "(/isaac-sim/python.sh) or use the Isaac Sim container."
            )

    isaac_available = is_isaac_sim_available()
    replicator_available = is_replicator_available()

    # Handle each mode
    if capture_mode == SensorDataCaptureMode.ISAAC_SIM:
        # Production mode - MUST have Isaac Sim
        if not isaac_available:
            raise RuntimeError(
                "╔══════════════════════════════════════════════════════════════════════════════╗\n"
                "║                   ❌  ISAAC SIM REQUIRED (Production Mode)                   ║\n"
                "╠══════════════════════════════════════════════════════════════════════════════╣\n"
                "║  Capture mode: ISAAC_SIM (production quality)                                ║\n"
                "║  Status: Isaac Sim NOT available                                             ║\n"
                "║                                                                              ║\n"
                "║  This mode requires Isaac Sim for real physics and sensor data.              ║\n"
                "║  Mock data is NOT suitable for training robots.                              ║\n"
                "║                                                                              ║\n"
                "║  To run with Isaac Sim:                                                      ║\n"
                "║    /isaac-sim/python.sh your_script.py                                       ║\n"
                "║                                                                              ║\n"
                "║  To allow mock data (development only):                                      ║\n"
                "║    export SENSOR_CAPTURE_MODE=mock_dev                                       ║\n"
                "╚══════════════════════════════════════════════════════════════════════════════╝"
            )
        if not replicator_available:
            raise RuntimeError(
                "╔══════════════════════════════════════════════════════════════════════════════╗\n"
                "║               ❌  REPLICATOR REQUIRED (Production Mode)                      ║\n"
                "╠══════════════════════════════════════════════════════════════════════════════╣\n"
                "║  Capture mode: ISAAC_SIM (production quality)                                ║\n"
                "║  Status: omni.replicator.core NOT available                                  ║\n"
                "║                                                                              ║\n"
                "║  Ensure the Replicator extension is enabled in Isaac Sim.                    ║\n"
                "║  This is required for real sensor data capture.                              ║\n"
                "╚══════════════════════════════════════════════════════════════════════════════╝"
            )
        # Try to initialize real capture
        capture = IsaacSimSensorCapture(config, verbose=verbose)
        if not capture.initialize():
            raise RuntimeError(
                "Isaac Sim sensor capture initialization failed. "
                "Ensure Isaac Sim is running with Replicator extension."
            )
        if verbose:
            logger.info(
                "[SENSOR-CAPTURE] ✅ [PRODUCTION] Using IsaacSimSensorCapture (production quality)"
            )
        return capture

    elif capture_mode == SensorDataCaptureMode.MOCK_DEV:
        # Development mode - explicitly allow mock
        if not allow_mock_capture:
            raise RuntimeError(
                "Mock sensor capture requested, but allow_mock_capture is False.\n"
                "Set allow_mock_capture=True (development only) or use ISAAC_SIM for production."
            )
        if not isaac_available:
            logger.warning("%s", "=" * 80)
            logger.warning("⚠️  WARNING: MOCK DATA MODE (Development Only)")
            logger.warning("%s", "=" * 80)
            logger.warning("Isaac Sim is not available. Using mock sensor data.")
            logger.warning("Mock data includes:")
            logger.warning("  - Random noise RGB images (NOT real sensor data)")
            logger.warning("  - Placeholder depth maps")
            logger.warning("  - No real physics validation")
            logger.warning("This data is NOT suitable for:")
            logger.warning("  - Training production ML models")
            logger.warning("  - Selling as training data")
            logger.warning("  - Real-world robot deployment")
            logger.warning("To use real Isaac Sim:")
            logger.warning("  /isaac-sim/python.sh your_script.py")
            logger.warning("%s", "=" * 80)
        capture = MockSensorCapture(config, verbose=verbose)
        capture.initialize()
        if verbose:
            logger.warning(
                "[SENSOR-CAPTURE] ⚠️  [TEST] Using MockSensorCapture (development only)"
            )
        return capture

    elif capture_mode == SensorDataCaptureMode.FAIL_CLOSED:
        # Fail-closed mode - require Isaac Sim unless explicitly overridden
        if not isaac_available:
            raise RuntimeError(
                "╔══════════════════════════════════════════════════════════════════════════════╗\n"
                "║                   ❌  ISAAC SIM REQUIRED (Fail-Closed Mode)                  ║\n"
                "╠══════════════════════════════════════════════════════════════════════════════╣\n"
                "║  Capture mode: FAIL_CLOSED (default production safe)                         ║\n"
                "║  Status: Isaac Sim NOT available                                             ║\n"
                "║                                                                              ║\n"
                "║  This mode prevents accidental generation of unusable training data.         ║\n"
                "║  Isaac Sim is required for production-quality episodes.                      ║\n"
                "║                                                                              ║\n"
                "║  To run with Isaac Sim:                                                      ║\n"
                "║    /isaac-sim/python.sh your_script.py                                       ║\n"
                "║                                                                              ║\n"
                "║  For development/testing (allows mock data):                                 ║\n"
                "║    export SENSOR_CAPTURE_MODE=mock_dev                                       ║\n"
                "╚══════════════════════════════════════════════════════════════════════════════╝"
            )
        if not replicator_available:
            raise RuntimeError(
                "╔══════════════════════════════════════════════════════════════════════════════╗\n"
                "║               ❌  REPLICATOR REQUIRED (Fail-Closed Mode)                     ║\n"
                "╠══════════════════════════════════════════════════════════════════════════════╣\n"
                "║  Capture mode: FAIL_CLOSED (default production safe)                         ║\n"
                "║  Status: omni.replicator.core NOT available                                  ║\n"
                "║                                                                              ║\n"
                "║  Ensure the Replicator extension is enabled in Isaac Sim.                    ║\n"
                "║  This is required for real sensor data capture.                              ║\n"
                "╚══════════════════════════════════════════════════════════════════════════════╝"
            )
        # Isaac Sim available - use it
        capture = IsaacSimSensorCapture(config, verbose=verbose)
        if not capture.initialize():
            raise RuntimeError(
                "Isaac Sim sensor capture initialization failed. "
                "Ensure Isaac Sim is running with Replicator extension."
            )
        if verbose:
            logger.info(
                "[SENSOR-CAPTURE] ✅ [PRODUCTION] Using IsaacSimSensorCapture (fail-closed mode)"
            )
        return capture

    else:
        raise ValueError(f"Unknown capture mode: {capture_mode}")


def require_isaac_sim_or_fail() -> None:
    """
    Raise an error if Isaac Sim is not available.

    Use this at the start of production scripts to prevent running
    with mock data accidentally.
    """
    if not is_isaac_sim_available():
        error_msg = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                     ❌  ISAAC SIM REQUIRED  ❌                                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  This script requires Isaac Sim for real physics and sensor data.            ║
║                                                                              ║
║  You are currently running with standard Python, which will produce          ║
║  MOCK DATA (random noise images, heuristic validation).                      ║
║                                                                              ║
║  To run with real Isaac Sim:                                                 ║
║    /isaac-sim/python.sh your_script.py                                       ║
║                                                                              ║
║  Or use Docker:                                                              ║
║    docker run --gpus all \\                                                  ║
║      nvcr.io/nvidia/isaac-sim:4.2.0 \\                                       ║
║      python.sh /app/your_script.py                                           ║
║                                                                              ║
║  If you want to test with mock data anyway, set:                             ║
║    ALLOW_MOCK_DATA=true                                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
        # Check if user explicitly allows mock data
        import os
        if os.environ.get("ALLOW_MOCK_DATA", "").lower() in ("true", "1", "yes"):
            logger.warning(
                "%s",
                error_msg.replace(
                    "❌  ISAAC SIM REQUIRED  ❌", "⚠️  MOCK DATA MODE  ⚠️"
                ),
            )
            logger.warning("[WARNING] ALLOW_MOCK_DATA=true - Proceeding with mock data")
            return

        raise RuntimeError(error_msg)


def check_sensor_capture_environment() -> Dict[str, Any]:
    """
    Check the sensor capture environment and return a status report.

    Returns:
        Dict with:
        - isaac_sim_available: bool
        - replicator_available: bool
        - recommended_capture: str ("real" or "mock")
        - warnings: List[str]
    """
    status = {
        "isaac_sim_available": is_isaac_sim_available(),
        "replicator_available": is_replicator_available(),
        "recommended_capture": "mock",
        "warnings": [],
    }

    if status["isaac_sim_available"] and status["replicator_available"]:
        status["recommended_capture"] = "real"
    else:
        status["warnings"].append(
            "Running outside Isaac Sim - sensor data will be placeholder (mock) data. "
            "For production training data, run with /isaac-sim/python.sh"
        )

    return status


# =============================================================================
# Main / Testing
# =============================================================================


if __name__ == "__main__":
    from tools.logging_config import init_logging

    init_logging()
    logger.info("Testing Sensor Data Capture")
    logger.info("%s", "=" * 60)

    # Test each data pack tier
    for tier in [DataPackTier.CORE, DataPackTier.PLUS, DataPackTier.FULL]:
        logger.info("--- Data Pack: %s ---", tier.value)

        capture = create_sensor_capture(
            data_pack=tier,
            num_cameras=2,
            resolution=(640, 480),
            fps=30.0,
            use_mock=True,
            allow_mock_capture=True,
            verbose=True,
        )

        # Initialize
        capture.initialize()

        # Simulate episode capture
        mock_objects = [
            {"id": "cup_01", "position": [0.5, 0.0, 0.85], "category": "cup"},
            {"id": "plate_01", "position": [0.3, 0.2, 0.82], "category": "plate"},
        ]

        # Capture a few frames
        episode_data = capture.capture_episode(
            episode_id=f"test_episode_{tier.value}",
            trajectory_states=[None] * 10,  # Mock 10 frames
            scene_objects=mock_objects,
        )

        logger.info("  Captured %s frames", episode_data.num_frames)
        logger.info("  Cameras: %s", episode_data.camera_ids)
        logger.info("  Has RGB: %s", episode_data.has_rgb)
        logger.info("  Has depth: %s", episode_data.has_depth)
        logger.info("  Has segmentation: %s", episode_data.has_segmentation)

        # Test export
        exporter = SensorDataExporter(Path("/tmp/sensor_test"), verbose=True)
        output_paths = exporter.export_episode(episode_data, episode_index=0)
        logger.info("  Exported files: %s", list(output_paths.keys()))

        capture.cleanup()

    logger.info("%s", "=" * 60)
    logger.info("Sensor data capture test complete!")
