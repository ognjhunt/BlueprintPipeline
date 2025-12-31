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

Compatible with:
- LeRobot v2.0 format (images as separate video files)
- RLDS format (observation dict with image keys)
- HuggingFace datasets (with image features)
"""

import json
import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Add parent to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# =============================================================================
# Data Pack Configuration
# =============================================================================


class DataPackTier(Enum):
    """Data pack tiers for episode datasets."""

    CORE = "core"  # RGB + state + actions + metadata
    PLUS = "plus"  # Core + depth + segmentation + bboxes
    FULL = "full"  # Plus + poses + contacts + privileged state


@dataclass
class CameraConfig:
    """Configuration for a single camera."""

    camera_id: str
    prim_path: str  # USD prim path (e.g., "/World/Cameras/wrist_camera")
    resolution: Tuple[int, int] = (640, 480)
    focal_length: float = 24.0  # mm
    sensor_width: float = 36.0  # mm
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

    def get_lerobot_key(self) -> str:
        """Get the LeRobot observation key for this camera."""
        return f"observation.images.{self.camera_type}"


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

    @classmethod
    def from_data_pack(
        cls,
        tier: DataPackTier,
        num_cameras: int = 1,
        resolution: Tuple[int, int] = (640, 480),
        fps: float = 30.0,
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
                num_cameras, resolution, capture_depth=False
            )

        elif tier == DataPackTier.PLUS:
            # Plus: RGB + depth + segmentation + bboxes
            config.cameras = cls._create_default_cameras(
                num_cameras,
                resolution,
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
        capture_depth: bool = False,
        capture_segmentation: bool = False,
        capture_bbox_2d: bool = False,
        capture_bbox_3d: bool = False,
        capture_normals: bool = False,
    ) -> List[CameraConfig]:
        """Create default camera configurations."""
        cameras = []

        # Standard camera configurations
        camera_specs = [
            ("wrist", "/World/Robot/wrist_camera", "wrist"),
            ("overhead", "/World/Cameras/overhead_camera", "overhead"),
            ("side", "/World/Cameras/side_camera", "side"),
            ("front", "/World/Cameras/front_camera", "front"),
        ]

        for i in range(min(num_cameras, len(camera_specs))):
            camera_id, prim_path, camera_type = camera_specs[i]
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
class FrameSensorData:
    """Sensor data captured for a single frame."""

    frame_index: int
    timestamp: float

    # RGB images per camera {camera_id: np.ndarray (H, W, 3)}
    rgb_images: Dict[str, np.ndarray] = field(default_factory=dict)

    # Depth maps per camera {camera_id: np.ndarray (H, W)}
    depth_maps: Dict[str, np.ndarray] = field(default_factory=dict)

    # Segmentation masks per camera {camera_id: np.ndarray (H, W)}
    semantic_masks: Dict[str, np.ndarray] = field(default_factory=dict)
    instance_masks: Dict[str, np.ndarray] = field(default_factory=dict)

    # 2D bounding boxes per camera {camera_id: List[BBox2D]}
    bboxes_2d: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    # 3D bounding boxes (camera-space) {camera_id: List[BBox3D]}
    bboxes_3d: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    # Surface normals per camera {camera_id: np.ndarray (H, W, 3)}
    normals: Dict[str, np.ndarray] = field(default_factory=dict)

    # Ground-truth object poses (world-space) {object_id: pose}
    object_poses: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Contact information
    contacts: List[Dict[str, Any]] = field(default_factory=list)

    # Privileged state (full physics state)
    privileged_state: Optional[Dict[str, Any]] = None


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


# =============================================================================
# Isaac Sim Sensor Data Capture
# =============================================================================


class IsaacSimSensorCapture:
    """
    Captures sensor data during Isaac Sim trajectory execution.

    Integrates with omni.replicator for efficient synthetic data generation.
    Supports all annotation types in Isaac Sim Replicator.
    """

    def __init__(
        self,
        config: SensorDataConfig,
        verbose: bool = True,
    ):
        self.config = config
        self.verbose = verbose
        self.initialized = False

        # Replicator handles (set during initialization)
        self._render_products: Dict[str, Any] = {}
        self._annotators: Dict[str, Dict[str, Any]] = {}
        self._writer = None

        # Try to import Isaac Sim modules
        self._rep = None
        self._omni = None

    def log(self, msg: str, level: str = "INFO") -> None:
        if self.verbose:
            print(f"[SENSOR-CAPTURE] [{level}] {msg}")

    def initialize(self, scene_path: Optional[str] = None) -> bool:
        """
        Initialize sensor capture in Isaac Sim.

        Args:
            scene_path: Optional USD scene path to load

        Returns:
            True if initialization successful
        """
        try:
            import omni.replicator.core as rep

            self._rep = rep
            self.log("Isaac Sim Replicator initialized")
        except ImportError:
            self.log("Isaac Sim Replicator not available - using mock capture", "WARNING")
            self.initialized = True
            return True

        try:
            # Set up cameras and annotators
            for camera_config in self.config.cameras:
                self._setup_camera(camera_config)

            self.initialized = True
            self.log(f"Initialized {len(self.config.cameras)} cameras")
            return True

        except Exception as e:
            self.log(f"Initialization failed: {e}", "ERROR")
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
                self.log(f"Frame capture error for {camera_id}: {e}", "WARNING")

        # Object poses (if enabled)
        if self.config.include_object_poses and scene_objects:
            frame_data.object_poses = self._capture_object_poses(scene_objects)

        # Contact information (if enabled)
        if self.config.include_contact_info:
            frame_data.contacts = self._capture_contacts()

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
        """Capture object poses from simulation."""
        poses = {}

        try:
            # Try to get poses from Isaac Sim
            if self._omni is not None:
                import omni.isaac.core.utils.stage as stage_utils

                for obj in scene_objects:
                    obj_id = obj.get("id", obj.get("name", ""))
                    prim_path = obj.get("prim_path", f"/World/Objects/{obj_id}")

                    try:
                        prim = stage_utils.get_prim_at_path(prim_path)
                        if prim.IsValid():
                            xform = prim.GetAttribute("xformOp:transform").Get()
                            if xform:
                                poses[obj_id] = {
                                    "position": list(xform.GetTranslation()),
                                    "rotation_quat": list(xform.GetRotation().GetQuat()),
                                    "prim_path": prim_path,
                                }
                    except Exception:
                        pass

        except Exception:
            # Fallback: use positions from scene_objects
            for obj in scene_objects:
                obj_id = obj.get("id", obj.get("name", ""))
                position = obj.get("position", [0, 0, 0])
                rotation = obj.get("rotation", [1, 0, 0, 0])

                poses[obj_id] = {
                    "position": [float(x) for x in position],
                    "rotation_quat": [float(x) for x in rotation],
                }

        return poses

    def _capture_contacts(self) -> List[Dict[str, Any]]:
        """Capture contact information from physics simulation."""
        contacts = []

        try:
            if self._omni is not None:
                import omni.physx as physx

                # Get contact report
                contact_data = physx.get_physx_interface().get_contact_report()

                for contact in contact_data:
                    contacts.append({
                        "body_a": contact.get("actor0", ""),
                        "body_b": contact.get("actor1", ""),
                        "position": list(contact.get("position", [0, 0, 0])),
                        "normal": list(contact.get("normal", [0, 0, 1])),
                        "force_magnitude": float(contact.get("impulse", 0)),
                    })

        except Exception:
            pass

        return contacts

    def _capture_privileged_state(
        self, scene_objects: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Capture full privileged physics state."""
        state = {
            "object_states": {},
            "robot_state": {},
            "scene_state": {},
        }

        # Object states with velocities
        if scene_objects:
            for obj in scene_objects:
                obj_id = obj.get("id", obj.get("name", ""))
                state["object_states"][obj_id] = {
                    "position": obj.get("position", [0, 0, 0]),
                    "rotation": obj.get("rotation", [1, 0, 0, 0]),
                    "linear_velocity": obj.get("linear_velocity", [0, 0, 0]),
                    "angular_velocity": obj.get("angular_velocity", [0, 0, 0]),
                    "is_grasped": obj.get("is_grasped", False),
                    "in_contact": obj.get("in_contact", False),
                }

        return state

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

        self.log(f"Capturing sensor data for episode {episode_id}")

        for i, state in enumerate(trajectory_states):
            timestamp = getattr(state, "timestamp", i / self.config.fps)
            frame_data = self.capture_frame(i, timestamp, scene_objects)
            episode_data.frames.append(frame_data)

        self.log(f"Captured {len(episode_data.frames)} frames")

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
    """

    def initialize(self, scene_path: Optional[str] = None) -> bool:
        self.initialized = True
        self.log("Using mock sensor capture (no Isaac Sim)")
        return True

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

        # Mock privileged state
        if self.config.include_privileged_state:
            frame_data.privileged_state = {
                "object_states": {},
                "robot_state": {"gripper_force": 10.0},
                "scene_state": {"gravity": [0, 0, -9.81]},
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
            print(f"[SENSOR-EXPORTER] [{level}] {msg}")

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


def create_sensor_capture(
    data_pack: Union[str, DataPackTier] = DataPackTier.CORE,
    num_cameras: int = 1,
    resolution: Tuple[int, int] = (640, 480),
    fps: float = 30.0,
    use_mock: bool = False,
    verbose: bool = True,
) -> IsaacSimSensorCapture:
    """
    Create a sensor capture instance.

    Args:
        data_pack: Data pack tier ("core", "plus", "full") or DataPackTier enum
        num_cameras: Number of cameras to configure
        resolution: Image resolution (width, height)
        fps: Frames per second
        use_mock: Use mock capture instead of Isaac Sim
        verbose: Print progress

    Returns:
        Configured sensor capture instance
    """
    if isinstance(data_pack, str):
        data_pack = DataPackTier(data_pack.lower())

    config = SensorDataConfig.from_data_pack(
        tier=data_pack,
        num_cameras=num_cameras,
        resolution=resolution,
        fps=fps,
    )

    if use_mock:
        return MockSensorCapture(config, verbose=verbose)
    else:
        return IsaacSimSensorCapture(config, verbose=verbose)


# =============================================================================
# Main / Testing
# =============================================================================


if __name__ == "__main__":
    print("Testing Sensor Data Capture")
    print("=" * 60)

    # Test each data pack tier
    for tier in [DataPackTier.CORE, DataPackTier.PLUS, DataPackTier.FULL]:
        print(f"\n--- Data Pack: {tier.value} ---")

        capture = create_sensor_capture(
            data_pack=tier,
            num_cameras=2,
            resolution=(640, 480),
            fps=30.0,
            use_mock=True,
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

        print(f"  Captured {episode_data.num_frames} frames")
        print(f"  Cameras: {episode_data.camera_ids}")
        print(f"  Has RGB: {episode_data.has_rgb}")
        print(f"  Has depth: {episode_data.has_depth}")
        print(f"  Has segmentation: {episode_data.has_segmentation}")

        # Test export
        exporter = SensorDataExporter(Path("/tmp/sensor_test"), verbose=True)
        output_paths = exporter.export_episode(episode_data, episode_index=0)
        print(f"  Exported files: {list(output_paths.keys())}")

        capture.cleanup()

    print("\n" + "=" * 60)
    print("Sensor data capture test complete!")
