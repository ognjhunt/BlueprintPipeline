"""
Writer configurations for Replicator synthetic data generation.

This module defines annotation writer configurations for different
data output types (RGB, depth, segmentation, bounding boxes, etc.).
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class WriterConfig:
    """Configuration for a single annotation writer."""

    name: str
    writer_type: str
    enabled: bool = True
    file_format: str = "png"
    annotator: Optional[str] = None
    parameters: dict[str, Any] = field(default_factory=dict)


def get_writers_for_policy(
    capture_config: dict[str, Any],
    output_dir: str,
) -> list[WriterConfig]:
    """
    Get writer configurations for a policy's capture requirements.

    Args:
        capture_config: Capture configuration from policy
        output_dir: Base output directory

    Returns:
        List of WriterConfig objects
    """
    writers = []

    annotations = capture_config.get("annotations", [])
    resolution = capture_config.get("resolution", [1280, 720])

    for annotation in annotations:
        writer = create_writer(annotation, resolution)
        if writer:
            writers.append(writer)

    return writers


def create_writer(
    annotation_type: str, resolution: tuple[int, int]
) -> Optional[WriterConfig]:
    """Create a writer configuration for an annotation type."""
    creators = {
        "rgb": _create_rgb_writer,
        "depth": _create_depth_writer,
        "semantic_segmentation": _create_semantic_writer,
        "instance_segmentation": _create_instance_writer,
        "bounding_box_2d": _create_bbox2d_writer,
        "bounding_box_3d": _create_bbox3d_writer,
        "object_pose": _create_pose_writer,
        "normals": _create_normals_writer,
        "joint_states": _create_joint_writer,
        "keypoints": _create_keypoints_writer,
        "cloth_keypoints": _create_cloth_keypoints_writer,
        "handle_keypoints": _create_handle_keypoints_writer,
        "barcode": _create_barcode_writer,
        "product_label": _create_product_label_writer,
    }

    creator = creators.get(annotation_type)
    if creator:
        return creator(resolution)

    return None


def _create_rgb_writer(resolution: tuple[int, int]) -> WriterConfig:
    """Create RGB image writer."""
    return WriterConfig(
        name="rgb",
        writer_type="rgb",
        enabled=True,
        file_format="png",
        annotator="rgb",
        parameters={
            "resolution": list(resolution),
            "colorspace": "sRGB",
            "antialiasing": True,
        },
    )


def _create_depth_writer(resolution: tuple[int, int]) -> WriterConfig:
    """Create depth map writer."""
    return WriterConfig(
        name="depth",
        writer_type="distance_to_camera",
        enabled=True,
        file_format="npy",
        annotator="distance_to_camera",
        parameters={
            "resolution": list(resolution),
            "near_clip": 0.1,
            "far_clip": 100.0,
            "output_linear": True,
        },
    )


def _create_semantic_writer(resolution: tuple[int, int]) -> WriterConfig:
    """Create semantic segmentation writer."""
    return WriterConfig(
        name="semantic_segmentation",
        writer_type="semantic_segmentation",
        enabled=True,
        file_format="png",
        annotator="semantic_segmentation",
        parameters={
            "resolution": list(resolution),
            "colorize": True,
            "include_unlabeled": False,
        },
    )


def _create_instance_writer(resolution: tuple[int, int]) -> WriterConfig:
    """Create instance segmentation writer."""
    return WriterConfig(
        name="instance_segmentation",
        writer_type="instance_segmentation",
        enabled=True,
        file_format="png",
        annotator="instance_segmentation",
        parameters={
            "resolution": list(resolution),
            "colorize": True,
            "include_background": False,
        },
    )


def _create_bbox2d_writer(resolution: tuple[int, int]) -> WriterConfig:
    """Create 2D bounding box writer."""
    return WriterConfig(
        name="bounding_box_2d",
        writer_type="bounding_box_2d_tight",
        enabled=True,
        file_format="json",
        annotator="bounding_box_2d_tight",
        parameters={
            "resolution": list(resolution),
            "include_occluded": True,
            "visibility_threshold": 0.1,
            "format": "coco",
        },
    )


def _create_bbox3d_writer(resolution: tuple[int, int]) -> WriterConfig:
    """Create 3D bounding box writer."""
    return WriterConfig(
        name="bounding_box_3d",
        writer_type="bounding_box_3d",
        enabled=True,
        file_format="json",
        annotator="bounding_box_3d",
        parameters={
            "coordinate_system": "camera",
            "include_orientation": True,
        },
    )


def _create_pose_writer(resolution: tuple[int, int]) -> WriterConfig:
    """Create object pose writer."""
    return WriterConfig(
        name="object_pose",
        writer_type="object_pose",
        enabled=True,
        file_format="json",
        annotator="object_pose",
        parameters={
            "coordinate_system": "world",
            "include_velocity": False,
            "format": "quaternion",
        },
    )


def _create_normals_writer(resolution: tuple[int, int]) -> WriterConfig:
    """Create surface normals writer."""
    return WriterConfig(
        name="normals",
        writer_type="normals",
        enabled=True,
        file_format="exr",
        annotator="normals",
        parameters={
            "resolution": list(resolution),
            "coordinate_system": "camera",
        },
    )


def _create_joint_writer(resolution: tuple[int, int]) -> WriterConfig:
    """Create joint states writer for articulated objects."""
    return WriterConfig(
        name="joint_states",
        writer_type="joint_states",
        enabled=True,
        file_format="json",
        annotator="articulation",
        parameters={
            "include_joint_names": True,
            "include_limits": True,
            "include_velocities": False,
        },
    )


def _create_keypoints_writer(resolution: tuple[int, int]) -> WriterConfig:
    """Create generic keypoints writer."""
    return WriterConfig(
        name="keypoints",
        writer_type="keypoints_2d",
        enabled=True,
        file_format="json",
        annotator="keypoints",
        parameters={
            "resolution": list(resolution),
            "visibility_check": True,
            "format": "coco",
        },
    )


def _create_cloth_keypoints_writer(resolution: tuple[int, int]) -> WriterConfig:
    """Create cloth-specific keypoints writer."""
    return WriterConfig(
        name="cloth_keypoints",
        writer_type="cloth_keypoints",
        enabled=True,
        file_format="json",
        annotator="cloth_keypoints",
        parameters={
            "resolution": list(resolution),
            "keypoint_types": ["corners", "edges", "center"],
            "include_fold_lines": True,
        },
    )


def _create_handle_keypoints_writer(resolution: tuple[int, int]) -> WriterConfig:
    """Create handle keypoints writer for doors/drawers."""
    return WriterConfig(
        name="handle_keypoints",
        writer_type="keypoints_2d",
        enabled=True,
        file_format="json",
        annotator="handle_keypoints",
        parameters={
            "resolution": list(resolution),
            "keypoint_names": ["handle_center", "grasp_point_1", "grasp_point_2"],
        },
    )


def _create_barcode_writer(resolution: tuple[int, int]) -> WriterConfig:
    """Create barcode annotation writer."""
    return WriterConfig(
        name="barcode",
        writer_type="barcode",
        enabled=True,
        file_format="json",
        annotator="barcode",
        parameters={
            "resolution": list(resolution),
            "barcode_types": ["qr", "ean13", "code128"],
            "include_decoded_value": True,
        },
    )


def _create_product_label_writer(resolution: tuple[int, int]) -> WriterConfig:
    """Create product label annotation writer."""
    return WriterConfig(
        name="product_label",
        writer_type="text_detection",
        enabled=True,
        file_format="json",
        annotator="text_detection",
        parameters={
            "resolution": list(resolution),
            "include_text_content": True,
            "language": "en",
        },
    )

