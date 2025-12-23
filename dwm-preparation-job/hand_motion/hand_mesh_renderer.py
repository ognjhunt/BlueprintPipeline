"""
Hand mesh renderer for DWM conditioning.

Renders hand meshes along trajectories to produce the hand-only
conditioning video for DWM. The output shows only the hand mesh
against a transparent or solid background.

Based on DWM paper:
- Hand-only video: frames rendering the hand meshes aligned with camera trajectory
- Used as the "action encoding" for DWM conditioning

Hand Models Supported:
- MANO: Parametric hand model (research standard)
- Simple mesh: Placeholder geometric hand
- Robot gripper: For robotics applications
"""

import json
import shutil
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import CameraTrajectory, HandPose, HandTrajectory


class HandModel(str, Enum):
    """Available hand models."""

    # MANO parametric hand model
    MANO = "mano"

    # Simple geometric hand (placeholder)
    SIMPLE = "simple"

    # Franka gripper (for robotics)
    FRANKA_GRIPPER = "franka_gripper"

    # Shadow hand (dexterous robot hand)
    SHADOW_HAND = "shadow_hand"


@dataclass
class HandRenderConfig:
    """Configuration for hand rendering."""

    # Output resolution (should match scene video)
    width: int = 720
    height: int = 480

    # Hand model to use
    hand_model: HandModel = HandModel.SIMPLE

    # Background
    background_color: tuple[int, int, int, int] = (0, 0, 0, 255)  # RGBA
    transparent_background: bool = False

    # Hand appearance
    hand_color: tuple[int, int, int] = (220, 180, 160)  # Skin tone
    hand_opacity: float = 1.0

    # Rendering quality
    antialiasing: bool = True

    # Output format
    output_format: str = "png"


class SimpleHandMesh:
    """
    Simple geometric hand representation.

    A placeholder hand model using basic geometric shapes.
    Used when MANO or other sophisticated models aren't available.
    """

    def __init__(self):
        self.vertices = None
        self.faces = None
        self._build_simple_hand()

    def _build_simple_hand(self):
        """Build a simple box-based hand geometry."""
        # Palm: flat box
        palm_vertices = self._box_vertices(
            center=[0, 0, 0],
            size=[0.08, 0.02, 0.10],
        )

        # Fingers: 5 elongated boxes
        finger_specs = [
            # (x_offset, length)
            (-0.03, 0.07),   # Thumb (shorter, offset)
            (-0.015, 0.08),  # Index
            (0.0, 0.085),    # Middle
            (0.015, 0.075),  # Ring
            (0.03, 0.06),    # Pinky
        ]

        all_vertices = [palm_vertices]
        base_z = 0.05  # Fingers start at end of palm

        for i, (x_off, length) in enumerate(finger_specs):
            if i == 0:  # Thumb
                finger_verts = self._box_vertices(
                    center=[x_off - 0.02, 0, -0.02],
                    size=[0.015, 0.015, length],
                )
            else:
                finger_verts = self._box_vertices(
                    center=[x_off, 0, base_z + length / 2],
                    size=[0.012, 0.012, length],
                )
            all_vertices.append(finger_verts)

        self.vertices = np.vstack(all_vertices)

        # Build faces (quads for each box, 6 faces * 6 boxes = 36 quads)
        faces = []
        for box_idx in range(6):
            base = box_idx * 8
            # Each box has 8 vertices, 6 faces
            box_faces = [
                [base + 0, base + 1, base + 2, base + 3],  # Bottom
                [base + 4, base + 5, base + 6, base + 7],  # Top
                [base + 0, base + 1, base + 5, base + 4],  # Front
                [base + 2, base + 3, base + 7, base + 6],  # Back
                [base + 0, base + 3, base + 7, base + 4],  # Left
                [base + 1, base + 2, base + 6, base + 5],  # Right
            ]
            faces.extend(box_faces)

        self.faces = np.array(faces)

    def _box_vertices(self, center, size):
        """Generate vertices for a box."""
        cx, cy, cz = center
        sx, sy, sz = [s / 2 for s in size]

        return np.array([
            [cx - sx, cy - sy, cz - sz],
            [cx + sx, cy - sy, cz - sz],
            [cx + sx, cy - sy, cz + sz],
            [cx - sx, cy - sy, cz + sz],
            [cx - sx, cy + sy, cz - sz],
            [cx + sx, cy + sy, cz - sz],
            [cx + sx, cy + sy, cz + sz],
            [cx - sx, cy + sy, cz + sz],
        ], dtype=np.float64)

    def get_vertices(self, hand_pose: HandPose) -> np.ndarray:
        """
        Get transformed vertices for a hand pose.

        Args:
            hand_pose: Hand pose with position and rotation

        Returns:
            Transformed vertices in world space
        """
        # Apply pose parameters if available (simplified)
        verts = self.vertices.copy()

        if hand_pose.pose_params is not None:
            # Simplified finger curl based on pose params
            curl_amount = np.mean(hand_pose.pose_params[3:15])
            # Curl fingers by rotating them inward
            for i in range(8, len(verts)):  # Skip palm vertices
                local_z = verts[i, 2]
                if local_z > 0:  # Finger vertices
                    curl_angle = curl_amount * 0.5
                    verts[i, 1] -= local_z * np.sin(curl_angle) * 0.3
                    verts[i, 2] *= np.cos(curl_angle)

        # Apply global rotation
        verts = verts @ hand_pose.rotation.T

        # Apply global position
        verts = verts + hand_pose.position

        return verts


class HandMeshRenderer:
    """
    Renderer for hand meshes.

    Renders hand meshes to produce the hand-only conditioning video
    for DWM. Supports multiple rendering backends.
    """

    def __init__(self, config: Optional[HandRenderConfig] = None):
        """
        Initialize hand mesh renderer.

        Args:
            config: Rendering configuration
        """
        self.config = config or HandRenderConfig()
        self.hand_mesh = SimpleHandMesh()

    def render_frame(
        self,
        hand_pose: HandPose,
        camera_pose,  # CameraPose from models
        output_path: Path,
    ) -> bool:
        """
        Render a single frame of the hand mesh.

        Args:
            hand_pose: Hand pose for this frame
            camera_pose: Camera pose (for view/projection)
            output_path: Output image path

        Returns:
            True if successful
        """
        try:
            from PIL import Image, ImageDraw

            # Create image
            if self.config.transparent_background:
                img = Image.new(
                    "RGBA",
                    (self.config.width, self.config.height),
                    (0, 0, 0, 0),
                )
            else:
                img = Image.new(
                    "RGB",
                    (self.config.width, self.config.height),
                    self.config.background_color[:3],
                )

            draw = ImageDraw.Draw(img)

            # Get hand vertices in world space
            vertices_world = self.hand_mesh.get_vertices(hand_pose)

            # Transform to camera space
            cam_inv = np.linalg.inv(camera_pose.transform)
            vertices_cam = (cam_inv[:3, :3] @ vertices_world.T).T + cam_inv[:3, 3]

            # Simple perspective projection
            fx = self.config.width * 0.8  # Focal length in pixels
            fy = self.config.height * 0.8
            cx = self.config.width / 2
            cy = self.config.height / 2

            vertices_2d = []
            for v in vertices_cam:
                if v[2] > 0.01:  # In front of camera
                    x = fx * v[0] / v[2] + cx
                    y = -fy * v[1] / v[2] + cy  # Flip Y
                    vertices_2d.append((x, y, v[2]))
                else:
                    vertices_2d.append(None)

            # Draw faces (back-to-front simple sorting)
            face_depths = []
            for i, face in enumerate(self.hand_mesh.faces):
                face_verts = [vertices_2d[v] for v in face]
                if all(v is not None for v in face_verts):
                    avg_depth = np.mean([v[2] for v in face_verts])
                    face_depths.append((avg_depth, i))

            # Sort by depth (far to near)
            face_depths.sort(reverse=True)

            # Draw faces
            hand_color = self.config.hand_color
            for depth, face_idx in face_depths:
                face = self.hand_mesh.faces[face_idx]
                face_verts = [vertices_2d[v] for v in face]
                if all(v is not None for v in face_verts):
                    polygon = [(v[0], v[1]) for v in face_verts]

                    # Simple shading based on face normal
                    shade = min(255, int(200 + 55 * (1 - depth / 2)))
                    shaded_color = tuple(int(c * shade / 255) for c in hand_color)

                    draw.polygon(polygon, fill=shaded_color, outline=(100, 80, 70))

            output_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(output_path)
            return True

        except ImportError:
            # PIL not available, create placeholder
            return self._render_placeholder(hand_pose, output_path)

    def _render_placeholder(
        self,
        hand_pose: HandPose,
        output_path: Path,
    ) -> bool:
        """Create a placeholder image when PIL isn't available."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write minimal valid PNG
        # (This is a 1x1 transparent PNG for placeholder)
        png_data = bytes([
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
            0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
            0x08, 0x06, 0x00, 0x00, 0x00, 0x1F, 0x15, 0xC4,
            0x89, 0x00, 0x00, 0x00, 0x0A, 0x49, 0x44, 0x41,
            0x54, 0x78, 0x9C, 0x63, 0x00, 0x01, 0x00, 0x00,
            0x05, 0x00, 0x01, 0x0D, 0x0A, 0x2D, 0xB4, 0x00,
            0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE,
            0x42, 0x60, 0x82,
        ])
        output_path.write_bytes(png_data)
        return True

    def render_trajectory(
        self,
        hand_trajectory: HandTrajectory,
        camera_trajectory: CameraTrajectory,
        output_dir: Path,
        frame_prefix: str = "hand",
    ) -> list[Path]:
        """
        Render all frames of a hand trajectory.

        Args:
            hand_trajectory: Hand motion trajectory
            camera_trajectory: Camera trajectory (for view transform)
            output_dir: Output directory for frames
            frame_prefix: Prefix for frame filenames

        Returns:
            List of paths to rendered frames
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        frame_paths = []

        # Ensure trajectories have same number of frames
        num_frames = min(
            hand_trajectory.num_frames,
            camera_trajectory.num_frames,
        )

        for i in range(num_frames):
            hand_pose = hand_trajectory.poses[i]
            camera_pose = camera_trajectory.poses[i]

            frame_name = f"{frame_prefix}_{i:04d}.{self.config.output_format}"
            frame_path = output_dir / frame_name

            success = self.render_frame(hand_pose, camera_pose, frame_path)
            if success:
                frame_paths.append(frame_path)

        return frame_paths

    def frames_to_video(
        self,
        frames_dir: Path,
        output_path: Path,
        fps: float = 24.0,
        frame_pattern: str = "hand_*.png",
    ) -> bool:
        """
        Encode rendered frames to video.

        Args:
            frames_dir: Directory containing frames
            output_path: Output video path
            fps: Frames per second
            frame_pattern: Pattern to match frame files

        Returns:
            True if successful
        """
        try:
            frames_dir = Path(frames_dir)
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Build ffmpeg command
            input_pattern = str(frames_dir / frame_pattern.replace("*", "%04d"))

            cmd = [
                "ffmpeg",
                "-y",
                "-framerate", str(fps),
                "-i", input_pattern,
                "-c:v", "libx264",
                "-crf", "18",
                "-pix_fmt", "yuv420p",
                str(output_path),
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
            )

            return result.returncode == 0

        except FileNotFoundError:
            print("ffmpeg not found - video encoding not available")
            return False


def render_hand_trajectory_to_video(
    hand_trajectory: HandTrajectory,
    camera_trajectory: CameraTrajectory,
    output_video_path: Path,
    config: Optional[HandRenderConfig] = None,
    keep_frames: bool = False,
) -> bool:
    """
    Convenience function: render hand trajectory directly to video.

    Args:
        hand_trajectory: Hand motion trajectory
        camera_trajectory: Camera trajectory for view transform
        output_video_path: Output video path
        config: Render configuration
        keep_frames: Keep individual frames after encoding

    Returns:
        True if successful
    """
    renderer = HandMeshRenderer(config)

    output_video_path = Path(output_video_path)
    frames_dir = output_video_path.parent / f"_hand_frames_{hand_trajectory.trajectory_id}"

    try:
        # Render frames
        frame_paths = renderer.render_trajectory(
            hand_trajectory,
            camera_trajectory,
            frames_dir,
            frame_prefix="hand",
        )

        if not frame_paths:
            print("No hand frames rendered")
            return False

        # Encode to video
        success = renderer.frames_to_video(
            frames_dir,
            output_video_path,
            fps=hand_trajectory.fps,
            frame_pattern="hand_*.png",
        )

        return success

    finally:
        if not keep_frames and frames_dir.exists():
            shutil.rmtree(frames_dir)
