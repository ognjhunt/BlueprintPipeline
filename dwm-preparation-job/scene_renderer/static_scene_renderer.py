"""
Static scene renderer for DWM conditioning.

Renders the static 3D scene along camera trajectories to produce
conditioning videos for DWM. The scene is rendered "frozen" - no
dynamics, just the static geometry from the reconstruction.

Supports multiple rendering backends:
- Isaac Sim (production quality, GPU-accelerated)
- PyRender (CPU/OpenGL, good for development)
- Blender (optional, high quality offline)

Based on DWM paper:
- Static scene video: frames rendered from static 3D scene along camera trajectory
- Resolution: 720x480 (DWM default)
- Frames: 49 (DWM default)
"""

import json
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import CameraPose, CameraTrajectory


class RenderBackend(str, Enum):
    """Available rendering backends."""

    # Isaac Sim via Omniverse
    ISAAC_SIM = "isaac_sim"

    # PyRender (OpenGL, works on CPU)
    PYRENDER = "pyrender"

    # Trimesh built-in renderer
    TRIMESH = "trimesh"

    # Blender via command line
    BLENDER = "blender"

    # Placeholder for when rendering isn't available
    MOCK = "mock"


@dataclass
class RenderConfig:
    """Configuration for rendering."""

    # Output resolution
    width: int = 720
    height: int = 480

    # Rendering quality
    samples: int = 32
    antialiasing: bool = True

    # Camera settings
    near_clip: float = 0.01
    far_clip: float = 100.0

    # Background
    background_color: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Output format
    output_format: str = "png"  # "png", "exr", "jpg"

    # Video encoding (if encoding frames to video)
    video_codec: str = "libx264"
    video_fps: float = 24.0
    video_quality: int = 23  # CRF for x264 (lower = better)


class BaseRenderer(ABC):
    """Abstract base class for scene renderers."""

    def __init__(self, config: RenderConfig):
        self.config = config

    @abstractmethod
    def load_scene(self, scene_path: Path) -> bool:
        """Load a 3D scene for rendering."""
        pass

    @abstractmethod
    def render_frame(
        self,
        camera_pose: CameraPose,
        output_path: Path,
        depth_path: Optional[Path] = None,
        segmentation_path: Optional[Path] = None,
    ) -> bool:
        """Render a single frame from the given camera pose."""
        pass

    def render_trajectory(
        self,
        trajectory: CameraTrajectory,
        output_dir: Path,
        frame_prefix: str = "frame",
        depth_dir: Optional[Path] = None,
        segmentation_dir: Optional[Path] = None,
    ) -> dict[str, list[Path]]:
        """
        Render all frames in a trajectory.

        Args:
            trajectory: Camera trajectory
            output_dir: Directory for output frames
            frame_prefix: Prefix for frame filenames
            depth_dir: Optional directory for depth outputs
            segmentation_dir: Optional directory for segmentation outputs

        Returns:
            Dictionary containing paths for color, depth, and segmentation frames
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if depth_dir:
            depth_dir = Path(depth_dir)
            depth_dir.mkdir(parents=True, exist_ok=True)
        if segmentation_dir:
            segmentation_dir = Path(segmentation_dir)
            segmentation_dir.mkdir(parents=True, exist_ok=True)

        color_paths = []
        depth_paths = []
        seg_paths = []
        for pose in trajectory.poses:
            frame_name = f"{frame_prefix}_{pose.frame_idx:04d}.{self.config.output_format}"
            frame_path = output_dir / frame_name

            depth_path = None
            seg_path = None
            if depth_dir:
                depth_path = depth_dir / f"{frame_prefix}_{pose.frame_idx:04d}.npy"
            if segmentation_dir:
                seg_path = segmentation_dir / f"{frame_prefix}_{pose.frame_idx:04d}.npy"

            success = self.render_frame(
                pose,
                frame_path,
                depth_path=depth_path,
                segmentation_path=seg_path,
            )
            if success:
                color_paths.append(frame_path)
                if depth_path:
                    depth_paths.append(depth_path)
                if seg_path:
                    seg_paths.append(seg_path)
            else:
                print(f"Warning: Failed to render frame {pose.frame_idx}")

        return {
            "color": color_paths,
            "depth": depth_paths,
            "segmentation": seg_paths,
        }


class MockRenderer(BaseRenderer):
    """
    Mock renderer for testing/development.

    Generates placeholder images instead of actual renders.
    """

    def __init__(self, config: RenderConfig):
        super().__init__(config)
        self.scene_loaded = False

    def load_scene(self, scene_path: Path) -> bool:
        """Mock scene loading."""
        self.scene_loaded = True
        return True

    def render_frame(
        self,
        camera_pose: CameraPose,
        output_path: Path,
        depth_path: Optional[Path] = None,
        segmentation_path: Optional[Path] = None,
    ) -> bool:
        """Generate a mock frame (solid color with frame info)."""
        try:
            from PIL import Image, ImageDraw

            # Create a gradient image
            img = Image.new(
                "RGB",
                (self.config.width, self.config.height),
                color=(50, 50, 80),
            )

            draw = ImageDraw.Draw(img)

            # Add frame info text
            text = f"Frame {camera_pose.frame_idx}"
            pos = f"Pos: ({camera_pose.position[0]:.2f}, {camera_pose.position[1]:.2f}, {camera_pose.position[2]:.2f})"

            draw.text((10, 10), text, fill=(255, 255, 255))
            draw.text((10, 30), pos, fill=(200, 200, 200))
            draw.text((10, 50), "[MOCK RENDER]", fill=(255, 200, 100))

            # Add a simple grid pattern
            for i in range(0, self.config.width, 50):
                draw.line([(i, 0), (i, self.config.height)], fill=(60, 60, 90))
            for i in range(0, self.config.height, 50):
                draw.line([(0, i), (self.config.width, i)], fill=(60, 60, 90))

            output_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(output_path)

            # Save mock depth/segmentation if requested
            if depth_path:
                depth_path.parent.mkdir(parents=True, exist_ok=True)
                depth = np.linspace(0, 1, num=self.config.width * self.config.height).reshape(
                    (self.config.height, self.config.width)
                )
                np.save(depth_path, depth)

            if segmentation_path:
                segmentation_path.parent.mkdir(parents=True, exist_ok=True)
                seg_map = np.zeros((self.config.height, self.config.width), dtype=np.int32)
                np.save(segmentation_path, seg_map)

            return True

        except ImportError:
            # PIL not available, create empty file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"")
            if depth_path:
                depth_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(depth_path, np.zeros((self.config.height, self.config.width)))
            if segmentation_path:
                segmentation_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(segmentation_path, np.zeros((self.config.height, self.config.width)))
            return True


class PyRenderRenderer(BaseRenderer):
    """
    PyRender-based renderer (OpenGL).

    Works on CPU, good for development and testing.
    Requires: pip install pyrender trimesh
    """

    def __init__(self, config: RenderConfig):
        super().__init__(config)
        self.scene = None
        self.renderer = None

    def load_scene(self, scene_path: Path) -> bool:
        """Load scene from GLB/GLTF/USD file."""
        try:
            import pyrender
            import trimesh

            scene_path = Path(scene_path)

            # Handle different file formats
            if scene_path.suffix.lower() in [".glb", ".gltf"]:
                mesh = trimesh.load(str(scene_path))
            elif scene_path.suffix.lower() in [".usda", ".usd", ".usdc"]:
                # USD requires conversion or special handling
                print(f"USD format detected - using mock render (USD support coming)")
                return False
            else:
                # Try to load as mesh
                mesh = trimesh.load(str(scene_path))

            # Create pyrender scene
            self.scene = pyrender.Scene()

            # Add meshes
            if isinstance(mesh, trimesh.Scene):
                for name, geometry in mesh.geometry.items():
                    if isinstance(geometry, trimesh.Trimesh):
                        pr_mesh = pyrender.Mesh.from_trimesh(geometry)
                        self.scene.add(pr_mesh)
            elif isinstance(mesh, trimesh.Trimesh):
                pr_mesh = pyrender.Mesh.from_trimesh(mesh)
                self.scene.add(pr_mesh)

            # Add default lighting
            light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
            self.scene.add(light, pose=np.eye(4))

            # Create offscreen renderer
            self.renderer = pyrender.OffscreenRenderer(
                self.config.width,
                self.config.height,
            )

            return True

        except Exception as e:
            print(f"Failed to load scene with PyRender: {e}")
            return False

    def render_frame(
        self,
        camera_pose: CameraPose,
        output_path: Path,
        depth_path: Optional[Path] = None,
        segmentation_path: Optional[Path] = None,
    ) -> bool:
        """Render a frame using PyRender."""
        if self.scene is None or self.renderer is None:
            return False

        try:
            import pyrender
            from PIL import Image

            # Create camera
            camera = pyrender.PerspectiveCamera(
                yfov=np.pi / 3.0,
                znear=self.config.near_clip,
                zfar=self.config.far_clip,
            )

            # Add camera to scene (removing any previous camera)
            for node in list(self.scene.camera_nodes):
                self.scene.remove_node(node)

            # PyRender uses OpenGL convention (same as our CameraPose)
            camera_node = self.scene.add(camera, pose=camera_pose.transform)

            # Render
            color, depth = self.renderer.render(self.scene)

            # Save image
            img = Image.fromarray(color)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(output_path)

            # Save depth map if requested
            if depth_path:
                depth_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(depth_path, depth)

            # Save dummy segmentation map if requested
            if segmentation_path:
                segmentation_path.parent.mkdir(parents=True, exist_ok=True)
                seg_map = np.zeros_like(depth, dtype=np.int32)
                np.save(segmentation_path, seg_map)

            # Remove camera for next frame
            self.scene.remove_node(camera_node)

            return True

        except Exception as e:
            print(f"PyRender frame render failed: {e}")
            return False


class IsaacSimRenderer(BaseRenderer):
    """
    Isaac Sim renderer (via Omniverse).

    Production quality, GPU-accelerated.
    Requires Isaac Sim installation.

    This renderer generates a script that can be run with Isaac Sim's
    Python interpreter for actual rendering.
    """

    def __init__(self, config: RenderConfig):
        super().__init__(config)
        self.scene_path = None

    def load_scene(self, scene_path: Path) -> bool:
        """Store scene path for Isaac Sim."""
        self.scene_path = Path(scene_path)
        return self.scene_path.exists()

    def render_frame(
        self,
        camera_pose: CameraPose,
        output_path: Path,
        depth_path: Optional[Path] = None,
        segmentation_path: Optional[Path] = None,
    ) -> bool:
        """
        Not directly implemented - use generate_render_script instead.

        Isaac Sim rendering requires running a script in the Isaac Sim
        Python environment.
        """
        return False

    def render_trajectory(
        self,
        trajectory: CameraTrajectory,
        output_dir: Path,
        frame_prefix: str = "frame",
        depth_dir: Optional[Path] = None,
        segmentation_dir: Optional[Path] = None,
    ) -> dict[str, list[Path]]:
        """
        Generate a script for Isaac Sim to render the trajectory.

        Returns expected output paths (actual rendering happens externally).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate the render script
        script_path = output_dir / "render_trajectory.py"
        self._generate_render_script(
            trajectory,
            output_dir,
            script_path,
            frame_prefix,
            depth_dir=depth_dir,
            segmentation_dir=segmentation_dir,
        )

        # Return expected frame paths
        expected_paths = {"color": [], "depth": [], "segmentation": []}
        for pose in trajectory.poses:
            frame_name = f"{frame_prefix}_{pose.frame_idx:04d}.{self.config.output_format}"
            expected_paths["color"].append(output_dir / frame_name)
            if depth_dir:
                expected_paths["depth"].append(
                    Path(depth_dir) / f"{frame_prefix}_{pose.frame_idx:04d}.npy"
                )
            if segmentation_dir:
                expected_paths["segmentation"].append(
                    Path(segmentation_dir) / f"{frame_prefix}_{pose.frame_idx:04d}.npy"
                )

        return expected_paths

    def _generate_render_script(
        self,
        trajectory: CameraTrajectory,
        output_dir: Path,
        script_path: Path,
        frame_prefix: str,
        depth_dir: Optional[Path] = None,
        segmentation_dir: Optional[Path] = None,
    ) -> None:
        """Generate Python script for Isaac Sim rendering."""
        poses_data = []
        for pose in trajectory.poses:
            poses_data.append({
                "frame_idx": pose.frame_idx,
                "transform": pose.transform.tolist(),
            })

        script = f'''#!/usr/bin/env python3
"""
Isaac Sim Trajectory Renderer
Generated for: {trajectory.trajectory_id}

Run with Isaac Sim:
    ./python.sh {script_path.name}
"""

import omni
from omni.isaac.kit import SimulationApp

# Initialize Isaac Sim
simulation_app = SimulationApp({{"headless": True}})

import numpy as np
from pathlib import Path
from pxr import Usd, UsdGeom, Gf

# Configuration
SCENE_PATH = "{self.scene_path}"
OUTPUT_DIR = Path("{output_dir}")
FRAME_PREFIX = "{frame_prefix}"
WIDTH = {self.config.width}
HEIGHT = {self.config.height}
WRITE_DEPTH = {bool(depth_dir)}
WRITE_SEGMENTATION = {bool(segmentation_dir)}

# Optional outputs
DEPTH_DIR = Path("{depth_dir}") if {bool(depth_dir)} else None
SEG_DIR = Path("{segmentation_dir}") if {bool(segmentation_dir)} else None

# Camera poses
POSES = {json.dumps(poses_data, indent=4)}


def setup_camera(stage, pose_matrix):
    """Create or update camera with given pose."""
    camera_path = "/World/DWMCamera"

    # Get or create camera
    camera = UsdGeom.Camera.Get(stage, camera_path)
    if not camera:
        camera = UsdGeom.Camera.Define(stage, camera_path)

    # Set focal length
    camera.GetFocalLengthAttr().Set({trajectory.focal_length})

    # Set transform
    xform = UsdGeom.Xformable(camera)
    xform.ClearXformOpOrder()

    matrix = Gf.Matrix4d(*[item for row in pose_matrix for item in row])
    xform.AddTransformOp().Set(matrix)

    return camera


def render_frame(stage, pose, output_path):
    """Render a single frame."""
    import omni.kit.viewport.utility as vp_utils
    from omni.kit.capture.viewport import CaptureExtension
    from omni.syntheticdata import SyntheticDataHelper

    # Setup camera
    pose_matrix = pose["transform"]
    setup_camera(stage, pose_matrix)

    # Capture frame
    viewport = vp_utils.get_active_viewport()
    viewport.set_active_camera("/World/DWMCamera")

    capture_ext = CaptureExtension()
    capture_ext.capture_frame(str(output_path), WIDTH, HEIGHT)

    # Capture depth/segmentation if requested
    render_product = viewport.get_render_product_path()
    sd_helper = SyntheticDataHelper()

    depth_path = None
    seg_path = None
    frame_idx = pose["frame_idx"]
    if WRITE_DEPTH and DEPTH_DIR:
        DEPTH_DIR.mkdir(parents=True, exist_ok=True)
        depth_path = DEPTH_DIR / f"{FRAME_PREFIX}_{frame_idx:04d}.npy"
    if WRITE_SEGMENTATION and SEG_DIR:
        SEG_DIR.mkdir(parents=True, exist_ok=True)
        seg_path = SEG_DIR / f"{FRAME_PREFIX}_{frame_idx:04d}.npy"

    if depth_path or seg_path:
        gt = sd_helper.get_groundtruth(
            render_product,
            ["depth", "instanceSegmentation"],
            wait_for_servers=True,
        )

        if depth_path and "depth" in gt:
            np.save(depth_path, gt["depth"])

        if seg_path and "instanceSegmentation" in gt:
            seg_data = gt["instanceSegmentation"]
            if isinstance(seg_data, dict):
                seg_data = seg_data.get("data")
            if seg_data is not None:
                np.save(seg_path, seg_data)


def main():
    """Main rendering loop."""
    from omni.isaac.core import World
    from omni.isaac.core.utils.stage import open_stage

    # Load scene
    open_stage(SCENE_PATH)
    stage = omni.usd.get_context().get_stage()

    # Render each pose
    for pose in POSES:
        frame_idx = pose["frame_idx"]
        output_path = OUTPUT_DIR / f"{{FRAME_PREFIX}}_{{frame_idx:04d}}.png"

        print(f"Rendering frame {{frame_idx}}...")
        render_frame(stage, pose, output_path)

    print(f"Rendered {{len(POSES)}} frames to {{OUTPUT_DIR}}")

    simulation_app.close()


if __name__ == "__main__":
    main()
'''

        script_path.write_text(script)
        script_path.chmod(0o755)


class SceneRenderer:
    """
    Main scene renderer interface.

    Automatically selects the best available backend and provides
    a unified interface for rendering static scene videos.
    """

    def __init__(
        self,
        backend: Optional[RenderBackend] = None,
        config: Optional[RenderConfig] = None,
    ):
        """
        Initialize scene renderer.

        Args:
            backend: Rendering backend (auto-detect if None)
            config: Render configuration
        """
        self.config = config or RenderConfig()

        if backend is None:
            backend = self._detect_backend()

        self.backend = backend
        self.renderer = self._create_renderer(backend)

    def _detect_backend(self) -> RenderBackend:
        """Auto-detect best available backend."""
        # Check for Isaac Sim
        try:
            import omni
            return RenderBackend.ISAAC_SIM
        except ImportError:
            pass

        # Check for PyRender
        try:
            import pyrender
            return RenderBackend.PYRENDER
        except ImportError:
            pass

        # Check for trimesh
        try:
            import trimesh
            return RenderBackend.TRIMESH
        except ImportError:
            pass

        # Fall back to mock
        return RenderBackend.MOCK

    def _create_renderer(self, backend: RenderBackend) -> BaseRenderer:
        """Create renderer instance for backend."""
        if backend == RenderBackend.ISAAC_SIM:
            return IsaacSimRenderer(self.config)
        elif backend == RenderBackend.PYRENDER:
            return PyRenderRenderer(self.config)
        elif backend == RenderBackend.MOCK:
            return MockRenderer(self.config)
        else:
            print(f"Backend {backend} not fully implemented, using mock")
            return MockRenderer(self.config)

    def load_scene(self, scene_path: Path) -> bool:
        """Load a scene for rendering."""
        return self.renderer.load_scene(scene_path)

    def render_trajectory_frames(
        self,
        trajectory: CameraTrajectory,
        output_dir: Path,
        frame_prefix: str = "static_scene",
    ) -> list[Path]:
        """
        Render trajectory to individual frames.

        Args:
            trajectory: Camera trajectory
            output_dir: Output directory for frames
            frame_prefix: Prefix for frame filenames

        Returns:
            List of paths to rendered frames
        """
        outputs = self.renderer.render_trajectory(
            trajectory,
            output_dir,
            frame_prefix,
        )
        if isinstance(outputs, dict):
            return outputs.get("color", [])
        return outputs

    def render_trajectory_outputs(
        self,
        trajectory: CameraTrajectory,
        output_dir: Path,
        frame_prefix: str = "static_scene",
        depth_dir: Optional[Path] = None,
        segmentation_dir: Optional[Path] = None,
    ) -> dict[str, list[Path]]:
        """
        Render trajectory to individual frames and optionally depth/segmentation.

        Args:
            trajectory: Camera trajectory
            output_dir: Output directory for RGB frames
            frame_prefix: Prefix for frame filenames
            depth_dir: Optional depth directory
            segmentation_dir: Optional segmentation directory

        Returns:
            Dict with keys color/depth/segmentation listing generated paths
        """
        outputs = self.renderer.render_trajectory(
            trajectory,
            output_dir,
            frame_prefix,
            depth_dir=depth_dir,
            segmentation_dir=segmentation_dir,
        )
        if isinstance(outputs, dict):
            return outputs
        return {"color": outputs, "depth": [], "segmentation": []}

    def frames_to_video(
        self,
        frames_dir: Path,
        output_path: Path,
        fps: float = None,
        frame_pattern: str = "*.png",
    ) -> bool:
        """
        Encode rendered frames to video.

        Args:
            frames_dir: Directory containing frames
            output_path: Output video path
            fps: Frames per second (use config default if None)
            frame_pattern: Pattern to match frame files

        Returns:
            True if successful
        """
        if fps is None:
            fps = self.config.video_fps

        # Use ffmpeg for encoding
        try:
            frames_dir = Path(frames_dir)
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Build ffmpeg command
            input_pattern = str(frames_dir / frame_pattern.replace("*", "%04d"))

            cmd = [
                "ffmpeg",
                "-y",  # Overwrite output
                "-framerate", str(fps),
                "-i", input_pattern,
                "-c:v", self.config.video_codec,
                "-crf", str(self.config.video_quality),
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


def render_trajectory_to_video(
    scene_path: Path,
    trajectory: CameraTrajectory,
    output_video_path: Path,
    config: Optional[RenderConfig] = None,
    backend: Optional[RenderBackend] = None,
    keep_frames: bool = False,
) -> bool:
    """
    Convenience function: render trajectory directly to video.

    Args:
        scene_path: Path to 3D scene file
        trajectory: Camera trajectory
        output_video_path: Output video path
        config: Render configuration
        backend: Rendering backend
        keep_frames: Keep individual frames after encoding

    Returns:
        True if successful
    """
    renderer = SceneRenderer(backend=backend, config=config)

    if not renderer.load_scene(scene_path):
        print(f"Failed to load scene: {scene_path}")
        return False

    # Create temp directory for frames
    output_video_path = Path(output_video_path)
    frames_dir = output_video_path.parent / f"_frames_{trajectory.trajectory_id}"

    try:
        # Render frames
        frame_paths = renderer.render_trajectory_frames(
            trajectory,
            frames_dir,
            frame_prefix="frame",
        )

        if not frame_paths:
            print("No frames rendered")
            return False

        # Encode to video
        success = renderer.frames_to_video(
            frames_dir,
            output_video_path,
            fps=trajectory.fps,
            frame_pattern="frame_*.png",
        )

        return success

    finally:
        # Cleanup frames if not keeping
        if not keep_frames and frames_dir.exists():
            shutil.rmtree(frames_dir)
