"""
Static scene renderer for DWM conditioning.

Renders the static 3D scene along camera trajectories to produce
conditioning videos for DWM. The scene is rendered "frozen" - no
dynamics, just the static geometry from the reconstruction.

DEPLOYMENT ARCHITECTURE:
========================
PRODUCTION (Cloud Run / Docker with GPU):
  - Uses ISAAC_SIM backend (RenderBackend.ISAAC_SIM)
  - Isaac Sim IS deployed via Dockerfile.isaacsim
  - GPU-accelerated rendering (L4/A100)
  - docker-compose.isaacsim.yaml configures the production environment

LOCAL DEVELOPMENT (without GPU):
  - Falls back to PYRENDER (CPU/OpenGL) when Isaac Sim is unavailable
  - Deterministic batch rendering supported for dataset reproducibility
  - Set environment or use for CI/CD iteration

Supported rendering backends:
- Isaac Sim (PRODUCTION - GPU-accelerated, highest quality)
- PyRender (CPU/OpenGL, good for development)
- Trimesh (built-in renderer)
- Blender (optional, high quality offline)

Based on DWM paper:
- Static scene video: frames rendered from static 3D scene along camera trajectory
- Resolution: 720x480 (DWM default)
- Frames: 49 (DWM default)
"""

import json
import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.config.seed_manager import set_global_seed
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

    # Deterministic rendering controls
    deterministic: bool = True
    deterministic_seed: int = 0


class BaseRenderer(ABC):
    """Abstract base class for scene renderers."""

    def __init__(self, config: RenderConfig):
        self.config = config

    def _apply_determinism(self) -> None:
        """Apply deterministic settings for reproducible rendering."""
        if self.config.deterministic:
            set_global_seed(self.config.deterministic_seed)

    def _frame_metadata(
        self,
        trajectory: CameraTrajectory,
        camera_pose: CameraPose,
        output_path: Path,
        depth_path: Optional[Path],
        segmentation_path: Optional[Path],
    ) -> dict[str, Any]:
        """Build consistent metadata for a rendered frame."""
        focal_length = (
            camera_pose.focal_length
            if camera_pose.focal_length is not None
            else trajectory.focal_length
        )

        return {
            "frame_idx": camera_pose.frame_idx,
            "timestamp": camera_pose.timestamp,
            "focal_length": focal_length,
            "transform": np.asarray(camera_pose.transform).tolist(),
            "position": np.asarray(camera_pose.position).tolist(),
            "rotation": np.asarray(camera_pose.rotation).tolist(),
            "output_path": str(output_path),
            "depth_path": str(depth_path) if depth_path else None,
            "segmentation_path": str(segmentation_path) if segmentation_path else None,
        }

    def _write_metadata(
        self,
        trajectory: CameraTrajectory,
        output_dir: Path,
        frames: list[dict[str, Any]],
    ) -> Path:
        """Write consistent frame metadata for reproducibility."""
        metadata = {
            "trajectory_id": trajectory.trajectory_id,
            "trajectory_type": trajectory.trajectory_type.value,
            "fps": trajectory.fps,
            "resolution": {
                "width": self.config.width,
                "height": self.config.height,
            },
            "renderer": type(self).__name__,
            "render_config": {
                "samples": self.config.samples,
                "antialiasing": self.config.antialiasing,
                "near_clip": self.config.near_clip,
                "far_clip": self.config.far_clip,
                "background_color": self.config.background_color,
                "output_format": self.config.output_format,
                "deterministic": self.config.deterministic,
                "deterministic_seed": self.config.deterministic_seed,
            },
            "frames": frames,
        }

        metadata_path = output_dir / "frame_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2))
        return metadata_path

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
    ) -> dict[str, Any]:
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

        self._apply_determinism()

        color_paths = []
        depth_paths = []
        seg_paths = []
        frame_metadata = []

        for pose in sorted(trajectory.poses, key=lambda item: item.frame_idx):
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
                frame_metadata.append(
                    self._frame_metadata(
                        trajectory,
                        pose,
                        frame_path,
                        depth_path,
                        seg_path,
                    )
                )
            else:
                print(f"Warning: Failed to render frame {pose.frame_idx}")

        metadata_path = self._write_metadata(
            trajectory,
            output_dir,
            frame_metadata,
        )

        return {
            "color": color_paths,
            "depth": depth_paths,
            "segmentation": seg_paths,
            "metadata_path": metadata_path,
        }


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
        import importlib.util

        if (
            importlib.util.find_spec("pyrender") is None
            or importlib.util.find_spec("trimesh") is None
        ):
            raise RuntimeError(
                "PyRender backend requested but pyrender/trimesh are not installed."
            )

        import pyrender
        import trimesh

        try:
            scene_path = Path(scene_path)
            if not scene_path.exists():
                raise FileNotFoundError(f"Scene file not found: {scene_path}")

            # Handle different file formats
            if scene_path.suffix.lower() in [".glb", ".gltf"]:
                mesh = trimesh.load(str(scene_path))
            elif scene_path.suffix.lower() in [".usda", ".usd", ".usdc"]:
                raise ValueError(
                    "USD rendering requires the Isaac Sim backend. "
                    "Run inside Isaac Sim or set render_backend=RenderBackend.ISAAC_SIM."
                )
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
            raise RuntimeError(f"Failed to load scene with PyRender: {e}") from e

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

        import pyrender
        from PIL import Image

        try:
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

    This renderer loads the USD scene directly, positions a camera from the
    provided trajectory, and captures color/depth/segmentation outputs.
    """

    CAMERA_PATH = "/World/DWMCamera"

    def __init__(self, config: RenderConfig):
        super().__init__(config)
        self.scene_path: Optional[Path] = None

    @staticmethod
    def _require_isaac_modules() -> None:
        """Ensure Isaac Sim modules are available."""
        import importlib.util

        if (
            importlib.util.find_spec("omni") is None
            or importlib.util.find_spec("isaacsim.simulation_app") is None
            or importlib.util.find_spec("pxr") is None
        ):
            raise RuntimeError(
                "Isaac Sim backend requested but required modules are unavailable. "
                "Run inside the Isaac Sim Python environment."
            )

    def load_scene(self, scene_path: Path) -> bool:
        """Store scene path for Isaac Sim."""
        self._require_isaac_modules()
        scene_path = Path(scene_path)
        if not scene_path.exists():
            raise FileNotFoundError(f"Scene file not found for Isaac Sim rendering: {scene_path}")

        self.scene_path = scene_path
        return True

    @staticmethod
    def _update_camera(stage, pose_matrix: np.ndarray, focal_length: float) -> None:
        """Create/update the camera for the current pose."""
        from pxr import Gf, UsdGeom

        pose_matrix = np.asarray(pose_matrix, dtype=float)
        camera = UsdGeom.Camera.Get(stage, IsaacSimRenderer.CAMERA_PATH)
        if not camera:
            camera = UsdGeom.Camera.Define(stage, IsaacSimRenderer.CAMERA_PATH)

        camera.GetFocalLengthAttr().Set(focal_length)

        xform = UsdGeom.Xformable(camera)
        xform.ClearXformOpOrder()
        matrix = Gf.Matrix4d(*pose_matrix.astype(float).flatten().tolist())
        xform.AddTransformOp().Set(matrix)

    def render_frame(
        self,
        camera_pose: CameraPose,
        output_path: Path,
        depth_path: Optional[Path] = None,
        segmentation_path: Optional[Path] = None,
    ) -> bool:
        """
        Render a single frame using Isaac Sim.

        This method initializes Isaac Sim, loads the scene, positions the camera,
        and captures the frame. For efficiency, prefer using render_trajectory()
        when rendering multiple frames.

        Args:
            camera_pose: Camera pose for the frame
            output_path: Path to save the rendered RGB image
            depth_path: Optional path to save depth map as .npy
            segmentation_path: Optional path to save segmentation mask as .npy

        Returns:
            True if rendering succeeded, False otherwise
        """
        self._require_isaac_modules()

        if self.scene_path is None:
            print("[IsaacSimRenderer] Error: Scene not loaded")
            return False

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        from isaacsim.simulation_app import SimulationApp
        import omni
        from isaacsim.core.api.utils.stage import open_stage
        import omni.kit.viewport.utility as vp_utils
        from omni.kit.capture.viewport import CaptureExtension

        try:
            # Create headless SimulationApp
            simulation_app = SimulationApp({"headless": True})

            try:
                # Load the scene
                open_stage(str(self.scene_path))
                stage = omni.usd.get_context().get_stage()

                # Setup viewport
                viewport = vp_utils.get_active_viewport()
                viewport.set_texture_resolution((self.config.width, self.config.height))
                viewport.set_active_camera(self.CAMERA_PATH)

                # Position camera
                self._update_camera(stage, camera_pose.transform, focal_length=35.0)
                simulation_app.update()

                # Capture frame
                capture_ext = CaptureExtension()
                capture_ext.capture_frame(str(output_path), self.config.width, self.config.height)
                simulation_app.update()

                # Capture depth and segmentation if requested
                if depth_path or segmentation_path:
                    import importlib.util

                    if importlib.util.find_spec("omni.syntheticdata") is None:
                        print("[IsaacSimRenderer] Warning: SyntheticDataHelper not available")
                    else:
                        try:
                            from omni.syntheticdata import SyntheticDataHelper

                            sd_helper = SyntheticDataHelper()
                            render_product = viewport.get_render_product_path()

                            sensor_types = []
                            if depth_path:
                                sensor_types.append("depth")
                            if segmentation_path:
                                sensor_types.append("instanceSegmentation")

                            gt = sd_helper.get_groundtruth(
                                render_product,
                                sensor_types,
                                wait_for_servers=True,
                            )

                            if depth_path and "depth" in gt:
                                depth_path = Path(depth_path)
                                depth_path.parent.mkdir(parents=True, exist_ok=True)
                                np.save(depth_path, gt["depth"])

                            if segmentation_path and "instanceSegmentation" in gt:
                                segmentation_path = Path(segmentation_path)
                                segmentation_path.parent.mkdir(parents=True, exist_ok=True)
                                seg_data = gt["instanceSegmentation"]
                                if isinstance(seg_data, dict):
                                    seg_data = seg_data.get("data", seg_data)
                                np.save(segmentation_path, seg_data)

                        except Exception as e:
                            print(
                                "[IsaacSimRenderer] Warning: "
                                f"Failed to capture depth/segmentation: {e}"
                            )

                return True

            finally:
                simulation_app.close()

        except Exception as e:
            print(f"[IsaacSimRenderer] Error: Rendering failed: {e}")
            return False

    def render_trajectory(
        self,
        trajectory: CameraTrajectory,
        output_dir: Path,
        frame_prefix: str = "frame",
        depth_dir: Optional[Path] = None,
        segmentation_dir: Optional[Path] = None,
    ) -> dict[str, Any]:
        """
        Render the full trajectory inside Isaac Sim.
        """
        self._require_isaac_modules()
        if self.scene_path is None:
            raise RuntimeError("Scene not loaded for Isaac Sim renderer")

        self._apply_determinism()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        depth_dir = Path(depth_dir) if depth_dir else None
        segmentation_dir = Path(segmentation_dir) if segmentation_dir else None

        color_paths: list[Path] = []
        depth_paths: list[Path] = []
        seg_paths: list[Path] = []
        frame_metadata: list[dict[str, Any]] = []

        import importlib.util

        from isaacsim.simulation_app import SimulationApp
        import omni
        from isaacsim.core.api.utils.stage import open_stage
        import omni.kit.viewport.utility as vp_utils
        from omni.kit.capture.viewport import CaptureExtension

        try:
            simulation_app = SimulationApp({"headless": True})
            try:
                open_stage(str(self.scene_path))
                stage = omni.usd.get_context().get_stage()

                viewport = vp_utils.get_active_viewport()
                viewport.set_texture_resolution((self.config.width, self.config.height))
                viewport.set_active_camera(self.CAMERA_PATH)

                capture_ext = CaptureExtension()
                sd_helper = None
                if importlib.util.find_spec("omni.syntheticdata") is not None:
                    from omni.syntheticdata import SyntheticDataHelper

                    sd_helper = SyntheticDataHelper()

                for pose in sorted(trajectory.poses, key=lambda item: item.frame_idx):
                    frame_name = f"{frame_prefix}_{pose.frame_idx:04d}.{self.config.output_format}"
                    output_path = output_dir / frame_name
                    depth_path = None
                    seg_path = None

                    if depth_dir:
                        depth_dir.mkdir(parents=True, exist_ok=True)
                        depth_path = depth_dir / f"{frame_prefix}_{pose.frame_idx:04d}.npy"
                    if segmentation_dir:
                        segmentation_dir.mkdir(parents=True, exist_ok=True)
                        seg_path = segmentation_dir / f"{frame_prefix}_{pose.frame_idx:04d}.npy"

                    self._update_camera(stage, pose.transform, trajectory.focal_length)
                    simulation_app.update()

                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    capture_ext.capture_frame(str(output_path), self.config.width, self.config.height)
                    simulation_app.update()

                    if (depth_path or seg_path) and sd_helper is not None:
                        render_product = viewport.get_render_product_path()
                        gt = sd_helper.get_groundtruth(
                            render_product,
                            ["depth", "instanceSegmentation"],
                            wait_for_servers=True,
                        )

                        if depth_path and "depth" in gt:
                            np.save(depth_path, gt["depth"])
                            depth_paths.append(depth_path)

                        if seg_path and "instanceSegmentation" in gt:
                            seg_data = gt["instanceSegmentation"]
                            if isinstance(seg_data, dict):
                                seg_data = seg_data.get("data")
                            if seg_data is not None:
                                np.save(seg_path, seg_data)
                                seg_paths.append(seg_path)
                    elif depth_path or seg_path:
                        print(
                            "[IsaacSimRenderer] Warning: "
                            "SyntheticDataHelper not available for depth/segmentation."
                        )

                    color_paths.append(output_path)
                    frame_metadata.append(
                        self._frame_metadata(
                            trajectory,
                            pose,
                            output_path,
                            depth_path,
                            seg_path,
                        )
                    )

            finally:
                simulation_app.close()

        except Exception as exc:
            raise RuntimeError(f"Isaac Sim rendering failed: {exc}") from exc

        metadata_path = self._write_metadata(
            trajectory,
            output_dir,
            frame_metadata,
        )

        return {
            "color": color_paths,
            "depth": depth_paths,
            "segmentation": seg_paths,
            "metadata_path": metadata_path,
        }


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
        import importlib.util

        if importlib.util.find_spec("omni") is not None:
            return RenderBackend.ISAAC_SIM

        if importlib.util.find_spec("pyrender") is not None:
            return RenderBackend.PYRENDER

        raise RuntimeError(
            "No supported renderer backend available. Install Isaac Sim or pyrender."
        )

    def _ensure_backend_available(self, backend: RenderBackend) -> None:
        """Validate that the selected backend can run."""
        if backend == RenderBackend.ISAAC_SIM:
            import importlib.util

            if (
                importlib.util.find_spec("omni") is None
                or importlib.util.find_spec("isaacsim.simulation_app") is None
            ):
                raise RuntimeError(
                    "Isaac Sim backend requested but required modules are missing. "
                    "Run within Isaac Sim or choose a different backend."
                )
        elif backend == RenderBackend.PYRENDER:
            import importlib.util

            if (
                importlib.util.find_spec("pyrender") is None
                or importlib.util.find_spec("trimesh") is None
            ):
                raise RuntimeError(
                    "PyRender backend requested but pyrender/trimesh are not installed."
                )
        else:
            raise ValueError(f"Unsupported render backend: {backend}")

    def _create_renderer(self, backend: RenderBackend) -> BaseRenderer:
        """Create renderer instance for backend."""
        self._ensure_backend_available(backend)
        if backend == RenderBackend.ISAAC_SIM:
            return IsaacSimRenderer(self.config)
        elif backend == RenderBackend.PYRENDER:
            return PyRenderRenderer(self.config)
        else:
            raise ValueError(f"Backend {backend} is not implemented")

    def load_scene(self, scene_path: Path) -> bool:
        """Load a scene for rendering."""
        success = self.renderer.load_scene(scene_path)
        if not success:
            raise RuntimeError(f"Renderer {self.backend} failed to load scene at {scene_path}")
        return success

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
    ) -> dict[str, Any]:
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
        return {"color": outputs, "depth": [], "segmentation": [], "metadata_path": None}

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
