#!/usr/bin/env python3
"""
Offline Rendering for Episode Generation (Isaac Sim Alternative).

This module provides sensor data capture using open-source renderers
when Isaac Sim is not available. Supports:

1. PyRender (default) - Fast OpenGL-based rendering
2. Blender (optional) - High-quality Cycles rendering
3. Open3D (fallback) - Basic visualization rendering

This replaces the random noise mock renderer with ACTUAL renders from
the USD/GLB scene, making episodes usable for training without Isaac Sim.

Usage:
    renderer = create_offline_renderer(backend="pyrender")
    renderer.load_scene(scene_usd_path, manifest)
    frame = renderer.capture_frame(camera_id, robot_state)
"""

import json
import os
import sys
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Add parent to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# =============================================================================
# Backend Detection
# =============================================================================

def _check_pyrender() -> bool:
    """Check if PyRender is available."""
    try:
        import pyrender
        import trimesh
        return True
    except ImportError:
        return False


def _check_open3d() -> bool:
    """Check if Open3D is available."""
    try:
        import open3d as o3d
        return True
    except ImportError:
        return False


def _check_blender() -> bool:
    """Check if Blender Python API is available (bpy)."""
    try:
        import bpy
        return True
    except ImportError:
        return False


def _check_pillow() -> bool:
    """Check if PIL/Pillow is available."""
    try:
        from PIL import Image
        return True
    except ImportError:
        return False


HAVE_PYRENDER = _check_pyrender()
HAVE_OPEN3D = _check_open3d()
HAVE_BLENDER = _check_blender()
HAVE_PILLOW = _check_pillow()


def get_available_backends() -> List[str]:
    """Get list of available rendering backends."""
    backends = []
    if HAVE_PYRENDER:
        backends.append("pyrender")
    if HAVE_OPEN3D:
        backends.append("open3d")
    if HAVE_BLENDER:
        backends.append("blender")
    return backends


def get_best_backend() -> Optional[str]:
    """Get the best available rendering backend."""
    if HAVE_PYRENDER:
        return "pyrender"
    if HAVE_OPEN3D:
        return "open3d"
    if HAVE_BLENDER:
        return "blender"
    return None


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class CameraParams:
    """Camera parameters for rendering."""

    camera_id: str
    position: np.ndarray  # [x, y, z] in world coordinates
    look_at: np.ndarray   # [x, y, z] target point
    up_vector: np.ndarray = field(default_factory=lambda: np.array([0, 1, 0]))
    fov: float = 60.0  # degrees
    resolution: Tuple[int, int] = (640, 480)
    near_clip: float = 0.01
    far_clip: float = 100.0


@dataclass
class RenderOutput:
    """Output from a render pass."""

    rgb: np.ndarray  # (H, W, 3) uint8
    depth: Optional[np.ndarray] = None  # (H, W) float32, meters
    segmentation: Optional[np.ndarray] = None  # (H, W) int32, instance IDs
    normals: Optional[np.ndarray] = None  # (H, W, 3) float32


@dataclass
class SceneObject:
    """Object in the scene for rendering."""

    object_id: str
    mesh_path: Path
    position: np.ndarray  # [x, y, z]
    rotation: np.ndarray  # quaternion [w, x, y, z]
    scale: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0]))
    semantic_class: str = "object"
    instance_id: int = 0


# =============================================================================
# Abstract Renderer Interface
# =============================================================================


class OfflineRenderer(ABC):
    """Abstract base class for offline renderers."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.scene_loaded = False
        self.objects: Dict[str, SceneObject] = {}
        self.cameras: Dict[str, CameraParams] = {}

    def log(self, msg: str, level: str = "INFO") -> None:
        if self.verbose:
            print(f"[OFFLINE-RENDER] [{level}] {msg}")

    @abstractmethod
    def load_mesh(self, mesh_path: Path, object_id: str) -> bool:
        """Load a mesh file (GLB, OBJ, PLY, etc.)."""
        pass

    @abstractmethod
    def set_object_transform(
        self,
        object_id: str,
        position: np.ndarray,
        rotation: np.ndarray,
        scale: np.ndarray,
    ) -> None:
        """Set transform for an object."""
        pass

    @abstractmethod
    def add_camera(self, params: CameraParams) -> None:
        """Add a camera to the scene."""
        pass

    @abstractmethod
    def render(
        self,
        camera_id: str,
        capture_depth: bool = False,
        capture_segmentation: bool = False,
    ) -> RenderOutput:
        """Render from specified camera."""
        pass

    def load_scene_from_manifest(
        self,
        manifest_path: Path,
        assets_root: Optional[Path] = None,
    ) -> bool:
        """Load scene from BlueprintPipeline manifest."""

        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
        except Exception as e:
            self.log(f"Failed to load manifest: {e}", "ERROR")
            return False

        if assets_root is None:
            assets_root = manifest_path.parent

        # Load objects
        objects = manifest.get("objects", [])
        for idx, obj in enumerate(objects):
            obj_id = obj.get("id", f"obj_{idx}")

            # Find mesh path
            asset_info = obj.get("asset", {})
            mesh_rel = asset_info.get("path", obj.get("asset_path"))
            if not mesh_rel:
                self.log(f"Object {obj_id} has no mesh path, skipping", "WARNING")
                continue

            mesh_path = assets_root / mesh_rel
            if not mesh_path.exists():
                # Try common alternatives
                for ext in [".glb", ".gltf", ".obj", ".ply"]:
                    alt_path = mesh_path.with_suffix(ext)
                    if alt_path.exists():
                        mesh_path = alt_path
                        break

            if not mesh_path.exists():
                self.log(f"Mesh not found for {obj_id}: {mesh_path}", "WARNING")
                continue

            # Get transform
            transform = obj.get("transform", {})
            pos = transform.get("position", {"x": 0, "y": 0, "z": 0})
            position = np.array([pos.get("x", 0), pos.get("y", 0), pos.get("z", 0)])

            rot = transform.get("rotation_quaternion", {"w": 1, "x": 0, "y": 0, "z": 0})
            rotation = np.array([rot.get("w", 1), rot.get("x", 0), rot.get("y", 0), rot.get("z", 0)])

            scl = transform.get("scale", {"x": 1, "y": 1, "z": 1})
            scale = np.array([scl.get("x", 1), scl.get("y", 1), scl.get("z", 1)])

            # Load mesh
            if self.load_mesh(mesh_path, obj_id):
                self.set_object_transform(obj_id, position, rotation, scale)

                self.objects[obj_id] = SceneObject(
                    object_id=obj_id,
                    mesh_path=mesh_path,
                    position=position,
                    rotation=rotation,
                    scale=scale,
                    semantic_class=obj.get("category", "object"),
                    instance_id=idx + 1,
                )
                self.log(f"Loaded object: {obj_id}")

        # Setup default cameras if not in manifest
        scene_info = manifest.get("scene", {})
        room = scene_info.get("room", {})
        bounds = room.get("bounds", {"width": 5, "depth": 5, "height": 3})

        # Create overhead camera
        center_x = bounds.get("width", 5) / 2
        center_z = bounds.get("depth", 5) / 2
        height = bounds.get("height", 3)

        self.add_camera(CameraParams(
            camera_id="overhead",
            position=np.array([center_x, height * 0.9, center_z]),
            look_at=np.array([center_x, 0, center_z]),
            up_vector=np.array([0, 0, -1]),
            fov=75.0,
            resolution=(640, 480),
        ))

        # Create front camera
        self.add_camera(CameraParams(
            camera_id="front",
            position=np.array([center_x, 1.5, -1.0]),
            look_at=np.array([center_x, 0.8, center_z / 2]),
            up_vector=np.array([0, 1, 0]),
            fov=60.0,
            resolution=(640, 480),
        ))

        self.scene_loaded = True
        self.log(f"Scene loaded: {len(self.objects)} objects, {len(self.cameras)} cameras")
        return True

    def capture_episode_frame(
        self,
        camera_ids: Optional[List[str]] = None,
        capture_depth: bool = False,
        capture_segmentation: bool = False,
    ) -> Dict[str, RenderOutput]:
        """Capture renders from multiple cameras."""

        if camera_ids is None:
            camera_ids = list(self.cameras.keys())

        outputs = {}
        for cam_id in camera_ids:
            if cam_id in self.cameras:
                outputs[cam_id] = self.render(
                    cam_id,
                    capture_depth=capture_depth,
                    capture_segmentation=capture_segmentation,
                )

        return outputs


# =============================================================================
# PyRender Implementation
# =============================================================================


class PyRenderRenderer(OfflineRenderer):
    """
    PyRender-based offline renderer.

    PyRender is a lightweight, easy-to-use renderer built on PyOpenGL.
    Supports:
    - GLB/GLTF mesh loading via trimesh
    - RGB + Depth rendering
    - Headless rendering (no display required)

    Install: pip install pyrender trimesh
    """

    def __init__(self, verbose: bool = True):
        super().__init__(verbose)

        if not HAVE_PYRENDER:
            raise ImportError(
                "PyRender not available. Install with: pip install pyrender trimesh"
            )

        import pyrender
        import trimesh

        self._pyrender = pyrender
        self._trimesh = trimesh

        # Create scene
        self._scene = pyrender.Scene(
            ambient_light=np.array([0.3, 0.3, 0.3, 1.0]),
            bg_color=np.array([0.9, 0.9, 0.9, 1.0]),
        )

        # Add lighting
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        light_pose = np.eye(4)
        light_pose[:3, 3] = [0, 5, 0]
        self._scene.add(light, pose=light_pose)

        # Add secondary light
        light2 = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=10.0)
        light2_pose = np.eye(4)
        light2_pose[:3, 3] = [2, 3, 2]
        self._scene.add(light2, pose=light2_pose)

        self._mesh_nodes: Dict[str, Any] = {}
        self._camera_nodes: Dict[str, Any] = {}

        self.log("PyRender renderer initialized")

    def load_mesh(self, mesh_path: Path, object_id: str) -> bool:
        """Load a mesh using trimesh."""
        try:
            # Load mesh
            mesh = self._trimesh.load(str(mesh_path), force='mesh')

            # Handle scene vs single mesh
            if isinstance(mesh, self._trimesh.Scene):
                # Combine all geometries
                meshes = []
                for name, geom in mesh.geometry.items():
                    if isinstance(geom, self._trimesh.Trimesh):
                        meshes.append(geom)
                if meshes:
                    mesh = self._trimesh.util.concatenate(meshes)
                else:
                    self.log(f"No valid geometry in {mesh_path}", "WARNING")
                    return False

            # Convert to pyrender mesh
            pr_mesh = self._pyrender.Mesh.from_trimesh(mesh, smooth=True)

            # Add to scene
            node = self._scene.add(pr_mesh)
            self._mesh_nodes[object_id] = node

            return True

        except Exception as e:
            self.log(f"Failed to load mesh {mesh_path}: {e}", "ERROR")
            return False

    def set_object_transform(
        self,
        object_id: str,
        position: np.ndarray,
        rotation: np.ndarray,
        scale: np.ndarray,
    ) -> None:
        """Set transform for an object."""
        if object_id not in self._mesh_nodes:
            return

        node = self._mesh_nodes[object_id]

        # Build transform matrix
        from scipy.spatial.transform import Rotation as R

        pose = np.eye(4)

        # Rotation from quaternion (w, x, y, z)
        quat = rotation[[1, 2, 3, 0]]  # scipy uses (x, y, z, w)
        rot_matrix = R.from_quat(quat).as_matrix()
        pose[:3, :3] = rot_matrix @ np.diag(scale)

        # Translation
        pose[:3, 3] = position

        # Update node
        self._scene.set_pose(node, pose)

    def add_camera(self, params: CameraParams) -> None:
        """Add a camera to the scene."""
        # Create camera
        camera = self._pyrender.PerspectiveCamera(
            yfov=np.radians(params.fov),
            aspectRatio=params.resolution[0] / params.resolution[1],
            znear=params.near_clip,
            zfar=params.far_clip,
        )

        # Compute camera pose (look-at)
        pose = self._look_at_matrix(
            params.position,
            params.look_at,
            params.up_vector,
        )

        # Add to scene
        node = self._scene.add(camera, pose=pose)
        self._camera_nodes[params.camera_id] = {
            "node": node,
            "camera": camera,
            "params": params,
        }
        self.cameras[params.camera_id] = params

    def _look_at_matrix(
        self,
        eye: np.ndarray,
        target: np.ndarray,
        up: np.ndarray,
    ) -> np.ndarray:
        """Create a look-at camera matrix."""
        forward = target - eye
        forward = forward / np.linalg.norm(forward)

        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)

        up_new = np.cross(right, forward)

        # PyRender uses -Z forward
        pose = np.eye(4)
        pose[:3, 0] = right
        pose[:3, 1] = up_new
        pose[:3, 2] = -forward
        pose[:3, 3] = eye

        return pose

    def render(
        self,
        camera_id: str,
        capture_depth: bool = False,
        capture_segmentation: bool = False,
    ) -> RenderOutput:
        """Render from specified camera."""
        if camera_id not in self._camera_nodes:
            raise ValueError(f"Unknown camera: {camera_id}")

        cam_info = self._camera_nodes[camera_id]
        params = cam_info["params"]

        # Create offscreen renderer
        renderer = self._pyrender.OffscreenRenderer(
            viewport_width=params.resolution[0],
            viewport_height=params.resolution[1],
        )

        try:
            # Render
            flags = self._pyrender.RenderFlags.RGBA
            if capture_depth:
                color, depth = renderer.render(
                    self._scene,
                    flags=flags,
                )
            else:
                color, _ = renderer.render(
                    self._scene,
                    flags=flags,
                )
                depth = None

            # Convert to RGB
            rgb = color[:, :, :3].astype(np.uint8)

            # Depth processing
            if depth is not None:
                # Convert to meters (pyrender returns in scene units)
                depth = depth.astype(np.float32)
                # Mask invalid depths
                depth[depth == 0] = np.nan

            return RenderOutput(
                rgb=rgb,
                depth=depth,
                segmentation=None,  # PyRender doesn't support instance seg natively
            )

        finally:
            renderer.delete()


# =============================================================================
# Open3D Implementation
# =============================================================================


class Open3DRenderer(OfflineRenderer):
    """
    Open3D-based offline renderer.

    Open3D provides visualization and rendering capabilities.
    Good for point clouds and simple mesh rendering.

    Install: pip install open3d
    """

    def __init__(self, verbose: bool = True):
        super().__init__(verbose)

        if not HAVE_OPEN3D:
            raise ImportError(
                "Open3D not available. Install with: pip install open3d"
            )

        import open3d as o3d
        self._o3d = o3d

        self._geometries: Dict[str, Any] = {}
        self._transforms: Dict[str, np.ndarray] = {}

        self.log("Open3D renderer initialized")

    def load_mesh(self, mesh_path: Path, object_id: str) -> bool:
        """Load a mesh using Open3D."""
        try:
            mesh = self._o3d.io.read_triangle_mesh(str(mesh_path))
            if not mesh.has_triangles():
                self.log(f"No triangles in mesh {mesh_path}", "WARNING")
                return False

            mesh.compute_vertex_normals()
            self._geometries[object_id] = mesh
            self._transforms[object_id] = np.eye(4)
            return True

        except Exception as e:
            self.log(f"Failed to load mesh {mesh_path}: {e}", "ERROR")
            return False

    def set_object_transform(
        self,
        object_id: str,
        position: np.ndarray,
        rotation: np.ndarray,
        scale: np.ndarray,
    ) -> None:
        """Set transform for an object."""
        if object_id not in self._geometries:
            return

        from scipy.spatial.transform import Rotation as R

        transform = np.eye(4)
        quat = rotation[[1, 2, 3, 0]]  # scipy uses (x, y, z, w)
        rot_matrix = R.from_quat(quat).as_matrix()
        transform[:3, :3] = rot_matrix @ np.diag(scale)
        transform[:3, 3] = position

        self._transforms[object_id] = transform

    def add_camera(self, params: CameraParams) -> None:
        """Add a camera configuration."""
        self.cameras[params.camera_id] = params

    def render(
        self,
        camera_id: str,
        capture_depth: bool = False,
        capture_segmentation: bool = False,
    ) -> RenderOutput:
        """Render using Open3D's offscreen renderer."""
        if camera_id not in self.cameras:
            raise ValueError(f"Unknown camera: {camera_id}")

        params = self.cameras[camera_id]

        # Create visualizer for offscreen rendering
        vis = self._o3d.visualization.Visualizer()
        vis.create_window(
            width=params.resolution[0],
            height=params.resolution[1],
            visible=False,
        )

        try:
            # Add transformed geometries
            for obj_id, geom in self._geometries.items():
                transformed = geom.transform(self._transforms[obj_id])
                vis.add_geometry(transformed)

            # Set camera
            ctr = vis.get_view_control()
            ctr.set_lookat(params.look_at)
            ctr.set_up(params.up_vector)
            ctr.set_front(params.position - params.look_at)
            ctr.set_zoom(0.5)

            # Render
            vis.poll_events()
            vis.update_renderer()

            # Capture
            rgb = np.asarray(vis.capture_screen_float_buffer(do_render=True))
            rgb = (rgb * 255).astype(np.uint8)

            depth = None
            if capture_depth:
                depth = np.asarray(vis.capture_depth_float_buffer(do_render=True))

            return RenderOutput(rgb=rgb, depth=depth)

        finally:
            vis.destroy_window()


# =============================================================================
# Factory Function
# =============================================================================


def create_offline_renderer(
    backend: Optional[str] = None,
    verbose: bool = True,
) -> OfflineRenderer:
    """
    Create an offline renderer instance.

    Args:
        backend: Renderer backend ("pyrender", "open3d", "blender").
                If None, uses the best available.
        verbose: Print debug info.

    Returns:
        OfflineRenderer instance.

    Example:
        renderer = create_offline_renderer()
        renderer.load_scene_from_manifest(Path("scene_manifest.json"))
        output = renderer.render("overhead", capture_depth=True)
        rgb_image = output.rgb  # (H, W, 3) uint8 array
    """
    if backend is None:
        backend = get_best_backend()

    if backend is None:
        raise ImportError(
            "No rendering backend available. Install one of:\n"
            "  pip install pyrender trimesh  # Recommended\n"
            "  pip install open3d\n"
        )

    if backend == "pyrender":
        return PyRenderRenderer(verbose=verbose)
    elif backend == "open3d":
        return Open3DRenderer(verbose=verbose)
    elif backend == "blender":
        raise NotImplementedError("Blender backend not yet implemented")
    else:
        raise ValueError(f"Unknown backend: {backend}")


# =============================================================================
# Integration with Sensor Data Capture
# =============================================================================


class OfflineRendererSensorCapture:
    """
    Sensor capture using offline rendering (replaces MockSensorCapture).

    This class provides the same interface as IsaacSimSensorCapture but
    uses PyRender/Open3D for actual rendering instead of random noise.
    """

    def __init__(
        self,
        cameras: List[Dict[str, Any]],
        data_tier: str = "core",
        verbose: bool = True,
    ):
        self.verbose = verbose
        self.cameras = cameras
        self.data_tier = data_tier
        self.initialized = False

        self._renderer: Optional[OfflineRenderer] = None
        self._capture_depth = data_tier in ["plus", "full"]
        self._capture_seg = data_tier in ["plus", "full"]

    def log(self, msg: str, level: str = "INFO") -> None:
        if self.verbose:
            print(f"[OFFLINE-CAPTURE] [{level}] {msg}")

    def initialize(
        self,
        scene_manifest_path: Optional[Path] = None,
        assets_root: Optional[Path] = None,
    ) -> bool:
        """Initialize the offline renderer with scene data."""
        try:
            self._renderer = create_offline_renderer(verbose=self.verbose)

            if scene_manifest_path:
                success = self._renderer.load_scene_from_manifest(
                    scene_manifest_path,
                    assets_root,
                )
                if not success:
                    self.log("Failed to load scene", "ERROR")
                    return False

            # Add cameras from config
            for cam_cfg in self.cameras:
                cam_id = cam_cfg.get("camera_id", "default")
                position = np.array(cam_cfg.get("position", [2, 2, 2]))
                look_at = np.array(cam_cfg.get("look_at", [0, 0, 0]))
                resolution = tuple(cam_cfg.get("resolution", [640, 480]))
                fov = cam_cfg.get("fov", 60.0)

                self._renderer.add_camera(CameraParams(
                    camera_id=cam_id,
                    position=position,
                    look_at=look_at,
                    resolution=resolution,
                    fov=fov,
                ))

            self.initialized = True
            self.log("Offline sensor capture initialized")
            return True

        except Exception as e:
            self.log(f"Initialization failed: {e}", "ERROR")
            return False

    def capture_frame(
        self,
        frame_idx: int,
        robot_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Capture a frame from all cameras."""
        if not self.initialized or not self._renderer:
            return self._fallback_frame()

        frame_data = {
            "frame_idx": frame_idx,
            "rgb_images": {},
            "depth_images": {},
            "segmentation": {},
        }

        for cam_id in self._renderer.cameras:
            try:
                output = self._renderer.render(
                    cam_id,
                    capture_depth=self._capture_depth,
                    capture_segmentation=self._capture_seg,
                )

                frame_data["rgb_images"][cam_id] = output.rgb

                if output.depth is not None:
                    frame_data["depth_images"][cam_id] = output.depth

                if output.segmentation is not None:
                    frame_data["segmentation"][cam_id] = output.segmentation

            except Exception as e:
                self.log(f"Render failed for camera {cam_id}: {e}", "WARNING")
                # Fallback to placeholder
                frame_data["rgb_images"][cam_id] = np.zeros((480, 640, 3), dtype=np.uint8)

        return frame_data

    def _fallback_frame(self) -> Dict[str, Any]:
        """Generate fallback frame when renderer not available."""
        return {
            "frame_idx": 0,
            "rgb_images": {"default": np.zeros((480, 640, 3), dtype=np.uint8)},
            "depth_images": {},
            "segmentation": {},
        }

    def update_object_pose(
        self,
        object_id: str,
        position: np.ndarray,
        rotation: np.ndarray,
    ) -> None:
        """Update an object's pose (for animation)."""
        if self._renderer and object_id in self._renderer.objects:
            obj = self._renderer.objects[object_id]
            self._renderer.set_object_transform(
                object_id,
                position,
                rotation,
                obj.scale,
            )

    def cleanup(self) -> None:
        """Cleanup resources."""
        self._renderer = None
        self.initialized = False


# =============================================================================
# Main / Testing
# =============================================================================


if __name__ == "__main__":
    print("=" * 60)
    print("Offline Renderer Test")
    print("=" * 60)

    # Check available backends
    print("\nAvailable backends:", get_available_backends())
    print("Best backend:", get_best_backend())

    if not get_available_backends():
        print("\nNo rendering backend available!")
        print("Install with: pip install pyrender trimesh")
        sys.exit(1)

    # Create renderer
    renderer = create_offline_renderer()

    # Add a test camera
    renderer.add_camera(CameraParams(
        camera_id="test",
        position=np.array([2.0, 2.0, 2.0]),
        look_at=np.array([0.0, 0.0, 0.0]),
        resolution=(640, 480),
    ))

    print("\nRenderer created successfully!")
    print(f"  Type: {type(renderer).__name__}")
    print(f"  Cameras: {list(renderer.cameras.keys())}")

    # Test render (will be empty scene)
    try:
        output = renderer.render("test")
        print(f"\nRender output:")
        print(f"  RGB shape: {output.rgb.shape}")
        print(f"  RGB dtype: {output.rgb.dtype}")
        print(f"  Depth: {output.depth is not None}")
        print("\n✅ Offline renderer working!")
    except Exception as e:
        print(f"\n❌ Render failed: {e}")
