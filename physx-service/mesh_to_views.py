#!/usr/bin/env python3
"""
Mesh to Multi-View Image Renderer.

Renders GLB/GLTF meshes to multiple view images for PhysX-Anything VLM analysis.
Uses trimesh with pyrender backend for high-quality rendering.

This module is used when the interactive-job or physx-service receives a GLB mesh
directly (e.g., from ZeroScene) and needs to generate view images for the VLM.
"""

import io
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


def load_mesh(mesh_path: Path):
    """
    Load a mesh from file.

    Supports GLB, GLTF, OBJ, PLY formats.
    Returns a trimesh.Trimesh or None on failure.
    """
    try:
        import trimesh
        mesh = trimesh.load(str(mesh_path), force='mesh')

        # Handle Scene objects (multiple meshes)
        if isinstance(mesh, trimesh.Scene):
            geometries = list(mesh.geometry.values())
            if geometries:
                mesh = trimesh.util.concatenate(geometries)
            else:
                return None

        return mesh
    except Exception as e:
        print(f"[MESH] Error loading mesh: {e}")
        return None


def load_mesh_from_bytes(mesh_bytes: bytes, file_type: str = "glb"):
    """
    Load a mesh from raw bytes.

    Args:
        mesh_bytes: Raw mesh file bytes
        file_type: File extension (glb, gltf, obj, etc.)

    Returns:
        trimesh.Trimesh or None
    """
    try:
        import trimesh
        mesh = trimesh.load(
            io.BytesIO(mesh_bytes),
            file_type=file_type,
            force='mesh'
        )

        if isinstance(mesh, trimesh.Scene):
            geometries = list(mesh.geometry.values())
            if geometries:
                mesh = trimesh.util.concatenate(geometries)
            else:
                return None

        return mesh
    except Exception as e:
        print(f"[MESH] Error loading mesh from bytes: {e}")
        return None


def compute_camera_positions(
    center: np.ndarray,
    distance: float,
    num_views: int = 4,
    elevation_degrees: float = 30.0,
) -> List[np.ndarray]:
    """
    Compute camera positions around an object.

    Args:
        center: Object center point
        distance: Distance from center to camera
        num_views: Number of views to generate
        elevation_degrees: Camera elevation angle above horizon

    Returns:
        List of camera position vectors
    """
    positions = []
    elevation_rad = np.radians(elevation_degrees)

    for i in range(num_views):
        azimuth = (2 * np.pi * i) / num_views

        # Camera position in spherical coordinates
        x = center[0] + distance * np.cos(azimuth) * np.cos(elevation_rad)
        y = center[1] + distance * np.sin(azimuth) * np.cos(elevation_rad)
        z = center[2] + distance * np.sin(elevation_rad)

        positions.append(np.array([x, y, z]))

    return positions


def look_at_matrix(eye: np.ndarray, target: np.ndarray, up: np.ndarray = None) -> np.ndarray:
    """
    Create a look-at camera transformation matrix.

    Args:
        eye: Camera position
        target: Point to look at
        up: Up vector (default: Z-up)

    Returns:
        4x4 transformation matrix
    """
    if up is None:
        up = np.array([0.0, 0.0, 1.0])

    forward = target - eye
    forward = forward / np.linalg.norm(forward)

    right = np.cross(forward, up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        # Handle case where forward is parallel to up
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, up)
        right_norm = np.linalg.norm(right)

    right = right / right_norm

    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

    mat = np.eye(4)
    mat[:3, 0] = right
    mat[:3, 1] = up
    mat[:3, 2] = -forward
    mat[:3, 3] = eye

    return mat


def render_views_trimesh(
    mesh,
    num_views: int = 4,
    resolution: Tuple[int, int] = (512, 512),
    elevation_degrees: float = 30.0,
) -> List[bytes]:
    """
    Render multiple views of a mesh using trimesh's built-in renderer.

    This is the simplest approach and works without GPU.

    Args:
        mesh: trimesh.Trimesh object
        num_views: Number of views to render
        resolution: Output image resolution (width, height)
        elevation_degrees: Camera elevation angle

    Returns:
        List of PNG image bytes
    """
    import trimesh

    images = []

    # Get mesh bounds and center
    bounds = mesh.bounds
    center = mesh.centroid
    size = np.max(bounds[1] - bounds[0])
    distance = size * 2.5

    for i in range(num_views):
        try:
            # Create a scene with the mesh
            scene = trimesh.Scene(mesh)

            # Compute camera angle
            angle = (360.0 * i) / num_views

            # Set camera
            scene.set_camera(
                angles=(np.radians(elevation_degrees), np.radians(angle), 0),
                distance=distance,
                center=center,
            )

            # Render to PNG bytes
            png_data = scene.save_image(resolution=resolution)
            if png_data:
                images.append(png_data)

        except Exception as e:
            print(f"[MESH] Error rendering view {i}: {e}")
            continue

    return images


def render_views_pyrender(
    mesh,
    num_views: int = 4,
    resolution: Tuple[int, int] = (512, 512),
    elevation_degrees: float = 30.0,
) -> List[bytes]:
    """
    Render multiple views of a mesh using pyrender.

    This provides higher quality rendering but requires pyrender to be installed.

    Args:
        mesh: trimesh.Trimesh object
        num_views: Number of views to render
        resolution: Output image resolution (width, height)
        elevation_degrees: Camera elevation angle

    Returns:
        List of PNG image bytes
    """
    try:
        import pyrender
        from PIL import Image
    except ImportError:
        print("[MESH] pyrender not available, falling back to trimesh")
        return render_views_trimesh(mesh, num_views, resolution, elevation_degrees)

    images = []

    # Get mesh bounds
    bounds = mesh.bounds
    center = mesh.centroid
    size = np.max(bounds[1] - bounds[0])
    distance = size * 2.5

    # Compute camera positions
    camera_positions = compute_camera_positions(
        center, distance, num_views, elevation_degrees
    )

    # Create pyrender scene
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0])

    # Add mesh to scene
    pyrender_mesh = pyrender.Mesh.from_trimesh(mesh)
    scene.add(pyrender_mesh)

    # Add light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(light, pose=np.eye(4))

    # Add ambient light
    ambient = pyrender.SpotLight(color=[1.0, 1.0, 1.0], intensity=1.0)
    scene.add(ambient, pose=np.eye(4))

    # Create camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)

    # Create renderer
    renderer = pyrender.OffscreenRenderer(resolution[0], resolution[1])

    try:
        for i, cam_pos in enumerate(camera_positions):
            try:
                # Compute camera pose (look at center)
                pose = look_at_matrix(cam_pos, center)

                # Add camera to scene
                camera_node = scene.add(camera, pose=pose)

                # Render
                color, _ = renderer.render(scene)

                # Convert to PNG bytes
                img = Image.fromarray(color)
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                images.append(buffer.getvalue())

                # Remove camera for next iteration
                scene.remove_node(camera_node)

            except Exception as e:
                print(f"[MESH] Error rendering view {i}: {e}")
                continue

    finally:
        renderer.delete()

    return images


def render_mesh_views(
    mesh_input,
    num_views: int = 4,
    resolution: Tuple[int, int] = (512, 512),
    elevation_degrees: float = 30.0,
    use_pyrender: bool = False,
) -> List[bytes]:
    """
    Main entry point for mesh rendering.

    Args:
        mesh_input: Path to mesh file, mesh bytes, or trimesh.Trimesh object
        num_views: Number of views to render
        resolution: Output image resolution
        elevation_degrees: Camera elevation angle
        use_pyrender: Use pyrender for higher quality (requires GPU/EGL)

    Returns:
        List of PNG image bytes
    """
    import trimesh

    # Load mesh if needed
    if isinstance(mesh_input, (str, Path)):
        mesh = load_mesh(Path(mesh_input))
    elif isinstance(mesh_input, bytes):
        mesh = load_mesh_from_bytes(mesh_input)
    elif isinstance(mesh_input, trimesh.Trimesh):
        mesh = mesh_input
    else:
        print(f"[MESH] Unsupported mesh input type: {type(mesh_input)}")
        return []

    if mesh is None:
        print("[MESH] Failed to load mesh")
        return []

    # Render views
    if use_pyrender:
        return render_views_pyrender(mesh, num_views, resolution, elevation_degrees)
    else:
        return render_views_trimesh(mesh, num_views, resolution, elevation_degrees)


def save_views(images: List[bytes], output_dir: Path, prefix: str = "view") -> List[Path]:
    """
    Save rendered view images to disk.

    Args:
        images: List of PNG image bytes
        output_dir: Output directory
        prefix: Filename prefix

    Returns:
        List of saved file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    for i, img_bytes in enumerate(images):
        path = output_dir / f"{prefix}_{i}.png"
        path.write_bytes(img_bytes)
        paths.append(path)

    return paths


# =============================================================================
# CLI for Testing
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render mesh to multi-view images")
    parser.add_argument("mesh_path", type=str, help="Path to mesh file (GLB, GLTF, OBJ)")
    parser.add_argument("--output-dir", type=str, default="./views", help="Output directory")
    parser.add_argument("--num-views", type=int, default=4, help="Number of views")
    parser.add_argument("--resolution", type=int, default=512, help="Image resolution")
    parser.add_argument("--elevation", type=float, default=30.0, help="Camera elevation (degrees)")
    parser.add_argument("--use-pyrender", action="store_true", help="Use pyrender for rendering")

    args = parser.parse_args()

    print(f"Loading mesh: {args.mesh_path}")
    images = render_mesh_views(
        args.mesh_path,
        num_views=args.num_views,
        resolution=(args.resolution, args.resolution),
        elevation_degrees=args.elevation,
        use_pyrender=args.use_pyrender,
    )

    if images:
        output_dir = Path(args.output_dir)
        paths = save_views(images, output_dir)
        print(f"Saved {len(paths)} views to {output_dir}")
        for p in paths:
            print(f"  - {p}")
    else:
        print("Failed to render any views")
