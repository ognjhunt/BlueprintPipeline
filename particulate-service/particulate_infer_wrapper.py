#!/usr/bin/env python3
"""
Particulate Inference Wrapper for BlueprintPipeline.

This script wraps the Particulate model to:
1. Load a GLB/OBJ mesh
2. Run feed-forward articulation inference
3. Export segmented mesh + URDF

Usage:
    python particulate_infer_wrapper.py \
        --input_mesh ./input.glb \
        --output_dir ./output \
        [--up_dir Y] \
        [--export_urdf] \
        [--export_glb]

Based on Particulate: Feed-Forward 3D Object Articulation
Paper: arXiv:2512.11798
"""

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET
from xml.dom import minidom

import numpy as np
import torch
import trimesh
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

# Add parent directory to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def log(msg: str, level: str = "INFO") -> None:
    """Log with prefix."""
    level_name = level.upper()
    log_level = getattr(logging, level_name, logging.INFO)
    logger.log(log_level, "[PARTICULATE] [%s] %s", level_name, msg)


# =============================================================================
# Model Loading
# =============================================================================

def load_model(config_path: Optional[Path] = None, device: str = "cuda"):
    """
    Load the Particulate model.

    Returns:
        Loaded model ready for inference
    """
    # Download checkpoint from HuggingFace
    log("Downloading model checkpoint from HuggingFace...")
    checkpoint_path = hf_hub_download(
        repo_id="rayli/Particulate",
        filename="model.pt"
    )
    log(f"Checkpoint: {checkpoint_path}")

    # Load config
    if config_path is None:
        config_path = SCRIPT_DIR / "configs" / "pat_B.yaml"

    if not config_path.exists():
        # Try to find config
        for candidate in [
            SCRIPT_DIR / "configs" / "pat_B.yaml",
            SCRIPT_DIR.parent / "configs" / "pat_B.yaml",
            Path("/opt/particulate/configs/pat_B.yaml"),
        ]:
            if candidate.exists():
                config_path = candidate
                break

    log(f"Loading config from: {config_path}")
    cfg = OmegaConf.load(config_path)

    # Import model class
    from particulate.models import PAT_B

    # Create and load model
    model = PAT_B(**cfg)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.to(device)
    model.eval()

    log(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    return model


# =============================================================================
# Mesh Processing
# =============================================================================

def load_mesh(mesh_path: Path) -> trimesh.Trimesh:
    """Load a mesh from GLB/OBJ file."""
    log(f"Loading mesh: {mesh_path}")

    mesh = trimesh.load(str(mesh_path), force='mesh')

    if isinstance(mesh, trimesh.Scene):
        # Concatenate all geometries
        geometries = list(mesh.geometry.values())
        if geometries:
            mesh = trimesh.util.concatenate(geometries)
        else:
            raise ValueError(f"No geometry found in {mesh_path}")

    log(f"Mesh loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    return mesh


def normalize_mesh(mesh: trimesh.Trimesh, up_dir: str = "Y") -> Tuple[trimesh.Trimesh, np.ndarray, float]:
    """
    Normalize mesh to [-0.5, 0.5]^3 bounding box.

    Args:
        mesh: Input mesh
        up_dir: Up direction ("X", "Y", "Z", "-X", "-Y", "-Z")

    Returns:
        (normalized_mesh, center, scale)
    """
    # Apply rotation to canonical up direction (Z-up for Particulate)
    rotation = np.eye(4)
    if up_dir == "Y":
        # Y-up to Z-up: rotate -90 degrees around X
        rotation[:3, :3] = trimesh.transformations.rotation_matrix(
            -np.pi / 2, [1, 0, 0]
        )[:3, :3]
    elif up_dir == "X":
        rotation[:3, :3] = trimesh.transformations.rotation_matrix(
            -np.pi / 2, [0, 1, 0]
        )[:3, :3]
    elif up_dir == "-Z":
        rotation[:3, :3] = trimesh.transformations.rotation_matrix(
            np.pi, [1, 0, 0]
        )[:3, :3]

    mesh = mesh.copy()
    mesh.apply_transform(rotation)

    # Normalize to [-0.5, 0.5]^3
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2
    scale = np.max(bounds[1] - bounds[0])

    mesh.vertices = (mesh.vertices - center) / scale

    return mesh, center, scale


def sample_points(mesh: trimesh.Trimesh, num_points: int = 40000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample points from mesh surface.

    50% uniform sampling, 50% from sharp edges.

    Returns:
        (points, normals) arrays of shape (num_points, 3)
    """
    # Uniform sampling
    num_uniform = num_points // 2
    points_uniform, face_indices = trimesh.sample.sample_surface(mesh, num_uniform)

    # Get normals for uniform samples
    normals_uniform = mesh.face_normals[face_indices]

    # Edge sampling (from high-curvature regions)
    num_edge = num_points - num_uniform
    try:
        # Find edges with high dihedral angles
        edges = mesh.edges_unique
        edge_angles = mesh.face_adjacency_angles

        # Sample more from sharp edges
        if len(edge_angles) > 0:
            sharp_mask = edge_angles > np.radians(30)  # >30 degree edges
            sharp_edges = mesh.face_adjacency_edges[sharp_mask] if np.any(sharp_mask) else edges[:100]

            if len(sharp_edges) > 0:
                # Sample points along sharp edges
                edge_vertices = mesh.vertices[sharp_edges]
                t = np.random.random((num_edge, 1))
                points_edge = edge_vertices[:, 0] * t + edge_vertices[:, 1] * (1 - t)

                # Use vertex normals averaged
                normals_edge = np.zeros_like(points_edge)
                for i, (v1, v2) in enumerate(sharp_edges[:num_edge]):
                    normals_edge[i] = (mesh.vertex_normals[v1] + mesh.vertex_normals[v2]) / 2
            else:
                points_edge, fi = trimesh.sample.sample_surface(mesh, num_edge)
                normals_edge = mesh.face_normals[fi]
        else:
            points_edge, fi = trimesh.sample.sample_surface(mesh, num_edge)
            normals_edge = mesh.face_normals[fi]

    except Exception:
        # Fallback to uniform sampling
        points_edge, fi = trimesh.sample.sample_surface(mesh, num_edge)
        normals_edge = mesh.face_normals[fi]

    # Combine
    points = np.vstack([points_uniform, points_edge[:num_edge]])
    normals = np.vstack([normals_uniform, normals_edge[:num_edge]])

    # Shuffle
    perm = np.random.permutation(len(points))
    points = points[perm]
    normals = normals[perm]

    return points.astype(np.float32), normals.astype(np.float32)


# =============================================================================
# Inference
# =============================================================================

def run_inference(
    model,
    mesh: trimesh.Trimesh,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Run Particulate inference on a mesh.

    Returns:
        Dictionary with:
        - part_ids: (N,) part assignments for each sampled point
        - motion_hierarchy: Dict[child_id -> parent_id]
        - is_part_revolute: Dict[part_id -> bool]
        - is_part_prismatic: Dict[part_id -> bool]
        - revolute_plucker: Dict[part_id -> (6,) Plucker coords]
        - revolute_range: Dict[part_id -> (lower, upper)]
        - prismatic_axis: Dict[part_id -> (3,) direction]
        - prismatic_range: Dict[part_id -> (lower, upper)]
    """
    log("Sampling points from mesh...")
    points, normals = sample_points(mesh, num_points=40000)

    # Prepare input tensor
    # Particulate expects: (batch, num_points, 6) for xyz + normals
    xyz = torch.from_numpy(points).unsqueeze(0).to(device)
    norm = torch.from_numpy(normals).unsqueeze(0).to(device)

    log(f"Running inference on {len(points)} points...")

    with torch.no_grad():
        # Model forward pass
        # Output structure depends on model architecture
        outputs = model(xyz, norm, output_all_hyps=True)

    # Parse outputs
    result = parse_model_outputs(outputs, points)
    log(f"Inference complete: {len(result.get('unique_part_ids', []))} parts detected")

    return result


def parse_model_outputs(outputs: Dict[str, torch.Tensor], points: np.ndarray) -> Dict[str, Any]:
    """Parse model outputs into structured articulation data."""
    result = {
        "part_ids": None,
        "unique_part_ids": [],
        "motion_hierarchy": {},
        "is_part_revolute": {},
        "is_part_prismatic": {},
        "revolute_plucker": {},
        "revolute_range": {},
        "prismatic_axis": {},
        "prismatic_range": {},
    }

    # Extract part segmentation
    if "part_ids" in outputs:
        part_ids = outputs["part_ids"].cpu().numpy().squeeze()
        result["part_ids"] = part_ids
        result["unique_part_ids"] = list(np.unique(part_ids))

    # Extract motion hierarchy
    if "motion_hierarchy" in outputs:
        hierarchy = outputs["motion_hierarchy"]
        if isinstance(hierarchy, torch.Tensor):
            # Convert tensor to dict
            hierarchy = hierarchy.cpu().numpy()
            for i in range(len(hierarchy)):
                if hierarchy[i] >= 0:
                    result["motion_hierarchy"][i] = int(hierarchy[i])
        elif isinstance(hierarchy, dict):
            result["motion_hierarchy"] = {int(k): int(v) for k, v in hierarchy.items()}

    # Extract joint types
    for part_id in result["unique_part_ids"]:
        # Revolute (rotation) joints
        if "is_part_revolute" in outputs:
            rev = outputs["is_part_revolute"]
            if isinstance(rev, torch.Tensor):
                result["is_part_revolute"][part_id] = bool(rev[part_id].item()) if part_id < len(rev) else False
            elif isinstance(rev, dict):
                result["is_part_revolute"][part_id] = rev.get(part_id, False)
        else:
            result["is_part_revolute"][part_id] = False

        # Prismatic (sliding) joints
        if "is_part_prismatic" in outputs:
            pris = outputs["is_part_prismatic"]
            if isinstance(pris, torch.Tensor):
                result["is_part_prismatic"][part_id] = bool(pris[part_id].item()) if part_id < len(pris) else False
            elif isinstance(pris, dict):
                result["is_part_prismatic"][part_id] = pris.get(part_id, False)
        else:
            result["is_part_prismatic"][part_id] = False

    # Extract axis information
    if "revolute_plucker" in outputs:
        plucker = outputs["revolute_plucker"].cpu().numpy()
        for part_id in result["unique_part_ids"]:
            if part_id < len(plucker):
                result["revolute_plucker"][part_id] = plucker[part_id].tolist()

    if "prismatic_axis" in outputs:
        paxis = outputs["prismatic_axis"].cpu().numpy()
        for part_id in result["unique_part_ids"]:
            if part_id < len(paxis):
                result["prismatic_axis"][part_id] = paxis[part_id].tolist()

    # Extract ranges
    if "revolute_range" in outputs:
        rrange = outputs["revolute_range"].cpu().numpy()
        for part_id in result["unique_part_ids"]:
            if part_id < len(rrange):
                result["revolute_range"][part_id] = tuple(rrange[part_id].tolist())

    if "prismatic_range" in outputs:
        prange = outputs["prismatic_range"].cpu().numpy()
        for part_id in result["unique_part_ids"]:
            if part_id < len(prange):
                result["prismatic_range"][part_id] = tuple(prange[part_id].tolist())

    return result


# =============================================================================
# Mesh Segmentation
# =============================================================================

def segment_mesh(
    mesh: trimesh.Trimesh,
    points: np.ndarray,
    part_ids: np.ndarray,
) -> Dict[int, trimesh.Trimesh]:
    """
    Segment mesh into parts based on point-wise predictions.

    Args:
        mesh: Original mesh
        points: Sampled points (N, 3)
        part_ids: Part assignments (N,)

    Returns:
        Dict mapping part_id -> submesh
    """
    log("Segmenting mesh into parts...")

    # Assign faces to parts based on nearest sampled point
    face_centers = mesh.triangles_center
    unique_parts = np.unique(part_ids)

    # Build KD-tree for fast lookup
    from scipy.spatial import cKDTree
    tree = cKDTree(points)

    # Find nearest sampled point for each face
    _, indices = tree.query(face_centers, k=1)
    face_parts = part_ids[indices]

    # Create submeshes
    submeshes = {}
    for part_id in unique_parts:
        face_mask = face_parts == part_id
        if not np.any(face_mask):
            continue

        # Extract faces for this part
        part_faces = mesh.faces[face_mask]

        # Get unique vertices used by these faces
        unique_verts = np.unique(part_faces.flatten())

        # Remap vertex indices
        vert_map = {old: new for new, old in enumerate(unique_verts)}
        new_faces = np.array([[vert_map[v] for v in face] for face in part_faces])

        # Create submesh
        submesh = trimesh.Trimesh(
            vertices=mesh.vertices[unique_verts],
            faces=new_faces,
        )
        submeshes[int(part_id)] = submesh
        log(f"  Part {part_id}: {len(submesh.vertices)} vertices, {len(submesh.faces)} faces")

    return submeshes


# =============================================================================
# URDF Export
# =============================================================================

def plucker_to_axis_point(plucker: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert Plucker coordinates to axis direction and point on axis.

    Plucker: (d, m) where d is direction, m is moment (d x p for point p on line)
    """
    plucker = np.array(plucker)
    direction = plucker[:3]
    moment = plucker[3:]

    # Normalize direction
    d_norm = np.linalg.norm(direction)
    if d_norm < 1e-6:
        return np.array([0, 0, 1]), np.array([0, 0, 0])

    direction = direction / d_norm

    # Find point on line: p = d x m / |d|^2
    point = np.cross(direction, moment)

    return direction, point


def export_urdf(
    submeshes: Dict[int, trimesh.Trimesh],
    articulation: Dict[str, Any],
    output_dir: Path,
    robot_name: str = "robot",
) -> Path:
    """
    Export articulated mesh to URDF.

    Args:
        submeshes: Dict mapping part_id -> submesh
        articulation: Articulation data from inference
        output_dir: Output directory
        robot_name: Name for the robot

    Returns:
        Path to URDF file
    """
    log(f"Exporting URDF to {output_dir}")

    meshes_dir = output_dir / "meshes"
    meshes_dir.mkdir(parents=True, exist_ok=True)

    unique_parts = list(submeshes.keys())
    motion_hierarchy = articulation.get("motion_hierarchy", {})

    # Write mesh files
    for part_id, submesh in submeshes.items():
        mesh_path = meshes_dir / f"part_{part_id}.obj"
        submesh.export(str(mesh_path))

    # Build URDF
    robot = ET.Element("robot", name=robot_name)

    # Find root parts
    child_ids = set(motion_hierarchy.keys())
    root_ids = [pid for pid in unique_parts if pid not in child_ids]
    if not root_ids:
        root_ids = [min(unique_parts)] if unique_parts else [0]

    # Create links
    for part_id in unique_parts:
        link = ET.SubElement(robot, "link", name=f"link_{part_id}")

        # Visual
        visual = ET.SubElement(link, "visual")
        visual_geom = ET.SubElement(visual, "geometry")
        ET.SubElement(visual_geom, "mesh", filename=f"meshes/part_{part_id}.obj")

        # Collision
        collision = ET.SubElement(link, "collision")
        collision_geom = ET.SubElement(collision, "geometry")
        ET.SubElement(collision_geom, "mesh", filename=f"meshes/part_{part_id}.obj")

        # Inertial
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "mass", value="1.0")
        ET.SubElement(inertial, "inertia",
                      ixx="0.1", ixy="0", ixz="0",
                      iyy="0.1", iyz="0", izz="0.1")

    # Create joints
    is_revolute = articulation.get("is_part_revolute", {})
    is_prismatic = articulation.get("is_part_prismatic", {})
    revolute_plucker = articulation.get("revolute_plucker", {})
    prismatic_axis = articulation.get("prismatic_axis", {})
    revolute_range = articulation.get("revolute_range", {})
    prismatic_range = articulation.get("prismatic_range", {})

    for child_id, parent_id in motion_hierarchy.items():
        if parent_id not in unique_parts or child_id not in unique_parts:
            continue

        # Determine joint type
        if is_revolute.get(child_id, False):
            joint_type = "revolute"
            if child_id in revolute_plucker:
                axis, _ = plucker_to_axis_point(revolute_plucker[child_id])
            else:
                axis = np.array([0, 0, 1])
            limits = revolute_range.get(child_id, (-3.14159, 3.14159))
        elif is_prismatic.get(child_id, False):
            joint_type = "prismatic"
            axis = np.array(prismatic_axis.get(child_id, [0, 0, 1]))
            limits = prismatic_range.get(child_id, (-0.5, 0.5))
        else:
            joint_type = "fixed"
            axis = np.array([0, 0, 1])
            limits = (0, 0)

        # Create joint element
        joint = ET.SubElement(robot, "joint",
                              name=f"joint_{parent_id}_{child_id}",
                              type=joint_type)

        ET.SubElement(joint, "parent", link=f"link_{parent_id}")
        ET.SubElement(joint, "child", link=f"link_{child_id}")
        ET.SubElement(joint, "origin", xyz="0 0 0", rpy="0 0 0")

        if joint_type != "fixed":
            # Normalize axis
            axis_norm = np.linalg.norm(axis)
            if axis_norm > 1e-6:
                axis = axis / axis_norm
            else:
                axis = np.array([0, 0, 1])

            ET.SubElement(joint, "axis", xyz=f"{axis[0]:.6f} {axis[1]:.6f} {axis[2]:.6f}")
            ET.SubElement(joint, "limit",
                          lower=str(limits[0]),
                          upper=str(limits[1]),
                          effort="1000",
                          velocity="100")

    # Format and write URDF
    urdf_str = ET.tostring(robot, encoding="unicode")
    urdf_pretty = minidom.parseString(urdf_str).toprettyxml(indent="  ")
    urdf_lines = [line for line in urdf_pretty.split("\n") if line.strip()]
    urdf_pretty = "\n".join(urdf_lines)

    urdf_path = output_dir / f"{robot_name}.urdf"
    urdf_path.write_text(urdf_pretty, encoding="utf-8")

    log(f"URDF exported: {urdf_path}")
    return urdf_path


# =============================================================================
# GLB Export
# =============================================================================

def export_segmented_glb(
    submeshes: Dict[int, trimesh.Trimesh],
    output_path: Path,
) -> Path:
    """Export segmented mesh as colored GLB."""
    log(f"Exporting segmented GLB to {output_path}")

    # Create scene with colored parts
    scene = trimesh.Scene()

    # Color palette
    colors = [
        [255, 0, 0, 255],      # Red
        [0, 255, 0, 255],      # Green
        [0, 0, 255, 255],      # Blue
        [255, 255, 0, 255],    # Yellow
        [255, 0, 255, 255],    # Magenta
        [0, 255, 255, 255],    # Cyan
        [255, 128, 0, 255],    # Orange
        [128, 0, 255, 255],    # Purple
    ]

    for i, (part_id, submesh) in enumerate(submeshes.items()):
        color = colors[i % len(colors)]
        submesh.visual.face_colors = np.tile(color, (len(submesh.faces), 1))
        scene.add_geometry(submesh, node_name=f"part_{part_id}")

    scene.export(str(output_path))
    log(f"GLB exported: {output_path}")
    return output_path


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Particulate Inference Wrapper")
    parser.add_argument("--input_mesh", type=str, required=True, help="Input mesh file (GLB/OBJ)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--up_dir", type=str, default="Y", choices=["X", "Y", "Z", "-X", "-Y", "-Z"],
                        help="Up direction of input mesh")
    parser.add_argument("--export_urdf", action="store_true", help="Export URDF file")
    parser.add_argument("--export_glb", action="store_true", help="Export segmented GLB")
    parser.add_argument("--config", type=str, default=None, help="Model config path")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")

    args = parser.parse_args()

    input_path = Path(args.input_mesh)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load mesh
    mesh = load_mesh(input_path)

    # Normalize mesh
    normalized_mesh, center, scale = normalize_mesh(mesh, args.up_dir)

    # Load model
    config_path = Path(args.config) if args.config else None
    model = load_model(config_path, args.device)

    # Run inference
    articulation = run_inference(model, normalized_mesh, args.device)

    # Sample points for segmentation
    points, _ = sample_points(normalized_mesh, num_points=40000)
    part_ids = articulation.get("part_ids", np.zeros(len(points), dtype=int))

    # Segment mesh
    submeshes = segment_mesh(normalized_mesh, points, part_ids)

    # Count articulated joints
    motion_hierarchy = articulation.get("motion_hierarchy", {})
    is_revolute = articulation.get("is_part_revolute", {})
    is_prismatic = articulation.get("is_part_prismatic", {})
    joint_count = sum(1 for cid in motion_hierarchy if is_revolute.get(cid) or is_prismatic.get(cid))

    # Export metadata
    metadata = {
        "part_count": len(submeshes),
        "joint_count": joint_count,
        "is_articulated": joint_count > 0,
        "unique_part_ids": articulation.get("unique_part_ids", []),
        "motion_hierarchy": articulation.get("motion_hierarchy", {}),
    }
    metadata_path = output_dir / "articulation.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    log(f"Metadata exported: {metadata_path}")

    # Export URDF
    if args.export_urdf or True:  # Always export URDF
        export_urdf(submeshes, articulation, output_dir, "robot")

    # Export segmented GLB
    if args.export_glb or True:  # Always export GLB
        export_segmented_glb(submeshes, output_dir / "segmented.glb")

    log("Inference complete!")
    log(f"  Parts: {len(submeshes)}")
    log(f"  Joints: {joint_count}")
    log(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
