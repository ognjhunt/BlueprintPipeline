#!/usr/bin/env python3
"""
Generate mock 3D-RE-GEN outputs for testing BlueprintPipeline.

This script creates a realistic 3D-RE-GEN output structure that can be used
for testing the pipeline without requiring actual 3D-RE-GEN reconstruction.

Usage:
    python fixtures/generate_mock_regen3d.py --scene-id test_kitchen --output-dir ./test_scenes
    python fixtures/generate_mock_regen3d.py --scene-id test_office --environment office

3D-RE-GEN (arXiv:2512.17459) is a modular, compositional pipeline for
"image → sim-ready 3D reconstruction" with explicit physical constraints.

Reference:
- Paper: https://arxiv.org/abs/2512.17459
- Project: https://3dregen.jdihlmann.com/
- GitHub: https://github.com/cgtuebingen/3D-RE-GEN
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional
import random
import math

# Object templates for different environment types
KITCHEN_OBJECTS = [
    {"id": "counter_0", "category": "counter", "size": [2.0, 0.9, 0.6], "position": [0, 0.45, -1.5], "is_floor_contact": True},
    {"id": "cabinet_0", "category": "cabinet", "size": [0.6, 0.7, 0.35], "position": [-0.8, 1.5, -1.5], "is_floor_contact": False},
    {"id": "cabinet_1", "category": "cabinet", "size": [0.6, 0.7, 0.35], "position": [0.0, 1.5, -1.5], "is_floor_contact": False},
    {"id": "fridge", "category": "refrigerator", "size": [0.9, 1.8, 0.7], "position": [1.5, 0.9, -1.5], "is_floor_contact": True},
    {"id": "dishwasher", "category": "dishwasher", "size": [0.6, 0.85, 0.6], "position": [-1.2, 0.425, -1.5], "is_floor_contact": True},
    {"id": "plate_0", "category": "plate", "size": [0.25, 0.02, 0.25], "position": [0, 0.91, -1.5], "is_floor_contact": False},
    {"id": "mug_0", "category": "mug", "size": [0.08, 0.1, 0.08], "position": [0.3, 0.95, -1.4], "is_floor_contact": False},
]

OFFICE_OBJECTS = [
    {"id": "desk_0", "category": "desk", "size": [1.5, 0.75, 0.8], "position": [0, 0.375, -1.2], "is_floor_contact": True},
    {"id": "chair_0", "category": "office_chair", "size": [0.6, 1.0, 0.6], "position": [0, 0.5, 0], "is_floor_contact": True},
    {"id": "monitor_0", "category": "monitor", "size": [0.6, 0.4, 0.05], "position": [0, 0.95, -1.4], "is_floor_contact": False},
    {"id": "keyboard_0", "category": "keyboard", "size": [0.45, 0.02, 0.15], "position": [0, 0.76, -1.0], "is_floor_contact": False},
    {"id": "cabinet_0", "category": "filing_cabinet", "size": [0.4, 1.2, 0.5], "position": [1.5, 0.6, -1.0], "is_floor_contact": True},
]

WAREHOUSE_OBJECTS = [
    {"id": "shelf_0", "category": "shelf", "size": [2.0, 2.0, 0.5], "position": [-2, 1.0, -2.0], "is_floor_contact": True},
    {"id": "shelf_1", "category": "shelf", "size": [2.0, 2.0, 0.5], "position": [2, 1.0, -2.0], "is_floor_contact": True},
    {"id": "box_0", "category": "box", "size": [0.4, 0.3, 0.4], "position": [-2, 0.5, -2.0], "is_floor_contact": False},
    {"id": "box_1", "category": "box", "size": [0.4, 0.3, 0.4], "position": [-2, 0.8, -2.0], "is_floor_contact": False},
    {"id": "pallet_0", "category": "pallet", "size": [1.2, 0.15, 1.0], "position": [0, 0.075, 0], "is_floor_contact": True},
]

ENVIRONMENT_OBJECTS = {
    "kitchen": KITCHEN_OBJECTS,
    "office": OFFICE_OBJECTS,
    "warehouse": WAREHOUSE_OBJECTS,
}


def create_minimal_glb() -> bytes:
    """Create a minimal valid GLB file (a simple cube)."""
    # This creates a minimal binary glTF file header
    # In production, you'd want actual mesh data

    # Minimal glTF JSON
    gltf_json = {
        "asset": {"version": "2.0", "generator": "mock-regen3d"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "Cube"}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 8, "type": "VEC3",
             "min": [-0.5, -0.5, -0.5], "max": [0.5, 0.5, 0.5]},
            {"bufferView": 1, "componentType": 5123, "count": 36, "type": "SCALAR"}
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 96},
            {"buffer": 0, "byteOffset": 96, "byteLength": 72}
        ],
        "buffers": [{"byteLength": 168}]
    }

    import struct

    # Cube vertices (8 corners)
    vertices = [
        -0.5, -0.5, -0.5,
         0.5, -0.5, -0.5,
         0.5,  0.5, -0.5,
        -0.5,  0.5, -0.5,
        -0.5, -0.5,  0.5,
         0.5, -0.5,  0.5,
         0.5,  0.5,  0.5,
        -0.5,  0.5,  0.5,
    ]

    # Cube indices (12 triangles = 6 faces)
    indices = [
        0, 1, 2, 0, 2, 3,  # Front
        4, 6, 5, 4, 7, 6,  # Back
        0, 4, 5, 0, 5, 1,  # Bottom
        2, 6, 7, 2, 7, 3,  # Top
        0, 3, 7, 0, 7, 4,  # Left
        1, 5, 6, 1, 6, 2,  # Right
    ]

    # Pack binary data
    vertex_data = struct.pack(f'{len(vertices)}f', *vertices)
    index_data = struct.pack(f'{len(indices)}H', *indices)
    binary_data = vertex_data + index_data

    # Create JSON chunk
    json_str = json.dumps(gltf_json, separators=(',', ':'))
    # Pad to 4-byte boundary
    while len(json_str) % 4 != 0:
        json_str += ' '
    json_bytes = json_str.encode('utf-8')

    # Create GLB header and chunks
    glb_header = struct.pack('<4sII', b'glTF', 2, 12 + 8 + len(json_bytes) + 8 + len(binary_data))
    json_chunk = struct.pack('<II', len(json_bytes), 0x4E4F534A) + json_bytes
    bin_chunk = struct.pack('<II', len(binary_data), 0x004E4942) + binary_data

    return glb_header + json_chunk + bin_chunk


def generate_transform_matrix(position: List[float], scale: List[float] = None) -> List[List[float]]:
    """Generate a 4x4 transform matrix."""
    scale = scale or [1.0, 1.0, 1.0]
    return [
        [scale[0], 0, 0, position[0]],
        [0, scale[1], 0, position[1]],
        [0, 0, scale[2], position[2]],
        [0, 0, 0, 1],
    ]


def generate_mock_regen3d(
    output_dir: Path,
    scene_id: str,
    environment_type: str = "kitchen",
    num_extra_objects: int = 0,
) -> Path:
    """Generate mock 3D-RE-GEN outputs.

    Args:
        output_dir: Base output directory
        scene_id: Scene identifier
        environment_type: Type of environment (kitchen, office, warehouse)
        num_extra_objects: Number of additional random objects to add

    Returns:
        Path to the generated regen3d directory
    """
    # Create scene directory structure
    scene_dir = output_dir / "scenes" / scene_id
    regen3d_dir = scene_dir / "regen3d"
    objects_dir = regen3d_dir / "objects"
    background_dir = regen3d_dir / "background"
    camera_dir = regen3d_dir / "camera"
    depth_dir = regen3d_dir / "depth"

    for d in [objects_dir, background_dir, camera_dir, depth_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Get objects for this environment type
    objects = ENVIRONMENT_OBJECTS.get(environment_type, KITCHEN_OBJECTS).copy()

    # Generate GLB data
    glb_data = create_minimal_glb()

    # Generate each object
    for obj in objects:
        obj_dir = objects_dir / obj["id"]
        obj_dir.mkdir(parents=True, exist_ok=True)

        # Write mesh
        (obj_dir / "mesh.glb").write_bytes(glb_data)

        # Write pose
        pose = {
            "transform_matrix": generate_transform_matrix(obj["position"]),
            "translation": obj["position"],
            "rotation_quaternion": [1.0, 0.0, 0.0, 0.0],
            "scale": [1.0, 1.0, 1.0],
            "confidence": 0.95,
            "is_floor_contact": obj.get("is_floor_contact", False),
        }
        (obj_dir / "pose.json").write_text(json.dumps(pose, indent=2))

        # Write bounds
        size = obj["size"]
        center = obj["position"]
        bounds = {
            "min": [center[0] - size[0]/2, center[1] - size[1]/2, center[2] - size[2]/2],
            "max": [center[0] + size[0]/2, center[1] + size[1]/2, center[2] + size[2]/2],
            "center": center,
            "size": size,
        }
        (obj_dir / "bounds.json").write_text(json.dumps(bounds, indent=2))

        # Write material
        material = {
            "base_color": [0.7, 0.7, 0.7],
            "metallic": 0.0,
            "roughness": 0.5,
            "material_type": "generic",
        }
        (obj_dir / "material.json").write_text(json.dumps(material, indent=2))

    # Generate background
    (background_dir / "mesh.glb").write_bytes(glb_data)
    bg_pose = {
        "transform_matrix": generate_transform_matrix([0, 0, 0]),
        "translation": [0, 0, 0],
        "rotation_quaternion": [1.0, 0.0, 0.0, 0.0],
        "scale": [1.0, 1.0, 1.0],
        "confidence": 1.0,
        "is_floor_contact": True,
    }
    (background_dir / "pose.json").write_text(json.dumps(bg_pose, indent=2))

    bg_bounds = {
        "min": [-5, 0, -5],
        "max": [5, 3, 5],
        "center": [0, 1.5, 0],
        "size": [10, 3, 10],
    }
    (background_dir / "bounds.json").write_text(json.dumps(bg_bounds, indent=2))

    # Generate camera parameters
    intrinsics = {
        "matrix": [
            [1000, 0, 960],
            [0, 1000, 540],
            [0, 0, 1],
        ],
        "width": 1920,
        "height": 1080,
    }
    (camera_dir / "intrinsics.json").write_text(json.dumps(intrinsics, indent=2))

    extrinsics = {
        "matrix": [
            [1, 0, 0, 0],
            [0, 1, 0, 1.5],
            [0, 0, 1, 3],
            [0, 0, 0, 1],
        ],
    }
    (camera_dir / "extrinsics.json").write_text(json.dumps(extrinsics, indent=2))

    # Generate scene info
    scene_info = {
        "scene_id": scene_id,
        "image_size": [1920, 1080],
        "coordinate_frame": "y_up",
        "meters_per_unit": 1.0,
        "confidence": 0.92,
        "version": "1.0",
        "environment_type": environment_type,
        "reconstruction_method": "3d-re-gen",
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
    (regen3d_dir / "scene_info.json").write_text(json.dumps(scene_info, indent=2))

    print(f"[MOCK-REGEN3D] Generated mock 3D-RE-GEN outputs for scene: {scene_id}")
    print(f"[MOCK-REGEN3D]   Output directory: {regen3d_dir}")
    print(f"[MOCK-REGEN3D]   Environment type: {environment_type}")
    print(f"[MOCK-REGEN3D]   Objects: {len(objects)}")

    return regen3d_dir


def main():
    parser = argparse.ArgumentParser(
        description="Generate mock 3D-RE-GEN outputs for testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--scene-id",
        default="test_kitchen",
        help="Scene identifier",
    )
    parser.add_argument(
        "--output-dir",
        default="./test_scenes",
        help="Output directory",
    )
    parser.add_argument(
        "--environment",
        default="kitchen",
        choices=["kitchen", "office", "warehouse"],
        help="Environment type",
    )
    parser.add_argument(
        "--extra-objects",
        type=int,
        default=0,
        help="Number of extra random objects to add",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    generate_mock_regen3d(
        output_dir=output_dir,
        scene_id=args.scene_id,
        environment_type=args.environment,
        num_extra_objects=args.extra_objects,
    )


if __name__ == "__main__":
    main()
