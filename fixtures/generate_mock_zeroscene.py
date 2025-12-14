#!/usr/bin/env python3
"""
Generate Mock ZeroScene Output for Testing.

Creates realistic ZeroScene-format outputs for pipeline testing when
the actual ZeroScene model code isn't available.

Usage:
    python fixtures/generate_mock_zeroscene.py --scene-id test_kitchen --output-dir ./test_scenes

This generates:
    output_dir/
        scenes/{scene_id}/
            zeroscene/
                scene_info.json
                objects/
                    obj_0/ (cabinet with door)
                    obj_1/ (counter)
                    obj_2/ (plate - manipulable)
                background/
                    mesh.glb
                camera/
                depth/
            images/
                room.jpg
"""

import argparse
import json
import struct
import zlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import math


# =============================================================================
# Mock Mesh Generation (Minimal GLB)
# =============================================================================


def create_minimal_glb(
    vertices: List[Tuple[float, float, float]],
    indices: List[Tuple[int, int, int]],
    name: str = "mesh",
) -> bytes:
    """Create a minimal valid GLB file with given vertices and indices.

    This creates a bare-minimum GLB that can be loaded by trimesh and
    converted to USD by the pipeline.
    """
    # Flatten vertices and indices
    flat_vertices = []
    for v in vertices:
        flat_vertices.extend(v)

    flat_indices = []
    for tri in indices:
        flat_indices.extend(tri)

    # Calculate bounds
    min_bounds = [min(v[i] for v in vertices) for i in range(3)]
    max_bounds = [max(v[i] for v in vertices) for i in range(3)]

    # Pack vertex data (float32)
    vertex_data = struct.pack(f'{len(flat_vertices)}f', *flat_vertices)

    # Pack index data (uint16)
    index_data = struct.pack(f'{len(flat_indices)}H', *flat_indices)

    # Pad to 4-byte alignment
    vertex_padding = (4 - len(vertex_data) % 4) % 4
    index_padding = (4 - len(index_data) % 4) % 4

    vertex_data += b'\x00' * vertex_padding
    index_data += b'\x00' * index_padding

    # Create buffer
    buffer_data = vertex_data + index_data

    # Create JSON structure
    gltf = {
        "asset": {"version": "2.0", "generator": "BlueprintPipeline Mock"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": name}],
        "meshes": [{
            "name": name,
            "primitives": [{
                "attributes": {"POSITION": 0},
                "indices": 1,
                "mode": 4  # TRIANGLES
            }]
        }],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,  # FLOAT
                "count": len(vertices),
                "type": "VEC3",
                "min": min_bounds,
                "max": max_bounds,
            },
            {
                "bufferView": 1,
                "componentType": 5123,  # UNSIGNED_SHORT
                "count": len(flat_indices),
                "type": "SCALAR",
            }
        ],
        "bufferViews": [
            {
                "buffer": 0,
                "byteOffset": 0,
                "byteLength": len(flat_vertices) * 4,
                "target": 34962  # ARRAY_BUFFER
            },
            {
                "buffer": 0,
                "byteOffset": len(vertex_data),
                "byteLength": len(flat_indices) * 2,
                "target": 34963  # ELEMENT_ARRAY_BUFFER
            }
        ],
        "buffers": [{"byteLength": len(buffer_data)}]
    }

    json_str = json.dumps(gltf, separators=(',', ':'))
    json_bytes = json_str.encode('utf-8')

    # Pad JSON to 4-byte alignment
    json_padding = (4 - len(json_bytes) % 4) % 4
    json_bytes += b' ' * json_padding

    # Build GLB
    # Header: magic, version, length
    # JSON chunk: length, type, data
    # BIN chunk: length, type, data

    total_length = 12 + 8 + len(json_bytes) + 8 + len(buffer_data)

    glb = bytearray()

    # Header
    glb.extend(b'glTF')  # magic
    glb.extend(struct.pack('<I', 2))  # version
    glb.extend(struct.pack('<I', total_length))  # length

    # JSON chunk
    glb.extend(struct.pack('<I', len(json_bytes)))  # chunk length
    glb.extend(b'JSON')  # chunk type
    glb.extend(json_bytes)

    # BIN chunk
    glb.extend(struct.pack('<I', len(buffer_data)))  # chunk length
    glb.extend(b'BIN\x00')  # chunk type
    glb.extend(buffer_data)

    return bytes(glb)


def create_box_mesh(
    center: Tuple[float, float, float] = (0, 0, 0),
    size: Tuple[float, float, float] = (1, 1, 1),
) -> Tuple[List[Tuple[float, float, float]], List[Tuple[int, int, int]]]:
    """Create a simple box mesh."""
    hx, hy, hz = size[0] / 2, size[1] / 2, size[2] / 2
    cx, cy, cz = center

    vertices = [
        (cx - hx, cy - hy, cz - hz),
        (cx + hx, cy - hy, cz - hz),
        (cx + hx, cy + hy, cz - hz),
        (cx - hx, cy + hy, cz - hz),
        (cx - hx, cy - hy, cz + hz),
        (cx + hx, cy - hy, cz + hz),
        (cx + hx, cy + hy, cz + hz),
        (cx - hx, cy + hy, cz + hz),
    ]

    indices = [
        # Front
        (0, 1, 2), (0, 2, 3),
        # Back
        (5, 4, 7), (5, 7, 6),
        # Left
        (4, 0, 3), (4, 3, 7),
        # Right
        (1, 5, 6), (1, 6, 2),
        # Top
        (3, 2, 6), (3, 6, 7),
        # Bottom
        (4, 5, 1), (4, 1, 0),
    ]

    return vertices, indices


def create_cylinder_mesh(
    center: Tuple[float, float, float] = (0, 0, 0),
    radius: float = 0.5,
    height: float = 1.0,
    segments: int = 12,
) -> Tuple[List[Tuple[float, float, float]], List[Tuple[int, int, int]]]:
    """Create a simple cylinder mesh (for plates, cups, etc.)."""
    cx, cy, cz = center
    half_h = height / 2

    vertices = []
    indices = []

    # Top center
    vertices.append((cx, cy + half_h, cz))
    # Bottom center
    vertices.append((cx, cy - half_h, cz))

    # Generate ring vertices
    for i in range(segments):
        angle = 2 * math.pi * i / segments
        x = cx + radius * math.cos(angle)
        z = cz + radius * math.sin(angle)
        vertices.append((x, cy + half_h, z))  # Top ring
        vertices.append((x, cy - half_h, z))  # Bottom ring

    # Generate faces
    for i in range(segments):
        top_curr = 2 + i * 2
        bot_curr = 3 + i * 2
        top_next = 2 + ((i + 1) % segments) * 2
        bot_next = 3 + ((i + 1) % segments) * 2

        # Top face
        indices.append((0, top_curr, top_next))
        # Bottom face
        indices.append((1, bot_next, bot_curr))
        # Side faces
        indices.append((top_curr, bot_curr, bot_next))
        indices.append((top_curr, bot_next, top_next))

    return vertices, indices


# =============================================================================
# Mock Scene Objects
# =============================================================================


# Kitchen scene objects
KITCHEN_OBJECTS = [
    {
        "id": "0",
        "category": "cabinet",
        "description": "Upper kitchen cabinet with door",
        "position": [-1.5, 1.5, 0.0],
        "size": [0.6, 0.8, 0.4],
        "sim_role": "articulated_furniture",
        "articulation_hint": "revolute",
        "material": {"type": "wood", "roughness": 0.6, "metallic": 0.0},
    },
    {
        "id": "1",
        "category": "counter",
        "description": "Kitchen counter with granite top",
        "position": [-1.5, 0.45, 0.0],
        "size": [0.6, 0.9, 0.6],
        "sim_role": "static",
        "material": {"type": "stone", "roughness": 0.3, "metallic": 0.0},
    },
    {
        "id": "2",
        "category": "refrigerator",
        "description": "Stainless steel refrigerator",
        "position": [1.5, 0.9, 0.0],
        "size": [0.8, 1.8, 0.7],
        "sim_role": "articulated_appliance",
        "articulation_hint": "revolute",
        "material": {"type": "metal", "roughness": 0.2, "metallic": 0.8},
    },
    {
        "id": "3",
        "category": "drawer_cabinet",
        "description": "Lower cabinet with drawers",
        "position": [0.0, 0.45, 0.0],
        "size": [0.6, 0.9, 0.6],
        "sim_role": "articulated_furniture",
        "articulation_hint": "prismatic",
        "material": {"type": "wood", "roughness": 0.5, "metallic": 0.0},
    },
    {
        "id": "4",
        "category": "plate",
        "description": "White ceramic dinner plate",
        "position": [-1.5, 1.0, 0.1],
        "size": [0.25, 0.02, 0.25],
        "sim_role": "manipulable_object",
        "material": {"type": "ceramic", "roughness": 0.1, "metallic": 0.0},
        "mesh_type": "cylinder",
    },
    {
        "id": "5",
        "category": "mug",
        "description": "Coffee mug",
        "position": [-1.3, 1.0, 0.15],
        "size": [0.08, 0.1, 0.08],
        "sim_role": "manipulable_object",
        "material": {"type": "ceramic", "roughness": 0.2, "metallic": 0.0},
        "mesh_type": "cylinder",
    },
]


def generate_scene_info(scene_id: str, environment_type: str) -> Dict[str, Any]:
    """Generate scene_info.json content."""
    return {
        "scene_id": scene_id,
        "version": "1.0",
        "coordinate_frame": "y_up",
        "meters_per_unit": 1.0,
        "image_size": [1920, 1080],
        "environment_type": environment_type,
        "source_image_path": f"scenes/{scene_id}/images/room.jpg",
        "confidence": 0.95,
        "reconstruction_method": "zeroscene",
        "created_at": datetime.utcnow().isoformat() + "Z",
    }


def generate_object_files(
    obj: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Generate all files for a single object."""
    obj_id = obj["id"]
    obj_dir = output_dir / f"obj_{obj_id}"
    obj_dir.mkdir(parents=True, exist_ok=True)

    position = obj["position"]
    size = obj["size"]

    # Generate mesh
    mesh_type = obj.get("mesh_type", "box")
    if mesh_type == "cylinder":
        vertices, indices = create_cylinder_mesh(
            center=(0, 0, 0),
            radius=size[0] / 2,
            height=size[1],
        )
    else:
        vertices, indices = create_box_mesh(
            center=(0, 0, 0),
            size=tuple(size),
        )

    glb_data = create_minimal_glb(vertices, indices, obj["category"])
    (obj_dir / "mesh.glb").write_bytes(glb_data)

    # Generate pose.json (identity rotation, translated to position)
    pose = {
        "transform_matrix": [
            [1.0, 0.0, 0.0, position[0]],
            [0.0, 1.0, 0.0, position[1]],
            [0.0, 0.0, 1.0, position[2]],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "translation": position,
        "rotation_quaternion": [1.0, 0.0, 0.0, 0.0],  # w, x, y, z
        "scale": [1.0, 1.0, 1.0],
        "confidence": 0.95,
    }
    (obj_dir / "pose.json").write_text(json.dumps(pose, indent=2))

    # Generate bounds.json
    half_size = [s / 2 for s in size]
    bounds = {
        "min": [position[i] - half_size[i] for i in range(3)],
        "max": [position[i] + half_size[i] for i in range(3)],
        "center": position,
        "size": size,
    }
    (obj_dir / "bounds.json").write_text(json.dumps(bounds, indent=2))

    # Generate material.json
    material_info = obj.get("material", {})
    material = {
        "base_color": [0.8, 0.8, 0.8],
        "metallic": material_info.get("metallic", 0.0),
        "roughness": material_info.get("roughness", 0.5),
        "material_type": material_info.get("type", "generic"),
    }
    (obj_dir / "material.json").write_text(json.dumps(material, indent=2))


def generate_background(output_dir: Path) -> None:
    """Generate background/room shell mesh."""
    bg_dir = output_dir / "background"
    bg_dir.mkdir(parents=True, exist_ok=True)

    # Create a simple room box (inverted)
    room_size = (6.0, 3.0, 5.0)
    vertices, indices = create_box_mesh(center=(0, 1.5, 0), size=room_size)

    # Invert normals by reversing triangle winding
    indices = [(i[0], i[2], i[1]) for i in indices]

    glb_data = create_minimal_glb(vertices, indices, "room_shell")
    (bg_dir / "mesh.glb").write_bytes(glb_data)

    # Pose (identity)
    pose = {
        "transform_matrix": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }
    (bg_dir / "pose.json").write_text(json.dumps(pose, indent=2))

    # Bounds
    bounds = {
        "min": [-3.0, 0.0, -2.5],
        "max": [3.0, 3.0, 2.5],
        "center": [0.0, 1.5, 0.0],
        "size": list(room_size),
    }
    (bg_dir / "bounds.json").write_text(json.dumps(bounds, indent=2))

    # Material
    material = {
        "base_color": [0.9, 0.9, 0.85],
        "metallic": 0.0,
        "roughness": 0.8,
        "material_type": "wall_paint",
    }
    (bg_dir / "material.json").write_text(json.dumps(material, indent=2))


def generate_camera(output_dir: Path, image_size: Tuple[int, int] = (1920, 1080)) -> None:
    """Generate camera parameters."""
    camera_dir = output_dir / "camera"
    camera_dir.mkdir(parents=True, exist_ok=True)

    # Standard camera intrinsics
    fx = fy = image_size[0] * 1.2  # Approximate focal length
    cx, cy = image_size[0] / 2, image_size[1] / 2

    intrinsics = {
        "matrix": [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        "image_size": list(image_size),
    }
    (camera_dir / "intrinsics.json").write_text(json.dumps(intrinsics, indent=2))

    # Camera looking at scene from front
    extrinsics = {
        "matrix": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.5],
            [0.0, 0.0, 1.0, 4.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }
    (camera_dir / "extrinsics.json").write_text(json.dumps(extrinsics, indent=2))


def generate_mock_image(output_path: Path, width: int = 800, height: int = 600) -> None:
    """Generate a placeholder image (simple PNG with kitchen colors)."""
    # Create a minimal valid PNG with a solid color
    # This is a placeholder - in real usage you'd have actual images

    # Simple PNG header + IHDR + minimal IDAT + IEND
    # Creates a small colored rectangle

    def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
        length = struct.pack('>I', len(data))
        crc = zlib.crc32(chunk_type + data) & 0xffffffff
        return length + chunk_type + data + struct.pack('>I', crc)

    # Create a simple 8x8 pattern that will be scaled
    small_w, small_h = 8, 6

    # Kitchen-like colors (warm beige)
    pixels = []
    for y in range(small_h):
        row = [0]  # Filter byte
        for x in range(small_w):
            # Gradient from beige to brown
            r = int(200 - y * 10)
            g = int(180 - y * 8)
            b = int(150 - y * 6)
            row.extend([r, g, b])
        pixels.append(bytes(row))

    raw_data = b''.join(pixels)
    compressed = zlib.compress(raw_data, 9)

    png = b'\x89PNG\r\n\x1a\n'
    png += png_chunk(b'IHDR', struct.pack('>IIBBBBB', small_w, small_h, 8, 2, 0, 0, 0))
    png += png_chunk(b'IDAT', compressed)
    png += png_chunk(b'IEND', b'')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(png)


def generate_mock_zeroscene(
    output_dir: Path,
    scene_id: str,
    environment_type: str = "kitchen",
    objects: Optional[List[Dict[str, Any]]] = None,
) -> Path:
    """Generate a complete mock ZeroScene output directory.

    Args:
        output_dir: Base output directory
        scene_id: Scene identifier
        environment_type: Type of environment
        objects: Optional custom objects (defaults to kitchen objects)

    Returns:
        Path to the zeroscene output directory
    """
    if objects is None:
        objects = KITCHEN_OBJECTS

    scene_dir = output_dir / "scenes" / scene_id
    zeroscene_dir = scene_dir / "zeroscene"
    images_dir = scene_dir / "images"

    # Create directories
    zeroscene_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    # Generate scene info
    scene_info = generate_scene_info(scene_id, environment_type)
    (zeroscene_dir / "scene_info.json").write_text(json.dumps(scene_info, indent=2))

    # Generate objects
    objects_dir = zeroscene_dir / "objects"
    for obj in objects:
        generate_object_files(obj, objects_dir)

    # Generate background
    generate_background(zeroscene_dir)

    # Generate camera
    generate_camera(zeroscene_dir)

    # Generate placeholder image
    generate_mock_image(images_dir / "room.jpg")

    print(f"Generated mock ZeroScene output at: {zeroscene_dir}")
    print(f"  - {len(objects)} objects")
    print(f"  - Background mesh")
    print(f"  - Camera parameters")
    print(f"  - Placeholder image")

    return zeroscene_dir


def main():
    parser = argparse.ArgumentParser(
        description="Generate mock ZeroScene outputs for testing"
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
        choices=["kitchen", "office", "warehouse", "laundry"],
        help="Environment type",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    generate_mock_zeroscene(
        output_dir=output_dir,
        scene_id=args.scene_id,
        environment_type=args.environment,
    )

    print(f"\nTo run the pipeline on this mock scene:")
    print(f"  python tools/run_local_pipeline.py --scene-dir {output_dir}/scenes/{args.scene_id}")


if __name__ == "__main__":
    main()
