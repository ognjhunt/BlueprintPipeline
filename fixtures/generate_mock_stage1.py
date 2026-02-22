#!/usr/bin/env python3
"""Generate mock Stage 1 outputs for BlueprintPipeline tests.

Usage:
    python fixtures/generate_mock_stage1.py --scene-id test_kitchen --output-dir ./test_scenes
"""

from __future__ import annotations

import argparse
import json
import struct
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

KITCHEN_OBJECTS = [
    {"id": "counter_0", "category": "counter", "size": [2.0, 0.9, 0.6], "position": [0, 0.45, -1.5]},
    {"id": "fridge", "category": "refrigerator", "size": [0.9, 1.8, 0.7], "position": [1.5, 0.9, -1.5]},
    {"id": "mug_0", "category": "mug", "size": [0.08, 0.1, 0.08], "position": [0.3, 0.95, -1.4]},
]

OFFICE_OBJECTS = [
    {"id": "desk_0", "category": "desk", "size": [1.5, 0.75, 0.8], "position": [0, 0.375, -1.2]},
    {"id": "chair_0", "category": "office_chair", "size": [0.6, 1.0, 0.6], "position": [0, 0.5, 0]},
    {"id": "monitor_0", "category": "monitor", "size": [0.6, 0.4, 0.05], "position": [0, 0.95, -1.4]},
]

WAREHOUSE_OBJECTS = [
    {"id": "shelf_0", "category": "shelf", "size": [2.0, 2.0, 0.5], "position": [-2, 1.0, -2.0]},
    {"id": "box_0", "category": "box", "size": [0.4, 0.3, 0.4], "position": [-2, 0.5, -2.0]},
    {"id": "pallet_0", "category": "pallet", "size": [1.2, 0.15, 1.0], "position": [0, 0.075, 0]},
]

ENVIRONMENT_OBJECTS = {
    "kitchen": KITCHEN_OBJECTS,
    "office": OFFICE_OBJECTS,
    "warehouse": WAREHOUSE_OBJECTS,
}


def create_minimal_glb() -> bytes:
    """Create a tiny valid GLB (cube)."""
    gltf_json = {
        "asset": {"version": "2.0", "generator": "mock-stage1"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "Cube"}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": 8,
                "type": "VEC3",
                "min": [-0.5, -0.5, -0.5],
                "max": [0.5, 0.5, 0.5],
            },
            {"bufferView": 1, "componentType": 5123, "count": 36, "type": "SCALAR"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 96},
            {"buffer": 0, "byteOffset": 96, "byteLength": 72},
        ],
        "buffers": [{"byteLength": 168}],
    }

    vertices = [
        -0.5,
        -0.5,
        -0.5,
        0.5,
        -0.5,
        -0.5,
        0.5,
        0.5,
        -0.5,
        -0.5,
        0.5,
        -0.5,
        -0.5,
        -0.5,
        0.5,
        0.5,
        -0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        -0.5,
        0.5,
        0.5,
    ]
    indices = [
        0,
        1,
        2,
        0,
        2,
        3,
        4,
        6,
        5,
        4,
        7,
        6,
        0,
        4,
        5,
        0,
        5,
        1,
        2,
        6,
        7,
        2,
        7,
        3,
        0,
        3,
        7,
        0,
        7,
        4,
        1,
        5,
        6,
        1,
        6,
        2,
    ]

    vertex_data = struct.pack(f"{len(vertices)}f", *vertices)
    index_data = struct.pack(f"{len(indices)}H", *indices)
    binary_data = vertex_data + index_data

    json_str = json.dumps(gltf_json, separators=(",", ":"))
    while len(json_str) % 4 != 0:
        json_str += " "
    json_bytes = json_str.encode("utf-8")

    glb_header = struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(json_bytes) + 8 + len(binary_data))
    json_chunk = struct.pack("<II", len(json_bytes), 0x4E4F534A) + json_bytes
    bin_chunk = struct.pack("<II", len(binary_data), 0x004E4942) + binary_data
    return glb_header + json_chunk + bin_chunk


def _build_manifest(scene_id: str, objects: List[Dict[str, Any]]) -> Dict[str, Any]:
    manifest_objects = []
    for obj in objects:
        asset_path = f"assets/objects/{obj['id']}/mesh.glb"
        manifest_objects.append(
            {
                "id": obj["id"],
                "name": obj["id"],
                "category": obj["category"],
                "description": f"mock {obj['category']}",
                "sim_role": "manipulable_object",
                "dimensions_est": {
                    "width": float(obj["size"][0]),
                    "depth": float(obj["size"][2]),
                    "height": float(obj["size"][1]),
                },
                "transform": {
                    "position": {
                        "x": float(obj["position"][0]),
                        "y": float(obj["position"][1]),
                        "z": float(obj["position"][2]),
                    },
                    "rotation_quaternion": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
                    "scale": {"x": 1.0, "y": 1.0, "z": 1.0},
                },
                "asset": {"path": asset_path},
                "physics": {"mass": 1.0},
                "physics_hints": {"material_type": "generic"},
                "articulation": {"type": "fixed"},
                "semantics": {"affordances": []},
                "relationships": [],
            }
        )

    return {
        "version": "1.0.0",
        "scene_id": scene_id,
        "scene": {
            "environment_type": "kitchen",
            "coordinate_frame": "y_up",
            "meters_per_unit": 1.0,
            "room": {"bounds": {"width": 5.0, "depth": 5.0, "height": 3.0}},
        },
        "objects": manifest_objects,
        "metadata": {
            "source_pipeline": "text-scene-adapter-job",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "marker_type": "stage1_complete",
        },
    }


def generate_mock_stage1(
    output_dir: Path,
    scene_id: str,
    environment_type: str = "kitchen",
) -> Path:
    """Generate canonical Stage 1 artifacts.

    Returns the scene directory path.
    """
    scene_dir = output_dir / "scenes" / scene_id
    assets_dir = scene_dir / "assets"
    layout_dir = scene_dir / "layout"
    seg_dir = scene_dir / "seg"
    textgen_dir = scene_dir / "textgen"

    for directory in (assets_dir, layout_dir, seg_dir, textgen_dir):
        directory.mkdir(parents=True, exist_ok=True)

    objects = list(ENVIRONMENT_OBJECTS.get(environment_type, KITCHEN_OBJECTS))
    glb = create_minimal_glb()

    for obj in objects:
        obj_dir = assets_dir / "objects" / obj["id"]
        obj_dir.mkdir(parents=True, exist_ok=True)
        (obj_dir / "mesh.glb").write_bytes(glb)

    manifest = _build_manifest(scene_id, objects)
    (assets_dir / "scene_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    layout_payload = {
        "scene_id": scene_id,
        "layout_version": "1.0.0",
        "objects": [
            {
                "id": obj["id"],
                "position": obj["position"],
                "size": obj["size"],
            }
            for obj in objects
        ],
    }
    (layout_dir / "scene_layout_scaled.json").write_text(
        json.dumps(layout_payload, indent=2),
        encoding="utf-8",
    )

    inventory_payload = {
        "scene_id": scene_id,
        "objects": [
            {
                "id": obj["id"],
                "category": obj["category"],
                "count": 1,
            }
            for obj in objects
        ],
    }
    (seg_dir / "inventory.json").write_text(json.dumps(inventory_payload, indent=2), encoding="utf-8")

    textgen_payload = {
        "schema_version": "v1",
        "scene_id": scene_id,
        "source_mode": "text",
        "text_backend": "hybrid_serial",
        "prompt": f"Mock {environment_type} scene",
        "quality_tier": "standard",
        "seed_count": 1,
        "provider_policy": "openrouter_qwen_primary",
        "constraints": {},
    }
    (textgen_dir / "package.json").write_text(json.dumps(textgen_payload, indent=2), encoding="utf-8")
    (textgen_dir / "request.normalized.json").write_text(
        json.dumps(textgen_payload, indent=2),
        encoding="utf-8",
    )
    (textgen_dir / ".textgen_complete").write_text(
        json.dumps(
            {
                "scene_id": scene_id,
                "status": "completed",
                "marker_type": "textgen_complete",
                "timestamp": datetime.utcnow().isoformat() + "Z",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (assets_dir / ".stage1_complete").write_text(
        json.dumps(
            {
                "scene_id": scene_id,
                "status": "completed",
                "marker_type": "stage1_complete",
                "timestamp": datetime.utcnow().isoformat() + "Z",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return scene_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate mock Stage 1 outputs")
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("test_scenes"))
    parser.add_argument(
        "--environment",
        choices=sorted(ENVIRONMENT_OBJECTS.keys()),
        default="kitchen",
        help="Environment profile to synthesize",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scene_dir = generate_mock_stage1(
        output_dir=args.output_dir,
        scene_id=args.scene_id,
        environment_type=args.environment,
    )
    print(f"Generated mock Stage 1 assets at {scene_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
