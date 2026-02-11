from __future__ import annotations

import json
from pathlib import Path

from tools.source_pipeline.adapter import build_manifest_layout_inventory


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_text_adapter_builds_canonical_artifacts_and_marker(tmp_path: Path) -> None:
    scene_id = "scene_text_001"
    assets_prefix = f"scenes/{scene_id}/assets"
    layout_prefix = f"scenes/{scene_id}/layout"
    seg_prefix = f"scenes/{scene_id}/seg"

    textgen_payload = {
        "schema_version": "v1",
        "scene_id": scene_id,
        "seed": 1,
        "quality_tier": "standard",
        "provider_used": "openai",
        "objects": [
            {
                "id": "obj_001",
                "name": "mug",
                "category": "mug",
                "description": "test mug",
                "sim_role": "manipulable_object",
                "asset_strategy": "generated",
                "transform": {
                    "position": {"x": 0.1, "y": 0.0, "z": -0.2},
                    "scale": {"x": 1.0, "y": 1.0, "z": 1.0},
                    "rotation_quaternion": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
                },
                "dimensions_est": {"width": 0.08, "height": 0.1, "depth": 0.08},
                "physics_hints": {"dynamic": True, "mass_kg": 0.35},
                "articulation": {"required": False, "backend_hint": "none"},
            }
        ],
    }

    source_request = {
        "schema_version": "v1",
        "scene_id": scene_id,
        "source_mode": "text",
        "prompt": "A kitchen scene",
        "quality_tier": "standard",
        "seed_count": 1,
    }

    result = build_manifest_layout_inventory(
        root=tmp_path,
        scene_id=scene_id,
        assets_prefix=assets_prefix,
        layout_prefix=layout_prefix,
        seg_prefix=seg_prefix,
        textgen_payload=textgen_payload,
        source_request=source_request,
    )

    manifest_path = Path(result["manifest_path"])
    layout_path = Path(result["layout_path"])
    inventory_path = Path(result["inventory_path"])
    completion_marker = Path(result["completion_marker"])

    assert manifest_path.is_file()
    assert layout_path.is_file()
    assert inventory_path.is_file()
    assert completion_marker.is_file()

    manifest = _load(manifest_path)
    layout = _load(layout_path)
    inventory = _load(inventory_path)
    marker = _load(completion_marker)

    assert manifest["metadata"]["source"]["type"] == "text"
    assert manifest["metadata"]["source"]["provider"] == "openai"
    assert manifest["metadata"]["source"]["seed"] == 1
    assert manifest["objects"][0]["asset"]["path"].endswith("/obj_001/model.usd")
    assert layout["scene_id"] == scene_id
    assert len(layout["objects"]) == 1
    assert inventory["source"] == "text_scene_gen"
    assert marker["marker_type"] == "stage1_complete"
