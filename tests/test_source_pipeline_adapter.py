from __future__ import annotations

import json
from pathlib import Path

from tools.source_pipeline.adapter import (
    build_manifest_layout_inventory,
    materialize_placeholder_assets,
)


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


def _base_source_request(scene_id: str = "scene_adapter") -> dict:
    return {
        "schema_version": "v1",
        "scene_id": scene_id,
        "source_mode": "text",
        "prompt": "Test scene",
        "quality_tier": "standard",
        "seed_count": 1,
    }


def _base_textgen_payload(scene_id: str = "scene_adapter", objects: list | None = None) -> dict:
    return {
        "schema_version": "v1",
        "scene_id": scene_id,
        "seed": 1,
        "quality_tier": "standard",
        "provider_used": "openai",
        "objects": objects or [],
    }


def _run_adapter(tmp_path: Path, scene_id: str, objects: list) -> dict:
    prefix = f"scenes/{scene_id}"
    result = build_manifest_layout_inventory(
        root=tmp_path,
        scene_id=scene_id,
        assets_prefix=f"{prefix}/assets",
        layout_prefix=f"{prefix}/layout",
        seg_prefix=f"{prefix}/seg",
        textgen_payload=_base_textgen_payload(scene_id, objects),
        source_request=_base_source_request(scene_id),
    )
    return result


def test_adapter_empty_objects_array(tmp_path: Path) -> None:
    result = _run_adapter(tmp_path, "scene_empty", [])

    manifest = _load(Path(result["manifest_path"]))
    layout = _load(Path(result["layout_path"]))
    inventory = _load(Path(result["inventory_path"]))

    assert manifest["objects"] == []
    assert layout["objects"] == []
    assert inventory["objects"] == []
    assert result["objects_count"] == 0


def test_adapter_multiple_objects_with_mixed_roles(tmp_path: Path) -> None:
    objects = [
        {
            "id": "obj_001",
            "name": "table",
            "category": "table",
            "sim_role": "static",
            "asset_strategy": "generated",
            "transform": {"position": {"x": 0.0, "y": 0.0, "z": 0.0}},
            "dimensions_est": {"width": 1.2, "height": 0.75, "depth": 0.7},
            "physics_hints": {"dynamic": False, "mass_kg": 15.0},
            "articulation": {"required": False, "backend_hint": "none"},
        },
        {
            "id": "obj_002",
            "name": "mug",
            "category": "mug",
            "sim_role": "manipulable_object",
            "asset_strategy": "generated",
            "transform": {"position": {"x": 0.1, "y": 0.75, "z": 0.0}},
            "dimensions_est": {"width": 0.08, "height": 0.10, "depth": 0.08},
            "physics_hints": {"dynamic": True, "mass_kg": 0.35},
            "articulation": {"required": False, "backend_hint": "none"},
        },
        {
            "id": "obj_003",
            "name": "cabinet",
            "category": "cabinet",
            "sim_role": "articulated_furniture",
            "asset_strategy": "retrieved",
            "transform": {"position": {"x": -1.5, "y": 0.0, "z": 0.0}},
            "dimensions_est": {"width": 0.8, "height": 1.8, "depth": 0.5},
            "physics_hints": {"dynamic": False, "mass_kg": 30.0},
            "articulation": {"required": True, "backend_hint": "particulate_first"},
        },
    ]

    result = _run_adapter(tmp_path, "scene_mixed", objects)
    manifest = _load(Path(result["manifest_path"]))

    assert result["objects_count"] == 3
    assert len(manifest["objects"]) == 3

    roles = {obj["id"]: obj["sim_role"] for obj in manifest["objects"]}
    assert roles["obj_001"] == "static"
    assert roles["obj_002"] == "manipulable_object"
    assert roles["obj_003"] == "articulated_furniture"

    # Articulated object should have required=True
    cab = [o for o in manifest["objects"] if o["id"] == "obj_003"][0]
    assert cab["articulation"]["required"] is True
    assert cab["articulation"]["backend_hint"] == "particulate_first"


def test_adapter_missing_optional_fields_uses_defaults(tmp_path: Path) -> None:
    """Objects with minimal fields should still produce valid output."""
    objects = [
        {
            "id": "obj_bare",
            # No name, category, description, transform, dimensions_est, physics_hints, articulation
        }
    ]

    result = _run_adapter(tmp_path, "scene_bare", objects)
    manifest = _load(Path(result["manifest_path"]))

    assert len(manifest["objects"]) == 1
    obj = manifest["objects"][0]
    assert obj["id"] == "obj_bare"
    assert obj["name"] == "obj_bare"  # defaults to id
    assert obj["category"] == "object"  # default
    assert obj["sim_role"] == "manipulable_object"  # default
    assert obj["transform"]["position"]["x"] == 0.0  # default
    assert obj["dimensions_est"]["width"] >= 0.02  # minimum


def test_adapter_placeholder_usd_is_valid_syntax(tmp_path: Path) -> None:
    """Verify placeholder USD files contain valid USDA structure."""
    objects = [
        {
            "id": "obj_usd",
            "category": "mug",
            "dimensions_est": {"width": 0.08, "height": 0.1, "depth": 0.08},
        }
    ]

    materialize_placeholder_assets(
        root=tmp_path,
        scene_id="scene_usd",
        assets_prefix="assets",
        objects=objects,
    )

    usd_path = tmp_path / "assets" / "obj_usd" / "model.usd"
    assert usd_path.is_file()

    content = usd_path.read_text(encoding="utf-8")
    assert content.startswith("#usda 1.0")
    assert 'defaultPrim = "Root"' in content
    assert 'def Xform "Root"' in content
    assert 'def Cube "Geom"' in content
    assert "xformOp:scale" in content

    # Metadata file should also exist
    meta_path = tmp_path / "assets" / "obj_usd" / "metadata.json"
    assert meta_path.is_file()
    meta = _load(meta_path)
    assert meta["id"] == "obj_usd"
    assert meta["class_name"] == "mug"


def test_adapter_regen3d_complete_marker_content(tmp_path: Path) -> None:
    """Verify .regen3d_complete marker has correct structure."""
    objects = [{"id": "obj_m", "category": "bottle"}]
    result = _run_adapter(tmp_path, "scene_marker", objects)

    marker = _load(Path(result["completion_marker"]))
    assert marker["scene_id"] == "scene_marker"
    assert marker["status"] == "completed"
    assert marker["source"] == "text_scene_adapter"
    assert marker["objects_count"] == 1
    assert marker["quality_tier"] == "standard"
    assert marker["provider"] == "openai"
    assert marker["seed"] == 1
    assert marker["marker_type"] == "stage1_complete"

    # Also check the .text_adapter_complete marker
    adapter_marker_path = tmp_path / "scenes" / "scene_marker" / "assets" / ".text_adapter_complete"
    assert adapter_marker_path.is_file()
    adapter_marker = _load(adapter_marker_path)
    assert adapter_marker["scene_id"] == "scene_marker"
    assert adapter_marker["status"] == "completed"


def test_adapter_objects_without_id_are_skipped(tmp_path: Path) -> None:
    """Objects missing 'id' should be filtered out."""
    objects = [
        {"id": "obj_valid", "category": "mug"},
        {"category": "plate"},  # no id
        {"id": "", "category": "bottle"},  # empty id
    ]
    result = _run_adapter(tmp_path, "scene_skip", objects)
    manifest = _load(Path(result["manifest_path"]))

    # Only obj_valid should appear (empty string id is skipped in adapter)
    valid_ids = [o["id"] for o in manifest["objects"]]
    assert "obj_valid" in valid_ids
    # The empty-id object and no-id object should not appear
    assert "" not in valid_ids
