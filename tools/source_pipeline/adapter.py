from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _dims_for_object(obj: Mapping[str, Any]) -> Dict[str, float]:
    dims = obj.get("dimensions_est") if isinstance(obj.get("dimensions_est"), Mapping) else {}
    width = max(0.02, _safe_float(dims.get("width"), 0.25))
    height = max(0.02, _safe_float(dims.get("height"), 0.25))
    depth = max(0.02, _safe_float(dims.get("depth"), 0.25))
    return {
        "width": width,
        "height": height,
        "depth": depth,
    }


def _write_placeholder_model_usd(path: Path, *, dims: Mapping[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sx = round(float(dims.get("width", 0.25)), 6)
    sy = round(float(dims.get("height", 0.25)), 6)
    sz = round(float(dims.get("depth", 0.25)), 6)
    payload = f'''#usda 1.0
(
    defaultPrim = "Root"
)

def Xform "Root"
{{
    def Cube "Geom"
    {{
        double size = 1
        double3 xformOp:scale = ({sx}, {sy}, {sz})
        uniform token[] xformOpOrder = ["xformOp:scale"]
    }}
}}
'''
    path.write_text(payload, encoding="utf-8")


def _materialize_metadata(
    path: Path,
    *,
    obj: Mapping[str, Any],
    dims: Mapping[str, float],
) -> None:
    metadata = {
        "id": obj.get("id"),
        "class_name": obj.get("category") or obj.get("name") or "object",
        "mesh_bounds": {
            "export": {
                "center": [0.0, 0.0, 0.0],
                "size": [dims["width"], dims["height"], dims["depth"]],
            }
        },
        "source": {
            "type": "text",
            "asset_strategy": obj.get("asset_strategy", "generated"),
        },
    }
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _object_transform_to_layout(obj: Mapping[str, Any]) -> Dict[str, Any]:
    transform = obj.get("transform") if isinstance(obj.get("transform"), Mapping) else {}
    position = transform.get("position") if isinstance(transform.get("position"), Mapping) else {}
    center3d = [
        _safe_float(position.get("x"), 0.0),
        _safe_float(position.get("y"), 0.0),
        _safe_float(position.get("z"), 0.0),
    ]

    layout_obj: Dict[str, Any] = {
        "id": str(obj.get("id")),
        "class_name": str(obj.get("category") or obj.get("name") or "object"),
        "center3d": center3d,
    }

    if "rotation_quaternion" in transform and isinstance(transform["rotation_quaternion"], Mapping):
        quat = transform["rotation_quaternion"]
        layout_obj["rotation_quaternion"] = {
            "w": _safe_float(quat.get("w"), 1.0),
            "x": _safe_float(quat.get("x"), 0.0),
            "y": _safe_float(quat.get("y"), 0.0),
            "z": _safe_float(quat.get("z"), 0.0),
        }

    if "obb" in obj and isinstance(obj["obb"], Mapping):
        layout_obj["obb"] = obj["obb"]

    return layout_obj


def materialize_placeholder_assets(
    *,
    root: Path,
    scene_id: str,
    assets_prefix: str,
    objects: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Create placeholder model assets for text-generated objects.

    Returns a list of provenance entries with generated asset paths.
    """

    assets_root = root / assets_prefix
    assets_root.mkdir(parents=True, exist_ok=True)

    provenance_assets: List[Dict[str, Any]] = []

    for obj in objects:
        oid = str(obj.get("id") or "")
        if not oid:
            continue
        obj_dir = assets_root / oid
        obj_dir.mkdir(parents=True, exist_ok=True)

        dims = _dims_for_object(obj)
        model_path = obj_dir / "model.usd"
        _write_placeholder_model_usd(model_path, dims=dims)

        metadata_path = obj_dir / "metadata.json"
        _materialize_metadata(metadata_path, obj=obj, dims=dims)

        provenance_assets.append(
            {
                "object_id": oid,
                "path": f"{assets_prefix}/{oid}/model.usd",
                "source": "generated" if obj.get("asset_strategy") != "retrieved" else "retrieved",
                "strategy": obj.get("asset_strategy", "generated"),
                "model_or_library": (
                    "partnet_mobility" if obj.get("asset_strategy") == "retrieved" else "textgen_placeholder"
                ),
            }
        )

    return provenance_assets


def _build_manifest(
    *,
    scene_id: str,
    objects: List[Dict[str, Any]],
    assets_prefix: str,
    source_request: Mapping[str, Any],
    provider_used: str,
    quality_tier: str,
    seed: int,
    provenance_assets: List[Dict[str, Any]],
) -> Dict[str, Any]:
    manifest_objects: List[Dict[str, Any]] = []

    for obj in objects:
        oid = str(obj.get("id") or "")
        if not oid:
            continue
        dims = _dims_for_object(obj)
        transform = obj.get("transform") if isinstance(obj.get("transform"), Mapping) else {}
        position = transform.get("position") if isinstance(transform.get("position"), Mapping) else {}
        scale = transform.get("scale") if isinstance(transform.get("scale"), Mapping) else {}

        manifest_obj: Dict[str, Any] = {
            "id": oid,
            "name": str(obj.get("name") or oid),
            "category": str(obj.get("category") or "object"),
            "description": str(obj.get("description") or "text-generated object"),
            "sim_role": str(obj.get("sim_role") or "manipulable_object"),
            "asset": {
                "path": f"{assets_prefix}/{oid}/model.usd",
                "source": "text_scene_gen",
                "format": "usd",
            },
            "transform": {
                "position": {
                    "x": _safe_float(position.get("x"), 0.0),
                    "y": _safe_float(position.get("y"), 0.0),
                    "z": _safe_float(position.get("z"), 0.0),
                },
                "scale": {
                    "x": _safe_float(scale.get("x"), 1.0),
                    "y": _safe_float(scale.get("y"), 1.0),
                    "z": _safe_float(scale.get("z"), 1.0),
                },
            },
            "dimensions_est": dims,
            "physics_hints": dict(obj.get("physics_hints") or {}),
            "articulation": {
                "required": bool((obj.get("articulation") or {}).get("required", False)),
                "backend_hint": str((obj.get("articulation") or {}).get("backend_hint", "none")),
            },
            "source": {
                "type": "text",
                "generation_tier": quality_tier,
                "provider": provider_used,
                "seed": seed,
            },
        }

        rotation_quaternion = transform.get("rotation_quaternion")
        if isinstance(rotation_quaternion, Mapping):
            manifest_obj["transform"]["rotation_quaternion"] = {
                "w": _safe_float(rotation_quaternion.get("w"), 1.0),
                "x": _safe_float(rotation_quaternion.get("x"), 0.0),
                "y": _safe_float(rotation_quaternion.get("y"), 0.0),
                "z": _safe_float(rotation_quaternion.get("z"), 0.0),
            }

        relationships = obj.get("relationships")
        if isinstance(relationships, list):
            manifest_obj["relationships"] = relationships

        manifest_objects.append(manifest_obj)

    return {
        "version": "1.0.0",
        "scene_id": scene_id,
        "scene": {
            "coordinate_frame": "y_up",
            "meters_per_unit": 1.0,
            "environment_type": str(source_request.get("constraints", {}).get("room_type", "generic")),
        },
        "objects": manifest_objects,
        "metadata": {
            "source": {
                "type": "text",
                "request_id": scene_id,
                "seed": seed,
                "provider": provider_used,
                "generation_tier": quality_tier,
            },
            "provenance": {
                "assets": provenance_assets,
                "request": source_request,
            },
            "source_pipeline": "text-scene-adapter-job",
        },
    }


def _build_layout(
    *,
    scene_id: str,
    objects: List[Dict[str, Any]],
) -> Dict[str, Any]:
    room_min = [-3.0, 0.0, -3.0]
    room_max = [3.0, 3.0, 3.0]

    layout_objects = [_object_transform_to_layout(obj) for obj in objects if obj.get("id")]

    return {
        "scene_id": scene_id,
        "objects": layout_objects,
        "room_box": {
            "min": room_min,
            "max": room_max,
        },
    }


def _build_inventory(
    *,
    scene_id: str,
    objects: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "scene_id": scene_id,
        "source": "text_scene_gen",
        "objects": [
            {
                "id": str(obj.get("id")),
                "name": str(obj.get("name") or obj.get("id") or "object"),
                "category": str(obj.get("category") or "object"),
                "sim_role": str(obj.get("sim_role") or "manipulable_object"),
                "description": str(obj.get("description") or ""),
                "asset_strategy": str(obj.get("asset_strategy") or "generated"),
                "articulation_required": bool((obj.get("articulation") or {}).get("required", False)),
            }
            for obj in objects
            if obj.get("id") is not None
        ],
    }


def build_manifest_layout_inventory(
    *,
    root: Path,
    scene_id: str,
    assets_prefix: str,
    layout_prefix: str,
    seg_prefix: str,
    textgen_payload: Mapping[str, Any],
    source_request: Mapping[str, Any],
) -> Dict[str, Any]:
    """Materialize canonical artifacts and compatibility markers from textgen payload."""

    objects = [obj for obj in (textgen_payload.get("objects") or []) if isinstance(obj, Mapping)]

    provenance_assets = materialize_placeholder_assets(
        root=root,
        scene_id=scene_id,
        assets_prefix=assets_prefix,
        objects=[dict(obj) for obj in objects],
    )

    quality_tier = str(textgen_payload.get("quality_tier") or "standard")
    provider_used = str(textgen_payload.get("provider_used") or "openai")
    seed = int(textgen_payload.get("seed") or 1)

    manifest = _build_manifest(
        scene_id=scene_id,
        objects=[dict(obj) for obj in objects],
        assets_prefix=assets_prefix,
        source_request=source_request,
        provider_used=provider_used,
        quality_tier=quality_tier,
        seed=seed,
        provenance_assets=provenance_assets,
    )
    layout = _build_layout(scene_id=scene_id, objects=[dict(obj) for obj in objects])
    inventory = _build_inventory(scene_id=scene_id, objects=[dict(obj) for obj in objects])

    assets_root = root / assets_prefix
    layout_root = root / layout_prefix
    seg_root = root / seg_prefix

    assets_root.mkdir(parents=True, exist_ok=True)
    layout_root.mkdir(parents=True, exist_ok=True)
    seg_root.mkdir(parents=True, exist_ok=True)

    manifest_path = assets_root / "scene_manifest.json"
    layout_path = layout_root / "scene_layout_scaled.json"
    inventory_path = seg_root / "inventory.json"

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    layout_path.write_text(json.dumps(layout, indent=2), encoding="utf-8")
    inventory_path.write_text(json.dumps(inventory, indent=2), encoding="utf-8")

    completion_marker = assets_root / ".regen3d_complete"
    completion_marker.write_text(
        json.dumps(
            {
                "scene_id": scene_id,
                "status": "completed",
                "source": "text_scene_adapter",
                "objects_count": len(objects),
                "quality_tier": quality_tier,
                "provider": provider_used,
                "seed": seed,
                "marker_type": "stage1_complete",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    adapter_complete = assets_root / ".text_adapter_complete"
    adapter_complete.write_text(
        json.dumps(
            {
                "scene_id": scene_id,
                "status": "completed",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "scene_id": scene_id,
        "objects_count": len(objects),
        "manifest_path": str(manifest_path),
        "layout_path": str(layout_path),
        "inventory_path": str(inventory_path),
        "completion_marker": str(completion_marker),
    }
