"""Utilities for working with canonical scene manifests.

Provides:
- ``validate``: Validate a scene_manifest.json against the schema.
- ``from_scene_assets``: Convert scene_assets.json + optional inventory.json
  into the canonical scene manifest structure.
- ``from_blueprint_recipe``: Convert a BlueprintRecipe scene plan and matched
  assets mapping into the canonical manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

try:
    import jsonschema
except ImportError as e:  # pragma: no cover - utility script
    jsonschema = None

SCHEMA_PATH = Path(__file__).with_name("manifest_schema.json")
DEFAULT_VERSION = "1.0.0"
DEFAULT_SCENE = {
    "coordinate_frame": "y_up",
    "meters_per_unit": 1.0,
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def normalize_sim_role(role: str | None) -> str:
    allowed = {
        "static",
        "interactive",
        "manipulable_object",
        "articulated_furniture",
        "articulated_appliance",
        "scene_shell",
        "background",
        "clutter",
        "deformable_object",
        "unknown",
    }
    if not role:
        return "unknown"
    role = str(role)
    if role in allowed:
        return role
    # Graceful downgrade for legacy labels
    if role in {"interactive", "dynamic"}:
        return "interactive"
    if role in {"deformable", "cloth", "soft_body"}:
        return "deformable_object"
    return "unknown"


def _coerce_number(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _canonical_transform(*sources: Mapping[str, Any] | None) -> Dict[str, Any]:
    position = {"x": 0.0, "y": 0.0, "z": 0.0}
    scale = {"x": 1.0, "y": 1.0, "z": 1.0}
    rotation_euler: Dict[str, float] | None = None
    rotation_quaternion: Dict[str, float] | None = None

    for source in sources:
        if not source:
            continue

        transform = source.get("transform") if isinstance(source, Mapping) else None
        transform = transform or (source if isinstance(source, Mapping) else None)

        pos = None
        if transform:
            pos = transform.get("position") or transform.get("translation")
        if isinstance(pos, Mapping):
            position.update({
                "x": _coerce_number(pos.get("x"), position["x"]),
                "y": _coerce_number(pos.get("y"), position["y"]),
                "z": _coerce_number(pos.get("z"), position["z"]),
            })
        if isinstance(pos, Iterable) and not isinstance(pos, (str, bytes, Mapping)):
            pos_list = list(pos)
            if len(pos_list) >= 3:
                position.update({
                    "x": _coerce_number(pos_list[0], position["x"]),
                    "y": _coerce_number(pos_list[1], position["y"]),
                    "z": _coerce_number(pos_list[2], position["z"]),
                })

        rot_euler = (transform or {}).get("rotation_euler") if transform else None
        if isinstance(rot_euler, Mapping):
            rotation_euler = {
                "roll": _coerce_number(rot_euler.get("roll"), 0.0),
                "pitch": _coerce_number(rot_euler.get("pitch"), 0.0),
                "yaw": _coerce_number(rot_euler.get("yaw"), 0.0),
            }

        rot_quat = (transform or {}).get("rotation") or (transform or {}).get("rotation_quaternion")
        if isinstance(rot_quat, Mapping):
            rotation_quaternion = {
                "w": _coerce_number(rot_quat.get("w"), 1.0),
                "x": _coerce_number(rot_quat.get("x"), 0.0),
                "y": _coerce_number(rot_quat.get("y"), 0.0),
                "z": _coerce_number(rot_quat.get("z"), 0.0),
            }

        scl = (transform or {}).get("scale") if transform else None
        if isinstance(scl, Mapping):
            scale.update({
                "x": _coerce_number(scl.get("x"), scale["x"]),
                "y": _coerce_number(scl.get("y"), scale["y"]),
                "z": _coerce_number(scl.get("z"), scale["z"]),
            })
        if isinstance(scl, Iterable) and not isinstance(scl, (str, bytes, Mapping)):
            scl_list = list(scl)
            if len(scl_list) >= 3:
                scale.update({
                    "x": _coerce_number(scl_list[0], scale["x"]),
                    "y": _coerce_number(scl_list[1], scale["y"]),
                    "z": _coerce_number(scl_list[2], scale["z"]),
                })

    transform_out: Dict[str, Any] = {"position": position, "scale": scale}
    if rotation_euler:
        transform_out["rotation_euler"] = rotation_euler
    if rotation_quaternion:
        transform_out["rotation_quaternion"] = rotation_quaternion
    return transform_out


def validate_manifest(manifest: Dict[str, Any]) -> None:
    if jsonschema is None:
        raise SystemExit(
            "jsonschema is required for validation. Install with: pip install jsonschema",
        )

    with SCHEMA_PATH.open("r") as f:
        schema = json.load(f)

    jsonschema.validate(instance=manifest, schema=schema)


# ---------------------------------------------------------------------------
# Converters
# ---------------------------------------------------------------------------


def manifest_from_scene_assets(scene_assets: Dict[str, Any], inventory: Dict[str, Any] | None = None) -> Dict[str, Any]:
    inventory = inventory or {}
    inv_map = {str(o.get("id")): o for o in inventory.get("objects", [])}

    scene_block = {**DEFAULT_SCENE}
    if inventory.get("environment_type") is not None:
        scene_block["environment_type"] = inventory.get("environment_type")
    if inventory.get("room"):
        scene_block["room"] = inventory.get("room")

    manifest = {
        "version": DEFAULT_VERSION,
        "scene_id": str(scene_assets.get("scene_id", "")),
        "scene": scene_block,
        "objects": [],
        "metadata": {
            "source_pipeline": "gemini",
        },
    }

    for obj in scene_assets.get("objects", []):
        oid = obj.get("id")
        inv = inv_map.get(str(oid), {})

        asset_path = obj.get("asset_path") or obj.get("interactive_output")
        background_candidate = obj.get("class_name") == "scene_background" or oid == "scene_background"

        entry = {
            "id": str(oid),
            "name": inv.get("display_name"),
            "category": obj.get("class_name") or inv.get("category"),
            "description": inv.get("short_description") or obj.get("object_phrase"),
            "sim_role": normalize_sim_role(obj.get("sim_role") or inv.get("sim_role") or obj.get("type")),
            "transform": _canonical_transform(obj, inv),
            "asset": {
                "path": asset_path,
                "format": "glb" if asset_path and asset_path.endswith(".glb") else None,
                "source": "scene_assets",
                "relative_path": obj.get("relative_path"),
                "pack_name": obj.get("pack_name") or inv.get("pack_name"),
                "candidates": inv.get("candidates") or obj.get("candidates"),
            },
            "placement_region": obj.get("placement_region"),
            "placement": {
                "polygon": obj.get("polygon") or inv.get("polygon"),
                "approx_location": obj.get("approx_location") or inv.get("approx_location"),
            },
            "articulation": {
                "physx_endpoint": obj.get("physx_endpoint"),
            },
            "physics_hints": inv.get("physics_hints") or obj.get("physics_hints"),
            "semantics": {
                "category": obj.get("class_name") or inv.get("category"),
                "short_description": inv.get("short_description") or obj.get("object_phrase"),
                "tags": inv.get("semantic_tags") or [],
            },
            "asset_generation": {
                "pipeline": obj.get("pipeline"),
                "inputs": {
                    "multiview_dir": obj.get("multiview_dir"),
                    "crop_path": obj.get("crop_path"),
                    "preferred_view": obj.get("preferred_view"),
                },
                "output": asset_path,
            },
            "source": {
                "scene_assets": obj,
                "inventory": inv if inv else None,
            },
        }

        # Drop empty nested mappings
        entry["asset"] = {k: v for k, v in entry["asset"].items() if v is not None}
        entry["articulation"] = {k: v for k, v in entry["articulation"].items() if v is not None}
        entry["physics_hints"] = entry["physics_hints"] or None
        entry["semantics"] = {k: v for k, v in entry["semantics"].items() if v}
        entry["placement"] = {k: v for k, v in entry["placement"].items() if v}
        entry["asset_generation"] = {
            k: v
            for k, v in entry["asset_generation"].items()
            if v and (not isinstance(v, dict) or any(v.values()))
        }
        entry["source"] = {k: v for k, v in entry["source"].items() if v}

        manifest["objects"].append(entry)

        if background_candidate:
            manifest["background"] = {
                "mesh": {"path": asset_path},
                "semantics": {"category": "scene_background"},
            }

    return manifest


def manifest_from_blueprint_recipe(scene_plan: Dict[str, Any], matched_assets: Dict[str, Any]) -> Dict[str, Any]:
    assets_by_id = {str(a.get("id") or a.get("object_id")): a for a in matched_assets.get("objects", matched_assets.get("assets", []))}

    scene_block = {**DEFAULT_SCENE}
    detected_env = (scene_plan.get("environment_analysis") or {}).get("detected_type")
    if detected_env is not None:
        scene_block["environment_type"] = detected_env
    if scene_plan.get("room"):
        scene_block["room"] = scene_plan.get("room")

    manifest = {
        "version": DEFAULT_VERSION,
        "scene_id": str(scene_plan.get("scene_id", "")),
        "scene": scene_block,
        "objects": [],
        "metadata": {
            "source_pipeline": "blueprint_recipe",
            "scene_plan_version": scene_plan.get("version"),
        },
    }

    for obj in scene_plan.get("objects", []):
        oid = obj.get("id")
        matched = assets_by_id.get(str(oid), {})
        asset_path = matched.get("asset_path") or matched.get("path")

        entry = {
            "id": str(oid),
            "name": obj.get("name"),
            "category": obj.get("category"),
            "description": obj.get("description"),
            "sim_role": normalize_sim_role(obj.get("sim_role")),
            "transform": _canonical_transform(obj.get("transform") or obj),
            "asset": {
                "path": asset_path,
                "format": matched.get("format"),
                "source": matched.get("source", "blueprint_recipe"),
                "pack_name": matched.get("pack_name"),
                "relative_path": matched.get("relative_path"),
                "variants": matched.get("variants"),
                "candidates": matched.get("candidates"),
                "simready_metadata": matched.get("simready_metadata"),
                "asset_id": matched.get("asset_id"),
                "inventory_id": matched.get("inventory_id"),
            },
            "articulation": obj.get("articulation"),
            "physics": obj.get("physics") or obj.get("physics_hints") or matched.get("physics"),
            "semantics": obj.get("semantics"),
            "placement_region": obj.get("placement_region"),
            "placement": obj.get("placement"),
            "relationships": obj.get("relationships"),
            "dimensions_est": obj.get("dimensions_est") or obj.get("dimensions"),
            "source": {
                "blueprint_recipe": obj,
                "matched_asset": matched,
            },
        }

        entry["asset"] = {k: v for k, v in entry["asset"].items() if v is not None}
        for key in ("articulation", "physics", "semantics", "placement", "relationships"):
            if isinstance(entry.get(key), dict):
                entry[key] = {k: v for k, v in entry[key].items() if v is not None}

        if obj.get("sim_role") == "background":
            manifest["background"] = {
                "mesh": {"path": asset_path},
                "semantics": obj.get("semantics"),
            }

        manifest["objects"].append(entry)

    if scene_plan.get("background") and "mesh" in scene_plan["background"]:
        manifest["background"] = scene_plan["background"]

    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scene manifest utilities")
    sub = parser.add_subparsers(dest="command", required=True)

    validate_cmd = sub.add_parser("validate", help="Validate a manifest against the schema")
    validate_cmd.add_argument("manifest", type=Path)

    sa_cmd = sub.add_parser("from_scene_assets", help="Build manifest from scene_assets.json + inventory.json")
    sa_cmd.add_argument("scene_assets", type=Path)
    sa_cmd.add_argument("--inventory", type=Path, default=None, help="Optional inventory.json path")
    sa_cmd.add_argument("--output", type=Path, default=None, help="Output path (defaults to stdout)")

    br_cmd = sub.add_parser("from_blueprint_recipe", help="Build manifest from BlueprintRecipe inputs")
    br_cmd.add_argument("scene_plan", type=Path, help="Scene plan JSON")
    br_cmd.add_argument("matched_assets", type=Path, help="Matched assets JSON")
    br_cmd.add_argument("--output", type=Path, default=None, help="Output path (defaults to stdout)")

    return parser.parse_args()


def write_output(manifest: Dict[str, Any], output: Path | None) -> None:
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(manifest, indent=2))
        print(f"[MANIFEST] Wrote {output}")
    else:
        json.dump(manifest, sys.stdout, indent=2)
        sys.stdout.write("\n")


def main() -> None:
    args = parse_args()

    if args.command == "validate":
        manifest = load_json(args.manifest)
        validate_manifest(manifest)
        print("[MANIFEST] Manifest is valid!")
        return

    if args.command == "from_scene_assets":
        scene_assets = load_json(args.scene_assets)
        inventory = load_json(args.inventory) if args.inventory else None
        manifest = manifest_from_scene_assets(scene_assets, inventory)
        write_output(manifest, args.output)
        return

    if args.command == "from_blueprint_recipe":
        scene_plan = load_json(args.scene_plan)
        matched_assets = load_json(args.matched_assets)
        manifest = manifest_from_blueprint_recipe(scene_plan, matched_assets)
        write_output(manifest, args.output)
        return


if __name__ == "__main__":
    main()
