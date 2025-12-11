"""Utilities for working with scene manifests.

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
from typing import Any, Dict

try:
    import jsonschema
except ImportError as e:  # pragma: no cover - utility script
    jsonschema = None

SCHEMA_PATH = Path(__file__).with_name("manifest_schema.json")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def normalize_sim_role(role: str | None) -> str:
    allowed = {
        "static",
        "manipulable_object",
        "articulated_furniture",
        "articulated_appliance",
        "scene_shell",
        "background",
        "unknown",
    }
    if not role:
        return "unknown"
    role = str(role)
    if role in allowed:
        return role
    # Graceful downgrade for legacy labels
    if role in {"interactive", "dynamic"}:
        return "manipulable_object"
    return "unknown"


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

    manifest = {
        "schema_version": "1.0",
        "source": "gemini",
        "scene_id": scene_assets.get("scene_id"),
        "objects": [],
    }

    for obj in scene_assets.get("objects", []):
        oid = obj.get("id")
        inv = inv_map.get(str(oid), {})

        asset_path = obj.get("asset_path") or obj.get("interactive_output")
        background_candidate = obj.get("class_name") == "scene_background" or oid == "scene_background"

        entry = {
            "id": oid,
            "sim_role": normalize_sim_role(obj.get("sim_role") or inv.get("sim_role") or obj.get("type")),
            "asset": {
                "path": asset_path,
                "format": "glb" if asset_path and asset_path.endswith(".glb") else None,
                "source": "scene_assets",
                "scene_asset_id": oid,
                "inventory_id": inv.get("id"),
            },
            "placement": {
                "polygon": obj.get("polygon") or inv.get("polygon"),
                "approx_location": obj.get("approx_location") or inv.get("approx_location"),
            },
            "articulation": {
                "physx_endpoint": obj.get("physx_endpoint"),
            },
            "physics": {
                "hints": inv.get("physics_hints") or obj.get("physics_hints"),
            },
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
        entry["physics"] = {k: v for k, v in entry["physics"].items() if v}
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

    manifest = {
        "schema_version": "1.0",
        "source": "blueprint_recipe",
        "scene_id": scene_plan.get("scene_id"),
        "objects": [],
    }

    for obj in scene_plan.get("objects", []):
        oid = obj.get("id")
        matched = assets_by_id.get(str(oid), {})
        asset_path = matched.get("asset_path") or matched.get("path")

        entry = {
            "id": oid,
            "sim_role": normalize_sim_role(obj.get("sim_role")),
            "asset": {
                "path": asset_path,
                "format": matched.get("format"),
                "source": matched.get("source", "blueprint_recipe"),
                "inventory_id": matched.get("inventory_id"),
            },
            "transform": obj.get("transform"),
            "articulation": obj.get("articulation"),
            "physics": obj.get("physics_hints") or matched.get("physics"),
            "semantics": obj.get("semantics"),
            "placement": obj.get("placement"),
            "source": {
                "blueprint_recipe": obj,
                "matched_asset": matched,
            },
        }

        entry["asset"] = {k: v for k, v in entry["asset"].items() if v is not None}
        for key in ("articulation", "physics", "semantics", "placement"):
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
