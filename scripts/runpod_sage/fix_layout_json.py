#!/usr/bin/env python3
"""Patch a SAGE layout JSON to make Stage 4a/4b robust.

This script is intentionally conservative: it only fills in *missing* keys that
downstream scripts expect, using best-effort matching against the already-placed
room objects.

Fixes:
  - Stage 4a: policy_analysis.minimum_required_objects entries often omit
    matched_object_ids/matched_object_types, causing strict assertions.
  - Stage 4b: pose augmentation expects updated_task_decomposition with concrete
    object IDs. Different upstream variants have expected this either at
    policy_analysis.updated_task_decomposition or at the top-level.

Usage:
  python fix_layout_json.py /path/to/layout_XXXX/layout_XXXX.json
"""
import json
import shutil
import sys

if len(sys.argv) != 2:
    print("Usage: python fix_layout_json.py /path/to/layout_XXXX/layout_XXXX.json", file=sys.stderr)
    raise SystemExit(2)

path = sys.argv[1]
shutil.copy2(path, path + ".bak")

d = json.load(open(path))


def _norm(s: str) -> str:
    return (s or "").strip().lower().replace(" ", "_")


def _get_objects(layout_dict):
    # Most common: {"rooms": [{"objects": [...]}]}
    rooms = layout_dict.get("rooms")
    if isinstance(rooms, list) and rooms:
        objects = rooms[0].get("objects")
        if isinstance(objects, list):
            return objects

    # SceneSmith bridge writes a single room dict: {"objects": [...]}
    objects = layout_dict.get("objects")
    if isinstance(objects, list):
        return objects

    # Fallbacks seen in some variants.
    room = layout_dict.get("room")
    if isinstance(room, dict):
        objects = room.get("objects")
        if isinstance(objects, list):
            return objects

    return []


objs = _get_objects(d)
pa = d.get("policy_analysis")
if not isinstance(pa, dict):
    pa = {}
    d["policy_analysis"] = pa

# Common type aliases
TYPE_MAP = {
    "kitchen_counter": "counter",
    "dining_table": "table",
    "mug": "mug",
    "cup": "mug",
    "sofa": "sofa",
    "couch": "sofa",
}


def _find_object_ids_by_type(object_type: str, limit: int):
    want = _norm(TYPE_MAP.get(object_type, object_type))
    matched = []
    for o in objs:
        otype = _norm(o.get("type", ""))
        if otype == want:
            oid = o.get("id")
            if oid:
                matched.append((oid, o.get("type", want)))
        if len(matched) >= limit:
            break

    # Fuzzy fallback: allow substring matches if exact match fails.
    if not matched and want:
        for o in objs:
            otype = _norm(o.get("type", ""))
            if want in otype or otype in want:
                oid = o.get("id")
                if oid:
                    matched.append((oid, o.get("type", want)))
            if len(matched) >= limit:
                break

    return matched


def _first_object_id_by_type(object_type: str):
    matches = _find_object_ids_by_type(object_type, limit=1)
    return matches[0][0] if matches else None


# --- Fix 1: matched_object_ids ---
mros = pa.get("minimum_required_objects")
if isinstance(mros, list):
    for mro in mros:
        if not isinstance(mro, dict):
            continue
        obj_type = mro.get("object_type")
        if not obj_type:
            continue

        # Only fill if missing or empty.
        have_ids = mro.get("matched_object_ids")
        have_types = mro.get("matched_object_types")
        if isinstance(have_ids, list) and have_ids and isinstance(have_types, list) and have_types:
            continue

        qty = mro.get("quantity", 1)
        try:
            qty_i = int(qty) if qty is not None else 1
        except Exception:
            qty_i = 1
        if qty_i < 1:
            qty_i = 1

        matches = _find_object_ids_by_type(obj_type, limit=qty_i)
        matched_ids = [mid for (mid, _t) in matches]
        matched_types = [t for (_mid, t) in matches]
        mro["matched_object_ids"] = matched_ids
        mro["matched_object_types"] = matched_types
        print(f"  MRO {obj_type}: matched {matched_ids}")
else:
    print("  NOTE: policy_analysis.minimum_required_objects missing or not a list; skipping Fix 1")

# --- Fix 2: updated_task_decomposition ---
td = pa.get("task_decomposition")
if isinstance(td, list) and td:
    # Start from existing if present; otherwise derive from task_decomposition.
    utd = pa.get("updated_task_decomposition")
    if not isinstance(utd, list) or not utd:
        utd = [dict(step) for step in td if isinstance(step, dict)]

    for step in utd:
        if not isinstance(step, dict):
            continue

        action = _norm(step.get("action", ""))
        target = step.get("target_object") or step.get("target") or step.get("object") or ""
        location = (
            step.get("location_object")
            or step.get("location")
            or step.get("destination_object")
            or step.get("destination")
            or ""
        )

        # Fill target_object_id if missing.
        if not step.get("target_object_id"):
            # Prefer explicit target in the step; otherwise use common pick/place objects.
            target_id = _first_object_id_by_type(str(target))
            if not target_id and action in ("pick", "place"):
                target_id = _first_object_id_by_type("mug") or _first_object_id_by_type(str(target))
            if target_id:
                step["target_object_id"] = target_id

        # Fill location_object_id if missing.
        if not step.get("location_object_id"):
            loc_id = _first_object_id_by_type(str(location))
            if not loc_id:
                if action == "pick":
                    loc_id = _first_object_id_by_type("counter")
                elif action == "place":
                    loc_id = _first_object_id_by_type("table")
                elif action == "navigate":
                    # "navigate to X" uses X as the location.
                    loc_id = _first_object_id_by_type(str(target))
            if loc_id:
                step["location_object_id"] = loc_id

    pa["updated_task_decomposition"] = utd
    # Compatibility: some variants expect this at the top-level.
    d["updated_task_decomposition"] = utd

    print("Added/updated updated_task_decomposition:")
    for s in utd:
        if not isinstance(s, dict):
            continue
        step_num = s.get("step", "?")
        act = s.get("action", "?")
        tid = s.get("target_object_id", "?")
        lid = s.get("location_object_id", "?")
        print(f"  Step {step_num}: {act} -> target={tid} location={lid}")
else:
    print("  NOTE: policy_analysis.task_decomposition missing/empty; skipping Fix 2")

with open(path, "w") as f:
    json.dump(d, f, indent=2)

print("Layout JSON patched successfully")
