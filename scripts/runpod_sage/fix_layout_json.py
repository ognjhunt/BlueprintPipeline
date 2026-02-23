#!/usr/bin/env python3
"""Patch layout JSON so Stage 4 has deterministic task/object fields.

This script can run in conservative mode (fill only missing IDs) or strict
remediation mode (`--require-stage4-fields`) where missing Stage 4-required
structures are synthesized from scene objects and task text.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


_SMALL_PICK_TOKENS: Sequence[str] = ("salt", "mug", "cup", "glass", "bottle", "can", "jar", "book", "plate", "bowl")
_SURFACE_TOKENS: Sequence[str] = ("dining_table", "table", "counter", "kitchen_island", "island", "desk", "bench")
_TYPE_ALIASES: Dict[str, str] = {
    "kitchen_counter": "counter",
    "countertop": "counter",
    "diningtable": "dining_table",
    "dining_table": "table",
    "kitchen_island": "island",
}


def _norm(value: Any) -> str:
    return str(value or "").strip().lower().replace(" ", "_")


def _norm_type(value: Any) -> str:
    t = _norm(value)
    return _TYPE_ALIASES.get(t, t)


def _extract_objects(layout: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rooms = layout.get("rooms")
    if isinstance(rooms, list) and rooms and isinstance(rooms[0], Mapping):
        objects = rooms[0].get("objects")
        if isinstance(objects, list):
            return [dict(o) for o in objects if isinstance(o, Mapping)]

    objects = layout.get("objects")
    if isinstance(objects, list):
        return [dict(o) for o in objects if isinstance(o, Mapping)]

    room = layout.get("room")
    if isinstance(room, Mapping):
        objects = room.get("objects")
        if isinstance(objects, list):
            return [dict(o) for o in objects if isinstance(o, Mapping)]

    return []


def _find_object_by_id(objects: Sequence[Mapping[str, Any]], object_id: str) -> Optional[Mapping[str, Any]]:
    want = str(object_id or "")
    if not want:
        return None
    for obj in objects:
        if str(obj.get("id", "")) == want:
            return obj
    return None


def _find_objects_by_type(objects: Sequence[Mapping[str, Any]], object_type: str, limit: int = 1) -> List[Mapping[str, Any]]:
    want = _norm_type(object_type)
    out: List[Mapping[str, Any]] = []
    if not want:
        return out

    for obj in objects:
        t = _norm_type(obj.get("type"))
        if t == want:
            out.append(obj)
            if len(out) >= limit:
                return out
    for obj in objects:
        t = _norm_type(obj.get("type"))
        if want in t or t in want:
            out.append(obj)
            if len(out) >= limit:
                return out
    return out


def _first_object_by_token(objects: Sequence[Mapping[str, Any]], tokens: Sequence[str]) -> Optional[Mapping[str, Any]]:
    norm_tokens = tuple(_norm_type(tok) for tok in tokens)
    for obj in objects:
        t = _norm_type(obj.get("type"))
        if any(tok in t for tok in norm_tokens):
            return obj
    return None


def _extract_task_targets(task_desc: str) -> Tuple[Optional[str], Optional[str]]:
    task = re.sub(r"[^a-z0-9_ ]+", " ", str(task_desc or "").lower())
    task = re.sub(r"\s+", " ", task).strip()
    if not task:
        return None, None

    pick_hint: Optional[str] = None
    place_hint: Optional[str] = None
    pick_match = re.search(
        r"(?:pick|grab|take|lift)\s+(?:up\s+)?(?:the\s+|a\s+|an\s+)?([a-z0-9_ ]+?)(?:\s+from|\s+on|\s+off|\s+and|\s+to|$)",
        task,
    )
    if pick_match:
        phrase = pick_match.group(1).strip()
        for tok in _SMALL_PICK_TOKENS:
            if tok in phrase:
                pick_hint = tok
                break
        if pick_hint is None and phrase:
            pick_hint = _norm_type(phrase.split()[-1])

    place_match = re.search(
        r"(?:place|put|set|move|drop)\s+.*?(?:on|onto|to|into)\s+(?:the\s+|a\s+|an\s+)?([a-z0-9_ ]+?)(?:[.,]|$)",
        task,
    )
    if place_match:
        phrase = " ".join(place_match.group(1).strip().split())
        for tok in sorted(_SURFACE_TOKENS, key=len, reverse=True):
            if tok.replace("_", " ") in phrase or tok in phrase:
                place_hint = tok
                break
        if place_hint is None and phrase:
            place_hint = _norm_type(phrase.split()[-1])

    return pick_hint, place_hint


def _infer_pick_place_objects(objects: Sequence[Mapping[str, Any]], task_desc: str) -> Tuple[Optional[Mapping[str, Any]], Optional[Mapping[str, Any]]]:
    pick_hint, place_hint = _extract_task_targets(task_desc)

    pick_obj: Optional[Mapping[str, Any]] = None
    place_obj: Optional[Mapping[str, Any]] = None
    if pick_hint:
        matches = _find_objects_by_type(objects, pick_hint, limit=1)
        pick_obj = matches[0] if matches else None
    if place_hint:
        matches = _find_objects_by_type(objects, place_hint, limit=1)
        place_obj = matches[0] if matches else None

    if pick_obj is None:
        pick_obj = _first_object_by_token(objects, _SMALL_PICK_TOKENS)
    if place_obj is None:
        place_obj = _first_object_by_token(objects, _SURFACE_TOKENS)
    if place_obj is None and objects:
        place_obj = objects[0]

    return pick_obj, place_obj


def _ensure_mro_entries(
    *,
    pa: Dict[str, Any],
    objects: Sequence[Mapping[str, Any]],
    pick_obj: Optional[Mapping[str, Any]],
    place_obj: Optional[Mapping[str, Any]],
    require_stage4_fields: bool,
) -> None:
    mros = pa.get("minimum_required_objects")
    if not isinstance(mros, list):
        if not require_stage4_fields:
            return
        mros = []
        pa["minimum_required_objects"] = mros

    if require_stage4_fields and not mros:
        if pick_obj is not None:
            mros.append(
                {
                    "object_type": str(pick_obj.get("type") or "object"),
                    "quantity": 1,
                    "matched_object_ids": [str(pick_obj.get("id", ""))] if pick_obj.get("id") else [],
                    "matched_object_types": [str(pick_obj.get("type") or "object")],
                }
            )
        if place_obj is not None:
            mros.append(
                {
                    "object_type": str(place_obj.get("type") or "surface"),
                    "quantity": 1,
                    "matched_object_ids": [str(place_obj.get("id", ""))] if place_obj.get("id") else [],
                    "matched_object_types": [str(place_obj.get("type") or "surface")],
                }
            )

    for mro in mros:
        if not isinstance(mro, dict):
            continue
        obj_type = str(mro.get("object_type") or "").strip()
        if not obj_type:
            continue
        qty_raw = mro.get("quantity", 1)
        try:
            qty = max(1, int(qty_raw))
        except Exception:
            qty = 1

        ids = mro.get("matched_object_ids")
        types = mro.get("matched_object_types")
        if isinstance(ids, list) and ids and isinstance(types, list) and types:
            continue

        matches = _find_objects_by_type(objects, obj_type, limit=qty)
        mro["matched_object_ids"] = [str(o.get("id", "")) for o in matches if o.get("id")]
        mro["matched_object_types"] = [str(o.get("type") or obj_type) for o in matches]


def _coerce_task_decomposition(
    *,
    pa: Dict[str, Any],
    pick_obj: Optional[Mapping[str, Any]],
    place_obj: Optional[Mapping[str, Any]],
    require_stage4_fields: bool,
) -> None:
    td = pa.get("task_decomposition")
    if not isinstance(td, list) or not td:
        if not require_stage4_fields:
            return
        pick_type = str((pick_obj or {}).get("type") or "object")
        place_type = str((place_obj or {}).get("type") or "surface")
        td = [
            {"step": 1, "action": "navigate", "target_object": pick_type, "location_object": pick_type},
            {"step": 2, "action": "pick", "target_object": pick_type, "location_object": pick_type},
            {"step": 3, "action": "navigate", "target_object": place_type, "location_object": place_type},
            {"step": 4, "action": "place", "target_object": pick_type, "location_object": place_type},
        ]
        pa["task_decomposition"] = td


def _resolve_step_target_id(
    *,
    step: Mapping[str, Any],
    objects: Sequence[Mapping[str, Any]],
    pick_obj: Optional[Mapping[str, Any]],
    place_obj: Optional[Mapping[str, Any]],
) -> Tuple[Optional[str], Optional[str]]:
    action = _norm(step.get("action"))
    target_hint = str(
        step.get("target_object")
        or step.get("target")
        or step.get("object")
        or ""
    )
    location_hint = str(
        step.get("location_object")
        or step.get("location")
        or step.get("destination_object")
        or step.get("destination")
        or ""
    )

    target_id: Optional[str] = str(step.get("target_object_id", "") or "") or None
    location_id: Optional[str] = str(step.get("location_object_id", "") or "") or None

    if target_id and _find_object_by_id(objects, target_id) is None:
        target_id = None
    if location_id and _find_object_by_id(objects, location_id) is None:
        location_id = None

    if target_id is None and target_hint:
        matches = _find_objects_by_type(objects, target_hint, limit=1)
        if matches and matches[0].get("id"):
            target_id = str(matches[0]["id"])
    if location_id is None and location_hint:
        matches = _find_objects_by_type(objects, location_hint, limit=1)
        if matches and matches[0].get("id"):
            location_id = str(matches[0]["id"])

    if target_id is None and action == "pick" and pick_obj and pick_obj.get("id"):
        target_id = str(pick_obj["id"])
    if location_id is None and action == "pick" and place_obj and place_obj.get("id"):
        # Prefer a support/surface hint if available, else fall back to pick object itself.
        surface_match = _first_object_by_token(objects, _SURFACE_TOKENS)
        if surface_match and surface_match.get("id"):
            location_id = str(surface_match["id"])
        elif pick_obj and pick_obj.get("id"):
            location_id = str(pick_obj["id"])
    if location_id is None and action in {"place", "navigate"} and place_obj and place_obj.get("id"):
        location_id = str(place_obj["id"])

    return target_id, location_id


def _ensure_updated_task_decomposition(
    *,
    layout: Dict[str, Any],
    pa: Dict[str, Any],
    objects: Sequence[Mapping[str, Any]],
    pick_obj: Optional[Mapping[str, Any]],
    place_obj: Optional[Mapping[str, Any]],
    require_stage4_fields: bool,
) -> None:
    td = pa.get("task_decomposition")
    if not isinstance(td, list):
        td = []
    utd = pa.get("updated_task_decomposition")
    if not isinstance(utd, list) or not utd:
        if td:
            utd = [dict(step) for step in td if isinstance(step, Mapping)]
        elif require_stage4_fields:
            utd = []
        else:
            utd = []

    out: List[Dict[str, Any]] = []
    for idx, step in enumerate(utd, start=1):
        if not isinstance(step, Mapping):
            continue
        entry = dict(step)
        entry.setdefault("step", int(step.get("step") or idx))
        action = _norm(entry.get("action"))
        if not action and idx == 2:
            entry["action"] = "pick"
            action = "pick"
        if not action and idx >= 4:
            entry["action"] = "place"
            action = "place"

        target_id, location_id = _resolve_step_target_id(
            step=entry,
            objects=objects,
            pick_obj=pick_obj,
            place_obj=place_obj,
        )
        if target_id:
            entry["target_object_id"] = target_id
        if location_id:
            entry["location_object_id"] = location_id
        out.append(entry)

    if require_stage4_fields and not out:
        base = [
            {"step": 1, "action": "navigate"},
            {"step": 2, "action": "pick"},
            {"step": 3, "action": "navigate"},
            {"step": 4, "action": "place"},
        ]
        for entry in base:
            target_id, location_id = _resolve_step_target_id(
                step=entry,
                objects=objects,
                pick_obj=pick_obj,
                place_obj=place_obj,
            )
            if target_id:
                entry["target_object_id"] = target_id
            if location_id:
                entry["location_object_id"] = location_id
            out.append(entry)

    pa["updated_task_decomposition"] = out
    layout["updated_task_decomposition"] = out


def _validate_stage4_contract(layout: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    pa = layout.get("policy_analysis")
    if not isinstance(pa, Mapping):
        errors.append("policy_analysis_missing")
        return errors
    mros = pa.get("minimum_required_objects")
    if not isinstance(mros, list) or not mros:
        errors.append("minimum_required_objects_missing")
    td = pa.get("task_decomposition")
    if not isinstance(td, list) or not td:
        errors.append("task_decomposition_missing")
    utd = pa.get("updated_task_decomposition")
    if not isinstance(utd, list) or not utd:
        errors.append("updated_task_decomposition_missing")
    return errors


def patch_layout_dict(layout: Dict[str, Any], *, task_desc: str, require_stage4_fields: bool) -> Dict[str, Any]:
    objects = _extract_objects(layout)
    pa = layout.get("policy_analysis")
    if not isinstance(pa, dict):
        pa = {}
        layout["policy_analysis"] = pa

    pick_obj, place_obj = _infer_pick_place_objects(objects, task_desc)
    _ensure_mro_entries(
        pa=pa,
        objects=objects,
        pick_obj=pick_obj,
        place_obj=place_obj,
        require_stage4_fields=require_stage4_fields,
    )
    _coerce_task_decomposition(
        pa=pa,
        pick_obj=pick_obj,
        place_obj=place_obj,
        require_stage4_fields=require_stage4_fields,
    )
    _ensure_updated_task_decomposition(
        layout=layout,
        pa=pa,
        objects=objects,
        pick_obj=pick_obj,
        place_obj=place_obj,
        require_stage4_fields=require_stage4_fields,
    )

    if require_stage4_fields:
        errors = _validate_stage4_contract(layout)
        if errors:
            raise RuntimeError(f"Stage 4 required fields missing after fix: {','.join(errors)}")
    return layout


def patch_layout_file(path: Path, *, task_desc: str, require_stage4_fields: bool) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Layout JSON not found: {path}")
    backup = path.with_suffix(path.suffix + ".bak")
    if not backup.exists():
        shutil.copy2(path, backup)

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Layout payload must be a dict: {path}")
    patched = patch_layout_dict(payload, task_desc=task_desc, require_stage4_fields=require_stage4_fields)
    path.write_text(json.dumps(patched, indent=2) + "\n", encoding="utf-8")
    return patched


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Patch layout JSON for deterministic Stage 4 fields.")
    parser.add_argument("layout_json", help="Path to layout_*/layout_*.json")
    parser.add_argument("--task-desc", default="", help="Task text used to infer pick/place anchors")
    parser.add_argument(
        "--require-stage4-fields",
        action="store_true",
        help="Fail if Stage 4 required fields cannot be synthesized",
    )
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    patched = patch_layout_file(
        Path(args.layout_json).expanduser().resolve(),
        task_desc=str(args.task_desc or ""),
        require_stage4_fields=bool(args.require_stage4_fields),
    )
    pa = patched.get("policy_analysis", {})
    mro_count = len(pa.get("minimum_required_objects", []) or []) if isinstance(pa, dict) else 0
    utd_count = len(pa.get("updated_task_decomposition", []) or []) if isinstance(pa, dict) else 0
    print(f"Layout JSON patched successfully: minimum_required_objects={mro_count} updated_task_decomposition={utd_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
