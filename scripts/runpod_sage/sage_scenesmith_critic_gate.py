#!/usr/bin/env python3
"""SceneSmith-style critic gate for SAGE layout JSONs.

Evaluates a generated SAGE layout using the lightweight SceneSmith critic
utilities in ``tools/source_pipeline/agent_loop.py`` and exits nonzero when the
scene does not meet configured thresholds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_str(value: Any, default: str = "") -> str:
    text = str(value or "").strip()
    return text if text else default


def _get_layout_objects(layout: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rooms = layout.get("rooms")
    if isinstance(rooms, list) and rooms:
        room0 = rooms[0]
        if isinstance(room0, Mapping):
            objs = room0.get("objects")
            if isinstance(objs, list):
                return [dict(o) for o in objs if isinstance(o, Mapping)]
    objs = layout.get("objects")
    if isinstance(objs, list):
        return [dict(o) for o in objs if isinstance(o, Mapping)]
    room = layout.get("room")
    if isinstance(room, Mapping):
        objs = room.get("objects")
        if isinstance(objs, list):
            return [dict(o) for o in objs if isinstance(o, Mapping)]
    return []


def _get_room_dimensions(layout: Mapping[str, Any]) -> Tuple[float, float, float]:
    rooms = layout.get("rooms")
    if isinstance(rooms, list) and rooms:
        room0 = rooms[0]
        if isinstance(room0, Mapping):
            dims = room0.get("dimensions")
            if isinstance(dims, Mapping):
                return (
                    max(1.0, _safe_float(dims.get("width"), 6.0)),
                    max(1.0, _safe_float(dims.get("length"), 6.0)),
                    max(2.0, _safe_float(dims.get("height"), 3.0)),
                )
    dims = layout.get("dimensions")
    if isinstance(dims, Mapping):
        return (
            max(1.0, _safe_float(dims.get("width"), 6.0)),
            max(1.0, _safe_float(dims.get("length"), 6.0)),
            max(2.0, _safe_float(dims.get("height"), 3.0)),
        )
    return 6.0, 6.0, 3.0


def _is_manipulable(obj_type: str, dims: Mapping[str, float]) -> bool:
    t = obj_type.lower()
    manipulable_tokens = (
        "mug",
        "cup",
        "bowl",
        "jar",
        "book",
        "plate",
        "spoon",
        "fork",
        "knife",
        "bottle",
        "can",
        "glass",
        "vase",
        "fruit",
        "apple",
        "banana",
    )
    if any(tok in t for tok in manipulable_tokens):
        return True
    volume = float(dims.get("width", 0.0)) * float(dims.get("depth", 0.0)) * float(dims.get("height", 0.0))
    return volume < 0.025


def _to_scene_critic_objects(layout_objects: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for idx, obj in enumerate(layout_objects, start=1):
        obj_id = _safe_str(obj.get("id"), f"obj_{idx:03d}")
        obj_type = _safe_str(obj.get("type") or obj.get("object_type") or obj.get("category"), "object")

        pos = obj.get("position") if isinstance(obj.get("position"), Mapping) else {}
        dims = obj.get("dimensions") if isinstance(obj.get("dimensions"), Mapping) else {}
        width = max(0.02, abs(_safe_float(dims.get("width"), 0.25)))
        depth = max(0.02, abs(_safe_float(dims.get("length"), _safe_float(dims.get("depth"), 0.25))))
        height = max(0.02, abs(_safe_float(dims.get("height"), 0.25)))

        # SAGE: z-up with footprint in x/y. SceneSmith critic helpers: y-up footprint x/z.
        x = _safe_float(pos.get("x"), 0.0)
        y = _safe_float(pos.get("z"), 0.0)
        z = _safe_float(pos.get("y"), 0.0)
        sim_role = "manipulable_object" if _is_manipulable(obj_type, {"width": width, "depth": depth, "height": height}) else "static"
        placement_stage = "manipulands" if sim_role == "manipulable_object" else "furniture"

        out.append(
            {
                "id": obj_id,
                "name": obj_type,
                "category": obj_type,
                "sim_role": sim_role,
                "placement_stage": placement_stage,
                "transform": {"position": {"x": float(x), "y": float(y), "z": float(z)}},
                "dimensions_est": {
                    "width": float(width),
                    "depth": float(depth),
                    "height": float(height),
                },
            }
        )
    return out


def _evaluate(
    *,
    task_desc: str,
    room_dims: Tuple[float, float, float],
    objects: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from tools.source_pipeline.agent_loop import _critic_score, compute_faithfulness_report  # type: ignore

    room_w, room_l, room_h = room_dims
    room_box = {"min": [0.0, 0.0, 0.0], "max": [float(room_w), float(room_h), float(room_l)]}
    critic = _critic_score(prompt=task_desc, objects=objects, room_box=room_box)
    faith = compute_faithfulness_report(task_desc, objects)
    result = dict(critic)
    result["faithfulness_report"] = faith
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="SceneSmith-style critic gate for SAGE layout JSON")
    ap.add_argument("--layout_json", required=True, help="Path to layout_*/layout_*.json")
    ap.add_argument("--task_desc", required=True, help="Task prompt/description used for semantic alignment")
    ap.add_argument("--report_path", required=True, help="Output JSON report path")
    ap.add_argument("--min_total_0_10", type=float, default=8.0)
    ap.add_argument("--min_faithfulness", type=float, default=0.80)
    ap.add_argument("--max_collision_rate", type=float, default=0.08)
    args = ap.parse_args()

    layout_path = Path(args.layout_json).expanduser().resolve()
    if not layout_path.is_file():
        raise FileNotFoundError(f"Layout JSON not found: {layout_path}")

    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    if not isinstance(layout, Mapping):
        raise ValueError(f"Layout JSON must be an object: {layout_path}")

    layout_objects = _get_layout_objects(layout)
    critic_objects = _to_scene_critic_objects(layout_objects)
    room_dims = _get_room_dimensions(layout)
    critic = _evaluate(task_desc=str(args.task_desc), room_dims=room_dims, objects=critic_objects)

    total = _safe_float(critic.get("total"), 0.0)
    collision_rate = _safe_float(critic.get("collision_rate"), 1.0)
    faith = critic.get("faithfulness_report") if isinstance(critic.get("faithfulness_report"), Mapping) else {}
    faith_score = _safe_float(faith.get("score"), 0.0)

    failures: List[str] = []
    if total < float(args.min_total_0_10):
        failures.append(f"critic total {total:.3f} < {float(args.min_total_0_10):.3f}")
    if faith_score < float(args.min_faithfulness):
        failures.append(f"faithfulness {faith_score:.3f} < {float(args.min_faithfulness):.3f}")
    if collision_rate > float(args.max_collision_rate):
        failures.append(f"collision_rate {collision_rate:.4f} > {float(args.max_collision_rate):.4f}")

    report = {
        "schema_version": "v1",
        "layout_json": str(layout_path),
        "task_desc": str(args.task_desc),
        "room_dimensions": {"width": room_dims[0], "length": room_dims[1], "height": room_dims[2]},
        "object_count": len(critic_objects),
        "thresholds": {
            "min_total_0_10": float(args.min_total_0_10),
            "min_faithfulness": float(args.min_faithfulness),
            "max_collision_rate": float(args.max_collision_rate),
        },
        "critic": critic,
        "all_pass": len(failures) == 0,
        "failures": failures,
    }

    out_path = Path(args.report_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if report["all_pass"]:
        print(f"[sage-critic-gate] PASS total={total:.3f} faith={faith_score:.3f} collision={collision_rate:.4f}")
        return 0
    print(f"[sage-critic-gate] FAIL {'; '.join(failures)}")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
