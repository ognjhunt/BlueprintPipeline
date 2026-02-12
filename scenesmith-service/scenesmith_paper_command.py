#!/usr/bin/env python3
"""Command bridge that runs official SceneSmith and emits Stage-1 objects.

Input: JSON request on stdin (SceneSmith wrapper contract).
Output: JSON object on stdout containing at least an ``objects`` list.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Mapping

_STATIC_CLASSES = {
    "bed",
    "bookshelf",
    "cabinet",
    "counter",
    "desk",
    "dishwasher",
    "dresser",
    "fridge",
    "nightstand",
    "shelf",
    "side_table",
    "sink",
    "sofa",
    "stove",
    "table",
    "tv_stand",
    "wardrobe",
    "washer",
}

_ARTICULATED_CLASSES = {
    "cabinet",
    "dishwasher",
    "drawer",
    "fridge",
    "microwave",
    "oven",
    "stove",
    "wardrobe",
    "washer",
}

_ARTICULATION_KEYWORDS = {
    "cabinet",
    "dishwasher",
    "drawer",
    "door",
    "fridge",
    "hinge",
    "microwave",
    "oven",
    "washer",
}

_PLACEMENT_AGENT_CONFIG_PREFIXES = (
    "furniture_agent",
    "wall_agent",
    "ceiling_agent",
    "manipuland_agent",
)

_VALID_PIPELINE_STAGES = {
    "floor_plan",
    "furniture",
    "wall_mounted",
    "ceiling_mounted",
    "manipuland",
}


def _safe_float(value: Any, *, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed


def _safe_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _is_truthy(raw: str | None, *, default: bool = False) -> bool:
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if not normalized:
        return default
    return normalized in {"1", "true", "yes", "on", "y"}


def _slug(value: str, *, default: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip())
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-")
    if not normalized:
        return default
    return normalized[:96]


def _coerce_optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on", "y"}:
            return True
        if normalized in {"0", "false", "no", "off", "n"}:
            return False
    return None


def _run_root(scene_id: str) -> Path:
    run_root = Path(os.getenv("SCENESMITH_PAPER_RUN_ROOT", "/tmp/scenesmith-paper-runs")).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    tag = uuid.uuid4().hex[:10]
    return run_root / f"{_slug(scene_id, default='scene')}-{tag}"


def _parse_extra_overrides(raw: str | None) -> List[str]:
    """Parse user-provided Hydra override strings from env.

    Supports either:
    - JSON array string: '["a=b", "c=d"]'
    - Delimited string: "a=b;;c=d"
    """
    if raw is None:
        return []
    value = raw.strip()
    if not value:
        return []

    if value.startswith("["):
        parsed = json.loads(value)
        if not isinstance(parsed, list):
            raise ValueError("SCENESMITH_PAPER_EXTRA_OVERRIDES JSON must be a list")
        overrides: List[str] = []
        for item in parsed:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                overrides.append(text)
        return overrides

    return [part.strip() for part in value.split(";;") if part.strip()]


def _classify_role(semantic_class: str, raw_obj: Mapping[str, Any]) -> str:
    category = semantic_class.strip().lower().replace(" ", "_")
    joint_type = str(raw_obj.get("joint_type") or "").strip().lower()
    if joint_type and joint_type not in {"none", "fixed"}:
        return "articulated_furniture"
    if category in _ARTICULATED_CLASSES:
        return "articulated_furniture"
    if bool(raw_obj.get("floor_object")) or category in _STATIC_CLASSES:
        return "static"
    return "manipulable_object"


def _infer_articulation_candidate(semantic_class: str, raw_obj: Mapping[str, Any]) -> tuple[bool, str]:
    category = semantic_class.strip().lower().replace(" ", "_")
    articulation = raw_obj.get("articulation")
    if isinstance(articulation, Mapping):
        explicit_candidate = _coerce_optional_bool(articulation.get("candidate"))
        if explicit_candidate is not None:
            return explicit_candidate, "input.articulation.candidate"

    explicit_requires = _coerce_optional_bool(raw_obj.get("requires_articulation"))
    if explicit_requires is True:
        return True, "input.requires_articulation"

    joint_type = str(raw_obj.get("joint_type") or "").strip().lower()
    if joint_type and joint_type not in {"none", "fixed"}:
        return True, "joint_type"

    if category in _ARTICULATED_CLASSES:
        return True, "category_default"

    text_parts = []
    for key in ("semantic_class", "class_name", "name", "id", "description"):
        value = raw_obj.get(key)
        if value is not None:
            text_parts.append(str(value).lower())
    if text_parts:
        text_blob = " ".join(text_parts)
        if any(keyword in text_blob for keyword in _ARTICULATION_KEYWORDS):
            return True, "semantic_keyword"

    return False, "none"


def _infer_articulation_required(
    *,
    raw_obj: Mapping[str, Any],
    sim_role: str,
    articulation_candidate: bool,
    force_generated_assets: bool,
) -> tuple[bool, str]:
    articulation = raw_obj.get("articulation")
    if isinstance(articulation, Mapping):
        explicit_required = _coerce_optional_bool(articulation.get("required"))
        if explicit_required is not None:
            return explicit_required, "input.articulation.required"

    explicit_requires = _coerce_optional_bool(raw_obj.get("requires_articulation"))
    if explicit_requires is not None:
        return explicit_requires, "input.requires_articulation"

    joint_type = str(raw_obj.get("joint_type") or "").strip().lower()
    if joint_type and joint_type not in {"none", "fixed"}:
        return True, "joint_type"

    if force_generated_assets:
        return False, "force_generated_assets"

    require_class_defaults = _is_truthy(
        os.getenv("SCENESMITH_PAPER_REQUIRE_ARTICULATION_CLASS_DEFAULTS"),
        default=False,
    )
    if require_class_defaults and articulation_candidate and sim_role.startswith("articulated"):
        return True, "class_default"

    if articulation_candidate:
        return False, "optional_candidate"
    return False, "none"


def _object_from_house_state(
    raw_obj: Mapping[str, Any],
    index: int,
    *,
    force_generated_assets: bool,
) -> Dict[str, Any]:
    obj_id = str(raw_obj.get("id") or f"obj_{index:03d}").strip() or f"obj_{index:03d}"
    semantic = str(raw_obj.get("semantic_class") or raw_obj.get("class_name") or "object").strip() or "object"
    category = semantic.lower().replace(" ", "_")
    sim_role = _classify_role(category, raw_obj)

    pose = raw_obj.get("pose") if isinstance(raw_obj.get("pose"), Mapping) else {}
    position = pose.get("position") if isinstance(pose.get("position"), Mapping) else {}
    orientation = pose.get("orientation") if isinstance(pose.get("orientation"), Mapping) else {}
    extent = raw_obj.get("extent") if isinstance(raw_obj.get("extent"), Mapping) else {}

    width = max(0.01, abs(_safe_float(extent.get("x"), default=0.3)))
    height = max(0.01, abs(_safe_float(extent.get("y"), default=0.3)))
    depth = max(0.01, abs(_safe_float(extent.get("z"), default=0.3)))

    placement_stage = "manipulands" if sim_role == "manipulable_object" else "furniture"
    articulation_candidate, candidate_source = _infer_articulation_candidate(category, raw_obj)
    articulation_required, requirement_source = _infer_articulation_required(
        raw_obj=raw_obj,
        sim_role=sim_role,
        articulation_candidate=articulation_candidate,
        force_generated_assets=force_generated_assets,
    )
    if articulation_required:
        backend_hint = "particulate_first"
    elif articulation_candidate:
        backend_hint = "particulate_optional"
    else:
        backend_hint = "none"

    return {
        "id": obj_id,
        "name": semantic,
        "category": category,
        "sim_role": sim_role,
        "description": f"{semantic} generated by official SceneSmith",
        "transform": {
            "position": {
                "x": round(_safe_float(position.get("x"), default=0.0), 4),
                "y": round(max(0.0, _safe_float(position.get("y"), default=0.0)), 4),
                "z": round(_safe_float(position.get("z"), default=0.0), 4),
            },
            "scale": {"x": 1.0, "y": 1.0, "z": 1.0},
            "rotation_quaternion": {
                "w": _safe_float(orientation.get("w"), default=1.0),
                "x": _safe_float(orientation.get("x"), default=0.0),
                "y": _safe_float(orientation.get("y"), default=0.0),
                "z": _safe_float(orientation.get("z"), default=0.0),
            },
        },
        "dimensions_est": {
            "width": round(width, 4),
            "height": round(height, 4),
            "depth": round(depth, 4),
        },
        "physics_hints": {
            "gravity": True,
            "pickable": sim_role == "manipulable_object",
            "collision_shape": "convex_hull",
        },
        "asset_strategy": "retrieved",
        "placement_stage": placement_stage,
        "articulation": {
            "candidate": articulation_candidate,
            "required": articulation_required,
            "backend_hint": backend_hint,
            "detection_source": candidate_source,
            "requirement_source": requirement_source,
        },
        "source_backend": "scenesmith_paper",
    }


def _collect_raw_objects(house_state: Mapping[str, Any]) -> List[Dict[str, Any]]:
    objects = house_state.get("objects")
    if isinstance(objects, list):
        return [dict(item) for item in objects if isinstance(item, Mapping)]

    combined: List[Dict[str, Any]] = []
    for key in ("furniture", "surface_objects", "ceiling_lights"):
        payload = house_state.get(key)
        if isinstance(payload, list):
            combined.extend(dict(item) for item in payload if isinstance(item, Mapping))
    return combined


def _load_house_state(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_house_state(run_dir: Path, scene_name: str) -> Path:
    direct_path = run_dir / "outputs" / scene_name / "scene_000" / "combined_house" / "house_state.json"
    if direct_path.is_file():
        return direct_path

    candidates = sorted((run_dir / "outputs").glob("*/scene_*/combined_house/house_state.json"))
    if candidates:
        return candidates[-1]

    raise FileNotFoundError(f"Unable to locate house_state.json under {run_dir}")


def _hydra_overrides(*, payload: Mapping[str, Any], run_dir: Path, scene_name: str) -> List[str]:
    prompt = str(payload.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("prompt is required")

    overrides = [
        f"hydra.run.dir={run_dir.as_posix()}",
        f"+name={scene_name}",
        f"experiment.prompts={json.dumps([prompt])}",
        "experiment.num_workers=1",
        "experiment.pipeline.parallel_rooms=false",
        "experiment.geometry_generation_server.port=17005",
        "experiment.hssd_retrieval_server.port=17006",
        "experiment.articulated_retrieval_server.port=17007",
        "experiment.materials_retrieval_server.port=17008",
        "experiment.objaverse_retrieval_server.port=17009",
    ]

    requested_stages = payload.get("pipeline_stages")
    if isinstance(requested_stages, list):
        normalized_stages = [
            str(stage).strip().lower()
            for stage in requested_stages
            if str(stage).strip().lower() in _VALID_PIPELINE_STAGES
        ]
        if normalized_stages:
            overrides.append(f"experiment.pipeline.start_stage={normalized_stages[0]}")
            overrides.append(f"experiment.pipeline.stop_stage={normalized_stages[-1]}")

    all_sam3d = _is_truthy(os.getenv("SCENESMITH_PAPER_ALL_SAM3D"), default=False)
    force_generated = all_sam3d or _is_truthy(
        os.getenv("SCENESMITH_PAPER_FORCE_GENERATED_ASSETS"),
        default=False,
    )
    disable_articulated = all_sam3d or _is_truthy(
        os.getenv("SCENESMITH_PAPER_DISABLE_ARTICULATED_STRATEGY"),
        default=False,
    )

    if force_generated:
        for prefix in _PLACEMENT_AGENT_CONFIG_PREFIXES:
            overrides.extend(
                [
                    f"{prefix}.asset_manager.general_asset_source=generated",
                    f"{prefix}.asset_manager.backend=sam3d",
                    f"{prefix}.asset_manager.router.strategies.generated.enabled=true",
                ]
            )

    if disable_articulated:
        for prefix in _PLACEMENT_AGENT_CONFIG_PREFIXES:
            overrides.extend(
                [
                    f"{prefix}.asset_manager.router.strategies.articulated.enabled=false",
                    f"{prefix}.asset_manager.articulated.sources.partnet_mobility.enabled=false",
                    f"{prefix}.asset_manager.articulated.sources.artvip.enabled=false",
                ]
            )

    image_backend = (
        str(os.getenv("SCENESMITH_PAPER_IMAGE_BACKEND", "")).strip().lower()
    )
    if image_backend in {"openai", "gemini"}:
        for prefix in _PLACEMENT_AGENT_CONFIG_PREFIXES:
            overrides.append(f"{prefix}.asset_manager.image_generation.backend={image_backend}")

    use_gemini_context_image = _is_truthy(
        os.getenv("SCENESMITH_PAPER_ENABLE_GEMINI_CONTEXT_IMAGE"),
        default=False,
    )
    if use_gemini_context_image:
        overrides.extend(
            [
                "furniture_agent.context_image_generation.enabled=true",
                "furniture_agent.asset_manager.image_generation.backend=gemini",
            ]
        )

    if _is_truthy(os.getenv("SCENESMITH_PAPER_ENABLE_FURNITURE_CONTEXT_IMAGE"), default=False):
        overrides.append("furniture_agent.context_image_generation.enabled=true")

    overrides.extend(_parse_extra_overrides(os.getenv("SCENESMITH_PAPER_EXTRA_OVERRIDES")))

    return overrides


def _run_official_scenesmith(payload: Mapping[str, Any]) -> Dict[str, Any]:
    repo_dir_raw = str(os.getenv("SCENESMITH_PAPER_REPO_DIR", "")).strip()
    if not repo_dir_raw:
        raise RuntimeError("SCENESMITH_PAPER_REPO_DIR is required")
    repo_dir = Path(repo_dir_raw).expanduser().resolve()
    if not repo_dir.is_dir():
        raise RuntimeError(f"SCENESMITH_PAPER_REPO_DIR not found: {repo_dir}")

    python_bin = str(os.getenv("SCENESMITH_PAPER_PYTHON_BIN", "python3")).strip() or "python3"
    timeout = _safe_int(os.getenv("SCENESMITH_PAPER_TIMEOUT_SECONDS"), default=5400)

    scene_id = str(payload.get("scene_id") or "scene").strip() or "scene"
    scene_name = _slug(scene_id, default="scene")
    run_dir = _run_root(scene_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    keep_run_dir = _is_truthy(os.getenv("SCENESMITH_PAPER_KEEP_RUN_DIR"), default=False)
    force_generated_assets = _is_truthy(os.getenv("SCENESMITH_PAPER_ALL_SAM3D"), default=False) or _is_truthy(
        os.getenv("SCENESMITH_PAPER_FORCE_GENERATED_ASSETS"),
        default=False,
    )

    try:
        overrides = _hydra_overrides(payload=payload, run_dir=run_dir, scene_name=scene_name)
        cmd: List[str] = [python_bin, "main.py"] + overrides
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")

        completed = subprocess.run(
            cmd,
            cwd=str(repo_dir),
            env=env,
            capture_output=True,
            text=True,
            timeout=max(30, timeout),
            check=False,
        )
        if completed.returncode != 0:
            stdout_tail = (completed.stdout or "")[-4000:]
            stderr_tail = (completed.stderr or "")[-4000:]
            raise RuntimeError(
                "Official SceneSmith failed "
                f"(exit={completed.returncode}). stdout_tail={stdout_tail!r} stderr_tail={stderr_tail!r}"
            )

        house_state_path = _find_house_state(run_dir, scene_name)
        house_state = _load_house_state(house_state_path)
        raw_objects = _collect_raw_objects(house_state)
        if not raw_objects:
            raise RuntimeError(f"Official SceneSmith returned zero objects (house_state={house_state_path})")

        objects = [
            _object_from_house_state(
                item,
                idx,
                force_generated_assets=force_generated_assets,
            )
            for idx, item in enumerate(raw_objects, start=1)
        ]
        constraints = payload.get("constraints")
        room_type = (
            str(constraints.get("room_type") or "generic_room")
            if isinstance(constraints, Mapping)
            else "generic_room"
        )

        return {
            "schema_version": "v1",
            "request_id": str(payload.get("request_id") or ""),
            "room_type": room_type,
            "objects": objects,
            "used_llm": True,
            "llm_attempts": 1,
            "fallback_strategy": "none",
            "paper_stack": {
                "enabled": True,
                "repo_dir": str(repo_dir),
                "run_dir": str(run_dir),
                "house_state_path": str(house_state_path),
                "object_count": len(objects),
                "backend": str(os.getenv("SCENESMITH_PAPER_BACKEND", "openai")),
                "model": str(os.getenv("SCENESMITH_PAPER_MODEL", "")),
            },
        }
    finally:
        if run_dir.exists() and not keep_run_dir:
            shutil.rmtree(run_dir, ignore_errors=True)


def _read_stdin_payload() -> Dict[str, Any]:
    raw = sys.stdin.read()
    if not raw.strip():
        raise ValueError("Expected JSON payload on stdin")
    parsed = json.loads(raw)
    if not isinstance(parsed, Mapping):
        raise ValueError("Input payload must be a JSON object")
    return dict(parsed)


def main() -> int:
    try:
        payload = _read_stdin_payload()
        response = _run_official_scenesmith(payload)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    json.dump(response, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
