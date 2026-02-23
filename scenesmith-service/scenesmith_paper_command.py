#!/usr/bin/env python3
"""Command bridge that runs official SceneSmith and emits Stage-1 objects.

Input: JSON request on stdin (SceneSmith wrapper contract).
Output: JSON object on stdout containing at least an ``objects`` list.
"""

from __future__ import annotations

import json
import os
import re
import signal
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

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

_CRITIC_PAYLOAD_KEYS = (
    "placement_stages",
    "critic_scores",
    "support_surfaces",
    "faithfulness_report",
    "quality_gate_report",
)

_OPENAI_API_WRAPPER = """\
import os
import runpy
import sys


def _configure_openai_agents() -> None:
    api = os.environ.get("SCENESMITH_PAPER_OPENAI_API", "").strip().lower()
    if not api:
        return

    try:
        from agents import set_default_openai_api, set_default_openai_client
        from openai import AsyncOpenAI
    except Exception:
        return

    if api in {"chat", "chat_completions"}:
        set_default_openai_api("chat_completions")
    elif api in {"responses"}:
        set_default_openai_api("responses")

    base_url = os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("OPENAI_API_KEY")
    if base_url and api_key:
        set_default_openai_client(AsyncOpenAI(base_url=base_url, api_key=api_key))


_configure_openai_agents()

args = sys.argv[1:]
sys.argv = ["main.py"] + args
runpy.run_path("main.py", run_name="__main__")
"""


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


def _resolve_existing_run_dir() -> Path | None:
    raw = str(os.getenv("SCENESMITH_PAPER_EXISTING_RUN_DIR", "")).strip()
    if not raw:
        return None
    path = Path(raw).expanduser().resolve()
    if not path.is_dir():
        raise RuntimeError(f"SCENESMITH_PAPER_EXISTING_RUN_DIR not found: {path}")
    return path


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


def _normalize_paper_openai_api(raw: str | None) -> str:
    value = str(raw or "").strip().lower()
    if not value:
        return ""
    if value in {"chat", "chat_completions"}:
        return "chat_completions"
    if value in {"responses"}:
        return "responses"
    raise ValueError(f"Unsupported SCENESMITH_PAPER_OPENAI_API={raw!r} (expected 'responses' or 'chat_completions')")


def _parse_model_chain(raw: str | None) -> List[str]:
    value = str(raw or "").strip()
    if not value:
        return []
    if value.startswith("["):
        parsed = json.loads(value)
        if not isinstance(parsed, list):
            raise ValueError("SCENESMITH_PAPER_MODEL_CHAIN JSON must be a list")
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [part.strip() for part in value.split(",") if part.strip()]


def _resolve_paper_model_attempts() -> List[str]:
    explicit_model = str(os.getenv("SCENESMITH_PAPER_MODEL", "")).strip()
    if explicit_model:
        return [explicit_model]

    chain = _parse_model_chain(os.getenv("SCENESMITH_PAPER_MODEL_CHAIN"))
    if chain:
        return chain

    # Empty string means "let upstream/default model resolve".
    return [""]


def _apply_paper_openai_env_overrides(
    env: Dict[str, str],
    *,
    model_override: str | None = None,
) -> None:
    """Apply paper-stack scoped OpenAI/Agents SDK overrides for the subprocess.

    This lets the parent process keep its own OpenAI config while the official
    SceneSmith stack uses a different key/base URL/model (e.g. OpenRouter).
    """

    api_key = str(os.getenv("SCENESMITH_PAPER_OPENAI_API_KEY", "")).strip()
    if api_key:
        env["OPENAI_API_KEY"] = api_key

    base_url = str(os.getenv("SCENESMITH_PAPER_OPENAI_BASE_URL", "")).strip()
    if base_url:
        env["OPENAI_BASE_URL"] = base_url

    tracing_key = str(os.getenv("SCENESMITH_PAPER_OPENAI_TRACING_KEY", "")).strip()
    if tracing_key:
        env["OPENAI_TRACING_KEY"] = tracing_key

    # Bridge selected model into the OpenAI Agents SDK default model.
    if model_override is None:
        model = str(os.getenv("SCENESMITH_PAPER_MODEL", "")).strip()
    else:
        model = str(model_override).strip()
    if model:
        env["OPENAI_DEFAULT_MODEL"] = model
        # Some stacks read OPENAI_MODEL instead.
        env["OPENAI_MODEL"] = model

    api = _normalize_paper_openai_api(os.getenv("SCENESMITH_PAPER_OPENAI_API"))
    if api:
        env["SCENESMITH_PAPER_OPENAI_API"] = api


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
    def _looks_like_object(candidate: Mapping[str, Any]) -> bool:
        identity_present = any(
            candidate.get(key) is not None
            for key in ("id", "object_id", "source_id", "semantic_class", "class_name", "category", "type", "name")
        )
        if not identity_present:
            return False

        if isinstance(candidate.get("extent"), Mapping):
            return True
        if isinstance(candidate.get("dimensions"), Mapping):
            return True
        if isinstance(candidate.get("pose"), Mapping):
            return True
        if isinstance(candidate.get("transform"), Mapping):
            return True
        if isinstance(candidate.get("position"), Mapping):
            return True
        return False

    def _normalize_candidate(candidate: Mapping[str, Any], *, fallback_index: int) -> Dict[str, Any]:
        out = dict(candidate)

        semantic = (
            out.get("semantic_class")
            or out.get("class_name")
            or out.get("category")
            or out.get("type")
            or out.get("name")
            or "object"
        )
        out["semantic_class"] = str(semantic).strip() or "object"

        obj_id = out.get("id")
        if obj_id is None or str(obj_id).strip() == "":
            alt_id = out.get("object_id") or out.get("source_id") or out.get("name")
            out["id"] = str(alt_id).strip() if alt_id else f"obj_{fallback_index:03d}"

        extent = out.get("extent")
        if not isinstance(extent, Mapping):
            dims = out.get("dimensions") if isinstance(out.get("dimensions"), Mapping) else {}
            out["extent"] = {
                "x": dims.get("x", dims.get("width", dims.get("w", 0.3))),
                "y": dims.get("y", dims.get("height", dims.get("h", 0.3))),
                "z": dims.get("z", dims.get("depth", dims.get("length", dims.get("l", 0.3)))),
            }

        pose = out.get("pose")
        if not isinstance(pose, Mapping):
            transform = out.get("transform") if isinstance(out.get("transform"), Mapping) else {}
            position = (
                transform.get("position")
                if isinstance(transform.get("position"), Mapping)
                else out.get("position")
                if isinstance(out.get("position"), Mapping)
                else {}
            )
            orientation = {}
            if isinstance(transform.get("rotation_quaternion"), Mapping):
                orientation = dict(transform.get("rotation_quaternion") or {})
            elif isinstance(transform.get("orientation"), Mapping):
                orientation = dict(transform.get("orientation") or {})
            elif isinstance(out.get("orientation"), Mapping):
                orientation = dict(out.get("orientation") or {})
            elif isinstance(out.get("rotation_quaternion"), Mapping):
                orientation = dict(out.get("rotation_quaternion") or {})
            out["pose"] = {"position": dict(position or {}), "orientation": dict(orientation or {})}

        if "floor_object" not in out and isinstance(out.get("sim_role"), str):
            out["floor_object"] = str(out.get("sim_role")).strip().lower() in {"static", "articulated_furniture"}

        articulation = out.get("articulation")
        if isinstance(articulation, Mapping):
            if "joint_type" not in out and articulation.get("joint_type") is not None:
                out["joint_type"] = articulation.get("joint_type")
            if "requires_articulation" not in out and articulation.get("required") is not None:
                out["requires_articulation"] = bool(articulation.get("required"))

        return out

    def _walk(node: Any) -> List[Mapping[str, Any]]:
        out: List[Mapping[str, Any]] = []
        if isinstance(node, Mapping):
            out.append(node)
            for value in node.values():
                out.extend(_walk(value))
        elif isinstance(node, list):
            for value in node:
                out.extend(_walk(value))
        return out

    def _dedupe_key(item: Mapping[str, Any]) -> str:
        item_id = str(item.get("id") or "").strip()
        if item_id:
            return f"id:{item_id.lower()}"
        pose = item.get("pose") if isinstance(item.get("pose"), Mapping) else {}
        position = pose.get("position") if isinstance(pose.get("position"), Mapping) else {}
        extent = item.get("extent") if isinstance(item.get("extent"), Mapping) else {}
        semantic = str(item.get("semantic_class") or "").strip().lower()
        return (
            "sig:"
            f"{semantic}:"
            f"{_safe_float(position.get('x'), default=0.0):.4f}:"
            f"{_safe_float(position.get('y'), default=0.0):.4f}:"
            f"{_safe_float(position.get('z'), default=0.0):.4f}:"
            f"{_safe_float(extent.get('x'), default=0.0):.4f}:"
            f"{_safe_float(extent.get('y'), default=0.0):.4f}:"
            f"{_safe_float(extent.get('z'), default=0.0):.4f}"
        )

    combined: List[Dict[str, Any]] = []
    seen: set[str] = set()

    def _append_if_object(candidate: Any) -> None:
        if not isinstance(candidate, Mapping):
            return
        if not _looks_like_object(candidate):
            return
        normalized = _normalize_candidate(candidate, fallback_index=len(combined) + 1)
        key = _dedupe_key(normalized)
        if key in seen:
            return
        seen.add(key)
        combined.append(normalized)

    for key in ("objects", "furniture", "surface_objects", "ceiling_lights"):
        payload = house_state.get(key)
        if isinstance(payload, list):
            for item in payload:
                _append_if_object(item)
    if combined:
        return combined

    for candidate in _walk(house_state):
        _append_if_object(candidate)
    return combined


def _load_house_state(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_house_state(run_dir: Path, scene_name: str) -> Path:
    direct_path = run_dir / "outputs" / scene_name / "scene_000" / "combined_house" / "house_state.json"
    if direct_path.is_file():
        return direct_path
    direct_no_outputs = run_dir / scene_name / "scene_000" / "combined_house" / "house_state.json"
    if direct_no_outputs.is_file():
        return direct_no_outputs

    roots = [run_dir / "outputs", run_dir]
    seen: set[str] = set()
    candidates: List[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        for pattern in (
            f"{scene_name}/scene_*/combined_house/house_state.json",
            "*/scene_*/combined_house/house_state.json",
            "scene_*/combined_house/house_state.json",
        ):
            for candidate in sorted(root.glob(pattern)):
                resolved = str(candidate.resolve())
                if resolved in seen:
                    continue
                seen.add(resolved)
                candidates.append(candidate)
    if candidates:
        candidates_sorted = sorted(candidates, key=lambda item: str(item))
        return candidates_sorted[-1]

    raise FileNotFoundError(f"Unable to locate house_state.json under {run_dir}")


def _find_house_state_optional(run_dir: Path, scene_name: str) -> Path | None:
    try:
        return _find_house_state(run_dir, scene_name)
    except FileNotFoundError:
        return None


def _tail_text(path: Path, *, max_bytes: int = 4000) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes), os.SEEK_SET)
            payload = handle.read()
    except OSError:
        return ""
    return payload.decode("utf-8", errors="replace")


def _signal_process_group(proc: subprocess.Popen[bytes], sig: int) -> None:
    if proc.poll() is not None:
        return
    try:
        if hasattr(os, "killpg"):
            os.killpg(proc.pid, sig)
            return
    except Exception:
        pass
    try:
        proc.send_signal(sig)
    except Exception:
        pass


def _terminate_process_group(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    _signal_process_group(proc, signal.SIGINT)
    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.2)

    _signal_process_group(proc, signal.SIGTERM)
    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.2)

    _signal_process_group(proc, signal.SIGKILL)


def _run_scenesmith_process(
    *,
    cmd: List[str],
    repo_dir: Path,
    env: Mapping[str, str],
    timeout_seconds: int,
    run_dir: Path,
    scene_name: str,
) -> Dict[str, Any]:
    timeout_seconds = max(30, int(timeout_seconds))
    poll_seconds = max(1, _safe_int(os.getenv("SCENESMITH_PAPER_WATCH_POLL_SECONDS"), default=2))
    grace_after_house_state = max(
        5,
        _safe_int(os.getenv("SCENESMITH_PAPER_POST_HOUSE_STATE_GRACE_SECONDS"), default=45),
    )
    prompt_markers = [
        token.strip().lower()
        for token in str(os.getenv("SCENESMITH_PAPER_EXIT_PROMPT_MARKERS", "Press Ctrl+C to exit")).split(";;")
        if token.strip()
    ]

    stdout_log = run_dir / "official_scenesmith.stdout.log"
    stderr_log = run_dir / "official_scenesmith.stderr.log"
    house_state_path: Path | None = None
    forced_exit_reason = ""
    timed_out = False

    with stdout_log.open("wb") as stdout_handle, stderr_log.open("wb") as stderr_handle:
        proc = subprocess.Popen(
            cmd,
            cwd=str(repo_dir),
            env=dict(env),
            stdout=stdout_handle,
            stderr=stderr_handle,
            start_new_session=True,
        )
        start = time.monotonic()
        house_state_seen_at: float | None = None
        prompt_seen = False

        while True:
            if proc.poll() is not None:
                break

            now = time.monotonic()
            if house_state_path is None:
                house_state_path = _find_house_state_optional(run_dir, scene_name)
                if house_state_path is not None:
                    house_state_seen_at = now

            if prompt_markers:
                tail_blob = (
                    _tail_text(stdout_log, max_bytes=8000) + "\n" + _tail_text(stderr_log, max_bytes=8000)
                ).lower()
                prompt_seen = any(marker in tail_blob for marker in prompt_markers)

            elapsed = now - start
            if elapsed >= timeout_seconds:
                timed_out = True
                forced_exit_reason = (
                    "timeout_after_house_state" if house_state_path is not None else "timeout_without_house_state"
                )
                _terminate_process_group(proc)
                break

            if house_state_seen_at is not None:
                should_force_exit = False
                if prompt_seen:
                    forced_exit_reason = "exit_prompt_after_house_state"
                    should_force_exit = True
                elif (now - house_state_seen_at) >= grace_after_house_state:
                    forced_exit_reason = "house_state_grace_elapsed"
                    should_force_exit = True
                if should_force_exit:
                    _terminate_process_group(proc)
                    break

            time.sleep(float(poll_seconds))

        if proc.poll() is None:
            _terminate_process_group(proc)

        try:
            returncode = int(proc.wait(timeout=5))
        except subprocess.TimeoutExpired:
            _signal_process_group(proc, signal.SIGKILL)
            returncode = int(proc.wait(timeout=5))

    if house_state_path is None:
        house_state_path = _find_house_state_optional(run_dir, scene_name)

    return {
        "returncode": returncode,
        "stdout_tail": _tail_text(stdout_log, max_bytes=4000),
        "stderr_tail": _tail_text(stderr_log, max_bytes=4000),
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
        "house_state_path": str(house_state_path) if house_state_path is not None else "",
        "forced_exit_reason": forced_exit_reason,
        "timed_out": timed_out,
    }


def _json_clone(value: Any) -> Any:
    # Keep return payload strictly JSON-safe.
    return json.loads(json.dumps(value))


def _extract_critic_payload(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in _CRITIC_PAYLOAD_KEYS:
        value = candidate.get(key)
        if key in {"placement_stages", "critic_scores", "support_surfaces"}:
            if isinstance(value, list):
                out[key] = _json_clone(value)
        else:
            if isinstance(value, Mapping):
                out[key] = _json_clone(dict(value))
    return out


def _iter_candidate_json_files(run_dir: Path) -> List[Path]:
    roots = [run_dir / "outputs", run_dir]
    include_tokens = ("quality", "critic", "package", "report", "result", "metadata", "house_state")
    seen: set[str] = set()
    out: List[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.json")):
            sp = str(path.resolve())
            if sp in seen:
                continue
            seen.add(sp)
            name = path.name.lower()
            if not any(tok in name for tok in include_tokens):
                continue
            try:
                if path.stat().st_size > 8_000_000:
                    continue
            except OSError:
                continue
            out.append(path)
    return out


_YAML_GRADE_RE = re.compile(
    r"^([\w][\w\s]*[\w])\s*:\s*\n\s+grade\s*:\s*(\d+)",
    re.MULTILINE,
)


def _collect_scenesmith_yaml_scores(run_dir: Path) -> Dict[str, Any]:
    """Scan SceneSmith ``scores.yaml`` files and synthesise critic payload.

    SceneSmith stores per-stage critic scores in YAML files at paths like::

        scene_000/room_kitchen/scene_states/<stage>/scores.yaml

    Each file contains rubric categories (Realism, Functionality, Layout,
    Holistic Completeness, Prompt Following, optionally Reachability) with
    integer grades on a 1-10 scale.

    Returns a dict that may contain ``quality_gate_report``,
    ``critic_scores``, and ``faithfulness_report`` in the format the
    downstream bridge expects.
    """
    yaml_files: List[Path] = []
    seen_paths: set[str] = set()
    for root in (run_dir, run_dir / "outputs"):
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("scores.yaml")):
            resolved = str(path.resolve())
            if resolved not in seen_paths:
                seen_paths.add(resolved)
                yaml_files.append(path)

    if not yaml_files:
        return {}

    all_stage_scores: List[Dict[str, Any]] = []
    all_grades: List[float] = []
    prompt_following_grades: List[float] = []

    for yaml_path in yaml_files:
        try:
            content = yaml_path.read_text(encoding="utf-8")
        except OSError:
            continue

        matches = _YAML_GRADE_RE.findall(content)
        if not matches:
            continue

        stage_name = yaml_path.parent.name
        stage_entry: Dict[str, Any] = {"stage": stage_name, "source": str(yaml_path)}
        grades_in_stage: List[float] = []

        for category, grade_str in matches:
            category = category.strip()
            grade = float(grade_str)
            stage_entry[category.lower().replace(" ", "_")] = grade
            grades_in_stage.append(grade)
            all_grades.append(grade)
            if category.lower() == "prompt following":
                prompt_following_grades.append(grade)

        if grades_in_stage:
            stage_entry["total"] = round(sum(grades_in_stage) / len(grades_in_stage), 3)
            all_stage_scores.append(stage_entry)

    if not all_stage_scores:
        return {}

    overall_avg = sum(all_grades) / len(all_grades) if all_grades else 0.0

    # quality_gate_report: pass if every stage average >= 6/10
    min_acceptable = 6.0
    checks = [
        {
            "stage": s["stage"],
            "score": s["total"],
            "passed": s["total"] >= min_acceptable,
        }
        for s in all_stage_scores
    ]
    all_pass = all(c["passed"] for c in checks)
    quality_gate_report: Dict[str, Any] = {
        "all_pass": all_pass,
        "status": "pass" if all_pass else "fail",
        "overall_score": round(overall_avg, 3),
        "checks": checks,
    }

    # faithfulness_report: use Prompt Following grades (0-10 scale)
    if prompt_following_grades:
        faithfulness_avg = sum(prompt_following_grades) / len(prompt_following_grades)
    else:
        faithfulness_avg = overall_avg
    faithfulness_report: Dict[str, Any] = {
        "score": round(faithfulness_avg, 3),
        "source": "scenesmith_yaml_scores",
    }

    return {
        "quality_gate_report": quality_gate_report,
        "critic_scores": all_stage_scores,
        "faithfulness_report": faithfulness_report,
    }


def _collect_critic_outputs(
    *,
    run_dir: Path,
    house_state: Mapping[str, Any],
    house_state_path: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    merged = _extract_critic_payload(house_state)
    source_files: List[str] = []
    if merged:
        source_files.append(str(house_state_path))

    for json_path in _iter_candidate_json_files(run_dir):
        if json_path.resolve() == house_state_path.resolve():
            continue
        try:
            parsed = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        candidates: List[Mapping[str, Any]] = []
        if isinstance(parsed, Mapping):
            candidates.append(parsed)
            package = parsed.get("package")
            if isinstance(package, Mapping):
                candidates.append(package)

        found_any = False
        for cand in candidates:
            extracted = _extract_critic_payload(cand)
            if not extracted:
                continue
            for key, value in extracted.items():
                if key not in merged:
                    merged[key] = value
            found_any = True
        if found_any:
            source_files.append(str(json_path))

        # Enough data collected; avoid scanning the whole run tree.
        if "critic_scores" in merged and "quality_gate_report" in merged and "faithfulness_report" in merged:
            break

    # Fallback: synthesise from SceneSmith YAML score files when JSON
    # scanning did not produce the three required critic keys.
    needed = {"critic_scores", "quality_gate_report", "faithfulness_report"}
    if not needed.issubset(merged.keys()):
        yaml_payload = _collect_scenesmith_yaml_scores(run_dir)
        for key, value in yaml_payload.items():
            if key not in merged:
                merged[key] = value
        if yaml_payload:
            source_files.append("scores.yaml (SceneSmith YAML synthesis)")

    summary = {
        "found_keys": sorted(list(merged.keys())),
        "source_files": source_files,
    }
    return merged, summary


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
        str(os.getenv("SCENESMITH_PAPER_IMAGE_BACKEND", "gemini")).strip().lower()
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


def _run_runtime_patch_script(*, repo_dir: Path, python_bin: str) -> None:
    if _is_truthy(os.getenv("SCENESMITH_PAPER_SKIP_RUNTIME_PATCHES"), default=False):
        return

    bp_root = Path(__file__).resolve().parents[1]
    patch_script = bp_root / "scripts" / "apply_scenesmith_paper_patches.sh"
    if not patch_script.exists():
        return

    patch_env = os.environ.copy()
    patch_env.setdefault("SCENESMITH_PAPER_REPO_DIR", str(repo_dir))
    patch_env.setdefault("SCENESMITH_PAPER_PYTHON_BIN", python_bin)
    patch_env.setdefault("PYTORCH_JIT", os.getenv("PYTORCH_JIT", "0") or "0")

    timeout = max(
        60,
        _safe_int(os.getenv("SCENESMITH_PAPER_PATCH_TIMEOUT_SECONDS"), default=1200),
    )
    proc = subprocess.run(
        ["bash", str(patch_script)],
        cwd=str(bp_root),
        env=patch_env,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "SceneSmith runtime patch script failed: "
            f"exit={proc.returncode} "
            f"stderr_tail={(proc.stderr or '')[-4000:]!r} "
            f"stdout_tail={(proc.stdout or '')[-4000:]!r}"
        )


def _run_official_scenesmith(payload: Mapping[str, Any]) -> Dict[str, Any]:
    repo_dir_raw = str(os.getenv("SCENESMITH_PAPER_REPO_DIR", "")).strip()
    if not repo_dir_raw:
        raise RuntimeError("SCENESMITH_PAPER_REPO_DIR is required")
    repo_dir = Path(repo_dir_raw).expanduser().resolve()
    if not repo_dir.is_dir():
        raise RuntimeError(f"SCENESMITH_PAPER_REPO_DIR not found: {repo_dir}")

    python_bin = str(os.getenv("SCENESMITH_PAPER_PYTHON_BIN", "python3")).strip() or "python3"
    os.environ.setdefault("PYTORCH_JIT", "0")
    _run_runtime_patch_script(repo_dir=repo_dir, python_bin=python_bin)

    timeout = _safe_int(os.getenv("SCENESMITH_PAPER_TIMEOUT_SECONDS"), default=5400)

    scene_id = str(payload.get("scene_id") or "scene").strip() or "scene"
    scene_name = _slug(scene_id, default="scene")
    existing_run_dir = _resolve_existing_run_dir()
    if existing_run_dir is None:
        root_run_dir = _run_root(scene_name)
        root_run_dir.mkdir(parents=True, exist_ok=True)
        cleanup_root_run_dir = True
    else:
        root_run_dir = existing_run_dir
        cleanup_root_run_dir = False
    model_attempts = _resolve_paper_model_attempts()

    keep_run_dir = _is_truthy(os.getenv("SCENESMITH_PAPER_KEEP_RUN_DIR"), default=False)
    force_generated_assets = _is_truthy(os.getenv("SCENESMITH_PAPER_ALL_SAM3D"), default=False) or _is_truthy(
        os.getenv("SCENESMITH_PAPER_FORCE_GENERATED_ASSETS"),
        default=False,
    )
    attempt_records: List[Dict[str, Any]] = []

    try:
        if existing_run_dir is not None:
            run_dir = existing_run_dir
            house_state_path = _find_house_state(run_dir, scene_name)
            house_state = _load_house_state(house_state_path)
            raw_objects = _collect_raw_objects(house_state)
            if not raw_objects:
                raise RuntimeError(f"Existing SceneSmith run returned zero objects (house_state={house_state_path})")
            critic_outputs, critic_summary = _collect_critic_outputs(
                run_dir=run_dir,
                house_state=house_state,
                house_state_path=house_state_path,
            )
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
            selected_model = model_attempts[0] if model_attempts else ""
            attempt_records.append(
                {
                    "attempt": 1,
                    "model": selected_model,
                    "status": "reused_existing_run",
                    "exit_code": 0,
                    "forced_exit_reason": "",
                    "timed_out": False,
                }
            )
            response: Dict[str, Any] = {
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
                    "model_selected": selected_model,
                    "scenesmith_exit_code": 0,
                    "scenesmith_nonzero_exit_accepted": False,
                    "scenesmith_forced_exit_reason": "",
                    "scenesmith_timed_out": False,
                    "scenesmith_existing_run_reused": True,
                    "scenesmith_stdout_log": str(run_dir / "official_scenesmith.stdout.log"),
                    "scenesmith_stderr_log": str(run_dir / "official_scenesmith.stderr.log"),
                    "model_attempts": attempt_records,
                    "critic_found_keys": critic_summary.get("found_keys", []),
                    "critic_source_files": critic_summary.get("source_files", []),
                },
            }
            if critic_outputs:
                response["critic_outputs"] = critic_outputs
                for key in _CRITIC_PAYLOAD_KEYS:
                    if key in critic_outputs:
                        response[key] = critic_outputs[key]
            return response

        for attempt_index, selected_model in enumerate(model_attempts, start=1):
            run_dir = (
                root_run_dir
                if len(model_attempts) == 1
                else root_run_dir / f"attempt-{attempt_index:02d}"
            )
            run_dir.mkdir(parents=True, exist_ok=True)

            overrides = _hydra_overrides(payload=payload, run_dir=run_dir, scene_name=scene_name)
            env = os.environ.copy()
            env.setdefault("PYTHONUNBUFFERED", "1")
            _apply_paper_openai_env_overrides(
                env,
                model_override=selected_model if selected_model else None,
            )

            openai_api = str(env.get("SCENESMITH_PAPER_OPENAI_API", "")).strip().lower()
            if openai_api:
                cmd: List[str] = [python_bin, "-c", _OPENAI_API_WRAPPER] + overrides
            else:
                cmd = [python_bin, "main.py"] + overrides

            process_report = _run_scenesmith_process(
                cmd=cmd,
                repo_dir=repo_dir,
                env=env,
                timeout_seconds=timeout,
                run_dir=run_dir,
                scene_name=scene_name,
            )
            exit_code = _safe_int(process_report.get("returncode"), default=1)
            stdout_tail = str(process_report.get("stdout_tail") or "")
            stderr_tail = str(process_report.get("stderr_tail") or "")
            forced_exit_reason = str(process_report.get("forced_exit_reason") or "")
            timed_out = bool(process_report.get("timed_out"))
            house_state_hint = str(process_report.get("house_state_path") or "").strip()
            accepted_nonzero_exit = bool(exit_code != 0 and house_state_hint)

            if exit_code != 0 and not accepted_nonzero_exit:
                attempt_records.append(
                    {
                        "attempt": attempt_index,
                        "model": selected_model,
                        "status": "failed",
                        "exit_code": int(exit_code),
                        "error": (
                            "Official SceneSmith failed "
                            f"(exit={exit_code}). "
                            f"stdout_tail={stdout_tail!r} stderr_tail={stderr_tail!r}"
                        ),
                    }
                )
                continue

            house_state_path = Path(house_state_hint) if house_state_hint else _find_house_state(run_dir, scene_name)
            house_state = _load_house_state(house_state_path)
            raw_objects = _collect_raw_objects(house_state)
            if not raw_objects:
                attempt_records.append(
                    {
                        "attempt": attempt_index,
                        "model": selected_model,
                        "status": "failed",
                        "error": f"Official SceneSmith returned zero objects (house_state={house_state_path})",
                    }
                )
                continue
            critic_outputs, critic_summary = _collect_critic_outputs(
                run_dir=run_dir,
                house_state=house_state,
                house_state_path=house_state_path,
            )

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

            attempt_records.append(
                {
                    "attempt": attempt_index,
                    "model": selected_model,
                    "status": "succeeded_with_nonzero_exit" if accepted_nonzero_exit else "succeeded",
                    "exit_code": int(exit_code),
                    "forced_exit_reason": forced_exit_reason,
                    "timed_out": timed_out,
                }
            )

            response: Dict[str, Any] = {
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
                    "model_selected": selected_model,
                    "scenesmith_exit_code": int(exit_code),
                    "scenesmith_nonzero_exit_accepted": accepted_nonzero_exit,
                    "scenesmith_forced_exit_reason": forced_exit_reason,
                    "scenesmith_timed_out": timed_out,
                    "scenesmith_existing_run_reused": False,
                    "scenesmith_stdout_log": str(process_report.get("stdout_log") or ""),
                    "scenesmith_stderr_log": str(process_report.get("stderr_log") or ""),
                    "model_attempts": attempt_records,
                    "critic_found_keys": critic_summary.get("found_keys", []),
                    "critic_source_files": critic_summary.get("source_files", []),
                },
            }
            if critic_outputs:
                response["critic_outputs"] = critic_outputs
                # Also expose common fields at top-level for compatibility with consumers
                # that read these directly.
                for key in _CRITIC_PAYLOAD_KEYS:
                    if key in critic_outputs:
                        response[key] = critic_outputs[key]
            return response

        raise RuntimeError(
            "Official SceneSmith failed for all model attempts: "
            + json.dumps(attempt_records, separators=(",", ":"))
        )
    finally:
        if cleanup_root_run_dir and root_run_dir.exists() and not keep_run_dir:
            shutil.rmtree(root_run_dir, ignore_errors=True)


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
