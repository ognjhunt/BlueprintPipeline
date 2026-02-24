#!/usr/bin/env python3
"""
SceneSmith (official paper stack) -> SAGE layout directory bridge.

Goal:
  - Keep SAGE stages 5-7 as the "factory" (grasps/plans/capture)
  - Allow generating a single SAGE-compatible layout dir from SceneSmith output

This script:
  1) Runs the official SceneSmith paper stack via:
       BlueprintPipeline/scenesmith-service/scenesmith_paper_command.py
     with an optional critic-loop accept/reject gate.
  2) Converts objects + transforms into a SAGE-style room dict JSON (z-up).
  3) Writes a minimal pose augmentation folder:
       pose_aug_0/meta.json (list with 1 variant)
       pose_aug_0/variant_000.json (layout dict with rooms[0])
  4) Writes meshes into generation/*.obj:
       - Default: key_only (manipulables + key surfaces) use SAM3D server
       - Others: primitive unit-box OBJ (scaled in Isaac Sim by dimensions)

Output:
  Prints the new layout_id to stdout (and logs to stderr).
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


_SURFACE_TOKENS = ("table", "counter", "desk", "island", "bench")


_PASS_TOKENS = {"pass", "passed", "ok", "success", "succeeded"}
_SAM3D_REQUIRED_MODULES = ("flask", "flask_cors", "psutil", "requests")
_FAIL_TOKENS = {"fail", "failed", "error", "invalid", "rejected"}


def _log(msg: str) -> None:
    print(f"[scenesmith->sage {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}", file=sys.stderr, flush=True)


def _safe_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _safe_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _mapping_or_empty(v: Any) -> Mapping[str, Any]:
    return v if isinstance(v, Mapping) else {}


def _first_mapping(raw: Mapping[str, Any], paths: Sequence[Sequence[str]]) -> Mapping[str, Any]:
    for path in paths:
        cursor: Any = raw
        ok = True
        for key in path:
            if not isinstance(cursor, Mapping):
                ok = False
                break
            cursor = cursor.get(key)
        if ok and isinstance(cursor, Mapping):
            return cursor
    return {}


def _to_xyz(v: Any) -> Tuple[float, float, float]:
    if isinstance(v, Mapping):
        return (
            _safe_float(v.get("x"), 0.0),
            _safe_float(v.get("y"), 0.0),
            _safe_float(v.get("z"), 0.0),
        )
    if (
        isinstance(v, Sequence)
        and not isinstance(v, (str, bytes))
        and len(v) >= 3
    ):
        return (_safe_float(v[0], 0.0), _safe_float(v[1], 0.0), _safe_float(v[2], 0.0))
    return 0.0, 0.0, 0.0


def _python_modules_report(python_bin: str, *, modules: Sequence[str]) -> List[str]:
    try:
        module_list = ", ".join(repr(str(m)) for m in modules)
        probe = (
            "import importlib.util, sys\n"
            f"mods = [{module_list}]\n"
            "missing = [m for m in mods if importlib.util.find_spec(m) is None]\n"
            "if missing:\n"
            "    print('\\n'.join(missing))\n"
            "    raise SystemExit(1)\n"
        )
        proc = subprocess.run(
            [python_bin, "-c", probe],
            capture_output=True,
            check=False,
            timeout=12,
            text=True,
        )
        if proc.returncode != 0:
            out = (proc.stdout or "").strip()
            if out:
                return [m.strip() for m in out.splitlines() if m.strip()]
            return [m for m in modules]
        return []
    except Exception:
        return list(modules)


def _python_install_modules(python_bin: str, modules: Sequence[str]) -> None:
    missing_pkgs = [str(m).replace("_", "-") for m in modules]
    if not missing_pkgs:
        return
    _log(f"Installing SAM3D deps in {python_bin}: {', '.join(missing_pkgs)}")
    proc = subprocess.run(
        [python_bin, "-m", "pip", "install", "--disable-pip-version-check", *missing_pkgs],
        capture_output=True,
        check=False,
        timeout=420,
        text=True,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"Failed to install SAM3D dependencies in {python_bin}: {err}")


def _python_has_modules(python_bin: str, *, modules: Sequence[str]) -> List[str]:
    return _python_modules_report(python_bin, modules=modules)


def _ensure_sam3d_modules(python_bin: str) -> None:
    missing = _python_has_modules(python_bin, modules=_SAM3D_REQUIRED_MODULES)
    if not missing:
        return

    auto_install = _is_truthy(os.getenv("SAM3D_AUTO_INSTALL_DEPS", "1"), default=True)
    if not auto_install:
        raise RuntimeError(
            f"SAM3D python={python_bin} missing required modules: {', '.join(sorted(missing))}. "
            f"Set SAM3D_AUTO_INSTALL_DEPS=1 to install automatically."
        )

    _python_install_modules(python_bin, _SAM3D_REQUIRED_MODULES)
    still_missing = _python_has_modules(python_bin, modules=_SAM3D_REQUIRED_MODULES)
    if still_missing:
        raise RuntimeError(
            "SAM3D dependency repair failed for "
            f"{python_bin}: still missing {', '.join(sorted(still_missing))}"
        )


def _is_truthy(v: Any, *, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _slug(s: str, default: str) -> str:
    import re

    text = str(s or "").strip()
    text = re.sub(r"[^a-zA-Z0-9_-]+", "_", text)
    text = re.sub(r"_{2,}", "_", text).strip("_")
    if not text:
        return default
    return text[:80]


def _snapshot_scenesmith_run(
    *,
    response: Mapping[str, Any],
    layout_dir: Path,
    attempt_label: str,
    persist_enabled: bool,
) -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {
        "attempt": str(attempt_label).strip(),
        "persist_enabled": bool(persist_enabled),
        "paper_stack_run_dir": "",
        "paper_stack_house_state_path": "",
        "status": "skipped",
    }
    if not persist_enabled:
        return snapshot

    paper_stack = response.get("paper_stack") if isinstance(response, Mapping) else {}
    if not isinstance(paper_stack, Mapping):
        snapshot["status"] = "no_paper_stack"
        return snapshot

    run_dir = Path(str(paper_stack.get("run_dir") or "")).expanduser()
    house_state_path = Path(str(paper_stack.get("house_state_path") or "")).expanduser()
    snapshot["paper_stack_run_dir"] = str(run_dir)
    snapshot["paper_stack_house_state_path"] = str(house_state_path)

    if not run_dir.is_dir():
        snapshot["status"] = "run_dir_missing"
        return snapshot

    try:
        scenesmith_run_dir = layout_dir / "scenesmith_run"
        scenesmith_run_dir.mkdir(parents=True, exist_ok=True)
        attempt_dir = scenesmith_run_dir / f"attempt_{snapshot['attempt']}"
        if attempt_dir.exists():
            shutil.rmtree(attempt_dir, ignore_errors=True)
        shutil.copytree(run_dir, attempt_dir)
        snapshot["status"] = "synced"
        snapshot["synced_to"] = str(attempt_dir)
    except Exception as exc:
        snapshot["status"] = "sync_error"
        snapshot["error"] = str(exc)
        _log(f"WARNING: SceneSmith run sync failed for attempt {snapshot['attempt']}: {exc}")

    return snapshot


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _single_room_layout_payload(room: Mapping[str, Any], *, layout_id: str) -> Dict[str, Any]:
    """Return a single-room layout wrapper expected by SAGE Stage 4/5 clients."""
    payload: Dict[str, Any] = {"layout_id": str(layout_id), "rooms": [dict(room)]}

    # Keep common top-level keys for scripts that still read them directly.
    for key in ("room_type", "dimensions", "seed", "scene_source", "policy_analysis"):
        value = room.get(key)
        if value is not None:
            payload[key] = value
    return payload


def _extract_critic_payload(response: Mapping[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    critic_outputs = response.get("critic_outputs")
    if isinstance(critic_outputs, Mapping):
        payload.update(dict(critic_outputs))
    for key in (
        "placement_stages",
        "critic_scores",
        "support_surfaces",
        "faithfulness_report",
        "quality_gate_report",
    ):
        value = response.get(key)
        if value is None or key in payload:
            continue
        if isinstance(value, (Mapping, list)):
            payload[key] = value
    return payload


def _json_compact(value: Any, max_len: int = 240) -> str:
    try:
        text = json.dumps(value, sort_keys=True)
    except Exception:
        text = repr(value)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


@dataclass
class CriticLoopConfig:
    enabled: bool
    max_attempts: int
    require_quality_pass: bool
    require_critic_score: bool
    require_faithfulness: bool
    min_faithfulness: float
    min_critic_total_0_10: float
    min_critic_total_0_1: float
    seed_stride: int
    allow_last_attempt_on_fail: bool


def _parse_quality_pass(quality_gate_report: Mapping[str, Any]) -> Optional[bool]:
    all_pass = quality_gate_report.get("all_pass")
    if isinstance(all_pass, bool):
        return all_pass
    if isinstance(all_pass, (int, float)):
        return bool(all_pass)

    status = quality_gate_report.get("status")
    if status is not None:
        s = str(status).strip().lower()
        if s in _PASS_TOKENS:
            return True
        if s in _FAIL_TOKENS:
            return False
        for tok in _FAIL_TOKENS:
            if tok in s:
                return False
        if "not pass" in s or "not_pass" in s or "notpassed" in s:
            return False
        for tok in _PASS_TOKENS:
            if tok in s:
                return True

    checks = quality_gate_report.get("checks")
    if isinstance(checks, list) and checks:
        has_false = False
        has_true = False
        for item in checks:
            if not isinstance(item, Mapping):
                continue
            passed = item.get("passed")
            if isinstance(passed, bool):
                has_true = has_true or passed
                has_false = has_false or (not passed)
        if has_false:
            return False
        if has_true:
            return True
    return None


def _extract_critic_total(critic_scores: Any) -> Tuple[Optional[float], Optional[str]]:
    if not isinstance(critic_scores, list):
        return None, None
    best_score: Optional[float] = None
    scale_hint: Optional[str] = None
    for entry in critic_scores:
        if not isinstance(entry, Mapping):
            continue
        candidate = None
        for key in ("total", "overall_score", "overall", "score", "final_score", "aggregate"):
            value = entry.get(key)
            if isinstance(value, (int, float)):
                candidate = float(value)
                break
            if isinstance(value, str):
                parsed = _safe_float(value, float("nan"))
                if math.isfinite(parsed):
                    candidate = float(parsed)
                    break
        if candidate is None or not math.isfinite(candidate):
            continue
        if best_score is None or candidate > best_score:
            best_score = candidate
    if best_score is not None:
        scale_hint = "0_1" if 0.0 <= best_score <= 1.5 else "0_10"
    return best_score, scale_hint


def _extract_faithfulness_score(faithfulness_report: Any) -> Optional[float]:
    if not isinstance(faithfulness_report, Mapping):
        return None
    for key in ("score", "overall_score", "overall", "faithfulness"):
        value = faithfulness_report.get(key)
        if isinstance(value, (int, float)):
            out = float(value)
            return (out / 10.0) if out > 1.5 else out
        if isinstance(value, str):
            out = _safe_float(value, float("nan"))
            if math.isfinite(out):
                return (out / 10.0) if out > 1.5 else out
    return None


def _evaluate_critic(response: Mapping[str, Any], config: CriticLoopConfig) -> Dict[str, Any]:
    critic_payload = _extract_critic_payload(response)
    quality_gate_report = critic_payload.get("quality_gate_report")
    quality_pass: Optional[bool] = None
    if isinstance(quality_gate_report, Mapping):
        quality_pass = _parse_quality_pass(quality_gate_report)

    critic_scores = critic_payload.get("critic_scores")
    critic_total, critic_scale = _extract_critic_total(critic_scores)
    critic_threshold = (
        config.min_critic_total_0_1 if critic_scale == "0_1" else config.min_critic_total_0_10
    )
    critic_pass: Optional[bool] = None
    if critic_total is not None:
        critic_pass = bool(critic_total >= critic_threshold)

    faithfulness = _extract_faithfulness_score(critic_payload.get("faithfulness_report"))
    faithfulness_pass: Optional[bool] = None
    if faithfulness is not None:
        faithfulness_pass = bool(faithfulness >= config.min_faithfulness)

    failures: List[str] = []
    if config.require_quality_pass:
        if quality_pass is None:
            failures.append("missing quality_gate_report pass/fail signal")
        elif not quality_pass:
            failures.append("quality gate report did not pass")
    if config.require_critic_score:
        if critic_pass is None:
            failures.append("missing critic total score")
        elif not critic_pass:
            failures.append(
                f"critic total {critic_total:.3f} below threshold {critic_threshold:.3f} ({critic_scale or 'unknown_scale'})"
            )
    if config.require_faithfulness:
        if faithfulness_pass is None:
            failures.append("missing faithfulness score")
        elif not faithfulness_pass:
            failures.append(f"faithfulness {faithfulness:.3f} below threshold {config.min_faithfulness:.3f}")

    return {
        "passed": len(failures) == 0,
        "failures": failures,
        "quality_pass": quality_pass,
        "critic_total": critic_total,
        "critic_scale": critic_scale,
        "critic_threshold": critic_threshold if critic_total is not None else None,
        "faithfulness": faithfulness,
        "faithfulness_threshold": config.min_faithfulness if faithfulness is not None else None,
        "critic_found_keys": sorted(list(critic_payload.keys())),
    }


def _write_unit_box_obj(path: Path) -> None:
    """
    Write a canonical unit box OBJ with:
      x,y in [-0.5, +0.5] (centered footprint)
      z in [0.0, 1.0]      (bottom at 0)
    This matches the convention used by the SAGE collector in this repo:
      - position.x/y is footprint center
      - position.z is bottom
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # 8 vertices, 12 triangles (two per face)
    obj = """# unit box (centered XY, bottom at Z=0)
v -0.5 -0.5 0.0
v  0.5 -0.5 0.0
v  0.5  0.5 0.0
v -0.5  0.5 0.0
v -0.5 -0.5 1.0
v  0.5 -0.5 1.0
v  0.5  0.5 1.0
v -0.5  0.5 1.0
f 1 2 3
f 1 3 4
f 5 8 7
f 5 7 6
f 1 5 6
f 1 6 2
f 2 6 7
f 2 7 3
f 3 7 8
f 3 8 4
f 4 8 5
f 4 5 1
"""
    path.write_text(obj, encoding="utf-8")


# SceneSmith (y-up) -> SAGE (z-up) basis transform:
#   x_g = x_s
#   y_g = -z_s
#   z_g = y_s
_A = (
    (1.0, 0.0, 0.0),
    (0.0, 0.0, -1.0),
    (0.0, 1.0, 0.0),
)


def _mat3_mul(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> List[List[float]]:
    out = [[0.0, 0.0, 0.0] for _ in range(3)]
    for i in range(3):
        for j in range(3):
            out[i][j] = float(a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j])
    return out


def _mat3_T(a: Sequence[Sequence[float]]) -> List[List[float]]:
    return [[float(a[j][i]) for j in range(3)] for i in range(3)]


def _quat_to_mat3(qw: float, qx: float, qy: float, qz: float) -> List[List[float]]:
    # Normalize (avoid drift)
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n <= 1e-12:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    return [
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ]


def _scenesmith_pos_to_sage(px: float, py: float, pz: float) -> Tuple[float, float, float]:
    return (float(px), float(-pz), float(py))


def _scenesmith_rot_to_sage_yaw_deg(q: Mapping[str, Any]) -> float:
    qw = _safe_float(q.get("w"), 1.0)
    qx = _safe_float(q.get("x"), 0.0)
    qy = _safe_float(q.get("y"), 0.0)
    qz = _safe_float(q.get("z"), 0.0)
    R_s = _quat_to_mat3(qw, qx, qy, qz)
    A = _A
    AT = _mat3_T(A)
    # Similarity transform: R_g = A * R_s * A^T
    R_g = _mat3_mul(_mat3_mul(A, R_s), AT)
    yaw = math.degrees(math.atan2(R_g[1][0], R_g[0][0]))
    # Wrap to [-180, 180]
    while yaw > 180.0:
        yaw -= 360.0
    while yaw < -180.0:
        yaw += 360.0
    return float(yaw)


def _is_surface_type(t: str) -> bool:
    s = str(t or "").strip().lower().replace(" ", "_")
    return any(tok in s for tok in _SURFACE_TOKENS)


def _pick_mesh_candidates(
    raw_objs: Sequence[Mapping[str, Any]],
    *,
    mesh_policy: str,
    max_meshes: int,
) -> List[int]:
    if max_meshes <= 0:
        return []
    if mesh_policy == "all":
        return list(range(min(len(raw_objs), max_meshes)))

    # key_only: manipulables first, then key surfaces.
    manipulables: List[int] = []
    surfaces: List[int] = []
    for idx, o in enumerate(raw_objs):
        role = str(o.get("sim_role", "")).strip().lower()
        t = str(o.get("category") or o.get("name") or "")
        if role == "manipulable_object":
            manipulables.append(idx)
        elif _is_surface_type(t):
            surfaces.append(idx)

    ordered = manipulables + [i for i in surfaces if i not in manipulables]
    return ordered[:max_meshes]


def _iter_nested_mappings(node: Any) -> List[Mapping[str, Any]]:
    out: List[Mapping[str, Any]] = []
    if isinstance(node, Mapping):
        out.append(node)
        for value in node.values():
            out.extend(_iter_nested_mappings(value))
    elif isinstance(node, list):
        for value in node:
            out.extend(_iter_nested_mappings(value))
    return out


def _looks_like_house_object(candidate: Mapping[str, Any]) -> bool:
    if isinstance(candidate.get("objects"), list):
        return False
    identity = (
        candidate.get("id")
        or candidate.get("object_id")
        or candidate.get("source_id")
        or candidate.get("semantic_class")
        or candidate.get("class_name")
        or candidate.get("category")
        or candidate.get("name")
    )
    if identity is None:
        return False
    if isinstance(candidate.get("pose"), Mapping):
        return True
    if isinstance(candidate.get("transform"), Mapping):
        return True
    if isinstance(candidate.get("position"), Mapping):
        return True
    if isinstance(candidate.get("extent"), Mapping):
        return True
    if isinstance(candidate.get("dimensions"), Mapping):
        return True
    return False


def _collect_glb_refs(node: Any) -> List[str]:
    refs: List[str] = []
    if isinstance(node, Mapping):
        for value in node.values():
            refs.extend(_collect_glb_refs(value))
    elif isinstance(node, list):
        for value in node:
            refs.extend(_collect_glb_refs(value))
    elif isinstance(node, str):
        text = node.strip()
        if text.lower().endswith(".glb"):
            refs.append(text)
    return refs


def _normalize_token(text: Any) -> str:
    token = str(text or "").strip().lower()
    token = re.sub(r"[^a-z0-9]+", "_", token).strip("_")
    token = re.sub(r"_{2,}", "_", token)
    return token


def _matches_confidently(stem: str, token: str) -> bool:
    if not token:
        return False
    if stem == token:
        return True
    if stem.startswith(token + "_"):
        return True
    return False


def _object_match_tokens(raw_obj: Mapping[str, Any]) -> List[str]:
    tokens: set[str] = set()
    for key in ("id", "name", "category"):
        raw = raw_obj.get(key)
        if raw is None:
            continue
        base = _normalize_token(raw)
        if not base:
            continue
        tokens.add(base)
        trimmed = re.sub(r"(?:_|-)?\d+$", "", base).strip("_")
        if trimmed:
            tokens.add(trimmed)

    blacklist = {"object", "generated", "asset", "model", "scene", "mesh"}
    filtered = [tok for tok in tokens if tok and tok not in blacklist and len(tok) >= 3]
    return sorted(filtered, key=len, reverse=True)


def _resolve_glb_path(path_str: str, *, base_dirs: Sequence[Path]) -> Optional[Path]:
    path = Path(path_str).expanduser()
    if path.is_absolute() and path.is_file():
        return path.resolve()
    for base_dir in base_dirs:
        candidate = (base_dir / path).resolve()
        if candidate.is_file():
            return candidate
    return None


def _collect_existing_glb_pool(
    *,
    response: Mapping[str, Any],
    raw_objects: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    paper_stack = response.get("paper_stack") if isinstance(response.get("paper_stack"), Mapping) else {}
    run_dir = Path(str(paper_stack.get("run_dir") or "")).expanduser()
    house_state_path = Path(str(paper_stack.get("house_state_path") or "")).expanduser()

    search_roots: List[Path] = []
    if house_state_path.is_file():
        search_roots.append(house_state_path.parent)
    if run_dir.is_dir():
        search_roots.append(run_dir)
        outputs_dir = run_dir / "outputs"
        if outputs_dir.is_dir():
            search_roots.append(outputs_dir)

    # Use a deduped list of base dirs for resolving relative refs.
    base_dirs: List[Path] = []
    seen_bases: set[str] = set()
    for root in search_roots:
        rp = str(root.resolve())
        if rp in seen_bases:
            continue
        seen_bases.add(rp)
        base_dirs.append(root.resolve())

    by_id: Dict[str, List[Path]] = {}
    all_paths: Dict[str, Path] = {}

    # 1) Prefer explicit GLB refs discoverable inside house_state.json.
    if house_state_path.is_file():
        try:
            house_state = _load_json(house_state_path)
        except Exception:
            house_state = {}
        for candidate in _iter_nested_mappings(house_state):
            if not _looks_like_house_object(candidate):
                continue
            oid_raw = candidate.get("id") or candidate.get("object_id") or candidate.get("source_id")
            oid = _normalize_token(oid_raw)
            refs = _collect_glb_refs(candidate)
            if not refs:
                continue
            resolved_refs: List[Path] = []
            for ref in refs:
                resolved = _resolve_glb_path(ref, base_dirs=base_dirs)
                if resolved is None:
                    continue
                resolved_refs.append(resolved)
                all_paths[str(resolved)] = resolved
            if oid and resolved_refs:
                by_id.setdefault(oid, [])
                for resolved in resolved_refs:
                    if resolved not in by_id[oid]:
                        by_id[oid].append(resolved)

    # 2) Fallback: index all GLBs under run dirs.
    for root in base_dirs:
        for path in root.rglob("*.glb"):
            try:
                resolved = path.resolve()
            except Exception:
                continue
            all_paths[str(resolved)] = resolved

    # Precompute normalized stems for token matching.
    stem_index: Dict[str, str] = {}
    for resolved in all_paths.values():
        stem_index[str(resolved)] = _normalize_token(resolved.stem)

    # Warm-match ids to paths by token if explicit refs were missing.
    for raw_obj in raw_objects:
        oid = _normalize_token(raw_obj.get("id"))
        if not oid or oid in by_id:
            continue
        tokens = _object_match_tokens(raw_obj)
        if not tokens:
            continue
        ranked_matches: List[Tuple[int, Path]] = []
        for resolved in all_paths.values():
            stem = stem_index.get(str(resolved), "")
            score = 0
            for token in tokens:
                if _matches_confidently(stem, token):
                    score = max(score, len(token))
            if score > 0:
                ranked_matches.append((score, resolved))
        ranked_matches.sort(key=lambda item: (-item[0], str(item[1])))
        matches = [item[1] for item in ranked_matches]
        if matches:
            by_id[oid] = matches

    return {
        "by_id": by_id,
        "all_paths": list(all_paths.values()),
        "stems": stem_index,
        "used": set(),
    }


def _pick_existing_glb_for_object(pool: Mapping[str, Any], raw_obj: Mapping[str, Any]) -> Optional[Path]:
    used = pool.get("used")
    if not isinstance(used, set):
        return None

    by_id = pool.get("by_id")
    oid = _normalize_token(raw_obj.get("id"))
    if isinstance(by_id, Mapping) and oid:
        refs = by_id.get(oid)
        if isinstance(refs, list):
            for ref in refs:
                if isinstance(ref, Path) and ref.is_file() and str(ref) not in used:
                    used.add(str(ref))
                    return ref

    all_paths = pool.get("all_paths")
    stems = pool.get("stems")
    if not isinstance(all_paths, list) or not isinstance(stems, Mapping):
        return None
    tokens = _object_match_tokens(raw_obj)
    if not tokens:
        return None

    best_path: Optional[Path] = None
    best_score = 0
    for candidate in all_paths:
        if not isinstance(candidate, Path):
            continue
        if not candidate.is_file():
            continue
        if str(candidate) in used:
            continue
        stem = str(stems.get(str(candidate), ""))
        score = 0
        for token in tokens:
            if _matches_confidently(stem, token):
                score = max(score, len(token))
        if score > best_score:
            best_score = score
            best_path = candidate

    if best_path is None:
        return None
    used.add(str(best_path))
    return best_path


def _convert_glb_to_obj(glb_path: Path, out_obj_path: Path) -> None:
    try:
        import trimesh  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"trimesh required to convert GLB->OBJ: {exc}") from exc

    loaded = trimesh.load(str(glb_path), file_type="glb")
    if hasattr(loaded, "geometry") and hasattr(loaded, "dump"):
        mesh = loaded.dump(concatenate=True)
    else:
        mesh = loaded
    out_obj_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out_obj_path))


def _http_post_json(url: str, payload: Dict[str, Any], *, timeout_s: int) -> Tuple[int, Dict[str, Any]]:
    try:
        import requests  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"requests is required for SAM3D calls: {exc}") from exc

    r = requests.post(url, json=payload, timeout=timeout_s)
    status = int(r.status_code)
    try:
        data = r.json()
    except Exception:
        data = {"_raw": r.text}
    return status, data


def _http_get_bytes(url: str, *, timeout_s: int) -> Tuple[int, bytes, Dict[str, Any]]:
    try:
        import requests  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"requests is required for SAM3D calls: {exc}") from exc

    r = requests.get(url, timeout=timeout_s)
    status = int(r.status_code)
    if status == 200 and (r.headers.get("content-type") or "").lower().startswith("application"):
        return status, r.content, {}
    # Non-200 or JSON status.
    try:
        data = r.json()
    except Exception:
        data = {"_raw": r.text}
    return status, b"", data


def _sam3d_health(server_base: str) -> bool:
    try:
        import requests  # type: ignore
    except Exception:
        return False
    try:
        r = requests.get(server_base.rstrip("/") + "/health", timeout=3)
        return bool(r.status_code == 200)
    except Exception:
        return False


def _sam3d_shutdown(server_base: str) -> None:
    try:
        import requests  # type: ignore
    except Exception:
        return
    try:
        requests.post(server_base.rstrip("/") + "/shutdown", timeout=5)
    except Exception:
        pass


def _sam3d_start_if_needed(server_base: str) -> None:
    """
    Best-effort: start sam3d_server.py if the health endpoint is down.

    This keeps SceneSmith generation and SAM3D mesh synthesis sequential on a
    single GPU, avoiding concurrent VRAM allocation.
    """
    if _sam3d_health(server_base):
        return

    # Try to launch the server as a detached subprocess.
    port = int(server_base.rsplit(":", 1)[-1])
    bp_root = Path(__file__).resolve().parents[2]
    server_script = bp_root / "scripts" / "runpod_sage" / "sam3d_server.py"
    if not server_script.exists():
        raise FileNotFoundError(f"Cannot start SAM3D server; missing: {server_script}")

    env_python = str(os.getenv("SAM3D_PYTHON_BIN", "")).strip()
    python_candidates: List[str] = []
    if env_python:
        env_python_path = Path(env_python).expanduser()
        if env_python_path.exists():
            python_candidates.append(str(env_python_path.resolve()))
        else:
            resolved = shutil.which(env_python)
            if resolved:
                python_candidates.append(resolved)
            else:
                raise FileNotFoundError(f"SAM3D_PYTHON_BIN not found: {env_python}")

    python_candidates.extend(
        [
            "/workspace/miniconda3/envs/sage/bin/python",
            "/workspace/miniconda3/envs/sage/bin/python3",
            "/workspace/miniconda3/bin/python",
            str(sys.executable),
            "/usr/bin/python3.11",
            "/usr/bin/python3",
        ]
    )

    resolved_candidates: List[str] = []
    for candidate in python_candidates:
        if not candidate:
            continue
        candidate = str(Path(candidate).expanduser())
        if candidate not in resolved_candidates:
            resolved_candidates.append(candidate)

    python_bin = ""
    for candidate in resolved_candidates:
        if shutil.which(candidate):
            missing = _python_has_modules(candidate, modules=_SAM3D_REQUIRED_MODULES)
            if not missing:
                python_bin = candidate
                break
            if not python_bin:
                python_bin = candidate

    if not python_bin:
        python_bin = "python3.11" if shutil.which("python3.11") else "python3"
    _ensure_sam3d_modules(python_bin)

    image_backend = os.getenv("SAM3D_IMAGE_BACKEND", "gemini")
    checkpoint_dir = os.getenv("SAM3D_CHECKPOINT_DIR", "/workspace/sam3d/checkpoints/hf")

    log_path = Path("/tmp/sam3d_server_from_scenesmith.log")
    _log(
        "Starting SAM3D server for mesh generation "
        f"on :{port} (backend={image_backend}, python={python_bin})"
    )
    with log_path.open("ab") as f:
        proc = subprocess.Popen(
            [python_bin, str(server_script), "--port", str(port), "--image-backend", image_backend, "--checkpoint-dir", checkpoint_dir],
            stdout=f,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )

    deadline = time.time() + 300.0
    while time.time() < deadline:
        if _sam3d_health(server_base):
            _log("SAM3D server healthy")
            return
        if proc.poll() is not None:
            raise RuntimeError(f"SAM3D server died while starting (log={log_path})")
        time.sleep(2.0)
    raise TimeoutError(f"SAM3D server did not become healthy within 300s (log={log_path})")


def _sam3d_generate_obj(
    *,
    prompt: str,
    out_obj_path: Path,
    seed: int,
    server_base: str,
    timeout_total_s: int = 900,
) -> bool:
    gen_url = server_base.rstrip("/") + "/generate"
    job_base = server_base.rstrip("/") + "/job/"

    status, data = _http_post_json(gen_url, {"input_text": prompt, "seed": int(seed)}, timeout_s=30)
    if status not in (200, 202):
        raise RuntimeError(f"SAM3D /generate failed (status={status}) data={data}")
    job_id = str(data.get("job_id") or "").strip()
    if not job_id:
        raise RuntimeError(f"SAM3D /generate returned no job_id: {data}")

    deadline = time.time() + float(timeout_total_s)
    while time.time() < deadline:
        st, blob, j = _http_get_bytes(job_base + job_id, timeout_s=30)
        if st == 200 and blob:
            # Convert GLB -> OBJ
            try:
                import trimesh  # type: ignore
            except Exception as exc:
                raise RuntimeError(f"trimesh required to convert GLB->OBJ: {exc}") from exc

            loaded = trimesh.load(io.BytesIO(blob), file_type="glb")
            if hasattr(loaded, "geometry") and hasattr(loaded, "dump"):
                # Scene -> single mesh
                mesh = loaded.dump(concatenate=True)
            else:
                mesh = loaded
            out_obj_path.parent.mkdir(parents=True, exist_ok=True)
            mesh.export(str(out_obj_path))
            return True
        if st == 202:
            time.sleep(2.0)
            continue
        if st >= 400:
            raise RuntimeError(f"SAM3D job failed (status={st}) data={j}")
        time.sleep(2.0)
    raise TimeoutError(f"SAM3D job timed out after {timeout_total_s}s (prompt={prompt!r})")


def _run_scenesmith_paper_stack(
    *,
    scene_id: str,
    prompt: str,
    room_type: str,
    seed: int,
) -> Dict[str, Any]:
    bp_root = Path(__file__).resolve().parents[2]  # BlueprintPipeline/
    paper_cmd = bp_root / "scenesmith-service" / "scenesmith_paper_command.py"
    if not paper_cmd.exists():
        raise FileNotFoundError(f"Missing SceneSmith command bridge: {paper_cmd}")

    # Defaults expected in the hybrid image.
    os.environ.setdefault("SCENESMITH_PAPER_REPO_DIR", "/opt/scenesmith")
    os.environ.setdefault("SCENESMITH_PAPER_PYTHON_BIN", "/opt/scenesmith/.venv/bin/python")
    os.environ.setdefault("SCENESMITH_PAPER_OPENAI_API", "responses")
    os.environ.setdefault("SCENESMITH_PAPER_ALL_SAM3D", "true")
    os.environ.setdefault("SCENESMITH_PAPER_KEEP_RUN_DIR", "1")
    os.environ.setdefault("SCENESMITH_MP_START_METHOD", "fork")
    os.environ.setdefault("SCENESMITH_PAPER_TIMEOUT_SECONDS", "10800")
    os.environ.setdefault("OPENAI_USE_WEBSOCKET", "1")
    openai_base_url = str(os.environ.get("OPENAI_BASE_URL", "")).strip().lower()
    if (
        not os.environ.get("SCENESMITH_PAPER_OPENAI_WEBSOCKET_BASE_URL")
        and not os.environ.get("OPENAI_WEBSOCKET_BASE_URL")
        and (not openai_base_url or "api.openai.com" in openai_base_url)
    ):
        os.environ["SCENESMITH_PAPER_OPENAI_WEBSOCKET_BASE_URL"] = "wss://api.openai.com/ws/v1/realtime?provider=openai"

    # Alias GEMINI_API_KEY -> GOOGLE_API_KEY if not already set (Gemini image backend requires GOOGLE_API_KEY)
    if not os.environ.get("GOOGLE_API_KEY") and os.environ.get("GEMINI_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
    os.environ.setdefault("SCENESMITH_PAPER_SINGLE_RUN_LOCK", "1")
    os.environ.setdefault("SCENESMITH_PAPER_CLEAN_STALE_PROCESSES", "1")
    if not os.environ.get("SCENESMITH_PAPER_OPENAI_WEBSOCKET_BASE_URL"):
        ws_url = os.environ.get("OPENAI_WEBSOCKET_BASE_URL", "").strip()
        if ws_url:
            os.environ["SCENESMITH_PAPER_OPENAI_WEBSOCKET_BASE_URL"] = ws_url
    if not os.environ.get("OPENAI_WEBSOCKET_BASE_URL"):
        inherited_ws_url = os.environ.get("SCENESMITH_PAPER_OPENAI_WEBSOCKET_BASE_URL", "").strip()
        if inherited_ws_url:
            os.environ["OPENAI_WEBSOCKET_BASE_URL"] = inherited_ws_url
    if not os.environ.get("SCENESMITH_PAPER_OPENAI_USE_WEBSOCKET"):
        os.environ["SCENESMITH_PAPER_OPENAI_USE_WEBSOCKET"] = (
            os.environ.get("OPENAI_USE_WEBSOCKET")
            if os.environ.get("OPENAI_USE_WEBSOCKET")
            else ("1" if os.environ.get("SCENESMITH_PAPER_OPENAI_WEBSOCKET_BASE_URL") else "0")
        )
    if not os.environ.get("SCENESMITH_PAPER_OPENAI_BASE_URL"):
        paper_openai_base = os.environ.get("OPENAI_BASE_URL", "").strip()
        if paper_openai_base:
            os.environ["SCENESMITH_PAPER_OPENAI_BASE_URL"] = paper_openai_base
    os.environ.setdefault("PYTORCH_JIT", "0")

    payload = {
        "request_id": uuid.uuid4().hex,
        "scene_id": scene_id,
        "prompt": prompt,
        "quality_tier": "paper",
        "seed": int(seed),
        "constraints": {"room_type": room_type},
    }

    proc = subprocess.run(
        [sys.executable, str(paper_cmd)],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        env=os.environ.copy(),
        cwd=str(bp_root),
        check=False,
        timeout=int(os.getenv("SCENESMITH_PAPER_TIMEOUT_SECONDS", "10800")),
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "SceneSmith paper stack failed: "
            f"exit={proc.returncode} stderr_tail={(proc.stderr or '')[-4000:]!r} "
            f"stdout_tail={(proc.stdout or '')[-4000:]!r}"
        )
    out = json.loads(proc.stdout)
    if not isinstance(out, Mapping):
        raise RuntimeError("SceneSmith paper stack returned non-object JSON")
    return dict(out)


def _convert_objects_to_sage(
    raw_objs: Sequence[Mapping[str, Any]],
    *,
    margin_m: float,
    room_type: str,
    seed: int,
) -> Dict[str, Any]:
    objs: List[Dict[str, Any]] = []
    zero_pose_count = 0
    for idx, raw in enumerate(raw_objs, start=1):
        pos = _first_mapping(
            raw,
            [
                ("transform", "position"),
                ("transform", "translation"),
                ("pose", "position"),
                ("pose", "translation"),
                ("position",),
                ("translation",),
            ],
        )
        quat = _first_mapping(
            raw,
            [
                ("transform", "rotation_quaternion"),
                ("transform", "orientation"),
                ("pose", "rotation_quaternion"),
                ("pose", "orientation"),
                ("rotation_quaternion",),
                ("orientation",),
            ],
        )
        dims = _mapping_or_empty(raw.get("dimensions_est"))
        if not dims:
            dims = _mapping_or_empty(raw.get("dimensions"))

        px, py, pz = _to_xyz(pos)
        if px == 0.0 and py == 0.0 and pz == 0.0:
            zero_pose_count += 1
        xg, yg, zg = _scenesmith_pos_to_sage(px, py, pz)

        width_s = max(0.01, abs(_safe_float(dims.get("width"), 0.3)))
        height_s = max(0.01, abs(_safe_float(dims.get("height"), 0.3)))
        depth_s = max(0.01, abs(_safe_float(dims.get("depth"), 0.3)))
        yaw_deg = _scenesmith_rot_to_sage_yaw_deg(quat)

        obj_type = str(raw.get("category") or raw.get("name") or "object").strip() or "object"
        obj_id = str(raw.get("id") or f"obj_{idx:03d}").strip() or f"obj_{idx:03d}"
        source_id = f"sm_{idx:04d}_{_slug(obj_type, default='object')}"

        objs.append(
            {
                "type": obj_type,
                "id": obj_id,
                "source_id": source_id,
                "position": {"x": float(xg), "y": float(yg), "z": float(zg)},
                "rotation": {"x": 0.0, "y": 0.0, "z": float(yaw_deg)},
                "dimensions": {"width": float(width_s), "length": float(depth_s), "height": float(height_s)},
            }
        )

    if len(objs) > 1 and zero_pose_count == len(objs):
        raise RuntimeError(
            "SceneSmith object poses are all zero; refusing to build SAGE layout with degenerate placements. "
            "Expected non-zero transform/pose data in scenesmith_response.json or house_state-based snapshots."
        )

    # Shift into positive room coordinates and compute room bounds.
    min_x = 1e9
    min_y = 1e9
    max_x = -1e9
    max_y = -1e9
    max_z = 0.0
    for o in objs:
        p = o["position"]
        d = o["dimensions"]
        cx, cy, cz = float(p["x"]), float(p["y"]), float(p["z"])
        w, l, h = float(d["width"]), float(d["length"]), float(d["height"])
        min_x = min(min_x, cx - 0.5 * w)
        max_x = max(max_x, cx + 0.5 * w)
        min_y = min(min_y, cy - 0.5 * l)
        max_y = max(max_y, cy + 0.5 * l)
        max_z = max(max_z, cz + h)

    if not math.isfinite(min_x):
        min_x, min_y, max_x, max_y = 0.0, 0.0, 6.0, 6.0

    shift_x = float(margin_m - min_x)
    shift_y = float(margin_m - min_y)
    for o in objs:
        o["position"]["x"] = float(o["position"]["x"] + shift_x)
        o["position"]["y"] = float(o["position"]["y"] + shift_y)
        # Keep z as-is (SceneSmith already y>=0 -> z>=0).

    room_w = float((max_x - min_x) + 2.0 * margin_m)
    room_l = float((max_y - min_y) + 2.0 * margin_m)
    room_h = float(max(3.0, max_z + 0.5))

    if not objs:
        raise RuntimeError(
            f"SceneSmith produced zero objects for room_type={room_type!r} seed={seed}. "
            "This layout cannot be used for Stages 5-7 (no graspable objects)."
        )

    return {
        "room_type": room_type,
        "dimensions": {"width": room_w, "length": room_l, "height": room_h},
        "seed": int(seed),
        "objects": objs,
        "scene_source": "scenesmith",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SceneSmith paper stack and emit a SAGE layout dir")
    parser.add_argument("--results_dir", default=os.getenv("SAGE_RESULTS_DIR", "/workspace/SAGE/server/results"))
    parser.add_argument("--layout_id", default="")
    parser.add_argument("--room_type", default=os.getenv("ROOM_TYPE", "generic_room"))
    parser.add_argument("--task_desc", default=os.getenv("TASK_DESC", ""))
    parser.add_argument("--prompt", default=os.getenv("SCENESMITH_PROMPT", ""))
    parser.add_argument("--seed", type=int, default=int(os.getenv("SEED", "0") or "0"))

    parser.add_argument("--pose_aug_name", default="pose_aug_0")
    parser.add_argument("--mesh_policy", default=os.getenv("SCENESMITH_TO_SAGE_MESH_POLICY", "key_only"), choices=["key_only", "all"])
    parser.add_argument("--max_meshes", type=int, default=int(os.getenv("SCENESMITH_TO_SAGE_MAX_MESHES", "30")))
    parser.add_argument("--sam3d_server", default=os.getenv("SAM3D_SERVER_URL", "http://127.0.0.1:8080"))
    parser.add_argument(
        "--mesh_fallback",
        type=int,
        choices=[0, 1],
        default=_safe_int(os.getenv("SCENESMITH_TO_SAGE_MESH_FALLBACK", "0"), 0),
    )
    parser.add_argument("--stop_sam3d_before_paper", action="store_true", default=os.getenv("SCENESMITH_TO_SAGE_STOP_SAM3D_BEFORE_PAPER", "1") == "1")
    parser.add_argument(
        "--persist_scenesmith_run",
        type=int,
        choices=[0, 1],
        default=_safe_int(os.getenv("SCENESMITH_TO_SAGE_PERSIST_SCENESMITH_RUN", "1"), 1),
    )
    parser.add_argument(
        "--critic_loop_enabled",
        type=int,
        choices=[0, 1],
        default=_safe_int(os.getenv("SCENESMITH_CRITIC_LOOP_ENABLED", "1"), 1),
    )
    parser.add_argument(
        "--critic_max_attempts",
        type=int,
        default=_safe_int(os.getenv("SCENESMITH_CRITIC_MAX_ATTEMPTS", "4"), 4),
    )
    parser.add_argument(
        "--critic_require_quality_pass",
        type=int,
        choices=[0, 1],
        default=_safe_int(os.getenv("SCENESMITH_CRITIC_REQUIRE_QUALITY_PASS", "1"), 1),
    )
    parser.add_argument(
        "--critic_require_score",
        type=int,
        choices=[0, 1],
        default=_safe_int(os.getenv("SCENESMITH_CRITIC_REQUIRE_SCORE", "1"), 1),
    )
    parser.add_argument(
        "--critic_require_faithfulness",
        type=int,
        choices=[0, 1],
        default=_safe_int(os.getenv("SCENESMITH_CRITIC_REQUIRE_FAITHFULNESS", "1"), 1),
    )
    parser.add_argument(
        "--critic_min_score_0_10",
        type=float,
        default=_safe_float(os.getenv("SCENESMITH_CRITIC_MIN_SCORE_0_10"), 8.0),
    )
    parser.add_argument(
        "--critic_min_score_0_1",
        type=float,
        default=_safe_float(os.getenv("SCENESMITH_CRITIC_MIN_SCORE_0_1"), 0.80),
    )
    parser.add_argument(
        "--critic_min_faithfulness",
        type=float,
        default=_safe_float(os.getenv("SCENESMITH_CRITIC_MIN_FAITHFULNESS"), 0.80),
    )
    parser.add_argument(
        "--critic_seed_stride",
        type=int,
        default=_safe_int(os.getenv("SCENESMITH_CRITIC_SEED_STRIDE", "7919"), 7919),
    )
    parser.add_argument(
        "--critic_allow_last_attempt_on_fail",
        type=int,
        choices=[0, 1],
        default=_safe_int(os.getenv("SCENESMITH_CRITIC_ALLOW_LAST_ATTEMPT_ON_FAIL", "0"), 0),
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir).expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    layout_id = str(args.layout_id).strip()
    if not layout_id:
        layout_id = f"layout_{int(time.time())}_{uuid.uuid4().hex[:6]}"

    layout_dir = results_dir / layout_id
    if layout_dir.exists():
        raise FileExistsError(f"layout_dir already exists: {layout_dir}")

    prompt = str(args.prompt).strip()
    if not prompt:
        room_type = str(args.room_type).strip() or "generic_room"
        task = str(args.task_desc).strip()
        if task:
            prompt = f"A realistic {room_type}. {task}."
        else:
            prompt = f"A realistic {room_type}."

    seed = int(args.seed)
    if seed == 0:
        seed = random.randint(1, 10_000_000)

    critic_cfg = CriticLoopConfig(
        enabled=bool(args.critic_loop_enabled),
        max_attempts=max(1, int(args.critic_max_attempts)),
        require_quality_pass=bool(args.critic_require_quality_pass),
        require_critic_score=bool(args.critic_require_score),
        require_faithfulness=bool(args.critic_require_faithfulness),
        min_faithfulness=float(args.critic_min_faithfulness),
        min_critic_total_0_10=float(args.critic_min_score_0_10),
        min_critic_total_0_1=float(args.critic_min_score_0_1),
        seed_stride=max(1, int(args.critic_seed_stride)),
        allow_last_attempt_on_fail=bool(args.critic_allow_last_attempt_on_fail),
    )

    _log(
        "Generating SceneSmith scene: "
        f"layout_id={layout_id} room_type={args.room_type} seed={seed} "
        f"critic_loop={'on' if critic_cfg.enabled else 'off'} max_attempts={critic_cfg.max_attempts}"
    )
    # Avoid concurrent VRAM usage: stop SAM3D server during SceneSmith generation.
    if bool(args.stop_sam3d_before_paper) and _sam3d_health(str(args.sam3d_server)):
        _log("Stopping SAM3D server before SceneSmith paper stack (free VRAM)...")
        _sam3d_shutdown(str(args.sam3d_server))
        # Best-effort wait for it to go down.
        for _ in range(60):
            if not _sam3d_health(str(args.sam3d_server)):
                break
            time.sleep(1.0)

    chosen_response: Optional[Dict[str, Any]] = None
    chosen_seed: Optional[int] = None
    chosen_attempt: Optional[int] = None
    chosen_raw_objects: List[Mapping[str, Any]] = []
    attempt_summaries: List[Dict[str, Any]] = []
    persist_scenesmith_run = bool(args.persist_scenesmith_run)
    scenesmith_run_snapshots: List[Dict[str, Any]] = []

    last_response: Optional[Dict[str, Any]] = None
    last_seed: Optional[int] = None
    last_objects: List[Mapping[str, Any]] = []

    max_attempts = critic_cfg.max_attempts if critic_cfg.enabled else 1
    for attempt_idx in range(1, max_attempts + 1):
        attempt_seed = seed + ((attempt_idx - 1) * critic_cfg.seed_stride)
        attempt_scene_id = layout_id if attempt_idx == 1 else f"{layout_id}_a{attempt_idx:02d}"
        _log(
            f"SceneSmith attempt {attempt_idx}/{max_attempts}: "
            f"scene_id={attempt_scene_id} seed={attempt_seed}"
        )
        response = _run_scenesmith_paper_stack(
            scene_id=attempt_scene_id,
            prompt=prompt,
            room_type=str(args.room_type),
            seed=attempt_seed,
        )
        raw_objects = response.get("objects") if isinstance(response.get("objects"), list) else []
        last_response = response
        last_seed = attempt_seed
        last_objects = list(raw_objects)
        paper_stack = response.get("paper_stack") if isinstance(response, Mapping) else {}

        eval_report = _evaluate_critic(response, critic_cfg)
        failures = list(eval_report.get("failures", []))
        object_count = len(raw_objects)
        if object_count < 1:
            failures.append("SceneSmith returned 0 objects")
            eval_report["passed"] = False
            eval_report["failures"] = failures

        attempt_summary = {
            "attempt": attempt_idx,
            "seed": attempt_seed,
            "scene_id": attempt_scene_id,
            "object_count": object_count,
            "critic": eval_report,
            "accepted": False,
        }
        if isinstance(paper_stack, Mapping):
            attempt_summary["paper_stack"] = {
                "run_dir": str(paper_stack.get("run_dir") or ""),
                "house_state_path": str(paper_stack.get("house_state_path") or ""),
                "scenesmith_exit_code": paper_stack.get("scenesmith_exit_code"),
                "model_selected": str(paper_stack.get("model_selected") or ""),
            }
        attempt_summaries.append(attempt_summary)

        if not critic_cfg.enabled:
            chosen_response = response
            chosen_seed = attempt_seed
            chosen_attempt = attempt_idx
            chosen_raw_objects = list(raw_objects)
            attempt_summary["accepted"] = True
            if not eval_report.get("passed"):
                _log(
                    "Critic loop disabled; accepting first attempt even though critic checks did not pass: "
                    f"{'; '.join(failures) if failures else 'unknown reason'}"
                )
            break

        if eval_report.get("passed"):
            chosen_response = response
            chosen_seed = attempt_seed
            chosen_attempt = attempt_idx
            chosen_raw_objects = list(raw_objects)
            attempt_summary["accepted"] = True
            _log(
                "SceneSmith critic gate PASS: "
                f"attempt={attempt_idx} object_count={object_count} "
                f"quality_pass={eval_report.get('quality_pass')} "
                f"critic_total={eval_report.get('critic_total')} "
                f"faithfulness={eval_report.get('faithfulness')}"
            )
            break

        _log(
            "SceneSmith critic gate FAIL: "
            f"attempt={attempt_idx} reasons={'; '.join(failures) if failures else 'unknown'} "
            f"critic_keys={_json_compact(eval_report.get('critic_found_keys', []), max_len=160)}"
        )

    if chosen_response is None:
        if critic_cfg.allow_last_attempt_on_fail and last_response is not None and last_objects:
            chosen_response = last_response
            chosen_seed = last_seed
            chosen_attempt = max_attempts
            chosen_raw_objects = list(last_objects)
            if attempt_summaries:
                attempt_summaries[-1]["accepted"] = True
            _log(
                "WARNING: SceneSmith critic loop rejected all attempts; "
                "accepting last attempt due to critic_allow_last_attempt_on_fail=1."
            )
        else:
            last_reason = ""
            if attempt_summaries:
                last_failures = attempt_summaries[-1].get("critic", {}).get("failures", [])
                if isinstance(last_failures, list) and last_failures:
                    last_reason = f" Last failure: {last_failures[0]}"
            raise RuntimeError(
                f"SceneSmith critic loop rejected all {max_attempts} attempts."
                f"{last_reason}"
            )

    assert chosen_response is not None  # for type checkers
    assert chosen_seed is not None
    response = chosen_response
    raw_objects = chosen_raw_objects

    layout_dir.mkdir(parents=True, exist_ok=True)
    scenesmith_run_snapshots.append(
        _snapshot_scenesmith_run(
            response=response,
            layout_dir=layout_dir,
            attempt_label=str(chosen_attempt or 1),
            persist_enabled=bool(persist_scenesmith_run),
        )
    )

    # Convert objects to SAGE room dict.
    room = _convert_objects_to_sage(raw_objects, margin_m=0.6, room_type=str(args.room_type), seed=int(chosen_seed))
    objs = list(room.get("objects", []) or [])

    # Create layout dir structure.
    (layout_dir / "generation").mkdir(parents=True, exist_ok=True)
    (layout_dir / args.pose_aug_name).mkdir(parents=True, exist_ok=True)

    # Mesh generation (bounded).
    candidates = _pick_mesh_candidates(raw_objects, mesh_policy=str(args.mesh_policy), max_meshes=int(args.max_meshes))
    candidate_set = set(candidates)
    mesh_prompts: Dict[int, str] = {}
    for idx in candidates:
        raw = raw_objects[idx]
        name = str(raw.get("name") or raw.get("category") or "object").strip() or "object"
        mesh_prompts[idx] = name
    existing_glb_pool = _collect_existing_glb_pool(response=response, raw_objects=raw_objects)
    existing_glb_count = len(existing_glb_pool.get("all_paths", [])) if isinstance(existing_glb_pool, Mapping) else 0
    if existing_glb_count > 0:
        _log(f"Found {existing_glb_count} existing GLB assets from SceneSmith run; reusing when matched.")

    _log(f"Meshes: policy={args.mesh_policy} max_meshes={args.max_meshes} candidates={len(candidates)}/{len(objs)}")

    mesh_fallback_enabled = bool(args.mesh_fallback)
    for i, obj in enumerate(objs):
        source_id = str(obj.get("source_id") or "").strip()
        if not source_id:
            continue
        out_obj = layout_dir / "generation" / f"{source_id}.obj"
        raw_obj = raw_objects[i] if i < len(raw_objects) else {}
        wrote_mesh = False

        # First preference: reuse SceneSmith-generated GLBs from the existing run.
        existing_glb = _pick_existing_glb_for_object(existing_glb_pool, raw_obj)
        if existing_glb is not None:
            try:
                _convert_glb_to_obj(existing_glb, out_obj)
                _log(f"Reused SceneSmith GLB [{i+1}/{len(objs)}]: {existing_glb.name} -> {out_obj.name}")
                wrote_mesh = True
                continue
            except Exception as exc:
                if i in candidate_set:
                    if mesh_fallback_enabled:
                        _log(f"WARNING: Failed GLB->OBJ conversion for {existing_glb}: {exc} (falling back later)")
                    else:
                        raise RuntimeError(f"Required SAM3D input failed for candidate object index={i}: {exc}") from exc
                else:
                    _log(f"WARNING: Failed GLB->OBJ conversion for {existing_glb}: {exc}")

        if i in candidate_set:
            prompt_i = mesh_prompts.get(i, str(obj.get("type") or "object"))
            try:
                _sam3d_start_if_needed(str(args.sam3d_server))
                _log(f"SAM3D mesh [{i+1}/{len(objs)}]: {prompt_i!r} -> {out_obj.name}")
                _sam3d_generate_obj(prompt=prompt_i, out_obj_path=out_obj, seed=int(chosen_seed) + i, server_base=str(args.sam3d_server))
                wrote_mesh = True
                continue
            except Exception as exc:
                if mesh_fallback_enabled:
                    _log(f"WARNING: SAM3D mesh failed for {prompt_i!r}: {exc} (falling back to unit box)")
                else:
                    raise RuntimeError(f"SAM3D mesh generation failed for candidate object {prompt_i!r}: {exc}") from exc
        if not wrote_mesh and (mesh_fallback_enabled or i not in candidate_set):
            _write_unit_box_obj(out_obj)

    # Write room jsons and pose-aug meta/variant.
    room_json = layout_dir / "room_0.json"
    _write_json(room_json, room)

    layout_payload = _single_room_layout_payload(room, layout_id=layout_id)
    base_layout_json = layout_dir / f"{layout_id}.json"
    _write_json(base_layout_json, layout_payload)

    variant_json_rel = "variant_000.json"
    variant_json = layout_dir / args.pose_aug_name / variant_json_rel
    _write_json(variant_json, layout_payload)

    meta_json = layout_dir / args.pose_aug_name / "meta.json"
    _write_json(meta_json, [variant_json_rel])

    # Keep a copy of raw response for debugging.
    _write_json(layout_dir / "scenesmith_response.json", response)
    _write_json(
        layout_dir / "scenesmith_critic_attempts.json",
        {
            "layout_id": layout_id,
            "accepted_attempt": int(chosen_attempt or 1),
            "accepted_seed": int(chosen_seed),
            "persist_scenesmith_run": bool(persist_scenesmith_run),
            "scenesmith_run_snapshots": scenesmith_run_snapshots,
            "critic_loop": {
                "enabled": critic_cfg.enabled,
                "max_attempts": critic_cfg.max_attempts,
                "require_quality_pass": critic_cfg.require_quality_pass,
                "require_critic_score": critic_cfg.require_critic_score,
                "require_faithfulness": critic_cfg.require_faithfulness,
                "min_critic_total_0_10": critic_cfg.min_critic_total_0_10,
                "min_critic_total_0_1": critic_cfg.min_critic_total_0_1,
                "min_faithfulness": critic_cfg.min_faithfulness,
                "seed_stride": critic_cfg.seed_stride,
                "allow_last_attempt_on_fail": critic_cfg.allow_last_attempt_on_fail,
            },
            "attempts": attempt_summaries,
        },
    )
    critic_report: Dict[str, Any] = {}
    critic_outputs = response.get("critic_outputs")
    if isinstance(critic_outputs, Mapping):
        critic_report.update(dict(critic_outputs))
    for key in ("placement_stages", "critic_scores", "support_surfaces", "faithfulness_report", "quality_gate_report"):
        value = response.get(key)
        if value is None or key in critic_report:
            continue
        if isinstance(value, (Mapping, list)):
            critic_report[key] = value
    if critic_report:
        critic_path = layout_dir / "scenesmith_critic_report.json"
        _write_json(critic_path, critic_report)
        _log(f"Saved SceneSmith critic report: {critic_path}")

    # Print layout id only (consumed by bash).
    sys.stdout.write(layout_id + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
