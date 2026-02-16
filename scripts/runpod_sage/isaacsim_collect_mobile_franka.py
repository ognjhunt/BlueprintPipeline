#!/usr/bin/env python3
"""
Isaac Sim Stage 7 collector for SAGE mobile_franka demos.

This script MUST be executed with Isaac Sim Python (e.g. /workspace/isaacsim_env/bin/python3).

It:
1) Converts pose-aug scene meshes (OBJ+MTL+textures) to USD via omni.kit.asset_converter
2) Assembles a USD stage (room + objects + RidgebackFranka robot)
3) Replays planned trajectories (base pose + arm joints + gripper)
4) Captures real RGB-D via Replicator annotators
5) Writes robomimic-compatible dataset.hdf5 + metadata files
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _sanitize_sys_path_for_isaacsim() -> None:
    blocked = Path(os.environ.get("SAGE_SERVER_DIR", "/workspace/SAGE/server")).resolve()
    cleaned: List[str] = []
    for entry in sys.path:
        # Drop implicit CWD entry to avoid importing /workspace/SAGE/server/isaacsim by accident.
        if not entry:
            continue
        try:
            real = Path(entry).resolve()
        except Exception:
            cleaned.append(entry)
            continue
        if real == blocked:
            continue
        cleaned.append(entry)
    sys.path[:] = cleaned


_sanitize_sys_path_for_isaacsim()

# Make BlueprintPipeline helpers importable (SimReady Lite presets).
BP_DIR = os.environ.get("BP_DIR", "/workspace/BlueprintPipeline")
if BP_DIR not in sys.path:
    sys.path.insert(0, BP_DIR)

from scripts.runpod_sage.bp_simready_lite import recommend_simready_lite


def _log(msg: str) -> None:
    print(f"[isaacsim-collect {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}", flush=True)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _env_bool(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "1" if default else "0")).strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def _resolve_sensor_failure_policy(*, strict: bool, requested: str) -> str:
    policy = str(requested or "").strip().lower()
    if policy in {"fail", "warn"}:
        return policy
    # Auto policy: strict mode fails closed, non-strict warns.
    return "fail" if strict else "warn"


BUNDLE_RUNTIME_MISMATCH_MARKER = "BUNDLE_RUNTIME_MISMATCH"
BUNDLE_RUNTIME_MISSING_RUN_ID_MARKER = "BUNDLE_RUNTIME_MISSING_RUN_ID"
DYNAMIC_COLLISION_APPROX_ALLOWED = {"convexDecomposition", "convexHull"}


def _coerce_optional_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(int(value))
    if isinstance(value, str):
        raw = value.strip().lower()
        if raw in {"1", "true", "yes", "y", "on"}:
            return True
        if raw in {"0", "false", "no", "n", "off"}:
            return False
    return None


def _bundle_runtime_mismatches(bundle: Dict[str, Any], args: argparse.Namespace) -> List[str]:
    checks = (
        ("strict", "strict"),
        ("headless", "headless"),
        ("enable_cameras", "enable_cameras"),
    )
    mismatches: List[str] = []
    for bundle_key, arg_key in checks:
        expected = _coerce_optional_bool(bundle.get(bundle_key))
        actual = bool(getattr(args, arg_key))
        if expected is None:
            mismatches.append(
                f"{BUNDLE_RUNTIME_MISMATCH_MARKER}: missing/invalid bundle field '{bundle_key}' "
                f"(runtime {arg_key}={actual})"
            )
            continue
        if expected != actual:
            mismatches.append(
                f"{BUNDLE_RUNTIME_MISMATCH_MARKER}: bundle {bundle_key}={expected} "
                f"!= runtime {arg_key}={actual}"
            )
    return mismatches


def _enforce_bundle_runtime_parity(bundle: Dict[str, Any], args: argparse.Namespace) -> None:
    mismatches = _bundle_runtime_mismatches(bundle, args)
    if not mismatches:
        return
    message = "; ".join(mismatches)
    enforce = bool(args.strict) or _env_bool("SAGE_ENFORCE_BUNDLE_STRICT", default=True)
    if enforce:
        raise RuntimeError(message)
    _log(f"WARNING: {message} (continuing because strict enforcement is disabled)")


def _normalize_dynamic_collision_approximation(value: Any) -> str:
    token = str(value or "").strip()
    lower = token.lower()
    if lower in {"convexdecomposition", "convex_decomposition"}:
        return "convexDecomposition"
    if lower in {"convexhull", "convex_hull"}:
        return "convexHull"
    return "convexDecomposition"


def _is_valid_dynamic_collision_approximation(value: Any) -> bool:
    return str(value) in DYNAMIC_COLLISION_APPROX_ALLOWED


def _camera_sensor_qc_metrics(
    *,
    rgb_frame: np.ndarray,
    depth_raw: np.ndarray,
    min_valid_depth_px: int,
    min_rgb_std: float,
    min_depth_std: float,
) -> Dict[str, Any]:
    rgb_np = np.asarray(rgb_frame)
    depth_np = np.asarray(depth_raw, dtype=np.float32)
    rgb_std = float(rgb_np.std()) if rgb_np.size else 0.0
    rgb_max = float(rgb_np.max()) if rgb_np.size else 0.0
    valid_depth_mask = np.logical_and(np.isfinite(depth_np), depth_np > 0.0)
    valid_depth_px = int(valid_depth_mask.sum())
    valid_fraction = float(valid_depth_px / max(1, depth_np.size))
    depth_std = float(depth_np[valid_depth_mask].std()) if valid_depth_px > 0 else 0.0

    failures: List[str] = []
    if rgb_std <= float(min_rgb_std) and rgb_max <= 2.0:
        failures.append("degenerate_rgb")
    if valid_depth_px < int(min_valid_depth_px):
        failures.append("degenerate_depth_valid_px")
    elif depth_std <= float(min_depth_std):
        failures.append("low_depth_variance")

    return {
        "rgb_std": rgb_std,
        "rgb_max": rgb_max,
        "valid_depth_px": valid_depth_px,
        "valid_depth_fraction": valid_fraction,
        "depth_std": depth_std,
        "failures": failures,
    }


def _evaluate_dual_camera_sensor_qc(
    *,
    agentview_rgb: np.ndarray,
    agentview_depth_raw: np.ndarray,
    agentview2_rgb: np.ndarray,
    agentview2_depth_raw: np.ndarray,
    min_valid_depth_px: int,
    min_rgb_std: float,
    min_depth_std: float,
) -> Dict[str, Any]:
    qc_a = _camera_sensor_qc_metrics(
        rgb_frame=agentview_rgb,
        depth_raw=agentview_depth_raw,
        min_valid_depth_px=min_valid_depth_px,
        min_rgb_std=min_rgb_std,
        min_depth_std=min_depth_std,
    )
    qc_b = _camera_sensor_qc_metrics(
        rgb_frame=agentview2_rgb,
        depth_raw=agentview2_depth_raw,
        min_valid_depth_px=min_valid_depth_px,
        min_rgb_std=min_rgb_std,
        min_depth_std=min_depth_std,
    )

    failures: List[str] = []
    failures.extend([f"agentview:{item}" for item in qc_a["failures"]])
    failures.extend([f"agentview_2:{item}" for item in qc_b["failures"]])
    return {
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "agentview": qc_a,
        "agentview_2": qc_b,
    }


def _resolve_variant_json_path(raw_path: str, *, layout_dir: Path) -> Path:
    p = Path(raw_path)
    if p.exists():
        return p
    if not p.is_absolute():
        candidate = layout_dir / p
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Variant layout JSON not found: {raw_path}")


def _extract_room_dict(layout_or_room: Dict[str, Any], *, demo: Dict[str, Any], variant_path: Path) -> Dict[str, Any]:
    # Already a room payload.
    if isinstance(layout_or_room, dict) and isinstance(layout_or_room.get("objects"), list):
        return layout_or_room

    rooms = layout_or_room.get("rooms") if isinstance(layout_or_room, dict) else None
    if not isinstance(rooms, list) or len(rooms) == 0:
        raise RuntimeError(
            f"Unsupported variant payload shape in {variant_path}; expected room dict or layout with non-empty 'rooms'"
        )

    if len(rooms) == 1:
        return rooms[0]

    # Multi-room fallback: pick room containing pick/place object ids.
    pick_source_id = str((demo.get("pick_object") or {}).get("source_id", "") or "")
    place_source_id = str((demo.get("place_surface") or {}).get("source_id", "") or "")
    wanted_ids = {x for x in (pick_source_id, place_source_id) if x}
    if wanted_ids:
        for room in rooms:
            objects = room.get("objects") or []
            room_ids = {str((obj or {}).get("source_id", "") or "") for obj in objects}
            if wanted_ids & room_ids:
                return room

    # Last resort.
    return rooms[0]


def _resolve_ridgeback_franka_usd() -> str:
    local_root = Path(os.environ.get("ISAAC_ASSETS_ROOT", "/workspace/isaacsim_assets/Assets/Isaac/5.1"))
    local_usd = local_root / "Isaac/Robots/Clearpath/RidgebackFranka/ridgeback_franka.usd"
    if local_usd.exists():
        _log(f"Using local robot USD: {local_usd}")
        return str(local_usd)
    allow_remote = _env_bool("SAGE_ALLOW_REMOTE_ISAAC_ASSETS", default=False)
    if not allow_remote:
        raise FileNotFoundError(
            "Local RidgebackFranka USD missing and remote fallback disabled. "
            f"Set SAGE_ALLOW_REMOTE_ISAAC_ASSETS=1 to allow fallback. Missing: {local_usd}"
        )
    remote_usd = "/Isaac/Robots/Clearpath/RidgebackFranka/ridgeback_franka.usd"
    _log(
        "WARNING: Local robot USD missing "
        f"({local_usd}); falling back to remote path due to SAGE_ALLOW_REMOTE_ISAAC_ASSETS=1: {remote_usd}"
    )
    return remote_usd


def _has_nvidia_marker(path: Path) -> bool:
    if str(path) == "/NVIDIA":
        return True
    if any(part == "NVIDIA" for part in path.parts):
        return True
    return (path / "NVIDIA").exists()


def _pick_assets_root_with_marker(local_root: Path) -> Path:
    """
    Keep local ISAAC_ASSETS_ROOT for deterministic assets, but emit marker sanity
    diagnostics for Nucleus-style roots where '/NVIDIA' should be present.
    """
    if _has_nvidia_marker(local_root):
        _log(f"Asset root marker check: found '/NVIDIA' marker for {local_root}")
        return local_root

    _log(
        "WARNING: Asset root marker check did not find '/NVIDIA' marker under "
        f"{local_root}. Continuing with explicit local root."
    )
    return local_root


def _configure_local_assets_root() -> None:
    local_root = Path(os.environ.get("ISAAC_ASSETS_ROOT", "/workspace/isaacsim_assets/Assets/Isaac/5.1"))
    if not local_root.exists():
        _log(f"WARNING: local ISAAC_ASSETS_ROOT does not exist: {local_root}")
        return
    selected_root = _pick_assets_root_with_marker(local_root)
    try:
        import carb

        carb.settings.get_settings().set("/persistent/isaac/asset_root/default", str(selected_root))
    except BaseException as exc:
        _log(f"WARNING: failed to set carb asset_root to local path ({type(exc).__name__}: {exc})")
    if os.getenv("SAGE_ENABLE_NUCLEUS_ASSET_ROOT", "0") == "1":
        try:
            from omni.isaac.core.utils import nucleus as nucleus_utils

            set_root = getattr(nucleus_utils, "set_assets_root_path", None)
            if callable(set_root):
                set_root(str(selected_root))
        except BaseException as exc:
            _log(f"WARNING: failed to set nucleus asset root to local path ({type(exc).__name__}: {exc})")
    _log(f"Configured local Isaac assets root: {selected_root}")


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _git_commit_sha(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def _runtime_provenance(
    *,
    bp_root: Path,
    sensor_failure_policy: str,
    args: argparse.Namespace,
    run_id: str,
    bundle_path: Path,
    bundle_sha256: str,
) -> Dict[str, Any]:
    script_path = Path(__file__).resolve()
    env_snapshot = {
        "ISAAC_ASSETS_ROOT": os.environ.get("ISAAC_ASSETS_ROOT", ""),
        "SAGE_ALLOW_REMOTE_ISAAC_ASSETS": os.environ.get("SAGE_ALLOW_REMOTE_ISAAC_ASSETS", "0"),
        "SAGE_SENSOR_FAILURE_POLICY": os.environ.get("SAGE_SENSOR_FAILURE_POLICY", "auto"),
        "SAGE_RENDER_WARMUP_FRAMES": os.environ.get("SAGE_RENDER_WARMUP_FRAMES", ""),
        "SAGE_SENSOR_MIN_RGB_STD": os.environ.get("SAGE_SENSOR_MIN_RGB_STD", ""),
        "SAGE_SENSOR_MIN_DEPTH_STD": os.environ.get("SAGE_SENSOR_MIN_DEPTH_STD", ""),
        "SAGE_MIN_VALID_DEPTH_PX": os.environ.get("SAGE_MIN_VALID_DEPTH_PX", ""),
        "SAGE_SENSOR_CHECK_FRAME": os.environ.get("SAGE_SENSOR_CHECK_FRAME", ""),
        "SAGE_ENFORCE_BUNDLE_STRICT": os.environ.get("SAGE_ENFORCE_BUNDLE_STRICT", ""),
        "SAGE_RUN_ID": os.environ.get("SAGE_RUN_ID", ""),
    }
    return {
        "run_id": run_id,
        "collector_script": str(script_path),
        "collector_script_sha256": _file_sha256(script_path),
        "bp_git_commit": _git_commit_sha(bp_root),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "plan_bundle_path": str(bundle_path),
        "plan_bundle_sha256": bundle_sha256,
        "sensor_failure_policy_effective": sensor_failure_policy,
        "args": {
            "headless": bool(args.headless),
            "enable_cameras": bool(args.enable_cameras),
            "strict": bool(args.strict),
            "render_warmup_frames": int(args.render_warmup_frames),
            "sensor_check_frame": int(args.sensor_check_frame),
            "sensor_min_rgb_std": float(args.sensor_min_rgb_std),
            "sensor_min_depth_std": float(args.sensor_min_depth_std),
            "min_valid_depth_px": int(args.min_valid_depth_px),
            "camera_retry_steps": int(args.camera_retry_steps),
            "dome_light_intensity": float(args.dome_light_intensity),
            "sun_light_intensity": float(args.sun_light_intensity),
        },
        "env": env_snapshot,
    }


def _ensure_extension(ext_name: str) -> None:
    import omni.kit.app

    em = omni.kit.app.get_app().get_extension_manager()
    # This is synchronous and works in standalone.
    em.set_extension_enabled_immediate(ext_name, True)


async def _convert_mesh_to_usd(in_file: str, out_file: str, *, load_materials: bool = True) -> bool:
    _ensure_extension("omni.kit.asset_converter")
    import omni.kit.asset_converter

    converter_context = omni.kit.asset_converter.AssetConverterContext()
    converter_context.ignore_materials = not load_materials
    converter_context.ignore_animations = True
    converter_context.ignore_camera = True
    converter_context.ignore_light = True
    # Prefer a single USD per mesh.
    converter_context.merge_all_meshes = True
    # Best-effort unit conversion; some pipelines still need manual scaling.
    converter_context.use_meter_as_world_unit = True
    converter_context.baking_scales = True
    converter_context.use_double_precision_to_usd_transform_op = True

    instance = omni.kit.asset_converter.get_instance()
    task = instance.create_converter_task(in_file, out_file, None, converter_context)
    success = await task.wait_until_finished()
    if not success:
        raise RuntimeError(f"Asset conversion failed: {in_file} -> {out_file}: {task.get_error_message()}")
    return bool(success)


def _run_async(coro) -> Any:
    """
    Run an async conversion coroutine in Isaac Sim standalone context.

    Handles both "no loop running" and "loop already running" cases by polling
    kit updates.
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if not loop.is_running():
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)

    # Loop is already running: schedule and poll.
    fut = asyncio.ensure_future(coro, loop=loop)
    import omni.kit.app

    app = omni.kit.app.get_app()
    while not fut.done():
        # Advance kit one frame.
        app.update()
        time.sleep(0.01)
    return fut.result()


def _annotator_data(annotator, *, name: str):
    """Get data from a Replicator annotator, handling API differences."""
    result = annotator.get_data()
    if result is None:
        raise RuntimeError(f"Annotator '{name}' returned None")
    if isinstance(result, dict):
        if "data" in result:
            return result["data"]
        # Some builds may return dict payloads with alternate keys. Prefer any ndarray-like value.
        for v in result.values():
            if hasattr(v, "shape"):
                return v
        raise RuntimeError(f"Annotator '{name}' returned dict without usable data keys: {list(result.keys())}")
    return result


def _add_default_lighting(stage, *, dome_intensity: float, distant_intensity: float) -> None:
    """
    Add explicit scene lighting so headless captures do not depend on renderer defaults.
    Safe to call repeatedly; reuses existing prim paths.
    """
    from pxr import Gf, UsdGeom, UsdLux

    UsdGeom.Scope.Define(stage, "/World/Lights")

    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
    dome.CreateIntensityAttr().Set(float(max(0.0, dome_intensity)))
    dome.CreateColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))

    sun = UsdLux.DistantLight.Define(stage, "/World/Lights/Sun")
    sun.CreateIntensityAttr().Set(float(max(0.0, distant_intensity)))
    sun.CreateColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))
    # Gentle elevation so horizontal surfaces receive light.
    sun_xf = np.eye(4, dtype=np.float64)
    c = math.cos(-math.pi * 0.25)
    s = math.sin(-math.pi * 0.25)
    sun_xf[:3, :3] = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)
    _apply_xform(stage, "/World/Lights/Sun", sun_xf)


def _capture_camera_frames(
    *,
    annotators: Dict[str, Dict[str, Any]],
    world: Any,
    max_extra_steps: int,
    width: int,
    height: int,
    min_valid_depth_px: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Capture one frame from both cameras, retrying a few simulation steps until
    shapes are valid, RGB is not fully black, and depth has enough valid pixels.
    """
    last_reason = "unknown"
    for _ in range(max(1, int(max_extra_steps) + 1)):
        a_rgb = _annotator_data(annotators["agentview"]["rgb"], name="agentview_rgb")
        a_d = _annotator_data(annotators["agentview"]["depth"], name="agentview_depth")
        b_rgb = _annotator_data(annotators["agentview_2"]["rgb"], name="agentview_2_rgb")
        b_d = _annotator_data(annotators["agentview_2"]["depth"], name="agentview_2_depth")

        a_rgb_np = np.asarray(a_rgb)[..., :3].astype(np.uint8)
        a_d_raw = np.asarray(a_d, dtype=np.float32)
        a_d_np = np.nan_to_num(a_d_raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float16)
        b_rgb_np = np.asarray(b_rgb)[..., :3].astype(np.uint8)
        b_d_raw = np.asarray(b_d, dtype=np.float32)
        b_d_np = np.nan_to_num(b_d_raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float16)

        rgb_ok = a_rgb_np.shape == (height, width, 3) and b_rgb_np.shape == (height, width, 3)
        depth_shape_ok = a_d_np.shape == (height, width) and b_d_np.shape == (height, width)
        a_valid_depth_px = int(np.logical_and(np.isfinite(a_d_raw), a_d_raw > 0.0).sum())
        b_valid_depth_px = int(np.logical_and(np.isfinite(b_d_raw), b_d_raw > 0.0).sum())
        depth_ok = (
            depth_shape_ok
            and a_valid_depth_px >= int(min_valid_depth_px)
            and b_valid_depth_px >= int(min_valid_depth_px)
        )
        not_black = bool(
            a_rgb_np.size > 0
            and b_rgb_np.size > 0
            and float(a_rgb_np.max()) > 0.0
            and float(b_rgb_np.max()) > 0.0
        )

        if rgb_ok and depth_ok and not_black:
            return a_rgb_np, a_d_np, a_d_raw, b_rgb_np, b_d_np, b_d_raw

        last_reason = (
            f"rgb_shapes={a_rgb_np.shape}/{b_rgb_np.shape} "
            f"depth_shapes={a_d_np.shape}/{b_d_np.shape} "
            f"rgb_max={float(a_rgb_np.max()) if a_rgb_np.size else -1.0:.3f}/"
            f"{float(b_rgb_np.max()) if b_rgb_np.size else -1.0:.3f} "
            f"valid_depth_px={a_valid_depth_px}/{b_valid_depth_px} "
            f"depth_minmax={float(np.nanmin(a_d_raw)) if a_d_raw.size else float('nan'):.4f}/"
            f"{float(np.nanmax(a_d_raw)) if a_d_raw.size else float('nan'):.4f}"
        )
        world.step(render=True)

    raise RuntimeError(f"Camera capture did not become valid after retries: {last_reason}")


def _read_obj_bounds(obj_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    mins = np.array([1e9, 1e9, 1e9], dtype=np.float64)
    maxs = np.array([-1e9, -1e9, -1e9], dtype=np.float64)
    with obj_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("v "):
                continue
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            v = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
            mins = np.minimum(mins, v)
            maxs = np.maximum(maxs, v)
    if not np.isfinite(mins).all() or not np.isfinite(maxs).all():
        return np.zeros(3, dtype=np.float64), np.ones(3, dtype=np.float64)
    return mins, maxs


def _xform_matrix(
    *,
    translation: Tuple[float, float, float],
    yaw_rad: float,
    scale: Tuple[float, float, float],
) -> np.ndarray:
    tx, ty, tz = translation
    sx, sy, sz = scale
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    S = np.diag([sx, sy, sz]).astype(np.float64)
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R @ S
    M[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)
    return M


def _apply_xform(stage, prim_path: str, matrix: np.ndarray) -> None:
    from pxr import Gf, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Invalid prim path for xform: {prim_path}")
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    m = Gf.Matrix4d(*matrix.reshape(-1).tolist())
    xformable.AddTransformOp().Set(m)


def _define_camera(stage, prim_path: str, *, pos: np.ndarray, look_at: np.ndarray) -> None:
    from pxr import Gf, UsdGeom

    cam = UsdGeom.Camera.Define(stage, prim_path)
    cam.GetFocalLengthAttr().Set(35.0)
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 100.0))

    forward = (look_at - pos).astype(np.float64)
    forward = forward / max(np.linalg.norm(forward), 1e-9)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, up)
    right = right / max(np.linalg.norm(right), 1e-9)
    up2 = np.cross(right, forward)

    # USD camera looks down -Z in local space.
    R = np.stack([right, up2, -forward], axis=1)
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R
    M[:3, 3] = pos

    xformable = UsdGeom.Xformable(cam)
    xformable.ClearXformOpOrder()
    xformable.AddTransformOp().Set(Gf.Matrix4d(*M.reshape(-1).tolist()))


def _apply_simready_lite_physics(stage, prim_path: str, *, category: str, dims: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply mass/friction/restitution + collision approximation hints.
    """
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    try:
        from pxr import PhysxSchema  # type: ignore
        PHYSX = PhysxSchema
    except Exception:  # pragma: no cover
        PHYSX = None

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Invalid prim for physics: {prim_path}")

    mesh_prims: List[Any] = []
    for candidate in Usd.PrimRange(prim):
        if candidate and candidate.IsValid() and candidate.IsA(UsdGeom.Mesh):
            mesh_prims.append(candidate)
    collision_targets = mesh_prims if mesh_prims else [prim]

    preset = recommend_simready_lite(category=category, dimensions=dims)
    approximation = str(preset.collision_approximation)
    if preset.is_dynamic:
        approximation = _normalize_dynamic_collision_approximation(approximation)

    # Rigid body + mass
    rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(prim)
    rigid_body_api.CreateKinematicEnabledAttr().Set(not preset.is_dynamic)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr().Set(float(preset.mass_kg))

    mesh_collision_paths: List[str] = []
    for collision_target in collision_targets:
        collision_api = UsdPhysics.CollisionAPI.Apply(collision_target)
        collision_api.CreateCollisionEnabledAttr().Set(True)
        collision_target.CreateAttribute("physics:approximation", Sdf.ValueTypeNames.Token).Set(approximation)
        mesh_collision_paths.append(str(collision_target.GetPath()))

        if PHYSX is not None and approximation == "convexDecomposition":
            try:
                physx_collision = PHYSX.PhysxCollisionAPI.Apply(collision_target)
                physx_collision.CreateMaxConvexHullsAttr().Set(int(preset.collision_max_hulls))
            except Exception:
                pass

    # Material (friction/restitution) bound as physics material
    material_path = f"{prim_path}/PhysicsMaterial"
    material = UsdShade.Material.Define(stage, material_path)
    material_prim = material.GetPrim()
    mat_api = UsdPhysics.MaterialAPI.Apply(material_prim)
    mat_api.CreateStaticFrictionAttr().Set(float(preset.static_friction))
    mat_api.CreateDynamicFrictionAttr().Set(float(preset.dynamic_friction))
    mat_api.CreateRestitutionAttr().Set(float(preset.restitution))

    if PHYSX is not None:
        try:
            physx_mat_api = PHYSX.PhysxMaterialAPI.Apply(material_prim)
            physx_mat_api.CreateStaticFrictionAttr().Set(float(preset.static_friction))
            physx_mat_api.CreateDynamicFrictionAttr().Set(float(preset.dynamic_friction))
            physx_mat_api.CreateRestitutionAttr().Set(float(preset.restitution))
        except Exception:
            pass

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
    for collision_target in collision_targets:
        UsdShade.MaterialBindingAPI.Apply(collision_target).Bind(material)

    return {
        "mass_kg": preset.mass_kg,
        "static_friction": preset.static_friction,
        "dynamic_friction": preset.dynamic_friction,
        "restitution": preset.restitution,
        "is_dynamic": preset.is_dynamic,
        "collision_approximation": approximation,
        "collision_max_hulls": preset.collision_max_hulls,
        "mesh_collision_prims": mesh_collision_paths,
    }


def _validate_dynamic_collision_settings(stage, physics_report: Dict[str, Any]) -> Dict[str, Any]:
    from pxr import UsdPhysics

    total_dynamic_meshes = 0
    collision_enabled_meshes = 0
    bad_dynamic_approximations: List[Dict[str, str]] = []
    missing_collision_paths: List[str] = []

    for entry in physics_report.get("objects", []) or []:
        if not bool(entry.get("is_dynamic")):
            continue
        mesh_paths = list(entry.get("mesh_collision_prims") or [])
        for mesh_path in mesh_paths:
            total_dynamic_meshes += 1
            prim = stage.GetPrimAtPath(mesh_path)
            if not prim or not prim.IsValid():
                missing_collision_paths.append(str(mesh_path))
                continue

            coll_api = UsdPhysics.CollisionAPI(prim)
            enabled_attr = coll_api.GetCollisionEnabledAttr()
            enabled = False
            if enabled_attr and enabled_attr.IsValid():
                value = enabled_attr.Get()
                enabled = bool(value) if value is not None else False
            if enabled:
                collision_enabled_meshes += 1
            else:
                missing_collision_paths.append(str(mesh_path))

            approx_attr = prim.GetAttribute("physics:approximation")
            approx_value = approx_attr.Get() if approx_attr and approx_attr.IsValid() else ""
            if str(approx_value) not in DYNAMIC_COLLISION_APPROX_ALLOWED:
                bad_dynamic_approximations.append(
                    {"path": str(mesh_path), "approximation": str(approx_value)}
                )

    coverage = float(collision_enabled_meshes / max(1, total_dynamic_meshes))
    return {
        "total_dynamic_meshes": total_dynamic_meshes,
        "collision_enabled_meshes": collision_enabled_meshes,
        "collision_coverage": coverage,
        "missing_collision_count": len(missing_collision_paths),
        "missing_collision_paths": missing_collision_paths[:50],
        "bad_dynamic_approx_count": len(bad_dynamic_approximations),
        "bad_dynamic_approximations": bad_dynamic_approximations[:50],
    }


def _set_robot_root_pose(stage, robot_prim_path: str, x: float, y: float, yaw: float) -> None:
    M = np.eye(4, dtype=np.float64)
    c = math.cos(yaw)
    s = math.sin(yaw)
    M[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    M[:3, 3] = np.array([x, y, 0.0], dtype=np.float64)
    _apply_xform(stage, robot_prim_path, M)


def _init_dynamic_control(robot_prim_path: str):
    from omni.isaac.dynamic_control import _dynamic_control

    dc = _dynamic_control.acquire_dynamic_control_interface()
    art = dc.get_articulation(robot_prim_path)
    if art == _dynamic_control.INVALID_HANDLE:
        raise RuntimeError(f"Dynamic control cannot find articulation at {robot_prim_path}")
    dc.wake_up_articulation(art)
    dof_count = dc.get_articulation_dof_count(art)
    dof_names = []
    for i in range(dof_count):
        dof = dc.get_articulation_dof(art, i)
        dof_names.append(dc.get_dof_name(dof))
    return dc, art, dof_names


def _apply_joint_targets(dc, art, dof_names: List[str], arm_q: np.ndarray, gripper_width: float) -> None:
    # Heuristic mapping:
    # - Arm joints: first 7 dofs whose name contains 'joint' and not 'finger'/'wheel'.
    # - Gripper: dofs with 'finger' in name (set to width/2)
    # If we can't detect, fallback to first 7 and last 2.
    arm_idx = [i for i, n in enumerate(dof_names) if ("joint" in (n or "").lower()) and ("finger" not in (n or "").lower()) and ("wheel" not in (n or "").lower())]
    if len(arm_idx) < 7:
        arm_idx = list(range(min(7, len(dof_names))))
    arm_idx = arm_idx[:7]

    finger_idx = [i for i, n in enumerate(dof_names) if "finger" in (n or "").lower()]
    if not finger_idx and len(dof_names) >= 2:
        finger_idx = [len(dof_names) - 2, len(dof_names) - 1]

    # Build full target vector.
    targets = np.zeros((len(dof_names),), dtype=np.float32)
    # Keep other dofs at current state (best-effort).
    try:
        states = dc.get_articulation_dof_states(art, 0)
        if states and "pos" in states:
            targets[:] = np.asarray(states["pos"], dtype=np.float32)
    except Exception:
        pass

    for j, i in enumerate(arm_idx):
        if j < arm_q.shape[0]:
            targets[i] = float(arm_q[j])

    finger_pos = float(max(gripper_width, 0.0) * 0.5)
    for i in finger_idx[:2]:
        targets[i] = finger_pos

    dc.set_articulation_dof_position_targets(art, targets)


def _create_or_open_hdf5(path: Path):
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    f = h5py.File(str(path), "w")
    f.create_group("data")
    return f


def main() -> None:
    parser = argparse.ArgumentParser(description="Isaac Sim collector (mobile_franka)")
    parser.add_argument("--plan_bundle", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--headless", dest="headless", action="store_true", default=True)
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.add_argument("--enable_cameras", dest="enable_cameras", action="store_true", default=True)
    parser.add_argument("--disable_cameras", dest="enable_cameras", action="store_false")
    parser.add_argument("--strict", dest="strict", action="store_true", default=False)
    parser.add_argument("--no-strict", dest="strict", action="store_false")
    parser.add_argument("--render_warmup_frames", type=int, default=int(os.getenv("SAGE_RENDER_WARMUP_FRAMES", "100")))
    parser.add_argument("--sensor_check_frame", type=int, default=int(os.getenv("SAGE_SENSOR_CHECK_FRAME", "10")))
    parser.add_argument("--camera_retry_steps", type=int, default=int(os.getenv("SAGE_CAMERA_RETRY_STEPS", "90")))
    parser.add_argument("--sensor_min_rgb_std", type=float, default=float(os.getenv("SAGE_SENSOR_MIN_RGB_STD", "0.01")))
    parser.add_argument("--sensor_min_depth_std", type=float, default=float(os.getenv("SAGE_SENSOR_MIN_DEPTH_STD", "0.0001")))
    parser.add_argument("--min_valid_depth_px", type=int, default=int(os.getenv("SAGE_MIN_VALID_DEPTH_PX", "1024")))
    parser.add_argument("--dome_light_intensity", type=float, default=float(os.getenv("SAGE_DOME_LIGHT_INTENSITY", "3000")))
    parser.add_argument("--sun_light_intensity", type=float, default=float(os.getenv("SAGE_SUN_LIGHT_INTENSITY", "600")))
    parser.add_argument(
        "--sensor_failure_policy",
        choices=["auto", "fail", "warn"],
        default=os.getenv("SAGE_SENSOR_FAILURE_POLICY", "auto"),
    )
    args, unknown_args = parser.parse_known_args()
    sensor_failure_policy = _resolve_sensor_failure_policy(strict=bool(args.strict), requested=args.sensor_failure_policy)

    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    os.environ.setdefault("ACCEPT_EULA", "Y")
    os.environ.setdefault("PRIVACY_CONSENT", "Y")

    bundle_path = Path(args.plan_bundle)
    bundle = _load_json(bundle_path)
    _enforce_bundle_runtime_parity(bundle, args)

    run_id = str(bundle.get("run_id") or os.environ.get("SAGE_RUN_ID", "")).strip()
    if not run_id:
        msg = (
            f"{BUNDLE_RUNTIME_MISSING_RUN_ID_MARKER}: plan bundle missing run_id "
            "and SAGE_RUN_ID is unset"
        )
        if args.strict:
            raise RuntimeError(msg)
        run_id = f"non_strict_{int(time.time())}"
        _log(f"WARNING: {msg}; generated fallback run_id={run_id}")
    os.environ.setdefault("SAGE_RUN_ID", run_id)

    layout_id = str(bundle["layout_id"])
    results_dir = Path(bundle["results_dir"])
    layout_dir = results_dir / layout_id

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_h5_path = output_dir / "dataset.hdf5"
    h5_tmp_path = output_dir / "dataset.tmp.hdf5"
    h5 = None
    if final_h5_path.exists():
        final_h5_path.unlink()

    # Isaac Sim initialization MUST come before omni imports.
    # Prevent Kit from interpreting application-specific CLI args.
    if unknown_args:
        _log(f"Ignoring unknown CLI args for collector: {unknown_args}")
    sys.argv = [sys.argv[0]]
    from isaacsim.simulation_app import SimulationApp

    sim_app = SimulationApp({"headless": bool(args.headless)})
    try:
        import omni
        import omni.usd
        from pxr import UsdGeom

        _ensure_extension("omni.replicator.core")

        import omni.replicator.core as rep
        from omni.isaac.core import World
        from omni.isaac.core.utils.stage import add_reference_to_stage

        _configure_local_assets_root()

        # Camera config
        res_env = os.getenv("SAGE_CAPTURE_RES", "640,480")
        width, height = [int(x) for x in res_env.split(",")]

        # HDF5 output (atomic: write temp then rename).
        if h5_tmp_path.exists():
            h5_tmp_path.unlink()
        h5 = _create_or_open_hdf5(h5_tmp_path)

        runtime_provenance = _runtime_provenance(
            bp_root=Path(BP_DIR).resolve(),
            sensor_failure_policy=sensor_failure_policy,
            args=args,
            run_id=run_id,
            bundle_path=bundle_path,
            bundle_sha256=_file_sha256(bundle_path),
        )
        demo_metadata: Dict[str, Any] = {
            "run_id": run_id,
            "layout_id": layout_id,
            "num_demos": 0,
            "demos": [],
            "provenance": runtime_provenance,
        }
        step_decomp: Dict[str, Any] = {
            "run_id": run_id,
            "num_demos": 0,
            "phase_labels": [],
            "demos": [],
            "provenance": runtime_provenance,
        }

        cameras_manifest: Dict[str, Any] = {
            "run_id": run_id,
            "layout_id": layout_id,
            "resolution": [width, height],
            "modalities": ["rgb", "depth"] if args.enable_cameras else [],
            "cameras": [],
            "provenance": runtime_provenance,
        }

        # Cache OBJ->USD conversions per source_id.
        usd_cache = layout_dir / "usd_cache"
        usd_cache.mkdir(parents=True, exist_ok=True)

        def build_scene_from_room(room: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any], Dict[str, str]]:
            """
            Build /World scene: floor + objects.
            Returns: (robot_root_path, physics_report)
            """
            omni.usd.get_context().new_stage()
            stage = omni.usd.get_context().get_stage()
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

            # World root
            UsdGeom.Xform.Define(stage, "/World")

            # Default ground plane (stable collisions)
            objs_scope = UsdGeom.Scope.Define(stage, "/World/Objects")

            # Ensure scene has stable lighting for headless rendering.
            _add_default_lighting(
                stage,
                dome_intensity=float(args.dome_light_intensity),
                distant_intensity=float(args.sun_light_intensity),
            )

            physics_report: Dict[str, Any] = {"objects": []}
            source_to_prim: Dict[str, str] = {}

            gen_dir = layout_dir / "generation"
            for idx, obj in enumerate(room.get("objects", []) or []):
                source_id = obj.get("source_id", "")
                if not source_id:
                    continue
                obj_path = gen_dir / f"{source_id}.obj"
                if not obj_path.exists():
                    raise FileNotFoundError(f"Missing mesh: {obj_path}")

                usd_out = usd_cache / f"{source_id}.usd"
                if not usd_out.exists():
                    _log(f"Converting OBJ -> USD: {obj_path.name}")
                    try:
                        _run_async(_convert_mesh_to_usd(str(obj_path), str(usd_out), load_materials=True))
                    except Exception as exc:
                        # Material references are occasionally incomplete. Retry geometry-only conversion.
                        _log(f"WARNING: material-aware conversion failed for {obj_path.name}: {exc}")
                        _log(f"Retrying geometry-only OBJ -> USD: {obj_path.name}")
                        _run_async(_convert_mesh_to_usd(str(obj_path), str(usd_out), load_materials=False))

                prim_path = f"/World/Objects/obj_{idx:03d}"
                UsdGeom.Xform.Define(stage, prim_path)
                add_reference_to_stage(str(usd_out), prim_path + "/Mesh")
                source_to_prim[str(source_id)] = prim_path

                # Compute scale to match target dimensions.
                mins, maxs = _read_obj_bounds(obj_path)
                size = np.maximum(maxs - mins, 1e-6)
                dims = obj.get("dimensions", {}) or {}
                target = np.array(
                    [
                        float(dims.get("width", 1.0)),
                        float(dims.get("length", 1.0)),
                        float(dims.get("height", 1.0)),
                    ],
                    dtype=np.float64,
                )
                scale = (target / size).astype(np.float64)

                pos = obj.get("position", {}) or {}
                rot = obj.get("rotation", {}) or {}
                M = _xform_matrix(
                    translation=(float(pos.get("x", 0.0)), float(pos.get("y", 0.0)), float(pos.get("z", 0.0))),
                    yaw_rad=math.radians(float(rot.get("z", 0.0))),
                    scale=(float(scale[0]), float(scale[1]), float(scale[2])),
                )
                _apply_xform(stage, prim_path, M)

                # Apply SimReady Lite physics on the Xform prim.
                phys = _apply_simready_lite_physics(
                    stage,
                    prim_path,
                    category=str(obj.get("type", "")),
                    dims={
                        "width": float(dims.get("width", 0.0)),
                        "length": float(dims.get("length", 0.0)),
                        "height": float(dims.get("height", 0.0)),
                    },
                )
                physics_report["objects"].append({"prim_path": prim_path, "source_id": source_id, **phys})

            collision_validation = _validate_dynamic_collision_settings(stage, physics_report)
            physics_report["validation"] = collision_validation
            _log(
                "Collision validation: "
                f"dynamic_meshes={collision_validation['total_dynamic_meshes']} "
                f"coverage={collision_validation['collision_coverage']:.3f} "
                f"bad_dynamic_approx={collision_validation['bad_dynamic_approx_count']}"
            )
            collision_invalid = (
                collision_validation["bad_dynamic_approx_count"] > 0
                or collision_validation["missing_collision_count"] > 0
                or (
                    collision_validation["total_dynamic_meshes"] > 0
                    and collision_validation["collision_coverage"] < 0.999
                )
            )
            if args.strict and collision_invalid:
                raise RuntimeError(
                    "Dynamic collision validation failed: "
                    f"coverage={collision_validation['collision_coverage']:.3f}, "
                    f"missing={collision_validation['missing_collision_count']}, "
                    f"bad_dynamic_approx={collision_validation['bad_dynamic_approx_count']}"
                )

            # Robot
            robot_xform = "/World/Robot"
            robot_articulation = "/World/Robot/RidgebackFranka"
            UsdGeom.Xform.Define(stage, robot_xform)
            add_reference_to_stage(_resolve_ridgeback_franka_usd(), robot_articulation)

            return robot_xform, robot_articulation, physics_report, source_to_prim

        def find_end_effector_prim(stage, robot_articulation: str) -> Optional[str]:
            """
            Best-effort discovery of an end-effector prim under the robot.
            We use this for kinematic object carry (attach) during pick/place.
            """
            candidates: List[Tuple[int, str]] = []
            robot_prefix = robot_articulation.rstrip("/")
            for prim in stage.Traverse():
                p = str(prim.GetPath())
                if not p.startswith(robot_prefix):
                    continue
                name = prim.GetName().lower()
                score = 0
                if "panda_hand" in name or "hand" in name:
                    score += 5
                if "gripper" in name:
                    score += 4
                if "eef" in name or "ee" == name:
                    score += 4
                if "link7" in name:
                    score += 2
                if score > 0:
                    candidates.append((score, p))
            if not candidates:
                return None
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1]

        def setup_cameras(room: Dict[str, Any]) -> Dict[str, Any]:
            stage = omni.usd.get_context().get_stage()
            dims = room.get("dimensions", {}) or {}
            w = float(dims.get("width", 6.0))
            l = float(dims.get("length", 6.0))
            h = float(dims.get("height", 3.0))
            center = np.array([w * 0.5, l * 0.5, 0.8], dtype=np.float64)

            cam_scope = UsdGeom.Scope.Define(stage, "/World/Cameras")

            cam_a = "/World/Cameras/agentview"
            cam_b = "/World/Cameras/agentview_2"
            _define_camera(stage, cam_a, pos=np.array([w * 0.5, 0.5, h * 0.85], dtype=np.float64), look_at=center)
            _define_camera(stage, cam_b, pos=np.array([w * 0.5, l - 0.5, h * 0.85], dtype=np.float64), look_at=center)

            render_a = rep.create.render_product(cam_a, resolution=(width, height))
            render_b = rep.create.render_product(cam_b, resolution=(width, height))

            rgb_a = rep.AnnotatorRegistry.get_annotator("rgb")
            rgb_b = rep.AnnotatorRegistry.get_annotator("rgb")
            depth_name = None
            depth_a = None
            depth_b = None
            for candidate in ("distance_to_image_plane", "distance_to_camera"):
                try:
                    depth_a = rep.AnnotatorRegistry.get_annotator(candidate)
                    depth_b = rep.AnnotatorRegistry.get_annotator(candidate)
                    depth_name = candidate
                    break
                except Exception:
                    continue
            if depth_a is None or depth_b is None or depth_name is None:
                raise RuntimeError("Could not initialize depth annotator (distance_to_image_plane/camera)")
            _log(f"Using depth annotator: {depth_name}")

            rgb_a.attach([render_a])
            depth_a.attach([render_a])
            rgb_b.attach([render_b])
            depth_b.attach([render_b])

            cameras_manifest["cameras"] = [
                {"camera_id": "agentview", "prim_path": cam_a},
                {"camera_id": "agentview_2", "prim_path": cam_b},
            ]
            cameras_manifest["depth_annotator"] = depth_name
            return {
                "agentview": {"rgb": rgb_a, "depth": depth_a},
                "agentview_2": {"rgb": rgb_b, "depth": depth_b},
            }

        demos = bundle.get("demos", []) or []
        _log(
            f"collector start: run_id={run_id} layout_id={layout_id} demos={len(demos)} "
            f"headless={args.headless} cameras={args.enable_cameras} strict={args.strict} "
            f"warmup={args.render_warmup_frames} retry={args.camera_retry_steps} "
            f"rgb_std_min={args.sensor_min_rgb_std} depth_std_min={args.sensor_min_depth_std} "
            f"min_depth_px={args.min_valid_depth_px} "
            f"dome_light={args.dome_light_intensity} sun_light={args.sun_light_intensity} "
            f"sensor_policy={sensor_failure_policy} "
            f"enforce_bundle_strict={os.environ.get('SAGE_ENFORCE_BUNDLE_STRICT', '1')} "
            f"remote_assets={os.environ.get('SAGE_ALLOW_REMOTE_ISAAC_ASSETS', '0')}"
        )
        if args.strict and len(demos) < 1:
            raise RuntimeError("plan_bundle contains 0 demos")

        # Group demos by variant json to reuse scene builds.
        by_variant: Dict[str, List[Dict[str, Any]]] = {}
        for d in demos:
            by_variant.setdefault(str(d["variant_layout_json"]), []).append(d)

        demo_global_idx = 0
        for variant_json, demo_list in by_variant.items():
            variant_path = _resolve_variant_json_path(variant_json, layout_dir=layout_dir)
            layout_payload = _load_json(variant_path)
            room = _extract_room_dict(layout_payload, demo=demo_list[0], variant_path=variant_path)
            if args.strict and not (room.get("objects") or []):
                raise RuntimeError(
                    f"Resolved room has 0 objects in strict mode: {variant_path}. "
                    "This usually means variant JSON room extraction is wrong."
                )
            _log(f"Building scene for variant: {variant_path} (demos={len(demo_list)})")
            robot_xform, robot_articulation, physics_report, source_to_prim = build_scene_from_room(room)
            stage = omni.usd.get_context().get_stage()
            ee_prim_path = find_end_effector_prim(stage, robot_articulation)
            if args.strict and ee_prim_path is None:
                _log("WARNING: could not auto-detect end-effector prim; object carry will be disabled.")

            annotators = setup_cameras(room) if args.enable_cameras else {}

            # Add a simple ground plane (avoids Nucleus asset download).
            from pxr import UsdPhysics
            ground_xform = UsdGeom.Xform.Define(stage, "/World/GroundPlane")
            ground_plane = UsdGeom.Mesh.Define(stage, "/World/GroundPlane/Mesh")
            ground_plane.GetPointsAttr().Set(
                [(-50, -50, 0), (50, -50, 0), (50, 50, 0), (-50, 50, 0)]
            )
            ground_plane.GetFaceVertexCountsAttr().Set([4])
            ground_plane.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
            UsdPhysics.CollisionAPI.Apply(ground_plane.GetPrim())

            # Let the stage settle a few frames.
            world = World(stage_units_in_meters=1.0)
            world.reset()
            for _ in range(max(0, int(args.render_warmup_frames))):
                world.step(render=True)

            dc, art, dof_names = _init_dynamic_control(robot_articulation)

            for demo in demo_list:
                demo_idx = int(demo["demo_idx"])
                _log(f"Capturing demo {demo_idx} for variant {variant_path.name}")
                base = np.asarray(demo["trajectory_base"], dtype=np.float32)
                arm = np.asarray(demo["trajectory_arm"], dtype=np.float32)
                grip = np.asarray(demo["trajectory_gripper"], dtype=np.float32).reshape(-1)
                labels = list(demo.get("step_labels", []))
                pick_source_id = str(demo.get("pick_object", {}).get("source_id", ""))
                pick_prim_path = source_to_prim.get(pick_source_id) if pick_source_id else None
                carrying = False
                carry_R = None  # rotation+scale to preserve object size while carrying

                if args.strict and (base.shape[0] == 0 or arm.shape[0] == 0):
                    raise RuntimeError(f"Empty trajectories for demo {demo_idx}")
                if base.shape[0] != arm.shape[0]:
                    raise RuntimeError(f"Trajectory length mismatch for demo {demo_idx}: base={base.shape[0]} arm={arm.shape[0]}")

                T = int(base.shape[0])

                actions = np.concatenate([arm, base, grip.reshape(T, 1)], axis=1).astype(np.float32)
                states = actions.copy()
                rewards = np.zeros((T,), dtype=np.float32)
                dones = np.zeros((T,), dtype=np.bool_)
                dones[-1] = True

                # Create demo group + datasets early so we can stream-write large camera tensors.
                demo_name = f"demo_{demo_global_idx}"
                grp = h5["data"].create_group(demo_name)
                grp.create_dataset("states", data=states, compression="gzip", compression_opts=4)
                grp.create_dataset("actions", data=actions, compression="gzip", compression_opts=4)
                grp.create_dataset("rewards", data=rewards)
                grp.create_dataset("dones", data=dones)

                obs_grp = grp.create_group("obs")
                next_grp = grp.create_group("next_obs")

                # Proprioception obs (small; write in one shot)
                obs_grp.create_dataset("robot_joint_pos", data=arm.astype(np.float32), compression="gzip", compression_opts=4)
                obs_grp.create_dataset("base_pose", data=base.astype(np.float32), compression="gzip", compression_opts=4)
                obs_grp.create_dataset("gripper_width", data=grip.reshape(T, 1).astype(np.float32), compression="gzip", compression_opts=4)

                next_grp.create_dataset(
                    "robot_joint_pos",
                    data=np.concatenate([arm[1:], arm[-1:]], axis=0).astype(np.float32),
                    compression="gzip",
                    compression_opts=4,
                )
                next_grp.create_dataset(
                    "base_pose",
                    data=np.concatenate([base[1:], base[-1:]], axis=0).astype(np.float32),
                    compression="gzip",
                    compression_opts=4,
                )
                next_grp.create_dataset(
                    "gripper_width",
                    data=np.concatenate([grip[1:].reshape(-1, 1), grip[-1:].reshape(-1, 1)], axis=0).astype(np.float32),
                    compression="gzip",
                    compression_opts=4,
                )

                # Camera obs (large; stream per-frame)
                cam_dsets = {}
                cam_next_dsets = {}
                if args.enable_cameras:
                    cam_dsets["agentview_rgb"] = obs_grp.create_dataset(
                        "agentview_rgb",
                        shape=(T, height, width, 3),
                        dtype=np.uint8,
                        compression="gzip",
                        compression_opts=4,
                        chunks=(1, height, width, 3),
                    )
                    cam_dsets["agentview_depth"] = obs_grp.create_dataset(
                        "agentview_depth",
                        shape=(T, height, width),
                        dtype=np.float16,
                        compression="gzip",
                        compression_opts=4,
                        chunks=(1, height, width),
                    )
                    cam_dsets["agentview_2_rgb"] = obs_grp.create_dataset(
                        "agentview_2_rgb",
                        shape=(T, height, width, 3),
                        dtype=np.uint8,
                        compression="gzip",
                        compression_opts=4,
                        chunks=(1, height, width, 3),
                    )
                    cam_dsets["agentview_2_depth"] = obs_grp.create_dataset(
                        "agentview_2_depth",
                        shape=(T, height, width),
                        dtype=np.float16,
                        compression="gzip",
                        compression_opts=4,
                        chunks=(1, height, width),
                    )

                    cam_next_dsets["agentview_rgb"] = next_grp.create_dataset(
                        "agentview_rgb",
                        shape=(T, height, width, 3),
                        dtype=np.uint8,
                        compression="gzip",
                        compression_opts=4,
                        chunks=(1, height, width, 3),
                    )
                    cam_next_dsets["agentview_depth"] = next_grp.create_dataset(
                        "agentview_depth",
                        shape=(T, height, width),
                        dtype=np.float16,
                        compression="gzip",
                        compression_opts=4,
                        chunks=(1, height, width),
                    )
                    cam_next_dsets["agentview_2_rgb"] = next_grp.create_dataset(
                        "agentview_2_rgb",
                        shape=(T, height, width, 3),
                        dtype=np.uint8,
                        compression="gzip",
                        compression_opts=4,
                        chunks=(1, height, width, 3),
                    )
                    cam_next_dsets["agentview_2_depth"] = next_grp.create_dataset(
                        "agentview_2_depth",
                        shape=(T, height, width),
                        dtype=np.float16,
                        compression="gzip",
                        compression_opts=4,
                        chunks=(1, height, width),
                    )

                # Reset robot to first pose
                _set_robot_root_pose(stage, robot_xform, float(base[0, 0]), float(base[0, 1]), float(base[0, 2]))

                # Cache object rotation+scale for carry (world frame).
                if ee_prim_path and pick_prim_path:
                    try:
                        from pxr import Usd, UsdGeom

                        obj_prim = stage.GetPrimAtPath(pick_prim_path)
                        if obj_prim.IsValid():
                            obj_xf = UsdGeom.Xformable(obj_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                            M = np.array([[float(obj_xf[i][j]) for j in range(4)] for i in range(4)], dtype=np.float64)
                            carry_R = M[:3, :3].copy()
                    except Exception:
                        carry_R = None

                # Replay
                last_a_rgb = None
                last_a_d = None
                last_b_rgb = None
                last_b_d = None
                check_t = max(0, min(int(args.sensor_check_frame), T - 1))
                sensor_qc: Dict[str, Any] = {
                    "enabled": bool(args.enable_cameras),
                    "policy": sensor_failure_policy,
                    "check_frame": int(check_t) if args.enable_cameras else None,
                    "status": "not_checked" if args.enable_cameras else "disabled",
                    "failures": [],
                }
                for t in range(T):
                    _set_robot_root_pose(stage, robot_xform, float(base[t, 0]), float(base[t, 1]), float(base[t, 2]))
                    _apply_joint_targets(dc, art, dof_names, arm[t], float(grip[t]))
                    # Step once so robot joints update.
                    world.step(render=True)

                    # Kinematic carry: when gripper closes during grasp/lift/navigate, move pick object with EE.
                    if ee_prim_path and pick_prim_path:
                        label = labels[t] if t < len(labels) else ""
                        g = float(grip[t])
                        if (label in {"grasp", "lift", "navigate", "approach_place", "place", "retreat_place"}) and g <= 0.005:
                            carrying = True
                        if label == "place" and g >= 0.02:
                            carrying = False

                        if carrying:
                            try:
                                from pxr import Usd, UsdGeom, Gf

                                ee_prim = stage.GetPrimAtPath(ee_prim_path)
                                obj_prim = stage.GetPrimAtPath(pick_prim_path)
                                if ee_prim.IsValid() and obj_prim.IsValid():
                                    ee_xf = UsdGeom.Xformable(ee_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                                    M = np.array([[float(ee_xf[i][j]) for j in range(4)] for i in range(4)], dtype=np.float64)
                                    # Keep object size/orientation stable; only carry translation.
                                    if carry_R is not None:
                                        M[:3, :3] = carry_R
                                    # Small offset in +Z so it doesn't interpenetrate the gripper.
                                    M[2, 3] += 0.02
                                    _apply_xform(stage, pick_prim_path, M)
                                    # Step again so render products see updated object pose this frame.
                                    world.step(render=True)
                            except Exception:
                                pass

                    if args.enable_cameras:
                        a_rgb_np, a_d_np, a_d_raw, b_rgb_np, b_d_np, b_d_raw = _capture_camera_frames(
                            annotators=annotators,
                            world=world,
                            max_extra_steps=int(args.camera_retry_steps),
                            width=width,
                            height=height,
                            min_valid_depth_px=int(args.min_valid_depth_px),
                        )

                        cam_dsets["agentview_rgb"][t] = a_rgb_np
                        cam_dsets["agentview_depth"][t] = a_d_np
                        cam_dsets["agentview_2_rgb"][t] = b_rgb_np
                        cam_dsets["agentview_2_depth"][t] = b_d_np

                        # Write next_obs[t-1] = obs[t] as we go.
                        if t > 0:
                            cam_next_dsets["agentview_rgb"][t - 1] = a_rgb_np
                            cam_next_dsets["agentview_depth"][t - 1] = a_d_np
                            cam_next_dsets["agentview_2_rgb"][t - 1] = b_rgb_np
                            cam_next_dsets["agentview_2_depth"][t - 1] = b_d_np

                        last_a_rgb, last_a_d = a_rgb_np, a_d_np
                        last_b_rgb, last_b_d = b_rgb_np, b_d_np

                        if t == check_t:
                            qc = _evaluate_dual_camera_sensor_qc(
                                agentview_rgb=a_rgb_np,
                                agentview_depth_raw=a_d_raw,
                                agentview2_rgb=b_rgb_np,
                                agentview2_depth_raw=b_d_raw,
                                min_valid_depth_px=int(args.min_valid_depth_px),
                                min_rgb_std=float(args.sensor_min_rgb_std),
                                min_depth_std=float(args.sensor_min_depth_std),
                            )
                            sensor_qc.update(qc)
                            depth_failures = [
                                item
                                for item in qc["failures"]
                                if item.endswith("degenerate_depth_valid_px") or item.endswith("low_depth_variance")
                            ]
                            rgb_failures = [item for item in qc["failures"] if item.endswith("degenerate_rgb")]
                            if depth_failures:
                                msg = (
                                    f"Degenerate depth for demo {demo_idx} "
                                    f"(failures={','.join(depth_failures)})"
                                )
                                if sensor_failure_policy == "fail":
                                    raise RuntimeError(msg)
                                _log(f"WARNING: {msg} — continuing")
                            if rgb_failures:
                                msg = f"Degenerate RGB for demo {demo_idx} (failures={','.join(rgb_failures)})"
                                if sensor_failure_policy == "fail":
                                    raise RuntimeError(msg)
                                _log(f"WARNING: {msg} — continuing")

                if args.enable_cameras:
                    # Last next_obs repeats last obs.
                    cam_next_dsets["agentview_rgb"][T - 1] = last_a_rgb
                    cam_next_dsets["agentview_depth"][T - 1] = last_a_d
                    cam_next_dsets["agentview_2_rgb"][T - 1] = last_b_rgb
                    cam_next_dsets["agentview_2_depth"][T - 1] = last_b_d

                # attrs
                grp.attrs["layout_id"] = layout_id
                grp.attrs["variant_layout_json"] = str(variant_path)
                grp.attrs["pick_object_type"] = str(demo.get("pick_object", {}).get("type", ""))
                grp.attrs["place_surface_type"] = str(demo.get("place_surface", {}).get("type", ""))

                demo_metadata["demos"].append(
                    {
                        "demo_name": demo_name,
                        "demo_idx": demo_idx,
                        "num_steps": T,
                        "variant_layout_json": str(variant_path),
                        "sensor_qc": sensor_qc,
                    }
                )
                step_decomp["demos"].append(
                    {
                        "demo_name": demo_name,
                        "demo_idx": demo_idx,
                        "total_steps": T,
                        "phase_labels": sorted(set(labels)),
                    }
                )

                demo_global_idx += 1
                h5.flush()

        demo_metadata["num_demos"] = demo_global_idx
        step_decomp["num_demos"] = demo_global_idx
        step_decomp["phase_labels"] = [
            "approach_pick",
            "grasp",
            "lift",
            "navigate",
            "approach_place",
            "place",
            "retreat_place",
        ]

        # Write train/valid masks (robomimic convention)
        num_train = int(demo_global_idx * 0.8)
        mask = h5.create_group("mask")
        mask.create_dataset("train", data=np.arange(num_train, dtype=np.int64))
        mask.create_dataset("valid", data=np.arange(num_train, demo_global_idx, dtype=np.int64))

        h5.flush()
        h5.close()
        h5 = None
        os.replace(str(h5_tmp_path), str(final_h5_path))

        _write_json(output_dir / "demo_metadata.json", demo_metadata)
        _write_json(output_dir / "step_decomposition.json", step_decomp)
        _write_json(output_dir / "capture_manifest.json", cameras_manifest)

        _log(f"Wrote {demo_global_idx} demos to {final_h5_path}")

        if args.strict:
            expected = len(demos)
            if demo_global_idx < expected:
                raise RuntimeError(f"Only {demo_global_idx}/{expected} demos captured")

    except Exception as exc:
        _log(f"ERROR: Stage 7 collector failed: {exc}")
        _log(traceback.format_exc())
        raise
    finally:
        try:
            if "h5" in locals() and h5 is not None:
                h5.flush()
                h5.close()
        except Exception:
            pass
        try:
            tmp = output_dir / "dataset.tmp.hdf5"
            if tmp.exists() and not (output_dir / "dataset.hdf5").exists():
                tmp.unlink()
        except Exception:
            pass
        sim_app.close()


if __name__ == "__main__":
    main()
