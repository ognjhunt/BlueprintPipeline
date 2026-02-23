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
    min_depth_finite_ratio: float = 0.0,
    min_valid_depth_px: int,
    min_rgb_std: float,
    max_rgb_saturation_ratio: float = 1.0,
    min_depth_std: float,
    min_depth_range_m: float = 0.0,
) -> Dict[str, Any]:
    rgb_np = np.asarray(rgb_frame)
    depth_np = np.asarray(depth_raw, dtype=np.float32)
    rgb_std = float(rgb_np.std()) if rgb_np.size else 0.0
    rgb_max = float(rgb_np.max()) if rgb_np.size else 0.0
    rgb_sat_ratio = (
        float(np.mean(np.any(rgb_np >= 250, axis=-1)))
        if (rgb_np.size and rgb_np.ndim == 3 and rgb_np.shape[-1] >= 3)
        else 1.0
    )
    valid_depth_mask = np.logical_and(np.isfinite(depth_np), depth_np > 0.0)
    valid_depth_px = int(valid_depth_mask.sum())
    valid_fraction = float(valid_depth_px / max(1, depth_np.size))
    depth_values = depth_np[valid_depth_mask]
    depth_std = float(depth_values.std()) if valid_depth_px > 0 else 0.0
    depth_range_m = float(depth_values.max() - depth_values.min()) if valid_depth_px > 0 else 0.0

    failures: List[str] = []
    if rgb_std < float(min_rgb_std):
        failures.append("low_rgb_std")
    if rgb_sat_ratio > float(max_rgb_saturation_ratio):
        failures.append("rgb_saturation_exceeds_threshold")
    if valid_fraction < float(min_depth_finite_ratio):
        failures.append("degenerate_depth_finite_ratio")
    if valid_depth_px < int(min_valid_depth_px):
        failures.append("degenerate_depth_valid_px")
    if depth_std <= float(min_depth_std):
        failures.append("low_depth_variance")
    if depth_range_m <= float(min_depth_range_m):
        failures.append("low_depth_range")

    return {
        "rgb_std": rgb_std,
        "rgb_max": rgb_max,
        "rgb_saturation_ratio": rgb_sat_ratio,
        "valid_depth_px": valid_depth_px,
        "depth_finite_ratio": valid_fraction,
        "depth_std": depth_std,
        "depth_range_m": depth_range_m,
        "failures": failures,
    }


def _evaluate_dual_camera_sensor_qc(
    *,
    agentview_rgb: np.ndarray,
    agentview_depth_raw: np.ndarray,
    agentview2_rgb: np.ndarray,
    agentview2_depth_raw: np.ndarray,
    min_depth_finite_ratio: float = 0.0,
    min_valid_depth_px: int,
    min_rgb_std: float,
    max_rgb_saturation_ratio: float = 1.0,
    min_depth_std: float,
    min_depth_range_m: float = 0.0,
) -> Dict[str, Any]:
    qc_a = _camera_sensor_qc_metrics(
        rgb_frame=agentview_rgb,
        depth_raw=agentview_depth_raw,
        min_depth_finite_ratio=min_depth_finite_ratio,
        min_valid_depth_px=min_valid_depth_px,
        min_rgb_std=min_rgb_std,
        max_rgb_saturation_ratio=max_rgb_saturation_ratio,
        min_depth_std=min_depth_std,
        min_depth_range_m=min_depth_range_m,
    )
    qc_b = _camera_sensor_qc_metrics(
        rgb_frame=agentview2_rgb,
        depth_raw=agentview2_depth_raw,
        min_depth_finite_ratio=min_depth_finite_ratio,
        min_valid_depth_px=min_valid_depth_px,
        min_rgb_std=min_rgb_std,
        max_rgb_saturation_ratio=max_rgb_saturation_ratio,
        min_depth_std=min_depth_std,
        min_depth_range_m=min_depth_range_m,
    )

    failures: List[str] = []
    failures.extend([f"agentview:{item}" for item in qc_a["failures"]])
    failures.extend([f"agentview_2:{item}" for item in qc_b["failures"]])
    depth_finite_ratio = float(min(qc_a["depth_finite_ratio"], qc_b["depth_finite_ratio"]))
    rgb_std = float(min(qc_a["rgb_std"], qc_b["rgb_std"]))
    return {
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "depth_finite_ratio": depth_finite_ratio,
        "rgb_std": rgb_std,
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
        "SAGE_STRICT_SENSORS": os.environ.get("SAGE_STRICT_SENSORS", ""),
        "SAGE_RENDER_WARMUP_FRAMES": os.environ.get("SAGE_RENDER_WARMUP_FRAMES", ""),
        "SAGE_SENSOR_MIN_RGB_STD": os.environ.get("SAGE_SENSOR_MIN_RGB_STD", ""),
        "SAGE_SENSOR_MIN_DEPTH_STD": os.environ.get("SAGE_SENSOR_MIN_DEPTH_STD", ""),
        "SAGE_MIN_DEPTH_FINITE_RATIO": os.environ.get("SAGE_MIN_DEPTH_FINITE_RATIO", ""),
        "SAGE_MAX_RGB_SATURATION_RATIO": os.environ.get("SAGE_MAX_RGB_SATURATION_RATIO", ""),
        "SAGE_MIN_DEPTH_RANGE_M": os.environ.get("SAGE_MIN_DEPTH_RANGE_M", ""),
        "SAGE_MIN_VALID_DEPTH_PX": os.environ.get("SAGE_MIN_VALID_DEPTH_PX", ""),
        "SAGE_SENSOR_CHECK_FRAME": os.environ.get("SAGE_SENSOR_CHECK_FRAME", ""),
        "SAGE_ENFORCE_BUNDLE_STRICT": os.environ.get("SAGE_ENFORCE_BUNDLE_STRICT", ""),
        "SAGE_EXPORT_SCENE_USD": os.environ.get("SAGE_EXPORT_SCENE_USD", ""),
        "SAGE_EXPORT_DEMO_VIDEOS": os.environ.get("SAGE_EXPORT_DEMO_VIDEOS", ""),
        "SAGE_QUALITY_REPORT_PATH": os.environ.get("SAGE_QUALITY_REPORT_PATH", ""),
        "SAGE_CARRY_MODE": os.environ.get("SAGE_CARRY_MODE", ""),
        "SAGE_MIN_GRIPPER_CONTACT_FORCE": os.environ.get("SAGE_MIN_GRIPPER_CONTACT_FORCE", ""),
        "SAGE_GRIPPER_CLOSED_WIDTH_THRESHOLD": os.environ.get("SAGE_GRIPPER_CLOSED_WIDTH_THRESHOLD", ""),
        "SAGE_DOMAIN_RAND": os.environ.get("SAGE_DOMAIN_RAND", "0"),
        "SAGE_RUN_ID": os.environ.get("SAGE_RUN_ID", ""),
        "SAGE_REPLICATOR_FSD_MODE": os.environ.get("SAGE_REPLICATOR_FSD_MODE", ""),
        "SAGE_REPLICATOR_RT_SUBFRAMES": os.environ.get("SAGE_REPLICATOR_RT_SUBFRAMES", ""),
        "SAGE_REPLICATOR_PREWARM_STEPS": os.environ.get("SAGE_REPLICATOR_PREWARM_STEPS", ""),
        "SAGE_REPLICATOR_STEP_EACH_FRAME": os.environ.get("SAGE_REPLICATOR_STEP_EACH_FRAME", ""),
        "SAGE_ISAAC_RENDERER": os.environ.get("SAGE_ISAAC_RENDERER", ""),
        "SAGE_ISAAC_MULTI_GPU": os.environ.get("SAGE_ISAAC_MULTI_GPU", ""),
        "SAGE_ISAAC_MAX_GPU_COUNT": os.environ.get("SAGE_ISAAC_MAX_GPU_COUNT", ""),
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
            "strict_sensors": bool(getattr(args, "strict_sensors", False)),
            "render_warmup_frames": int(args.render_warmup_frames),
            "sensor_check_frame": int(args.sensor_check_frame),
            "sensor_min_rgb_std": float(args.sensor_min_rgb_std),
            "sensor_min_depth_std": float(args.sensor_min_depth_std),
            "min_depth_finite_ratio": float(getattr(args, "min_depth_finite_ratio", 0.98)),
            "max_rgb_saturation_ratio": float(getattr(args, "max_rgb_saturation_ratio", 0.85)),
            "min_depth_range_m": float(getattr(args, "min_depth_range_m", 0.05)),
            "min_valid_depth_px": int(args.min_valid_depth_px),
            "export_scene_usd": bool(getattr(args, "export_scene_usd", True)),
            "export_demo_videos": bool(getattr(args, "export_demo_videos", True)),
            "quality_report_path": str(getattr(args, "quality_report_path", "")),
            "camera_retry_steps": int(args.camera_retry_steps),
            "dome_light_intensity": float(args.dome_light_intensity),
            "sun_light_intensity": float(args.sun_light_intensity),
            "carry_mode": str(getattr(args, "carry_mode", "physics")),
            "min_gripper_contact_force": float(getattr(args, "min_gripper_contact_force", 0.5)),
            "gripper_closed_width_threshold": float(getattr(args, "gripper_closed_width_threshold", 0.01)),
            "domain_rand_enabled": _env_bool("SAGE_DOMAIN_RAND", default=False),
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


def _jitter_lighting_for_demo(
    stage,
    *,
    dome_base: float,
    sun_base: float,
    demo_idx: int,
    base_seed: int,
) -> Dict[str, Any]:
    """
    Randomize dome and sun lighting for visual diversity across demos.

    Dome intensity jittered +-30%, sun intensity +-40%, sun elevation +-15 deg.
    Uses a seeded RNG for reproducibility.
    """
    rng = np.random.RandomState(base_seed + demo_idx)
    dome_factor = 1.0 + rng.uniform(-0.30, 0.30)
    sun_factor = 1.0 + rng.uniform(-0.40, 0.40)
    dome_intensity = max(0.0, dome_base * dome_factor)
    sun_intensity = max(0.0, sun_base * sun_factor)

    from pxr import UsdLux

    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
    dome.CreateIntensityAttr().Set(float(dome_intensity))

    sun = UsdLux.DistantLight.Define(stage, "/World/Lights/Sun")
    sun.CreateIntensityAttr().Set(float(sun_intensity))

    # Jitter sun elevation +-15 degrees from the base -45 degree elevation.
    base_elev = -math.pi * 0.25
    elev_jitter = rng.uniform(-math.radians(15), math.radians(15))
    elev = base_elev + elev_jitter
    sun_xf = np.eye(4, dtype=np.float64)
    c = math.cos(elev)
    s = math.sin(elev)
    sun_xf[:3, :3] = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)
    _apply_xform(stage, "/World/Lights/Sun", sun_xf)

    return {
        "dome_intensity": float(dome_intensity),
        "dome_factor": float(dome_factor),
        "sun_intensity": float(sun_intensity),
        "sun_factor": float(sun_factor),
        "sun_elevation_deg": float(math.degrees(elev)),
    }


def _jitter_cameras_for_demo(
    stage,
    *,
    cam_a_base_pos: np.ndarray,
    cam_b_base_pos: np.ndarray,
    center_base: np.ndarray,
    demo_idx: int,
    base_seed: int,
) -> Dict[str, Any]:
    """
    Randomize camera positions for visual diversity across demos.

    XY jitter +-0.15m, Z jitter +-0.10m, look-at jitter +-0.10m.
    Uses a seeded RNG (offset from lighting seed) for reproducibility.
    """
    rng = np.random.RandomState(base_seed + demo_idx + 10000)

    xy_jitter_a = rng.uniform(-0.15, 0.15, size=2)
    z_jitter_a = rng.uniform(-0.10, 0.10)
    xy_jitter_b = rng.uniform(-0.15, 0.15, size=2)
    z_jitter_b = rng.uniform(-0.10, 0.10)
    look_jitter = rng.uniform(-0.10, 0.10, size=3)

    pos_a = cam_a_base_pos.copy()
    pos_a[:2] += xy_jitter_a
    pos_a[2] += z_jitter_a

    pos_b = cam_b_base_pos.copy()
    pos_b[:2] += xy_jitter_b
    pos_b[2] += z_jitter_b

    look_at = center_base.copy() + look_jitter

    _define_camera(stage, "/World/Cameras/agentview", pos=pos_a, look_at=look_at)
    _define_camera(stage, "/World/Cameras/agentview_2", pos=pos_b, look_at=look_at)

    return {
        "cam_a_offset": [float(xy_jitter_a[0]), float(xy_jitter_a[1]), float(z_jitter_a)],
        "cam_b_offset": [float(xy_jitter_b[0]), float(xy_jitter_b[1]), float(z_jitter_b)],
        "look_at_offset": [float(look_jitter[0]), float(look_jitter[1]), float(look_jitter[2])],
    }


def _capture_camera_frames(
    *,
    annotators: Dict[str, Dict[str, Any]],
    world: Any,
    max_extra_steps: int,
    width: int,
    height: int,
    min_valid_depth_px: int,
    prefer_replicator_step: bool = False,
    replicator_rt_subframes: int = 4,
    prime_before_read: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Capture one frame from both cameras, retrying a few simulation steps until
    shapes are valid, RGB is not fully black, and depth has enough valid pixels.
    """
    def _step_for_capture() -> str:
        if bool(prefer_replicator_step):
            try:
                import omni.replicator.core as rep

                rep.orchestrator.step(rt_subframes=max(1, int(replicator_rt_subframes)))
                return "replicator"
            except Exception:
                pass
        world.step(render=True)
        return "world"

    last_reason = "unknown"
    if bool(prime_before_read):
        _step_for_capture()
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
        step_mode = _step_for_capture()
        last_reason = f"{last_reason} step={step_mode}"

    raise RuntimeError(f"Camera capture did not become valid after retries: {last_reason}")


def _render_product_path(render_product: Any) -> str:
    if isinstance(render_product, str):
        return render_product
    for attr in ("path", "render_product_path", "prim_path"):
        value = getattr(render_product, attr, None)
        if isinstance(value, str) and value:
            return value
    text = str(render_product or "").strip()
    return text


def _read_render_product_resolution(stage: Any, render_product_path: str) -> Optional[Tuple[int, int]]:
    if not render_product_path:
        return None
    prim = stage.GetPrimAtPath(render_product_path)
    if not prim or not prim.IsValid():
        return None
    for attr_name in ("resolution", "renderProductResolution", "dataWindowNDC"):
        attr = prim.GetAttribute(attr_name)
        if attr is None or not attr.IsValid():
            continue
        value = attr.Get()
        if value is None:
            continue
        if hasattr(value, "__len__") and len(value) >= 2:
            try:
                return int(value[0]), int(value[1])
            except Exception:
                continue
    return None


def _enforce_render_resolution_contract(
    *,
    stage: Any,
    render_product_path: str,
    expected_width: int,
    expected_height: int,
    strict_sensors: bool,
) -> None:
    actual = _read_render_product_resolution(stage, render_product_path)
    if actual is None:
        msg = f"Could not read render product resolution for {render_product_path}"
        if strict_sensors:
            raise RuntimeError(msg)
        _log(f"WARNING: {msg}")
        return
    if actual == (int(expected_width), int(expected_height)):
        return
    msg = (
        f"Render resolution contract mismatch at {render_product_path}: "
        f"expected=({expected_width}, {expected_height}) actual={actual}"
    )
    if strict_sensors:
        raise RuntimeError(msg)
    _log(f"WARNING: {msg}")


def _prepare_stage7_output_dir(output_dir: Path) -> Dict[str, Any]:
    """
    Remove stale files from prior runs so strict contract checks cannot be
    satisfied by mixed-run leftovers.
    """
    removed: List[str] = []

    def _unlink(path: Path) -> None:
        try:
            if path.is_file() or path.is_symlink():
                path.unlink()
                removed.append(str(path))
        except Exception as exc:
            raise RuntimeError(f"Failed to remove stale artifact {path}: {exc}") from exc

    singleton_files = [
        output_dir / "dataset.hdf5",
        output_dir / "dataset.tmp.hdf5",
        output_dir / "demo_metadata.json",
        output_dir / "step_decomposition.json",
        output_dir / "capture_manifest.json",
        output_dir / "quality_report.json",
        output_dir / "artifact_manifest.json",
        output_dir / "embodiment_manifest.json",
    ]
    for path in singleton_files:
        if path.exists():
            _unlink(path)

    for scene_path in sorted(output_dir.glob("scene_*.usd")):
        _unlink(scene_path)

    videos_dir = output_dir / "videos"
    if videos_dir.exists() and videos_dir.is_dir():
        for mp4 in sorted(videos_dir.glob("demo_*.mp4")):
            _unlink(mp4)
        # Remove now-empty videos dir to avoid stale directory-only manifests.
        try:
            if not any(videos_dir.iterdir()):
                videos_dir.rmdir()
        except Exception:
            pass

    return {"removed_count": int(len(removed)), "removed_paths": removed}


def _resolve_dof_groups(dof_names: List[str]) -> Tuple[List[int], List[int]]:
    arm_idx = [
        i
        for i, n in enumerate(dof_names)
        if ("joint" in (n or "").lower()) and ("finger" not in (n or "").lower()) and ("wheel" not in (n or "").lower())
    ]
    if len(arm_idx) < 7:
        arm_idx = list(range(min(7, len(dof_names))))
    arm_idx = arm_idx[:7]

    finger_idx = [i for i, n in enumerate(dof_names) if "finger" in (n or "").lower()]
    if not finger_idx and len(dof_names) >= 2:
        finger_idx = [len(dof_names) - 2, len(dof_names) - 1]
    return arm_idx, finger_idx[:2]


def _extract_state_vector(states: Any, *keys: str) -> Optional[np.ndarray]:
    if states is None:
        return None
    if isinstance(states, dict):
        for key in keys:
            if key in states:
                return np.asarray(states[key], dtype=np.float32).reshape(-1)
    dtype_names = getattr(getattr(states, "dtype", None), "names", None)
    if dtype_names:
        for key in keys:
            if key in dtype_names:
                return np.asarray(states[key], dtype=np.float32).reshape(-1)
    return None


def _read_robot_dof_observation(
    dc: Any,
    art: Any,
    dof_names: List[str],
    arm_idx: List[int],
    finger_idx: List[int],
) -> Dict[str, Any]:
    from omni.isaac.dynamic_control import _dynamic_control

    n = len(dof_names)
    zeros = np.zeros((max(1, len(arm_idx)),), dtype=np.float32)
    try:
        state_flags = int(getattr(_dynamic_control, "STATE_ALL", 0))
        states = dc.get_articulation_dof_states(art, state_flags)
    except Exception:
        return {
            "arm_vel": zeros.copy(),
            "arm_effort": zeros.copy(),
            "gripper_force": 0.0,
            "source": "missing",
        }

    vel_full = _extract_state_vector(states, "vel", "velocity")
    eff_full = _extract_state_vector(states, "effort", "tau", "force")
    had_effort_signal = eff_full is not None
    if vel_full is None:
        vel_full = np.zeros((n,), dtype=np.float32)
    if eff_full is None:
        eff_full = np.zeros((n,), dtype=np.float32)

    def _slice_with_bounds(vec: np.ndarray, idxs: List[int], width: int) -> np.ndarray:
        out = np.zeros((max(1, width),), dtype=np.float32)
        for j, idx in enumerate(idxs[:width]):
            if 0 <= int(idx) < int(vec.shape[0]):
                out[j] = float(vec[int(idx)])
        return out

    arm_width = max(1, len(arm_idx))
    arm_vel = _slice_with_bounds(vel_full, arm_idx, arm_width)
    arm_eff = _slice_with_bounds(eff_full, arm_idx, arm_width)
    grip_eff = [abs(float(eff_full[i])) for i in finger_idx if 0 <= int(i) < int(eff_full.shape[0])]
    gripper_force = float(sum(grip_eff))
    source = "physx" if had_effort_signal else "missing_effort_signal"
    return {
        "arm_vel": arm_vel,
        "arm_effort": arm_eff,
        "gripper_force": gripper_force,
        "source": source,
    }


def _quat_wxyz_from_rot(R: np.ndarray) -> Tuple[float, float, float, float]:
    m = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = float(np.trace(m))
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    else:
        if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
    return float(w), float(x), float(y), float(z)


def _remove_prim_if_exists(stage: Any, prim_path: str) -> None:
    prim = stage.GetPrimAtPath(prim_path)
    if prim and prim.IsValid():
        stage.RemovePrim(prim_path)


def _create_fixed_carry_joint(
    *,
    stage: Any,
    joint_path: str,
    ee_prim_path: str,
    obj_prim_path: str,
    z_offset_m: float = 0.02,
) -> None:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    ee_prim = stage.GetPrimAtPath(ee_prim_path)
    obj_prim = stage.GetPrimAtPath(obj_prim_path)
    if not (ee_prim and ee_prim.IsValid() and obj_prim and obj_prim.IsValid()):
        raise RuntimeError(f"Cannot create carry joint; invalid prims ee={ee_prim_path} obj={obj_prim_path}")

    ee_xf = UsdGeom.Xformable(ee_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    obj_xf = UsdGeom.Xformable(obj_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    ee_m = np.array([[float(ee_xf[i][j]) for j in range(4)] for i in range(4)], dtype=np.float64)
    obj_m = np.array([[float(obj_xf[i][j]) for j in range(4)] for i in range(4)], dtype=np.float64)
    rel = np.linalg.inv(ee_m) @ obj_m
    rel[2, 3] += float(z_offset_m)
    qw, qx, qy, qz = _quat_wxyz_from_rot(rel[:3, :3])

    constraints_root = stage.GetPrimAtPath("/World/CarryConstraints")
    if not (constraints_root and constraints_root.IsValid()):
        UsdGeom.Scope.Define(stage, "/World/CarryConstraints")

    _remove_prim_if_exists(stage, joint_path)
    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(ee_prim_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(obj_prim_path)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(float(rel[0, 3]), float(rel[1, 3]), float(rel[2, 3])))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(float(qw), Gf.Vec3f(float(qx), float(qy), float(qz))))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))


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


def _add_reference_prim(stage, prim_path: str, asset_path: str) -> None:
    """
    Add a USD reference using low-level USD APIs.

    This avoids Kit command routing (`AddReference`) that has intermittently
    asserted in headless runs under `carb.assets`.
    """
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        prim = UsdGeom.Xform.Define(stage, prim_path).GetPrim()
    refs = prim.GetReferences()
    ok = refs.AddReference(str(asset_path))
    if not ok:
        raise RuntimeError(f"Failed to add USD reference: prim={prim_path} asset={asset_path}")


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
    arm_idx, finger_idx = _resolve_dof_groups(dof_names)

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
    for i in finger_idx:
        targets[i] = finger_pos

    dc.set_articulation_dof_position_targets(art, targets)


def _export_demo_videos(
    hdf5_path: Path,
    output_dir: Path,
    num_demos: int,
    fps: int = 10,
) -> Dict[str, Any]:
    """Extract per-demo MP4 videos from HDF5 RGB data (agentview camera)."""
    report: Dict[str, Any] = {
        "attempted_demos": int(num_demos),
        "exported_demos": 0,
        "videos": [],
        "errors": [],
    }
    try:
        import h5py
    except ImportError:
        _log("WARNING: h5py not available — skipping video export")
        report["errors"].append("missing_h5py")
        return report

    iio_v3 = None
    iio_v2 = None
    try:
        import imageio.v3 as iio_v3  # type: ignore[assignment]
    except Exception:
        iio_v3 = None
    try:
        import imageio as iio_v2  # type: ignore[assignment]
    except Exception:
        iio_v2 = None
    if iio_v3 is None and iio_v2 is None:
        _log("WARNING: imageio not available — skipping video export")
        report["errors"].append("missing_imageio")
        return report

    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    exported = 0

    def _write_mp4(path: Path, frames: np.ndarray) -> Optional[Exception]:
        last_exc: Optional[Exception] = None
        if iio_v3 is not None:
            try:
                iio_v3.imwrite(str(path), frames, fps=fps, codec="libx264")
                return None
            except Exception as exc:
                last_exc = exc
            try:
                iio_v3.imwrite(str(path), frames, fps=fps)
                return None
            except Exception as exc:
                last_exc = exc
        if iio_v2 is not None:
            frame_list = [frame for frame in frames]
            try:
                iio_v2.mimwrite(str(path), frame_list, fps=fps, codec="libx264")
                return None
            except Exception as exc:
                last_exc = exc
            try:
                iio_v2.mimwrite(str(path), frame_list, fps=fps)
                return None
            except Exception as exc:
                last_exc = exc
        return last_exc

    try:
        with h5py.File(str(hdf5_path), "r") as f:
            data = f["data"]
            for i in range(num_demos):
                demo_key = f"demo_{i}"
                if demo_key not in data:
                    continue
                demo = data[demo_key]
                if "obs" not in demo or "agentview_rgb" not in demo["obs"]:
                    continue

                rgb = demo["obs"]["agentview_rgb"][:]  # (T, H, W, 3)
                if rgb.size == 0 or rgb.shape[0] <= 0:
                    report["errors"].append(f"{demo_key}:empty_rgb_tensor")
                    continue
                if rgb.max() == 0:
                    _log(f"WARNING: demo {i} RGB is all-black, skipping video")
                    report["errors"].append(f"{demo_key}:all_black_rgb")
                    continue

                mp4_path = videos_dir / f"demo_{i}.mp4"
                write_err = _write_mp4(mp4_path, rgb)
                if write_err is not None:
                    report["errors"].append(f"{demo_key}:video_write_failed:{write_err}")
                    _log(f"WARNING: failed to export {demo_key} video: {write_err}")
                    continue
                if not mp4_path.exists() or mp4_path.stat().st_size <= 0:
                    report["errors"].append(f"{demo_key}:video_not_written")
                    continue
                exported += 1
                report["videos"].append(
                    {
                        "demo_name": demo_key,
                        "path": str(mp4_path),
                        "frames": int(rgb.shape[0]),
                        "size_bytes": int(mp4_path.stat().st_size),
                    }
                )

        _log(f"Exported {exported}/{num_demos} demo videos to {videos_dir}")
        report["exported_demos"] = int(exported)
    except Exception as e:
        _log(f"WARNING: Video export failed: {e}")
        report["errors"].append(f"video_export_failed:{e}")
    return report


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _validate_usd_loadable(path: Path) -> bool:
    try:
        from pxr import Usd

        stage = Usd.Stage.Open(str(path))
        return stage is not None
    except Exception:
        return False


def _artifact_entry(path: Path, *, root: Path) -> Dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
    }


def _write_root_scalar(group: Any, key: str, value: Any) -> None:
    if isinstance(value, str):
        group.attrs[key] = value
    elif isinstance(value, bool):
        group.create_dataset(key, data=np.asarray(bool(value), dtype=np.bool_))
    elif isinstance(value, int):
        group.create_dataset(key, data=np.asarray(int(value), dtype=np.int64))
    else:
        group.create_dataset(key, data=np.asarray(float(value), dtype=np.float64))


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
    parser.add_argument(
        "--strict-sensors",
        dest="strict_sensors",
        action="store_true",
        default=_env_bool("SAGE_STRICT_SENSORS", default=True),
    )
    parser.add_argument("--no-strict-sensors", dest="strict_sensors", action="store_false")
    parser.add_argument("--render_warmup_frames", type=int, default=int(os.getenv("SAGE_RENDER_WARMUP_FRAMES", "100")))
    parser.add_argument("--sensor_check_frame", type=int, default=int(os.getenv("SAGE_SENSOR_CHECK_FRAME", "10")))
    parser.add_argument("--camera_retry_steps", type=int, default=int(os.getenv("SAGE_CAMERA_RETRY_STEPS", "90")))
    parser.add_argument("--sensor_min_rgb_std", type=float, default=float(os.getenv("SAGE_SENSOR_MIN_RGB_STD", "5.0")))
    parser.add_argument("--min-rgb-std", dest="min_rgb_std_override", type=float, default=None)
    parser.add_argument("--sensor_min_depth_std", type=float, default=float(os.getenv("SAGE_SENSOR_MIN_DEPTH_STD", "0.0001")))
    parser.add_argument("--min-depth-finite-ratio", type=float, default=float(os.getenv("SAGE_MIN_DEPTH_FINITE_RATIO", "0.98")))
    parser.add_argument("--max-rgb-saturation-ratio", type=float, default=float(os.getenv("SAGE_MAX_RGB_SATURATION_RATIO", "0.85")))
    parser.add_argument("--min-depth-range-m", type=float, default=float(os.getenv("SAGE_MIN_DEPTH_RANGE_M", "0.05")))
    parser.add_argument("--min_valid_depth_px", type=int, default=int(os.getenv("SAGE_MIN_VALID_DEPTH_PX", "1024")))
    parser.add_argument(
        "--export-scene-usd",
        dest="export_scene_usd",
        action="store_true",
        default=_env_bool("SAGE_EXPORT_SCENE_USD", default=True),
    )
    parser.add_argument("--no-export-scene-usd", dest="export_scene_usd", action="store_false")
    parser.add_argument(
        "--export-demo-videos",
        dest="export_demo_videos",
        action="store_true",
        default=_env_bool("SAGE_EXPORT_DEMO_VIDEOS", default=True),
    )
    parser.add_argument("--no-export-demo-videos", dest="export_demo_videos", action="store_false")
    parser.add_argument("--quality-report-path", default=os.getenv("SAGE_QUALITY_REPORT_PATH", ""))
    parser.add_argument(
        "--carry-mode",
        choices=["physics", "kinematic", "auto"],
        default=os.getenv("SAGE_CARRY_MODE", "physics"),
    )
    parser.add_argument(
        "--min-gripper-contact-force",
        type=float,
        default=float(os.getenv("SAGE_MIN_GRIPPER_CONTACT_FORCE", "0.5")),
    )
    parser.add_argument(
        "--gripper-closed-width-threshold",
        type=float,
        default=float(os.getenv("SAGE_GRIPPER_CLOSED_WIDTH_THRESHOLD", "0.01")),
    )
    parser.add_argument("--dome_light_intensity", type=float, default=float(os.getenv("SAGE_DOME_LIGHT_INTENSITY", "3000")))
    parser.add_argument("--sun_light_intensity", type=float, default=float(os.getenv("SAGE_SUN_LIGHT_INTENSITY", "600")))
    parser.add_argument(
        "--sensor_failure_policy",
        choices=["auto", "fail", "warn"],
        default=os.getenv("SAGE_SENSOR_FAILURE_POLICY", "auto"),
    )
    args, unknown_args = parser.parse_known_args()
    if args.min_rgb_std_override is not None:
        args.sensor_min_rgb_std = float(args.min_rgb_std_override)
    sensor_failure_policy = _resolve_sensor_failure_policy(
        strict=bool(args.strict or args.strict_sensors),
        requested=args.sensor_failure_policy,
    )

    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    os.environ.setdefault("ACCEPT_EULA", "Y")
    os.environ.setdefault("PRIVACY_CONSENT", "Y")

    bundle_path = Path(args.plan_bundle)
    bundle = _load_json(bundle_path)
    _enforce_bundle_runtime_parity(bundle, args)

    # --- Embodiment config: try to import from trajectory_solver, fall back inline ---
    # Must run before Isaac Sim init so _robot_config_dict is always available.
    _ts_path = str(Path(BP_DIR) / "episode-generation-job")
    try:
        if _ts_path not in sys.path:
            sys.path.insert(0, _ts_path)
        from trajectory_solver import FRANKA_CONFIG, robot_config_to_dict as _rcd  # type: ignore
        _robot_config_dict = _rcd(
            FRANKA_CONFIG,
            urdf_path=str(Path(BP_DIR) / "tools/geniesim_adapter/robot_assets/robots/franka/panda.urdf"),
        )
        _log("Embodiment config loaded from trajectory_solver.FRANKA_CONFIG")
    except Exception as _import_err:
        _log(f"WARNING: Could not import trajectory_solver for embodiment metadata: {_import_err}")
        _robot_config_dict = {
            "robot_type": "franka",
            "num_joints": 7,
            "joint_names": [
                "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
                "panda_joint5", "panda_joint6", "panda_joint7",
            ],
            "joint_limits": {
                "lower": [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
                "upper": [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
            },
            "default_joint_positions": [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
            "gripper_joint_names": ["panda_finger_joint1", "panda_finger_joint2"],
            "gripper_limits": [0.0, 0.04],
            "urdf_path": "",
        }

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
    cleanup_report = _prepare_stage7_output_dir(output_dir)
    if int(cleanup_report.get("removed_count", 0)) > 0:
        _log(
            f"Cleared {cleanup_report['removed_count']} stale Stage 7 artifacts from {output_dir} "
            "before collection"
        )
    quality_report_path = Path(args.quality_report_path) if args.quality_report_path else (output_dir / "quality_report.json")
    if quality_report_path.exists() and quality_report_path.is_file():
        quality_report_path.unlink()
        cleanup_report["removed_count"] = int(cleanup_report.get("removed_count", 0)) + 1
        cleanup_report.setdefault("removed_paths", []).append(str(quality_report_path))
    artifact_manifest_path = output_dir / "artifact_manifest.json"
    final_h5_path = output_dir / "dataset.hdf5"
    h5_tmp_path = output_dir / "dataset.tmp.hdf5"
    h5 = None
    if final_h5_path.exists():
        final_h5_path.unlink()

    # Camera config (parsed before SimulationApp init so render resolution can be set in launch config).
    res_env = os.getenv("SAGE_CAPTURE_RES", "640,480")
    try:
        width, height = [int(x) for x in res_env.split(",")]
    except Exception:
        width, height = (640, 480)
        _log(f"WARNING: invalid SAGE_CAPTURE_RES='{res_env}', using {width}x{height}")

    # Isaac Sim initialization MUST come before omni imports.
    # Prevent Kit from interpreting application-specific CLI args.
    if unknown_args:
        _log(f"Ignoring unknown CLI args for collector: {unknown_args}")
    sys.argv = [sys.argv[0]]
    from isaacsim.simulation_app import SimulationApp

    sim_renderer = str(os.getenv("SAGE_ISAAC_RENDERER", "RaytracedLighting")).strip() or "RaytracedLighting"
    sim_multi_gpu = _env_bool("SAGE_ISAAC_MULTI_GPU", default=False)
    try:
        sim_max_gpu_count = max(1, int(os.getenv("SAGE_ISAAC_MAX_GPU_COUNT", "1")))
    except Exception:
        sim_max_gpu_count = 1
    sim_cfg = {
        "headless": bool(args.headless),
        "renderer": sim_renderer,
        "width": int(width),
        "height": int(height),
        "sync_loads": True,
        "multi_gpu": bool(sim_multi_gpu),
        "max_gpu_count": int(sim_max_gpu_count if sim_multi_gpu else 1),
    }
    _log(
        "SimulationApp launch config: "
        f"headless={sim_cfg['headless']} renderer={sim_cfg['renderer']} "
        f"res={sim_cfg['width']}x{sim_cfg['height']} multi_gpu={sim_cfg['multi_gpu']} "
        f"max_gpu_count={sim_cfg['max_gpu_count']}"
    )
    sim_app = SimulationApp(sim_cfg)
    try:
        import omni
        import omni.usd
        from pxr import UsdGeom

        _ensure_extension("omni.replicator.core")

        import omni.replicator.core as rep
        from omni.isaac.core import World

        try:
            rep.orchestrator.set_capture_on_play(False)
        except Exception as rep_cfg_err:
            _log(f"WARNING: failed to configure Replicator capture_on_play: {rep_cfg_err}")

        _configure_local_assets_root()

        # Headless RTX render quality: disable DLSS (causes upscale artifacts at
        # low resolution), use direct lighting to avoid noisy path-tracing, and
        # set the internal render resolution to at least match the capture
        # resolution so the RGB annotator returns clean frames.
        if args.headless:
            try:
                import carb
                settings = carb.settings.get_settings()
                # Replicator RGB pipelines can emit empty LdrColor host buffers in
                # headless mode when Fabric Scene Delegate is enabled.
                fsd_mode = str(os.getenv("SAGE_REPLICATOR_FSD_MODE", "off")).strip().lower()
                if fsd_mode in {"off", "0", "false", "disable", "disabled"}:
                    settings.set("/app/useFabricSceneDelegate", False)
                    _log("Replicator compatibility: /app/useFabricSceneDelegate=0")
                elif fsd_mode in {"on", "1", "true", "enable", "enabled"}:
                    settings.set("/app/useFabricSceneDelegate", True)
                    _log("Replicator compatibility: /app/useFabricSceneDelegate=1")
                # Disable DLSS — it requires minimum 300px and causes uniform-brightness
                # artifacts in headless mode.
                settings.set("/rtx/post/dlss/enabled", False)
                # Keep DLSS mode explicit for SDG pipelines where driver/runtime can
                # silently override post settings.
                try:
                    settings.set("/rtx/post/dlss/execMode", int(os.getenv("SAGE_DLSS_EXEC_MODE", "2")))
                except Exception:
                    settings.set("/rtx/post/dlss/execMode", 2)
                # Use higher sample count for less noisy output.
                settings.set("/rtx/pathtracing/spp", 64)
                settings.set("/rtx/pathtracing/totalSpp", 64)
                # Ensure internal render resolution matches capture resolution.
                settings.set("/rtx/renderResolution/width", width)
                settings.set("/rtx/renderResolution/height", height)
                # Enable tone mapping for proper LDR output.
                settings.set("/rtx/post/tonemap/enabled", True)
                _log(
                    "Headless RTX settings applied: "
                    f"DLSS=off, spp=64, res={width}x{height}, "
                    f"dlss_exec_mode={os.getenv('SAGE_DLSS_EXEC_MODE', '2')}"
                )
            except Exception as rtx_err:
                _log(f"WARNING: Could not apply headless RTX settings: {rtx_err}")

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
            "robot_type": _robot_config_dict.get("robot_type", "franka"),
            "platform": "ridgeback_franka",
            "arm_joint_names": _robot_config_dict.get("joint_names", []),
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

            # Validate and repair object Z-heights: objects placed on surfaces
            # (place_id != "floor") should have Z ≈ surface top, not Z ≈ 0.
            surface_heights: Dict[str, float] = {}
            all_objects = room.get("objects", []) or []
            for obj in all_objects:
                oid = obj.get("id", "")
                otype = (obj.get("type", "") or "").lower()
                dims = obj.get("dimensions", {}) or {}
                h = float(dims.get("height", 0.0))
                z = float((obj.get("position", {}) or {}).get("z", 0.0))
                if otype in ("table", "counter", "desk", "shelf", "dresser", "nightstand", "cabinet"):
                    surface_heights[oid] = z + h

            for obj in all_objects:
                place_id = obj.get("place_id", "") or ""
                if place_id and place_id != "floor" and place_id in surface_heights:
                    pos = obj.get("position", {}) or {}
                    obj_z = float(pos.get("z", 0.0))
                    surface_top = surface_heights[place_id]
                    if obj_z < surface_top * 0.5 and surface_top > 0.1:
                        _log(
                            f"WARNING: Z-repair: {obj.get('id','?')} ({obj.get('type','?')}) "
                            f"z={obj_z:.3f}m -> {surface_top:.3f}m (placed on {place_id})"
                        )
                        pos["z"] = surface_top

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
                _add_reference_prim(stage, prim_path + "/Mesh", str(usd_out))
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
            _add_reference_prim(stage, robot_articulation, _resolve_ridgeback_franka_usd())

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

        def setup_cameras(room: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, str], str]:
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
            render_a_path = _render_product_path(render_a)
            render_b_path = _render_product_path(render_b)
            _enforce_render_resolution_contract(
                stage=stage,
                render_product_path=render_a_path,
                expected_width=width,
                expected_height=height,
                strict_sensors=bool(args.strict_sensors),
            )
            _enforce_render_resolution_contract(
                stage=stage,
                render_product_path=render_b_path,
                expected_width=width,
                expected_height=height,
                strict_sensors=bool(args.strict_sensors),
            )

            rgb_a = rep.AnnotatorRegistry.get_annotator("rgb")
            rgb_b = rep.AnnotatorRegistry.get_annotator("rgb")
            depth_name = None
            depth_a = None
            depth_b = None
            # RTX-based depth annotators produce valid data only with a display-attached
            # GPU.  In headless mode the basic rasterized "depth" annotator (z-buffer) is
            # the most reliable fallback.  Try it last so RTX variants are preferred when
            # they actually work.
            for candidate in ("distance_to_image_plane", "distance_to_camera", "depth"):
                try:
                    depth_a = rep.AnnotatorRegistry.get_annotator(candidate)
                    depth_b = rep.AnnotatorRegistry.get_annotator(candidate)
                    depth_name = candidate
                    break
                except Exception:
                    continue
            if depth_a is None or depth_b is None or depth_name is None:
                raise RuntimeError("Could not initialize depth annotator (distance_to_image_plane/camera/depth)")
            _log(f"Using depth annotator: {depth_name}")

            rgb_a.attach([render_a])
            depth_a.attach([render_a])
            rgb_b.attach([render_b])
            depth_b.attach([render_b])

            # Camera intrinsics: computed from known USD camera parameters.
            # focal_length=35mm (set in _define_camera), sensor_width=20.955mm (Isaac Sim default).
            _focal_mm = 35.0
            _sensor_w_mm = 20.955
            _fx = (_focal_mm / _sensor_w_mm) * width
            _fy = _fx  # square pixels
            _cx = width / 2.0
            _cy = height / 2.0
            _intrinsics = {
                "fx": _fx, "fy": _fy, "cx": _cx, "cy": _cy,
                "focal_length_mm": _focal_mm, "sensor_width_mm": _sensor_w_mm,
                "width": width, "height": height,
            }
            cameras_manifest["cameras"] = [
                {
                    "camera_id": "agentview",
                    "prim_path": cam_a,
                    "position_world": [w * 0.5, 0.5, h * 0.85],
                    "look_at_world": center.tolist(),
                    "intrinsics": _intrinsics,
                },
                {
                    "camera_id": "agentview_2",
                    "prim_path": cam_b,
                    "position_world": [w * 0.5, l - 0.5, h * 0.85],
                    "look_at_world": center.tolist(),
                    "intrinsics": _intrinsics,
                },
            ]
            cameras_manifest["depth_annotator"] = depth_name
            return (
                {
                    "agentview": {"rgb": rgb_a, "depth": depth_a},
                    "agentview_2": {"rgb": rgb_b, "depth": depth_b},
                },
                {"agentview": render_a, "agentview_2": render_b},
                {"agentview": render_a_path, "agentview_2": render_b_path},
                str(depth_name),
            )

        demos = bundle.get("demos", []) or []
        _log(
            f"collector start: run_id={run_id} layout_id={layout_id} demos={len(demos)} "
            f"headless={args.headless} cameras={args.enable_cameras} strict={args.strict} "
            f"warmup={args.render_warmup_frames} retry={args.camera_retry_steps} "
            f"strict_sensors={args.strict_sensors} "
            f"rgb_std_min={args.sensor_min_rgb_std} depth_std_min={args.sensor_min_depth_std} "
            f"depth_finite_ratio_min={args.min_depth_finite_ratio} "
            f"depth_range_min_m={args.min_depth_range_m} "
            f"max_rgb_saturation_ratio={args.max_rgb_saturation_ratio} "
            f"min_depth_px={args.min_valid_depth_px} "
            f"export_scene_usd={args.export_scene_usd} export_demo_videos={args.export_demo_videos} "
            f"quality_report_path={quality_report_path} "
            f"dome_light={args.dome_light_intensity} sun_light={args.sun_light_intensity} "
            f"carry_mode={args.carry_mode} min_gripper_contact_force={args.min_gripper_contact_force} "
            f"gripper_closed_width_threshold={args.gripper_closed_width_threshold} "
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
        warmup_sensor_checks: List[Dict[str, Any]] = []
        expected_scene_names: List[str] = []
        # Will be populated by _init_dynamic_control() on first variant; used for embodiment HDF5.
        runtime_dof_names: List[str] = _robot_config_dict.get("joint_names", [])
        for variant_json, demo_list in by_variant.items():
            variant_path = _resolve_variant_json_path(variant_json, layout_dir=layout_dir)
            expected_scene_names.append(f"scene_{variant_path.stem}.usd")
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
            if args.strict and str(args.carry_mode) == "physics" and ee_prim_path is None:
                raise RuntimeError(
                    "carry_mode=physics requires a valid end-effector prim, but none was discovered"
                )

            annotators: Dict[str, Any] = {}
            render_products: Dict[str, Any] = {}
            depth_name = ""
            if args.enable_cameras:
                annotators, render_products, _render_paths, depth_name = setup_cameras(room)

            # Store base camera positions for domain randomization reset.
            _room_dims = room.get("dimensions", {}) or {}
            _rw = float(_room_dims.get("width", 6.0))
            _rl = float(_room_dims.get("length", 6.0))
            _rh = float(_room_dims.get("height", 3.0))
            _cam_a_base_pos = np.array([_rw * 0.5, 0.5, _rh * 0.85], dtype=np.float64)
            _cam_b_base_pos = np.array([_rw * 0.5, _rl - 0.5, _rh * 0.85], dtype=np.float64)
            _center_base = np.array([_rw * 0.5, _rl * 0.5, 0.8], dtype=np.float64)
            _domain_rand_enabled = _env_bool("SAGE_DOMAIN_RAND", default=False)
            # Use stable SHA-derived seed (Python's hash() is process-randomized).
            _domain_rand_seed = int(hashlib.sha256(str(run_id).encode("utf-8")).hexdigest()[:8], 16)

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
            replicator_rt_subframes = max(1, int(os.getenv("SAGE_REPLICATOR_RT_SUBFRAMES", "8")))
            replicator_step_each_frame = _env_bool("SAGE_REPLICATOR_STEP_EACH_FRAME", default=False)
            replicator_prewarm_steps = max(0, int(os.getenv("SAGE_REPLICATOR_PREWARM_STEPS", "4")))
            if args.enable_cameras and args.headless and replicator_prewarm_steps > 0:
                for _ in range(replicator_prewarm_steps):
                    try:
                        rep.orchestrator.step(rt_subframes=replicator_rt_subframes)
                    except Exception:
                        world.step(render=True)
                _log(
                    f"Replicator prewarm complete: steps={replicator_prewarm_steps} "
                    f"rt_subframes={replicator_rt_subframes}"
                )

            # Probe depth annotator after warmup — if the selected annotator produces
            # all-inf/NaN (common with RTX annotators in headless mode), fall back to
            # the basic rasterized "depth" (z-buffer) annotator automatically.
            if args.enable_cameras and depth_name != "depth":
                try:
                    world.step(render=True)
                    probe = _annotator_data(annotators["agentview"]["depth"], name="depth_probe")
                    probe_raw = np.asarray(probe, dtype=np.float32)
                    probe_valid = int(np.logical_and(np.isfinite(probe_raw), probe_raw > 0.0).sum())
                    _log(f"Depth probe ({depth_name}): {probe_valid}/{probe_raw.size} valid pixels")
                    if probe_valid < int(args.min_valid_depth_px):
                        _log(f"WARNING: {depth_name} produced {probe_valid} valid pixels in headless "
                             f"mode — falling back to rasterized 'depth' annotator")
                        try:
                            fallback_a = rep.AnnotatorRegistry.get_annotator("depth")
                            fallback_b = rep.AnnotatorRegistry.get_annotator("depth")
                            fallback_a.attach([render_products["agentview"]])
                            fallback_b.attach([render_products["agentview_2"]])
                            annotators["agentview"]["depth"] = fallback_a
                            annotators["agentview_2"]["depth"] = fallback_b
                            depth_name = "depth"
                            cameras_manifest["depth_annotator"] = "depth"
                            cameras_manifest["depth_fallback_reason"] = "headless_rtx_failure"
                            _log(f"Switched to rasterized 'depth' annotator successfully")
                        except Exception as fb_err:
                            msg = (
                                "Rasterized depth fallback also failed. "
                                "This runtime/GPU mode cannot provide valid depth capture "
                                f"(original={depth_name}, error={fb_err})"
                            )
                            if bool(args.strict_sensors) or sensor_failure_policy == "fail":
                                raise RuntimeError(msg)
                            _log(f"WARNING: {msg}")
                except Exception as probe_err:
                    msg = f"Depth probe failed: {probe_err}"
                    if bool(args.strict_sensors) or sensor_failure_policy == "fail":
                        raise RuntimeError(msg)
                    _log(f"WARNING: {msg}")

            if args.enable_cameras:
                try:
                    warm_a_rgb, _, warm_a_d_raw, warm_b_rgb, _, warm_b_d_raw = _capture_camera_frames(
                        annotators=annotators,
                        world=world,
                        max_extra_steps=int(args.camera_retry_steps),
                        width=width,
                        height=height,
                        min_valid_depth_px=int(args.min_valid_depth_px),
                        prefer_replicator_step=bool(args.headless),
                        replicator_rt_subframes=int(replicator_rt_subframes),
                        prime_before_read=True,
                    )
                except Exception as warmup_exc:
                    detail = str(warmup_exc)
                    if "LdrColorSDhost" in detail or "renderVar" in detail:
                        raise RuntimeError(f"RGB render var unavailable during warmup: {detail}")
                    raise RuntimeError(f"Camera warmup capture failed: {detail}")

                warmup_qc = _evaluate_dual_camera_sensor_qc(
                    agentview_rgb=warm_a_rgb,
                    agentview_depth_raw=warm_a_d_raw,
                    agentview2_rgb=warm_b_rgb,
                    agentview2_depth_raw=warm_b_d_raw,
                    min_depth_finite_ratio=float(args.min_depth_finite_ratio),
                    min_valid_depth_px=int(args.min_valid_depth_px),
                    min_rgb_std=float(args.sensor_min_rgb_std),
                    max_rgb_saturation_ratio=float(args.max_rgb_saturation_ratio),
                    min_depth_std=float(args.sensor_min_depth_std),
                    min_depth_range_m=float(args.min_depth_range_m),
                )
                warmup_qc["variant_layout_json"] = str(variant_path)
                warmup_qc["depth_annotator"] = str(depth_name)
                warmup_sensor_checks.append(warmup_qc)
                if warmup_qc["status"] != "pass":
                    msg = (
                        f"Warmup sensor QC failed for {variant_path.name}: "
                        f"{','.join(warmup_qc.get('failures', []))}"
                    )
                    if bool(args.strict_sensors) or sensor_failure_policy == "fail":
                        raise RuntimeError(msg)
                    _log(f"WARNING: {msg}")

            # Export the assembled USD scene to disk (before trajectory replay).
            scene_usd_path = output_dir / f"scene_{variant_path.stem}.usd"
            if bool(args.export_scene_usd):
                try:
                    stage.Export(str(scene_usd_path))
                    if not scene_usd_path.exists():
                        raise RuntimeError("scene USD export did not create a file")
                    if not _validate_usd_loadable(scene_usd_path):
                        raise RuntimeError(f"scene USD is not loadable: {scene_usd_path}")
                    _log(f"Exported assembled scene to {scene_usd_path}")
                except Exception as e:
                    msg = f"Failed to export scene USD: {e}"
                    if args.strict:
                        raise RuntimeError(msg)
                    _log(f"WARNING: {msg}")

            dc, art, dof_names = _init_dynamic_control(robot_articulation)
            runtime_dof_names = list(dof_names)  # persist across variants for embodiment metadata

            for demo in demo_list:
                demo_idx = int(demo["demo_idx"])
                _log(f"Capturing demo {demo_idx} for variant {variant_path.name}")

                # Per-demo domain randomization (lighting + camera jitter).
                lighting_jitter_info: Optional[Dict[str, Any]] = None
                camera_jitter_info: Optional[Dict[str, Any]] = None
                if _domain_rand_enabled:
                    lighting_jitter_info = _jitter_lighting_for_demo(
                        stage,
                        dome_base=float(args.dome_light_intensity),
                        sun_base=float(args.sun_light_intensity),
                        demo_idx=demo_idx,
                        base_seed=_domain_rand_seed,
                    )
                    _log(f"  lighting jitter: dome={lighting_jitter_info['dome_intensity']:.0f} sun={lighting_jitter_info['sun_intensity']:.0f} elev={lighting_jitter_info['sun_elevation_deg']:.1f}deg")
                    camera_jitter_info = _jitter_cameras_for_demo(
                        stage,
                        cam_a_base_pos=_cam_a_base_pos,
                        cam_b_base_pos=_cam_b_base_pos,
                        center_base=_center_base,
                        demo_idx=demo_idx,
                        base_seed=_domain_rand_seed,
                    )
                    _log(f"  camera jitter: a={camera_jitter_info['cam_a_offset']} b={camera_jitter_info['cam_b_offset']}")

                base = np.asarray(demo["trajectory_base"], dtype=np.float32)
                arm = np.asarray(demo["trajectory_arm"], dtype=np.float32)
                grip = np.asarray(demo["trajectory_gripper"], dtype=np.float32).reshape(-1)
                labels = list(demo.get("step_labels", []))
                pick_source_id = str(demo.get("pick_object", {}).get("source_id", ""))
                pick_prim_path = source_to_prim.get(pick_source_id) if pick_source_id else None
                carrying = False
                carry_mode_active = ""
                carry_modes_used: List[str] = []
                carry_joint_path = f"/World/CarryConstraints/demo_{demo_global_idx:04d}_joint"
                carry_R = None  # rotation+scale to preserve object size while carrying

                if args.strict and (base.shape[0] == 0 or arm.shape[0] == 0):
                    raise RuntimeError(f"Empty trajectories for demo {demo_idx}")
                if base.shape[0] != arm.shape[0]:
                    raise RuntimeError(f"Trajectory length mismatch for demo {demo_idx}: base={base.shape[0]} arm={arm.shape[0]}")

                T = int(base.shape[0])
                arm_idx, finger_idx = _resolve_dof_groups(dof_names)
                arm_width = int(arm.shape[1]) if (arm.ndim == 2 and int(arm.shape[1]) > 0) else max(1, len(arm_idx))
                arm_vel_hist = np.zeros((T, arm_width), dtype=np.float32)
                arm_eff_hist = np.zeros((T, arm_width), dtype=np.float32)
                gripper_force_hist = np.zeros((T, 1), dtype=np.float32)
                gripper_contact_hist = np.zeros((T, 1), dtype=np.float32)
                dynamics_source_counts: Dict[str, int] = {}

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
                sensor_frame_failures = 0
                check_t = max(0, min(int(args.sensor_check_frame), T - 1))
                sensor_qc: Dict[str, Any] = {
                    "enabled": bool(args.enable_cameras),
                    "policy": sensor_failure_policy,
                    "strict_sensors": bool(args.strict_sensors),
                    "check_frame": int(check_t) if args.enable_cameras else None,
                    "status": "not_checked" if args.enable_cameras else "disabled",
                    "failures": [],
                    "thresholds": {
                        "min_depth_finite_ratio": float(args.min_depth_finite_ratio),
                        "min_valid_depth_px": int(args.min_valid_depth_px),
                        "min_rgb_std": float(args.sensor_min_rgb_std),
                        "max_rgb_saturation_ratio": float(args.max_rgb_saturation_ratio),
                        "min_depth_std": float(args.sensor_min_depth_std),
                        "min_depth_range_m": float(args.min_depth_range_m),
                    },
                }
                frame_qc_records: List[Dict[str, Any]] = []
                for t in range(T):
                    _set_robot_root_pose(stage, robot_xform, float(base[t, 0]), float(base[t, 1]), float(base[t, 2]))
                    _apply_joint_targets(dc, art, dof_names, arm[t], float(grip[t]))
                    # Step once so robot joints update.
                    world.step(render=True)

                    # Carry logic for object transport:
                    # - physics: attach a fixed joint between EE and object
                    # - kinematic: legacy _apply_xform fallback
                    if ee_prim_path and pick_prim_path:
                        label = labels[t] if t < len(labels) else ""
                        g = float(grip[t])
                        close_thresh = float(args.gripper_closed_width_threshold)
                        carry_labels = {"grasp", "lift", "navigate", "approach_place", "place", "retreat_place"}
                        should_grasp = (label in carry_labels) and (g <= close_thresh)
                        should_release = (label == "place") and (g >= max(0.02, close_thresh * 2.0))

                        if should_grasp and not carrying:
                            requested_mode = str(args.carry_mode)
                            if requested_mode in {"physics", "auto"}:
                                try:
                                    _create_fixed_carry_joint(
                                        stage=stage,
                                        joint_path=carry_joint_path,
                                        ee_prim_path=ee_prim_path,
                                        obj_prim_path=pick_prim_path,
                                        z_offset_m=0.02,
                                    )
                                    carrying = True
                                    carry_mode_active = "physics_joint"
                                except Exception as carry_exc:
                                    msg = (
                                        f"Failed to enable physics carry joint for demo={demo_idx} frame={t}: {carry_exc}"
                                    )
                                    if requested_mode == "physics":
                                        raise RuntimeError(msg)
                                    _log(f"WARNING: {msg}; falling back to kinematic carry")
                                    carrying = True
                                    carry_mode_active = "kinematic"
                            else:
                                carrying = True
                                carry_mode_active = "kinematic"
                            if carry_mode_active and carry_mode_active not in carry_modes_used:
                                carry_modes_used.append(carry_mode_active)

                        if should_release and carrying:
                            if carry_mode_active == "physics_joint":
                                _remove_prim_if_exists(stage, carry_joint_path)
                            carrying = False
                            carry_mode_active = ""

                        if carrying and carry_mode_active == "kinematic":
                            try:
                                from pxr import Usd, UsdGeom

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
                            except Exception as carry_exc:
                                if args.strict and str(args.carry_mode) in {"kinematic", "auto"}:
                                    raise RuntimeError(
                                        f"Kinematic carry failed for demo={demo_idx} frame={t}: {carry_exc}"
                                    )
                                _log(f"WARNING: kinematic carry failed for demo={demo_idx} frame={t}: {carry_exc}")

                    dyn_obs = _read_robot_dof_observation(dc, art, dof_names, arm_idx, finger_idx)
                    dyn_source = str(dyn_obs.get("source", "unknown"))
                    dynamics_source_counts[dyn_source] = int(dynamics_source_counts.get(dyn_source, 0)) + 1
                    arm_vel = np.asarray(dyn_obs.get("arm_vel", []), dtype=np.float32).reshape(-1)
                    arm_eff = np.asarray(dyn_obs.get("arm_effort", []), dtype=np.float32).reshape(-1)
                    if arm_vel.size > 0:
                        w = min(int(arm_vel.size), int(arm_width))
                        arm_vel_hist[t, :w] = arm_vel[:w]
                    if arm_eff.size > 0:
                        w = min(int(arm_eff.size), int(arm_width))
                        arm_eff_hist[t, :w] = arm_eff[:w]
                    grip_force = float(dyn_obs.get("gripper_force", 0.0))
                    gripper_force_hist[t, 0] = grip_force
                    contact_from_force = (
                        grip_force >= float(args.min_gripper_contact_force)
                        and float(grip[t]) <= float(args.gripper_closed_width_threshold)
                    )
                    contact_from_carry = bool(carrying)
                    gripper_contact_hist[t, 0] = 1.0 if (contact_from_force or contact_from_carry) else 0.0

                    if args.enable_cameras:
                        frame_capture_repeated = False
                        frame_capture_error = ""
                        try:
                            a_rgb_np, a_d_np, a_d_raw, b_rgb_np, b_d_np, b_d_raw = _capture_camera_frames(
                                annotators=annotators,
                                world=world,
                                max_extra_steps=int(args.camera_retry_steps),
                                width=width,
                                height=height,
                                min_valid_depth_px=int(args.min_valid_depth_px),
                                prefer_replicator_step=bool(args.headless and replicator_step_each_frame),
                                replicator_rt_subframes=int(replicator_rt_subframes),
                            )
                        except RuntimeError as sensor_exc:
                            sensor_frame_failures += 1
                            frame_capture_repeated = True
                            frame_capture_error = str(sensor_exc)
                            _log(
                                f"WARNING: sensor capture failed demo={demo_idx} frame={t}: "
                                f"{sensor_exc}; repeating previous frame "
                                f"(failures={sensor_frame_failures}/{T})"
                            )
                            if last_a_rgb is not None:
                                a_rgb_np, a_d_np = last_a_rgb, last_a_d
                                b_rgb_np, b_d_np = last_b_rgb, last_b_d
                            else:
                                a_rgb_np = np.zeros((height, width, 3), dtype=np.uint8)
                                a_d_np = np.zeros((height, width), dtype=np.float32)
                                b_rgb_np = np.zeros((height, width, 3), dtype=np.uint8)
                                b_d_np = np.zeros((height, width), dtype=np.float32)
                            a_d_raw = np.zeros((height, width), dtype=np.float32)
                            b_d_raw = np.zeros((height, width), dtype=np.float32)

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

                        qc = _evaluate_dual_camera_sensor_qc(
                            agentview_rgb=a_rgb_np,
                            agentview_depth_raw=a_d_raw,
                            agentview2_rgb=b_rgb_np,
                            agentview2_depth_raw=b_d_raw,
                            min_depth_finite_ratio=float(args.min_depth_finite_ratio),
                            min_valid_depth_px=int(args.min_valid_depth_px),
                            min_rgb_std=float(args.sensor_min_rgb_std),
                            max_rgb_saturation_ratio=float(args.max_rgb_saturation_ratio),
                            min_depth_std=float(args.sensor_min_depth_std),
                            min_depth_range_m=float(args.min_depth_range_m),
                        )
                        qc["frame_idx"] = int(t)
                        qc["capture_repeated"] = bool(frame_capture_repeated)
                        if frame_capture_error:
                            qc["capture_error"] = frame_capture_error
                        frame_qc_records.append(qc)
                        if t == check_t:
                            sensor_qc["check_frame_qc"] = qc

                        if qc["status"] != "pass":
                            reason = f"frame={t} failures={','.join(qc.get('failures', []))}"
                            if bool(frame_capture_repeated):
                                sensor_qc.setdefault("warnings", []).append(
                                    f"frame={t} repeated_capture:{frame_capture_error or 'capture_failed'}"
                                )
                            else:
                                sensor_qc["failures"].append(reason)
                                msg = f"Sensor QC failed for demo {demo_idx}: {reason}"
                                if bool(args.strict_sensors) or sensor_failure_policy == "fail":
                                    raise RuntimeError(msg)
                                _log(f"WARNING: {msg}")

                if carry_mode_active == "physics_joint":
                    _remove_prim_if_exists(stage, carry_joint_path)
                    carry_mode_active = ""
                    carrying = False

                # Check sensor frame failure ratio for this demo.
                if sensor_frame_failures > 0:
                    failure_ratio = sensor_frame_failures / max(1, T)
                    if failure_ratio > 0.5:
                        raise RuntimeError(
                            f"Demo {demo_idx} unsalvageable: {sensor_frame_failures}/{T} "
                            f"({failure_ratio:.0%}) sensor frames failed"
                        )
                    _log(
                        f"Demo {demo_idx}: {sensor_frame_failures}/{T} "
                        f"({failure_ratio:.1%}) sensor frames repeated from previous frame"
                    )

                if args.enable_cameras:
                    # Last next_obs repeats last obs.
                    cam_next_dsets["agentview_rgb"][T - 1] = last_a_rgb
                    cam_next_dsets["agentview_depth"][T - 1] = last_a_d
                    cam_next_dsets["agentview_2_rgb"][T - 1] = last_b_rgb
                    cam_next_dsets["agentview_2_depth"][T - 1] = last_b_d
                    if frame_qc_records:
                        depth_ratios = [float(item.get("depth_finite_ratio", 0.0)) for item in frame_qc_records]
                        rgb_stds = [float(item.get("rgb_std", 0.0)) for item in frame_qc_records]
                        sat_ratios = [
                            float(
                                max(
                                    item.get("agentview", {}).get("rgb_saturation_ratio", 0.0),
                                    item.get("agentview_2", {}).get("rgb_saturation_ratio", 0.0),
                                )
                            )
                            for item in frame_qc_records
                        ]
                        depth_ranges = [
                            float(
                                min(
                                    item.get("agentview", {}).get("depth_range_m", 0.0),
                                    item.get("agentview_2", {}).get("depth_range_m", 0.0),
                                )
                            )
                            for item in frame_qc_records
                        ]
                        sensor_qc.update(
                            {
                                "status": "pass" if not sensor_qc["failures"] else "fail",
                                "frames_checked": int(len(frame_qc_records)),
                                "repeated_frames": int(sensor_frame_failures),
                                "depth_finite_ratio_min": float(min(depth_ratios)),
                                "depth_finite_ratio_mean": float(np.mean(depth_ratios)),
                                "rgb_std_min": float(min(rgb_stds)),
                                "rgb_std_mean": float(np.mean(rgb_stds)),
                                "rgb_saturation_ratio_max": float(max(sat_ratios)),
                                "depth_range_m_min": float(min(depth_ranges)),
                            }
                        )

                # Additional dynamics/contact observations (non-breaking extra keys).
                obs_grp.create_dataset("robot_joint_vel", data=arm_vel_hist.astype(np.float32), compression="gzip", compression_opts=4)
                obs_grp.create_dataset("robot_joint_effort", data=arm_eff_hist.astype(np.float32), compression="gzip", compression_opts=4)
                obs_grp.create_dataset("gripper_force", data=gripper_force_hist.astype(np.float32), compression="gzip", compression_opts=4)
                obs_grp.create_dataset("gripper_contact", data=gripper_contact_hist.astype(np.float32), compression="gzip", compression_opts=4)

                next_grp.create_dataset(
                    "robot_joint_vel",
                    data=np.concatenate([arm_vel_hist[1:], arm_vel_hist[-1:]], axis=0).astype(np.float32),
                    compression="gzip",
                    compression_opts=4,
                )
                next_grp.create_dataset(
                    "robot_joint_effort",
                    data=np.concatenate([arm_eff_hist[1:], arm_eff_hist[-1:]], axis=0).astype(np.float32),
                    compression="gzip",
                    compression_opts=4,
                )
                next_grp.create_dataset(
                    "gripper_force",
                    data=np.concatenate([gripper_force_hist[1:], gripper_force_hist[-1:]], axis=0).astype(np.float32),
                    compression="gzip",
                    compression_opts=4,
                )
                next_grp.create_dataset(
                    "gripper_contact",
                    data=np.concatenate([gripper_contact_hist[1:], gripper_contact_hist[-1:]], axis=0).astype(np.float32),
                    compression="gzip",
                    compression_opts=4,
                )

                sensor_qc["carry_mode_requested"] = str(args.carry_mode)
                sensor_qc["carry_modes_used"] = carry_modes_used
                sensor_qc["dynamics"] = {
                    "joint_effort_abs_max": float(np.max(np.abs(arm_eff_hist))) if arm_eff_hist.size else 0.0,
                    "joint_effort_abs_mean": float(np.mean(np.abs(arm_eff_hist))) if arm_eff_hist.size else 0.0,
                    "gripper_force_max": float(np.max(gripper_force_hist)) if gripper_force_hist.size else 0.0,
                    "gripper_contact_ratio": float(np.mean(gripper_contact_hist)) if gripper_contact_hist.size else 0.0,
                    "sources": dynamics_source_counts,
                }

                # attrs
                grp.attrs["layout_id"] = layout_id
                grp.attrs["variant_layout_json"] = str(variant_path)
                grp.attrs["pick_object_type"] = str(demo.get("pick_object", {}).get("type", ""))
                grp.attrs["place_surface_type"] = str(demo.get("place_surface", {}).get("type", ""))

                demo_meta_entry: Dict[str, Any] = {
                    "demo_name": demo_name,
                    "demo_idx": demo_idx,
                    "num_steps": T,
                    "variant_layout_json": str(variant_path),
                    "sensor_qc": sensor_qc,
                }
                if lighting_jitter_info is not None:
                    demo_meta_entry["lighting_jitter"] = lighting_jitter_info
                if camera_jitter_info is not None:
                    demo_meta_entry["camera_jitter"] = camera_jitter_info
                demo_metadata["demos"].append(demo_meta_entry)
                step_decomp["demos"].append(
                    {
                        "demo_name": demo_name,
                        "demo_idx": demo_idx,
                        "total_steps": T,
                        "phase_labels": sorted(set(labels)),
                    }
                )

                # Reset lighting + cameras to base values after per-demo jitter
                # to prevent cumulative drift between demos.
                if _domain_rand_enabled:
                    _add_default_lighting(
                        stage,
                        dome_intensity=float(args.dome_light_intensity),
                        distant_intensity=float(args.sun_light_intensity),
                    )
                    _define_camera(stage, "/World/Cameras/agentview", pos=_cam_a_base_pos, look_at=_center_base)
                    _define_camera(stage, "/World/Cameras/agentview_2", pos=_cam_b_base_pos, look_at=_center_base)

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

        demo_qc_records = [
            d.get("sensor_qc", {}) for d in demo_metadata.get("demos", []) if isinstance(d.get("sensor_qc"), dict)
        ]
        depth_finite_values = [
            float(q.get("depth_finite_ratio_min", q.get("depth_finite_ratio", 0.0)))
            for q in demo_qc_records
            if q.get("enabled")
        ]
        rgb_std_values = [
            float(q.get("rgb_std_min", q.get("rgb_std", 0.0)))
            for q in demo_qc_records
            if q.get("enabled")
        ]
        gripper_contact_ratio_values = [
            float((q.get("dynamics") or {}).get("gripper_contact_ratio", 0.0))
            for q in demo_qc_records
            if isinstance(q, dict)
        ]
        joint_effort_abs_max_values = [
            float((q.get("dynamics") or {}).get("joint_effort_abs_max", 0.0))
            for q in demo_qc_records
            if isinstance(q, dict)
        ]
        failed_demos = [
            {
                "demo_idx": int(d.get("demo_idx", -1)),
                "demo_name": str(d.get("demo_name", "")),
                "failures": list((d.get("sensor_qc") or {}).get("failures", [])),
            }
            for d in demo_metadata.get("demos", [])
            if isinstance(d.get("sensor_qc"), dict) and (d["sensor_qc"].get("status") == "fail")
        ]
        quality_status = "disabled"
        if args.enable_cameras:
            quality_status = "pass" if not failed_demos else "fail"

        run_depth_finite_ratio = float(min(depth_finite_values)) if depth_finite_values else 0.0
        run_rgb_std = float(min(rgb_std_values)) if rgb_std_values else 0.0
        run_gripper_contact_ratio = float(np.mean(gripper_contact_ratio_values)) if gripper_contact_ratio_values else 0.0
        run_joint_effort_abs_max = float(max(joint_effort_abs_max_values)) if joint_effort_abs_max_values else 0.0

        metadata_grp = h5.create_group("metadata")
        prov_grp = metadata_grp.create_group("provenance")
        _write_root_scalar(prov_grp, "run_id", str(run_id))
        quality_grp = metadata_grp.create_group("quality")
        _write_root_scalar(quality_grp, "depth_finite_ratio", run_depth_finite_ratio)
        _write_root_scalar(quality_grp, "rgb_std", run_rgb_std)
        _write_root_scalar(quality_grp, "gripper_contact_ratio", run_gripper_contact_ratio)
        _write_root_scalar(quality_grp, "joint_effort_abs_max", run_joint_effort_abs_max)
        _write_root_scalar(quality_grp, "strict_sensors", bool(args.strict_sensors))

        # --- Embodiment identity: makes dataset self-describing for cross-embodiment training ---
        import h5py  # noqa: PLC0415 — local import mirrors pattern used elsewhere in this file
        emb_grp = metadata_grp.create_group("embodiment")
        emb_grp.attrs["robot_type"] = _robot_config_dict.get("robot_type", "franka")
        emb_grp.attrs["platform"] = "ridgeback_franka"
        emb_grp.attrs["urdf_path"] = _robot_config_dict.get("urdf_path", "")
        emb_grp.attrs["num_joints"] = _robot_config_dict.get("num_joints", 7)
        _vlen_str = h5py.special_dtype(vlen=str)
        emb_grp.create_dataset(
            "arm_joint_names",
            data=np.array(_robot_config_dict.get("joint_names", []), dtype=object),
            dtype=_vlen_str,
        )
        emb_grp.create_dataset(
            "runtime_dof_names",
            data=np.array(runtime_dof_names, dtype=object),
            dtype=_vlen_str,
        )
        emb_grp.create_dataset(
            "gripper_joint_names",
            data=np.array(_robot_config_dict.get("gripper_joint_names", []), dtype=object),
            dtype=_vlen_str,
        )
        _jl = _robot_config_dict.get("joint_limits", {})
        if _jl.get("lower"):
            emb_grp.create_dataset(
                "joint_limits_lower",
                data=np.array(_jl["lower"], dtype=np.float32),
            )
            emb_grp.create_dataset(
                "joint_limits_upper",
                data=np.array(_jl["upper"], dtype=np.float32),
            )
        _gl = _robot_config_dict.get("gripper_limits", [0.0, 0.04])
        emb_grp.attrs["gripper_limit_min"] = float(_gl[0])
        emb_grp.attrs["gripper_limit_max"] = float(_gl[1])

        h5.flush()
        h5.close()
        h5 = None
        os.replace(str(h5_tmp_path), str(final_h5_path))

        video_report: Dict[str, Any] = {"attempted_demos": 0, "exported_demos": 0, "videos": [], "errors": []}
        if args.enable_cameras and bool(args.export_demo_videos):
            video_report = _export_demo_videos(final_h5_path, output_dir, demo_global_idx)

        expected_scene_set = set(expected_scene_names if bool(args.export_scene_usd) else [])
        actual_scene_paths = sorted(output_dir.glob("scene_*.usd"))
        actual_scene_set = {p.name for p in actual_scene_paths}
        videos_dir = output_dir / "videos"
        video_files = sorted(videos_dir.glob("demo_*.mp4")) if videos_dir.exists() else []
        video_name_set = {p.name for p in video_files}
        expected_video_set = {f"demo_{i}.mp4" for i in range(int(demo_global_idx))}
        missing_videos = sorted(expected_video_set - video_name_set)
        unexpected_videos = sorted(video_name_set - expected_video_set)
        missing_scenes = sorted(expected_scene_set - actual_scene_set)
        unexpected_scenes = sorted(actual_scene_set - expected_scene_set) if expected_scene_set else []

        scene_paths_serialized = [str(p) for p in actual_scene_paths]
        artifact_contract_issues: List[str] = []
        if args.enable_cameras and bool(args.export_demo_videos) and bool(args.strict or args.strict_sensors):
            if missing_videos or unexpected_videos:
                artifact_contract_issues.append(
                    f"video_set_mismatch missing={missing_videos} unexpected={unexpected_videos}"
                )
            if len(video_files) != int(demo_global_idx):
                artifact_contract_issues.append(
                    f"video_count_mismatch expected={demo_global_idx} found={len(video_files)} in {videos_dir}"
                )
        if bool(args.export_scene_usd) and bool(args.strict):
            if missing_scenes or unexpected_scenes:
                artifact_contract_issues.append(
                    f"scene_set_mismatch missing={missing_scenes} unexpected={unexpected_scenes}"
                )
            if len(scene_paths_serialized) < 1:
                artifact_contract_issues.append("No scene_*.usd was exported in strict mode")
        if artifact_contract_issues and bool(args.strict or args.strict_sensors):
            raise RuntimeError("; ".join(artifact_contract_issues))

        quality_report: Dict[str, Any] = {
            "run_id": run_id,
            "layout_id": layout_id,
            "status": quality_status,
            "strict": bool(args.strict),
            "strict_sensors": bool(args.strict_sensors),
            "sensor_failure_policy": sensor_failure_policy,
            "quality_report_path": str(quality_report_path),
            "thresholds": {
                "min_depth_finite_ratio": float(args.min_depth_finite_ratio),
                "min_valid_depth_px": int(args.min_valid_depth_px),
                "min_rgb_std": float(args.sensor_min_rgb_std),
                "max_rgb_saturation_ratio": float(args.max_rgb_saturation_ratio),
                "min_depth_std": float(args.sensor_min_depth_std),
                "min_depth_range_m": float(args.min_depth_range_m),
                "min_gripper_contact_force": float(args.min_gripper_contact_force),
                "gripper_closed_width_threshold": float(args.gripper_closed_width_threshold),
            },
            "num_demos": int(demo_global_idx),
            "warmup_sensor_checks": warmup_sensor_checks,
            "demos": demo_metadata.get("demos", []),
            "failed_demos": failed_demos,
            "summary": {
                "depth_finite_ratio_min": run_depth_finite_ratio,
                "rgb_std_min": run_rgb_std,
                "gripper_contact_ratio_mean": run_gripper_contact_ratio,
                "joint_effort_abs_max": run_joint_effort_abs_max,
            },
            "stale_cleanup": cleanup_report,
            "artifact_contract": {
                "expected_video_names": sorted(expected_video_set),
                "actual_video_names": sorted(video_name_set),
                "missing_videos": missing_videos,
                "unexpected_videos": unexpected_videos,
                "expected_scene_names": sorted(expected_scene_set),
                "actual_scene_names": sorted(actual_scene_set),
                "missing_scenes": missing_scenes,
                "unexpected_scenes": unexpected_scenes,
            },
            "scene_exports": scene_paths_serialized,
            "video_report": video_report,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        demo_metadata_path = output_dir / "demo_metadata.json"
        step_decomp_path = output_dir / "step_decomposition.json"
        capture_manifest_path = output_dir / "capture_manifest.json"
        embodiment_manifest_path = output_dir / "embodiment_manifest.json"
        _write_json(demo_metadata_path, demo_metadata)
        _write_json(step_decomp_path, step_decomp)
        _write_json(capture_manifest_path, cameras_manifest)

        # --- Write embodiment_manifest.json ---
        _usd_path_str = ""
        try:
            _usd_path_str = str(_resolve_ridgeback_franka_usd())
        except Exception:
            pass
        embodiment_manifest: Dict[str, Any] = {
            "version": "1.0",
            "run_id": run_id,
            "layout_id": layout_id,
            "robot_type": _robot_config_dict.get("robot_type", "franka"),
            "platform": "ridgeback_franka",
            "arm_joint_names": _robot_config_dict.get("joint_names", []),
            "runtime_dof_names": runtime_dof_names,
            "joint_limits": _robot_config_dict.get("joint_limits", {}),
            "default_joint_positions": _robot_config_dict.get("default_joint_positions", []),
            "gripper_joint_names": _robot_config_dict.get("gripper_joint_names", []),
            "gripper_limits": _robot_config_dict.get("gripper_limits", [0.0, 0.04]),
            "num_joints": _robot_config_dict.get("num_joints", 7),
            "urdf_path": _robot_config_dict.get("urdf_path", ""),
            "usd_path": _usd_path_str,
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        _write_json(embodiment_manifest_path, embodiment_manifest)

        _write_json(quality_report_path, quality_report)

        artifact_manifest: Dict[str, Any] = {
            "run_id": run_id,
            "layout_id": layout_id,
            "status": "ok",
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "files": [],
            "counts": {
                "demos": int(demo_global_idx),
                "scene_usd": int(len(scene_paths_serialized)),
                "videos": int(len(video_files)),
            },
            "quality_report_path": str(quality_report_path),
            "stale_cleanup": cleanup_report,
        }
        core_paths = [
            final_h5_path,
            demo_metadata_path,
            step_decomp_path,
            capture_manifest_path,
            embodiment_manifest_path,
            quality_report_path,
        ]
        for path in core_paths:
            if path.exists():
                artifact_manifest["files"].append(_artifact_entry(path, root=output_dir))
        for scene_path_str in scene_paths_serialized:
            scene_path = Path(scene_path_str)
            artifact_manifest["files"].append(_artifact_entry(scene_path, root=output_dir))
        for mp4 in video_files:
            artifact_manifest["files"].append(_artifact_entry(mp4, root=output_dir))
        _write_json(artifact_manifest_path, artifact_manifest)

        _log(f"Wrote {demo_global_idx} demos to {final_h5_path}")

        if args.strict:
            expected = len(demos)
            if demo_global_idx < expected:
                raise RuntimeError(f"Only {demo_global_idx}/{expected} demos captured")
        if args.enable_cameras and bool(args.strict or args.strict_sensors) and failed_demos:
            raise RuntimeError(
                f"Sensor QC failed for {len(failed_demos)} demos (see {quality_report_path})"
            )

    except Exception as exc:
        _log(f"ERROR: Stage 7 collector failed: {exc}")
        _log(traceback.format_exc())
        try:
            fail_report = {
                "run_id": run_id if "run_id" in locals() else "",
                "layout_id": layout_id if "layout_id" in locals() else "",
                "status": "error",
                "error": str(exc),
                "stale_cleanup": cleanup_report if "cleanup_report" in locals() else {},
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            _write_json(quality_report_path, fail_report)
        except Exception:
            pass
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
