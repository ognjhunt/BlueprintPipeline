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
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

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
        return loop.run_until_complete(coro)

    # Loop is already running: schedule and poll.
    fut = asyncio.ensure_future(coro)
    import omni.kit.app

    app = omni.kit.app.get_app()
    while not fut.done():
        # Advance kit one frame.
        app.update()
        time.sleep(0.01)
    return fut.result()


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
    from pxr import Sdf, UsdPhysics, UsdShade

    try:
        from pxr import PhysxSchema  # type: ignore
        PHYSX = PhysxSchema
    except Exception:  # pragma: no cover
        PHYSX = None

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Invalid prim for physics: {prim_path}")

    # Collision is usually authored on geometry prims; try a common child path.
    collision_prim = stage.GetPrimAtPath(prim_path + "/Mesh")
    if collision_prim and collision_prim.IsValid():
        collision_target = collision_prim
    else:
        collision_target = prim

    preset = recommend_simready_lite(category=category, dimensions=dims)

    # Rigid body + mass
    rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(prim)
    rigid_body_api.CreateKinematicEnabledAttr().Set(not preset.is_dynamic)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr().Set(float(preset.mass_kg))

    # Collision + approximation
    collision_api = UsdPhysics.CollisionAPI.Apply(collision_target)
    collision_api.CreateCollisionEnabledAttr().Set(True)
    collision_target.CreateAttribute("physics:approximation", Sdf.ValueTypeNames.Token).Set(preset.collision_approximation)

    if PHYSX is not None and preset.collision_approximation == "convexDecomposition":
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

    UsdShade.MaterialBindingAPI(prim).Bind(material, UsdShade.Tokens.physics)

    return {
        "mass_kg": preset.mass_kg,
        "static_friction": preset.static_friction,
        "dynamic_friction": preset.dynamic_friction,
        "restitution": preset.restitution,
        "is_dynamic": preset.is_dynamic,
        "collision_approximation": preset.collision_approximation,
        "collision_max_hulls": preset.collision_max_hulls,
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
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--enable_cameras", action="store_true", default=True)
    parser.add_argument("--strict", action="store_true", default=True)
    args = parser.parse_args()

    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    os.environ.setdefault("ACCEPT_EULA", "Y")
    os.environ.setdefault("PRIVACY_CONSENT", "Y")

    bundle_path = Path(args.plan_bundle)
    bundle = _load_json(bundle_path)

    layout_id = str(bundle["layout_id"])
    results_dir = Path(bundle["results_dir"])
    layout_dir = results_dir / layout_id

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Isaac Sim initialization MUST come before omni imports.
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

        # Camera config
        res_env = os.getenv("SAGE_CAPTURE_RES", "640,480")
        width, height = [int(x) for x in res_env.split(",")]

        # HDF5 output
        h5_path = output_dir / "dataset.hdf5"
        h5 = _create_or_open_hdf5(h5_path)

        demo_metadata: Dict[str, Any] = {"layout_id": layout_id, "num_demos": 0, "demos": []}
        step_decomp: Dict[str, Any] = {"num_demos": 0, "phase_labels": [], "demos": []}

        cameras_manifest: Dict[str, Any] = {
            "layout_id": layout_id,
            "resolution": [width, height],
            "modalities": ["rgb", "depth"] if args.enable_cameras else [],
            "cameras": [],
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
                    _run_async(_convert_mesh_to_usd(str(obj_path), str(usd_out), load_materials=True))

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

            # Robot
            robot_xform = "/World/Robot"
            robot_articulation = "/World/Robot/RidgebackFranka"
            UsdGeom.Xform.Define(stage, robot_xform)
            add_reference_to_stage(
                "/Isaac/Robots/Clearpath/RidgebackFranka/ridgeback_franka.usd",
                robot_articulation,
            )

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
            depth_a = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
            rgb_b = rep.AnnotatorRegistry.get_annotator("rgb")
            depth_b = rep.AnnotatorRegistry.get_annotator("distance_to_camera")

            rgb_a.attach([render_a])
            depth_a.attach([render_a])
            rgb_b.attach([render_b])
            depth_b.attach([render_b])

            cameras_manifest["cameras"] = [
                {"camera_id": "agentview", "prim_path": cam_a},
                {"camera_id": "agentview_2", "prim_path": cam_b},
            ]
            return {
                "agentview": {"rgb": rgb_a, "depth": depth_a},
                "agentview_2": {"rgb": rgb_b, "depth": depth_b},
            }

        demos = bundle.get("demos", []) or []
        if args.strict and len(demos) < 1:
            raise RuntimeError("plan_bundle contains 0 demos")

        # Group demos by variant json to reuse scene builds.
        by_variant: Dict[str, List[Dict[str, Any]]] = {}
        for d in demos:
            by_variant.setdefault(str(d["variant_layout_json"]), []).append(d)

        demo_global_idx = 0
        for variant_json, demo_list in by_variant.items():
            room = _load_json(Path(variant_json))
            robot_xform, robot_articulation, physics_report, source_to_prim = build_scene_from_room(room)
            stage = omni.usd.get_context().get_stage()
            ee_prim_path = find_end_effector_prim(stage, robot_articulation)
            if args.strict and ee_prim_path is None:
                _log("WARNING: could not auto-detect end-effector prim; object carry will be disabled.")

            annotators = setup_cameras(room) if args.enable_cameras else {}

            # Let the stage settle a few frames.
            world = World(stage_units_in_meters=1.0)
            world.scene.add_default_ground_plane()
            world.reset()
            for _ in range(10):
                world.step(render=True)

            dc, art, dof_names = _init_dynamic_control(robot_articulation)

            for demo in demo_list:
                demo_idx = int(demo["demo_idx"])
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
                        a_rgb = annotators["agentview"]["rgb"].get_data()["data"]
                        a_d = annotators["agentview"]["depth"].get_data()["data"]
                        b_rgb = annotators["agentview_2"]["rgb"].get_data()["data"]
                        b_d = annotators["agentview_2"]["depth"].get_data()["data"]
                        a_rgb_np = np.asarray(a_rgb)[..., :3].astype(np.uint8)
                        a_d_np = np.asarray(a_d, dtype=np.float32).astype(np.float16)
                        b_rgb_np = np.asarray(b_rgb)[..., :3].astype(np.uint8)
                        b_d_np = np.asarray(b_d, dtype=np.float32).astype(np.float16)

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

                        # Strict sensor sanity on first frame only.
                        if args.strict and t == 0:
                            if float(a_rgb_np.std()) <= 1.0:
                                raise RuntimeError(f"Degenerate RGB for demo {demo_idx} (std={float(a_rgb_np.std()):.3f})")
                            finite = np.isfinite(a_d_np)
                            if not bool(finite.any()) or float(a_d_np[finite].std()) <= 1e-4:
                                raise RuntimeError("Degenerate depth for demo %d" % demo_idx)

                if args.enable_cameras:
                    # Last next_obs repeats last obs.
                    cam_next_dsets["agentview_rgb"][T - 1] = last_a_rgb
                    cam_next_dsets["agentview_depth"][T - 1] = last_a_d
                    cam_next_dsets["agentview_2_rgb"][T - 1] = last_b_rgb
                    cam_next_dsets["agentview_2_depth"][T - 1] = last_b_d

                # attrs
                grp.attrs["layout_id"] = layout_id
                grp.attrs["variant_layout_json"] = str(variant_json)
                grp.attrs["pick_object_type"] = str(demo.get("pick_object", {}).get("type", ""))
                grp.attrs["place_surface_type"] = str(demo.get("place_surface", {}).get("type", ""))

                demo_metadata["demos"].append(
                    {
                        "demo_name": demo_name,
                        "demo_idx": demo_idx,
                        "num_steps": T,
                        "variant_layout_json": str(variant_json),
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

        _write_json(output_dir / "demo_metadata.json", demo_metadata)
        _write_json(output_dir / "step_decomposition.json", step_decomp)
        _write_json(output_dir / "capture_manifest.json", cameras_manifest)

        _log(f"Wrote {demo_global_idx} demos to {h5_path}")

        if args.strict:
            expected = len(demos)
            if demo_global_idx < expected:
                raise RuntimeError(f"Only {demo_global_idx}/{expected} demos captured")

    finally:
        sim_app.close()


if __name__ == "__main__":
    main()
