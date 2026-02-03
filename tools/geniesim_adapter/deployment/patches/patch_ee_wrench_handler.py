#!/usr/bin/env python3
"""
Patch the Genie Sim server's grpc_server.py to add get_ee_wrench.

This RPC computes an EE wrench from PhysX contact reports and (optionally)
the robot's EE pose if available.

Environment:
  GENIESIM_EE_PRIM_PATH: Explicit USD prim path for the end-effector.
    If set, this path is used first when resolving the EE transform
    for torque computation.
"""
import os
import re
import sys
import textwrap

GENIESIM_ROOT = os.environ.get("GENIESIM_ROOT", "/opt/geniesim")
GRPC_SERVER = os.path.join(
    GENIESIM_ROOT,
    "source", "data_collection", "server", "grpc_server.py",
)

PATCH_MARKER = "BlueprintPipeline ee_wrench patch"

HANDLER = textwrap.dedent("""\

    # --- BEGIN BlueprintPipeline ee_wrench patch ---
    def get_ee_wrench(self, request, context):
        \"\"\"Return an EE wrench estimate from PhysX contacts.\"\"\"
        try:
            try:
                from aimdk.protocol import geniesim_grpc_pb2 as _pb2
            except Exception:
                try:
                    import geniesim_grpc_pb2 as _pb2
                except Exception:
                    from tools.geniesim_adapter import geniesim_grpc_pb2 as _pb2

            EEWrenchRsp = _pb2.EEWrenchRsp
            ContactPoint = _pb2.ContactPoint
        except Exception:
            return None

        include_contacts = getattr(request, "include_contacts", False)
        force = [0.0, 0.0, 0.0]
        torque = [0.0, 0.0, 0.0]
        contacts = []
        source = "physx_contact_report"

        # Try to resolve an EE transform (for torque computation and frame conversion)
        ee_position = None
        ee_rotation = None
        try:
            server_fn = getattr(self, "server_function", None)
            cmd = getattr(server_fn, "command_controller", None) or server_fn
            robot = getattr(cmd, "robot", None)
            def _ee_warn(msg):
                logger = getattr(self, "logger", None)
                if logger and hasattr(logger, "warning"):
                    logger.warning(msg)
                else:
                    try:
                        import carb
                        carb.log_warn(msg)
                    except Exception:
                        print(msg)
            # Allow explicit override via env var for deterministic EE frame resolution.
            # Example: export GENIESIM_EE_PRIM_PATH="/World/Robot/ee_link"
            ee_prim_env = os.getenv("GENIESIM_EE_PRIM_PATH", "").strip()
            ee_prim_path = ee_prim_env or None
            for attr in (
                "end_effector_prim_path",
                "ee_prim_path",
                "gripper_prim_path",
                "hand_prim_path",
                "end_effector_path",
            ):
                if ee_prim_path:
                    break
                ee_prim_path = getattr(cmd, attr, None) or getattr(robot, attr, None)
                if ee_prim_path:
                    break
            if ee_prim_path:
                try:
                    import omni.usd
                    from pxr import UsdGeom
                    stage = omni.usd.get_context().get_stage()
                    if stage:
                        prim = stage.GetPrimAtPath(ee_prim_path)
                        if prim and prim.IsValid():
                            xformable = UsdGeom.Xformable(prim)
                            matrix = xformable.ComputeLocalToWorldTransform(UsdGeom.TimeCode.Default())
                            ee_position = [matrix[3][0], matrix[3][1], matrix[3][2]]
                            ee_rotation = [
                                [matrix[0][0], matrix[0][1], matrix[0][2]],
                                [matrix[1][0], matrix[1][1], matrix[1][2]],
                                [matrix[2][0], matrix[2][1], matrix[2][2]],
                            ]
                        elif ee_prim_env:
                            _ee_warn(
                                "GENIESIM_EE_PRIM_PATH is set but prim is missing/invalid: %s"
                                % ee_prim_path
                            )
                    elif ee_prim_env:
                        _ee_warn(
                            "GENIESIM_EE_PRIM_PATH is set but USD stage is unavailable: %s"
                            % ee_prim_path
                        )
                except Exception as exc:
                    if ee_prim_env:
                        _ee_warn(
                            "GENIESIM_EE_PRIM_PATH resolution failed for %s: %s"
                            % (ee_prim_path, exc)
                        )
                    ee_position = ee_position
                    ee_rotation = ee_rotation

            if ee_position is None or ee_rotation is None:
                get_ee_pose = getattr(robot, "get_ee_pose", None)
                if callable(get_ee_pose):
                    ee_pose = get_ee_pose()
                    if isinstance(ee_pose, (list, tuple)) and len(ee_pose) >= 1:
                        ee_position = ee_pose[0]
                    if isinstance(ee_pose, (list, tuple)) and len(ee_pose) >= 2:
                        quat = ee_pose[1]
                        if quat is not None and len(quat) >= 4:
                            # Quaternion expected as [rw, rx, ry, rz]
                            w, x, y, z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
                            ee_rotation = [
                                [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
                                [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
                                [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y],
                            ]
        except Exception:
            ee_position = None
            ee_rotation = None

        try:
            force_world = [0.0, 0.0, 0.0]
            torque_world = [0.0, 0.0, 0.0]
            from omni.physx import get_physx_interface
            physx = get_physx_interface()
            contact_data = None
            try:
                from omni.physx import get_physx_simulation_interface
                sim = get_physx_simulation_interface()
                if sim is not None and hasattr(sim, "get_contact_report"):
                    contact_data = sim.get_contact_report()
            except Exception:
                contact_data = None

            if contact_data is None and physx is not None and hasattr(physx, "get_contact_report"):
                contact_data = physx.get_contact_report()

            if contact_data:
                for contact in contact_data:
                    if isinstance(contact, dict):
                        body_a = str(contact.get("actor0", ""))
                        body_b = str(contact.get("actor1", ""))
                        impulse = float(contact.get("impulse", 0.0))
                        position = contact.get("position", [0, 0, 0])
                        normal = contact.get("normal", [0, 0, 1])
                        force_vector = contact.get("force_vector") or contact.get("impulse_vector")
                        tangent_impulse = contact.get("tangent_impulse") or contact.get("tangent_impulse_vector")
                        friction = contact.get("friction") or contact.get("friction_coefficient")
                        contact_area = contact.get("contact_area")
                    else:
                        body_a = str(getattr(contact, "actor0", ""))
                        body_b = str(getattr(contact, "actor1", ""))
                        impulse = float(getattr(contact, "impulse", 0.0))
                        position = getattr(contact, "position", [0, 0, 0])
                        normal = getattr(contact, "normal", [0, 0, 1])
                        force_vector = getattr(contact, "force_vector", None) or getattr(contact, "impulse_vector", None)
                        tangent_impulse = getattr(contact, "tangent_impulse", None) or getattr(contact, "tangent_impulse_vector", None)
                        friction = getattr(contact, "friction", None) or getattr(contact, "friction_coefficient", None)
                        contact_area = getattr(contact, "contact_area", None)

                    # Filter to gripper/hand contacts
                    lower_a = body_a.lower()
                    lower_b = body_b.lower()
                    if not any(kw in lower_a or kw in lower_b for kw in ("gripper", "finger", "hand", "wrist")):
                        continue

                    if force_vector is None and normal is not None:
                        try:
                            force_vector = [float(impulse) * float(n) for n in normal]
                        except Exception:
                            force_vector = None

                    if force_vector is not None:
                        force_world[0] += float(force_vector[0])
                        force_world[1] += float(force_vector[1])
                        force_world[2] += float(force_vector[2])

                        if ee_position is not None and position is not None:
                            try:
                                rx = float(position[0]) - float(ee_position[0])
                                ry = float(position[1]) - float(ee_position[1])
                                rz = float(position[2]) - float(ee_position[2])
                                fx, fy, fz = float(force_vector[0]), float(force_vector[1]), float(force_vector[2])
                                torque_world[0] += ry * fz - rz * fy
                                torque_world[1] += rz * fx - rx * fz
                                torque_world[2] += rx * fy - ry * fx
                            except Exception:
                                pass

                    if include_contacts:
                        contacts.append(ContactPoint(
                            body_a=body_a,
                            body_b=body_b,
                            normal_force=impulse,
                            penetration_depth=0.0,
                            position=list(position) if position is not None else [],
                            normal=list(normal) if normal is not None else [],
                            force_vector=list(force_vector) if force_vector is not None else [],
                            tangent_impulse=list(tangent_impulse) if tangent_impulse is not None else [],
                            friction=float(friction) if friction is not None else 0.0,
                            contact_area=float(contact_area) if contact_area is not None else 0.0,
                        ))

            # Convert to EE frame if rotation available
            if ee_rotation is not None:
                try:
                    rt = [
                        [ee_rotation[0][0], ee_rotation[1][0], ee_rotation[2][0]],
                        [ee_rotation[0][1], ee_rotation[1][1], ee_rotation[2][1]],
                        [ee_rotation[0][2], ee_rotation[1][2], ee_rotation[2][2]],
                    ]
                    force = [
                        rt[0][0] * force_world[0] + rt[0][1] * force_world[1] + rt[0][2] * force_world[2],
                        rt[1][0] * force_world[0] + rt[1][1] * force_world[1] + rt[1][2] * force_world[2],
                        rt[2][0] * force_world[0] + rt[2][1] * force_world[1] + rt[2][2] * force_world[2],
                    ]
                    torque = [
                        rt[0][0] * torque_world[0] + rt[0][1] * torque_world[1] + rt[0][2] * torque_world[2],
                        rt[1][0] * torque_world[0] + rt[1][1] * torque_world[1] + rt[1][2] * torque_world[2],
                        rt[2][0] * torque_world[0] + rt[2][1] * torque_world[1] + rt[2][2] * torque_world[2],
                    ]
                    frame = "end_effector"
                except Exception:
                    force = force_world
                    torque = torque_world
                    frame = "world"
            else:
                force = force_world
                torque = torque_world
                frame = "world"

            return EEWrenchRsp(
                force=force,
                torque=torque,
                frame=frame,
                source=source,
                contacts=contacts,
            )
        except Exception:
            return EEWrenchRsp(
                force=force,
                torque=torque,
                frame="end_effector",
                source="unavailable",
                contacts=[],
            )
    # --- END BlueprintPipeline ee_wrench patch ---
""")


def patch_file() -> None:
    if not os.path.isfile(GRPC_SERVER):
        print(f"[PATCH] grpc_server.py not found at {GRPC_SERVER}")
        print("[PATCH] Skipping ee_wrench patch (server source not available)")
        sys.exit(0)

    with open(GRPC_SERVER, "r") as f:
        content = f.read()

    if PATCH_MARKER in content:
        print("[PATCH] ee_wrench already patched — skipping")
        sys.exit(0)

    class_match = re.search(r"^class\\s+SimObservationServiceServicer\\b.*?:", content, re.MULTILINE)
    if not class_match:
        print("[PATCH] SimObservationServiceServicer class not found — skipping")
        sys.exit(0)

    insert_start = class_match.end()
    next_class = re.search(r"^class\\s+\\w+", content[insert_start:], re.MULTILINE)
    insert_at = insert_start + next_class.start() if next_class else len(content)

    method_indent = "    "
    method_match = re.search(r"^([ \\t]+)def\\s+\\w+\\(self", content[insert_start:insert_at], re.MULTILINE)
    if method_match:
        method_indent = method_match.group(1)

    indented_handler = "\\n".join(
        (method_indent + line) if line.strip() else line
        for line in HANDLER.splitlines()
    ) + "\\n"

    patched = content[:insert_at].rstrip() + "\\n\\n" + indented_handler + content[insert_at:]

    with open(GRPC_SERVER, "w") as f:
        f.write(patched)

    print("[PATCH] Injected get_ee_wrench into grpc_server.py")


if __name__ == "__main__":
    patch_file()
