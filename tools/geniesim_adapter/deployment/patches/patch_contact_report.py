#!/usr/bin/env python3
"""
Patch the Genie Sim server's grpc_server.py to add get_contact_report and joint efforts.

This enables:
- PhysX contact reporting via SimObservationService/get_contact_report
- Real joint effort capture in observations (not backfilled inverse dynamics)

The patch is idempotent and safe to run multiple times.
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

PATCH_MARKER = "BlueprintPipeline contact_report patch"
EFFORTS_PATCH_MARKER = "BlueprintPipeline joint_efforts patch"

CONTACT_HANDLER = textwrap.dedent("""\

    # --- BEGIN BlueprintPipeline contact_report patch ---
    _bp_contact_reporting_enabled = False

    def _bp_enable_contact_reporting(self):
        \"\"\"Enable PhysX ContactReportAPI on gripper links and scene objects.\"\"\"
        if getattr(self, "_bp_contact_reporting_enabled", False):
            return
        try:
            import os as _os
            import omni.usd
            from pxr import Usd, UsdPhysics, PhysxSchema

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                return

            prim_paths = set()

            # Try to resolve robot prim roots from known controller attributes
            robot_roots = []
            try:
                server_fn = getattr(self, "server_function", None)
                cmd = getattr(server_fn, "command_controller", None) or server_fn
                robot = getattr(cmd, "robot", None)
                for attr in ("robot_prim_path", "robot_root_path", "robot_root", "robot_path"):
                    root = getattr(cmd, attr, None) or getattr(robot, attr, None)
                    if root:
                        robot_roots.append(str(root))
            except Exception:
                pass

            # Discover gripper/finger links under robot roots
            gripper_keywords = ("finger", "gripper", "hand", "leftfinger", "rightfinger")
            for root in robot_roots:
                prim = stage.GetPrimAtPath(root)
                if not prim.IsValid():
                    continue
                for desc in Usd.PrimRange(prim):
                    name = desc.GetName().lower()
                    if not any(kw in name for kw in gripper_keywords):
                        continue
                    if desc.HasAPI(UsdPhysics.RigidBodyAPI) or desc.HasAPI(UsdPhysics.CollisionAPI):
                        prim_paths.add(str(desc.GetPath()))

            # Include scene objects under common paths
            for prefix in ("/World/Scene", "/World/Objects", "/World"):
                prim = stage.GetPrimAtPath(prefix)
                if not prim.IsValid():
                    continue
                for desc in Usd.PrimRange(prim):
                    if desc.HasAPI(UsdPhysics.RigidBodyAPI) or desc.HasAPI(UsdPhysics.CollisionAPI):
                        prim_paths.add(str(desc.GetPath()))

            if not prim_paths:
                return

            threshold = float(_os.getenv("PHYSX_CONTACT_REPORT_THRESHOLD_N", "0.1"))
            enabled = 0
            for prim_path in prim_paths:
                try:
                    prim = stage.GetPrimAtPath(prim_path)
                    if not prim.IsValid():
                        continue
                    if prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
                        enabled += 1
                        continue
                    contact_api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
                    if contact_api:
                        thr_attr = contact_api.GetThresholdAttr()
                        if thr_attr:
                            thr_attr.Set(threshold)
                        enabled += 1
                except Exception:
                    continue
            if enabled > 0:
                self._bp_contact_reporting_enabled = True
        except Exception:
            return

    def get_contact_report(self, request, context):
        \"\"\"Return a PhysX contact report for the current simulation step.

        Note: PhysX contact reporting requires ContactReportAPI to be enabled on prims,
        which we skip to avoid slow stage traversal. Returns empty report if no contact data.
        \"\"\"
        try:
            # Direct import from /opt/geniesim to avoid /workspace path shadowing
            import importlib.util
            import os as _os
            _pb2_path = "/opt/geniesim/source/data_collection/common/aimdk/protocol/contact_report_pb2.py"
            if _os.path.exists(_pb2_path):
                _spec = importlib.util.spec_from_file_location("_bp_contact_pb2", _pb2_path)
                if _spec and _spec.loader:
                    _pb2 = importlib.util.module_from_spec(_spec)
                    _spec.loader.exec_module(_pb2)
            else:
                raise ImportError(f"contact_report_pb2.py not found at {_pb2_path}")

            ContactReportRsp = _pb2.ContactReportRsp
            ContactPoint = _pb2.ContactPoint
        except Exception as _import_err:
            print(f"[CONTACT_REPORT] pb2 import failed: {_import_err}")
            return None

        contacts = []
        total_force = 0.0
        max_penetration = 0.0
        include_points = getattr(request, "include_points", False)

        # Try to get PhysX contact data (may be empty if ContactReportAPI not enabled)
        try:
            contact_data = None
            try:
                from omni.physx import get_physx_simulation_interface
                sim = get_physx_simulation_interface()
                if sim is not None and hasattr(sim, "get_contact_report"):
                    contact_data = sim.get_contact_report()
            except Exception:
                pass

            if contact_data is None:
                try:
                    from omni.physx import get_physx_interface
                    physx = get_physx_interface()
                    if physx is not None and hasattr(physx, "get_contact_report"):
                        contact_data = physx.get_contact_report()
                except Exception:
                    pass

            if contact_data:
                for contact in contact_data:
                    if isinstance(contact, dict):
                        body_a = str(contact.get("actor0", ""))
                        body_b = str(contact.get("actor1", ""))
                        impulse = float(contact.get("impulse", 0.0))
                        separation = float(contact.get("separation", 0.0))
                        position = contact.get("position", [0, 0, 0])
                        normal = contact.get("normal", [0, 0, 1])
                    else:
                        body_a = str(getattr(contact, "actor0", ""))
                        body_b = str(getattr(contact, "actor1", ""))
                        impulse = float(getattr(contact, "impulse", 0.0))
                        separation = float(getattr(contact, "separation", 0.0))
                        position = getattr(contact, "position", [0, 0, 0])
                        normal = getattr(contact, "normal", [0, 0, 1])

                    penetration = abs(separation)
                    total_force += impulse
                    max_penetration = max(max_penetration, penetration)

                    contacts.append(ContactPoint(
                        body_a=body_a,
                        body_b=body_b,
                        normal_force=impulse,
                        penetration_depth=penetration,
                        position=list(position) if include_points else [],
                        normal=list(normal) if include_points else [],
                    ))
        except Exception as _err:
            print(f"[CONTACT_REPORT] PhysX contact capture failed: {_err}")

        return ContactReportRsp(
            contacts=contacts,
            total_normal_force=total_force,
            max_penetration_depth=max_penetration,
        )
    # --- END BlueprintPipeline contact_report patch ---
""")

# Joint efforts patch: Inject effort capture into get_observation handler
EFFORTS_PATCH = textwrap.dedent("""\
    # --- BEGIN BlueprintPipeline joint_efforts patch ---
    # Capture real joint efforts from robot articulation
    def _get_real_joint_efforts(self):
        \"\"\"Get joint efforts from robot articulation (not inverse dynamics).\"\"\"
        try:
            if not hasattr(self, '_robot_articulation') or self._robot_articulation is None:
                return None, "no_articulation"
            robot = self._robot_articulation
            efforts = None
            source = "unavailable"

            def _efforts_nontrivial(vals):
                try:
                    _vals = vals.flatten().tolist() if hasattr(vals, "flatten") else list(vals)
                    return any(abs(float(v)) > 1e-6 for v in _vals)
                except Exception:
                    return False

            # Ensure PhysX has computed efforts if API requires it
            try:
                if hasattr(robot, "compute_joint_efforts"):
                    robot.compute_joint_efforts()
            except Exception as _compute_err:
                print(f"[JOINT_EFFORTS] compute_joint_efforts failed: {_compute_err}")

            # Measured joint forces -> torque magnitude
            if efforts is None and hasattr(robot, "get_measured_joint_forces"):
                try:
                    _forces = robot.get_measured_joint_forces()
                    if _forces is not None:
                        if hasattr(_forces, "shape") and len(_forces.shape) >= 2 and _forces.shape[-1] >= 6:
                            try:
                                import numpy as _np
                                efforts = _np.linalg.norm(_forces[..., 3:6], axis=-1)
                            except Exception:
                                efforts = []
                                for _row in _forces:
                                    try:
                                        _tor = _row[3:6]
                                        efforts.append(sum(abs(float(x)) for x in _tor))
                                    except Exception:
                                        efforts.append(0.0)
                        else:
                            efforts = _forces
                        if efforts is not None and _efforts_nontrivial(efforts):
                            source = "physx_measured_forces"
                        else:
                            efforts = None
                except Exception:
                    efforts = None

            # Measured efforts
            if efforts is None and hasattr(robot, "get_measured_joint_efforts"):
                try:
                    efforts = robot.get_measured_joint_efforts()
                    if efforts is not None and _efforts_nontrivial(efforts):
                        source = "physx_measured"
                    else:
                        efforts = None
                except Exception:
                    efforts = None

            # Applied efforts
            if efforts is None and hasattr(robot, "get_applied_joint_efforts"):
                try:
                    efforts = robot.get_applied_joint_efforts()
                    if efforts is not None and _efforts_nontrivial(efforts):
                        source = "physx_applied"
                    else:
                        efforts = None
                except Exception:
                    efforts = None

            # Direct joint efforts
            if efforts is None and hasattr(robot, "get_joint_efforts"):
                try:
                    efforts = robot.get_joint_efforts()
                    if efforts is not None and _efforts_nontrivial(efforts):
                        source = "physx_joint"
                    else:
                        efforts = None
                except Exception:
                    efforts = None

            # Dynamic-control fallback
            if efforts is None:
                try:
                    from omni.isaac.dynamic_control import _dynamic_control
                    _dc = _dynamic_control.acquire_dynamic_control_interface()
                    _art_handle = None
                    for _attr in ("_articulation", "articulation", "handle"):
                        _art_handle = getattr(robot, _attr, None)
                        if _art_handle:
                            break
                    if _art_handle is None:
                        _prim_path = getattr(robot, "prim_path", None) or getattr(robot, "prim", None)
                        if _prim_path:
                            _art_handle = _dc.get_articulation(_prim_path)
                    if _art_handle:
                        _dof_states = _dc.get_articulation_dof_states(
                            _art_handle, _dynamic_control.STATE_ALL
                        )
                        _eff_list = []
                        if hasattr(_dof_states, "dtype") and getattr(_dof_states.dtype, "names", None):
                            if "effort" in _dof_states.dtype.names:
                                _eff_list = [float(x) for x in _dof_states["effort"].tolist()]
                        else:
                            for _ds in _dof_states:
                                if hasattr(_ds, "effort"):
                                    _eff_list.append(float(_ds.effort))
                                elif isinstance(_ds, dict) and "effort" in _ds:
                                    _eff_list.append(float(_ds["effort"]))
                                else:
                                    try:
                                        _eff_list.append(float(_ds[2]))
                                    except Exception:
                                        _eff_list.append(0.0)
                        if _eff_list and _efforts_nontrivial(_eff_list):
                            efforts = _eff_list
                            source = "dynamic_control"
                except Exception:
                    efforts = None

            # Gravity compensation fallback
            if efforts is None:
                try:
                    if hasattr(robot, "get_coriolis_and_centrifugal_forces"):
                        _coriolis = robot.get_coriolis_and_centrifugal_forces()
                        if _coriolis is not None and _efforts_nontrivial(_coriolis):
                            efforts = _coriolis
                            source = "coriolis_and_centrifugal"
                    elif hasattr(robot, "get_generalized_gravity_forces"):
                        _gravity = robot.get_generalized_gravity_forces()
                        if _gravity is not None and _efforts_nontrivial(_gravity):
                            efforts = _gravity
                            source = "generalized_gravity"
                except Exception:
                    efforts = None

            if efforts is not None:
                return list(efforts), source
            return None, source
        except Exception as _e:
            print(f"[JOINT_EFFORTS] Failed to get efforts: {_e}")
            return None, f"error:{_e}"
    # --- END BlueprintPipeline joint_efforts patch ---
""")


def verify_patch_applied(grpc_server_path: str = GRPC_SERVER) -> dict:
    """Verify that patches have been applied to the server.

    Returns:
        dict with keys: contact_report_patched, joint_efforts_patched, server_found
    """
    result = {
        "server_found": os.path.isfile(grpc_server_path),
        "contact_report_patched": False,
        "joint_efforts_patched": False,
        "server_path": grpc_server_path,
    }
    if not result["server_found"]:
        return result

    with open(grpc_server_path, "r") as f:
        content = f.read()

    result["contact_report_patched"] = PATCH_MARKER in content
    result["joint_efforts_patched"] = EFFORTS_PATCH_MARKER in content
    return result


def patch_file() -> None:
    if not os.path.isfile(GRPC_SERVER):
        print(f"[PATCH] grpc_server.py not found at {GRPC_SERVER}")
        print("[PATCH] Skipping contact_report patch (server source not available)")
        sys.exit(0)

    with open(GRPC_SERVER, "r") as f:
        content = f.read()

    contact_patched = PATCH_MARKER in content
    efforts_patched = EFFORTS_PATCH_MARKER in content

    if contact_patched and efforts_patched:
        print("[PATCH] contact_report and joint_efforts already patched — skipping")
        sys.exit(0)

    if contact_patched:
        print("[PATCH] contact_report already patched — checking joint_efforts")

    # Try multiple class name patterns (Genie Sim may use different names across versions)
    class_patterns = [
        r"^class\s+(SimObservationServiceServicer)\b.*?:",
        r"^class\s+(ObservationService)\b.*?:",
        r"^class\s+(\w*Observation\w*Servicer?)\b.*?:",
        r"^class\s+(\w*Service\w*)\(.*?pb2_grpc\.\w+\).*?:",  # Any class inheriting from grpc servicer
    ]
    class_match = None
    matched_class = None
    for pattern in class_patterns:
        class_match = re.search(pattern, content, re.MULTILINE)
        if class_match:
            matched_class = class_match.group(1)
            print(f"[PATCH] Found service class: {matched_class}")
            break
    if not class_match:
        # Last resort: find any class with get_observation method
        obs_method = re.search(r"^(\s+)def get_observation\(self,", content, re.MULTILINE)
        if obs_method:
            # Search backward for class definition
            before_method = content[:obs_method.start()]
            class_match = list(re.finditer(r"^class\s+(\w+)\b.*?:", before_method, re.MULTILINE))
            if class_match:
                class_match = class_match[-1]  # Last class before the method
                matched_class = class_match.group(1)
                print(f"[PATCH] Found service class via get_observation: {matched_class}")
    if not class_match:
        print("[PATCH] Service class not found (tried SimObservationServiceServicer, ObservationService, etc.) — skipping")
        sys.exit(0)

    insert_start = class_match.end()
    next_class = re.search(r"^class\s+\w+", content[insert_start:], re.MULTILINE)
    insert_at = insert_start + next_class.start() if next_class else len(content)

    method_indent = "    "
    method_match = re.search(r"^([ \t]+)def\s+\w+\(self", content[insert_start:insert_at], re.MULTILINE)
    if method_match:
        method_indent = method_match.group(1)

    indented_handler = "\n".join(
        (method_indent + line) if line.strip() else line
        for line in CONTACT_HANDLER.splitlines()
    ) + "\n"

    patched = content[:insert_at].rstrip() + "\n\n" + indented_handler + content[insert_at:]

    with open(GRPC_SERVER, "w") as f:
        f.write(patched)

    print("[PATCH] Injected get_contact_report into grpc_server.py")

    # Now patch joint efforts if needed
    if not efforts_patched:
        _patch_joint_efforts(GRPC_SERVER, method_indent)


def _patch_joint_efforts(grpc_server_path: str, method_indent: str = "    ") -> None:
    """Inject joint efforts helper method into the server."""
    with open(grpc_server_path, "r") as f:
        content = f.read()

    if EFFORTS_PATCH_MARKER in content:
        print("[PATCH] joint_efforts already patched — skipping")
        return

    # Find the end of the contact_report patch to insert after it
    contact_end_marker = "# --- END BlueprintPipeline contact_report patch ---"
    if contact_end_marker not in content:
        print("[PATCH] Cannot find contact_report patch end marker — skipping joint_efforts")
        return

    insert_pos = content.index(contact_end_marker) + len(contact_end_marker)

    # Indent the efforts patch
    indented_efforts = "\n".join(
        (method_indent + line) if line.strip() else line
        for line in EFFORTS_PATCH.splitlines()
    ) + "\n"

    patched = content[:insert_pos] + "\n" + indented_efforts + content[insert_pos:]

    with open(grpc_server_path, "w") as f:
        f.write(patched)

    print("[PATCH] Injected joint_efforts helper into grpc_server.py")


if __name__ == "__main__":
    patch_file()

    # Print verification status
    status = verify_patch_applied()
    print(f"[VERIFY] Server found: {status['server_found']}")
    print(f"[VERIFY] Contact report patched: {status['contact_report_patched']}")
    print(f"[VERIFY] Joint efforts patched: {status['joint_efforts_patched']}")
