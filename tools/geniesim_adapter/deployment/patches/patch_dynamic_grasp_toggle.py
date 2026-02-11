#!/usr/bin/env python3
"""Patch: Dynamic grasp toggle — make target objects dynamic during grasp only.

ROOT CAUSE: GENIESIM_KEEP_OBJECTS_KINEMATIC=1 prevents objects from falling
through surfaces, but also prevents PhysX from generating real contact forces
and varying joint efforts during manipulation. This produces frozen effort
data and empty contact reports.

FIX: Add a gRPC-accessible command to toggle specific objects between
kinematic and dynamic states. The client (local_framework.py) sends
toggle commands at grasp start and release end. The actual USD prim
modification runs on the sim thread via on_command_step() dispatch
to avoid the known USD threading deadlock.

PROTOCOL:
  1. Client sends: set_object_dynamic(prim_path, is_dynamic=True) before grasp
  2. Server queues the request
  3. on_command_step() processes it on sim thread:
     - Sets physics:kinematicEnabled = (not is_dynamic) on the USD prim
     - Zeros velocity to prevent explosive forces
  4. Client sends: set_object_dynamic(prim_path, is_dynamic=False) after release

This patch must run AFTER patch_keep_objects_kinematic.py.
"""
import sys
import os
import re

CC_PATH = os.environ.get(
    "GENIESIM_CC_PATH",
    "/opt/geniesim/source/data_collection/server/command_controller.py",
)

GRPC_PATH = os.environ.get(
    "GENIESIM_GRPC_PATH",
    "/opt/geniesim/source/data_collection/server/grpc_server.py",
)

MARKER = "# BPv_dynamic_grasp_toggle"
RSP_GUARD_MARKER = "# BPv_dynamic_grasp_toggle_rsp_guard"
DISPATCH_MARKER = f"{MARKER} -- set_task_metric dispatch"
LEGACY_DISPATCH_MARKER = f"{MARKER} — set_task_metric dispatch"


def _inject_into_set_task_metric(src: str, injected: str, marker: str):
    """Inject code at the top of set_task_metric() unless marker already exists."""
    if marker in src:
        return src, False
    _stm_match = re.search(
        r'(def set_task_metric\(self[^)]*\)[^:]*:)\s*\n',
        src,
    )
    if not _stm_match:
        return src, False
    _inject_after = _stm_match.end()
    return src[:_inject_after] + injected + src[_inject_after:], True


def apply():
    # --- Patch command_controller.py: add toggle handler on sim thread ---
    with open(CC_PATH) as f:
        cc_src = f.read()

    if MARKER in cc_src:
        print("[PATCH] dynamic_grasp_toggle: already applied to command_controller.py")
    else:
        # Find the on_command_step method to inject our handler
        inject_point = "def on_command_step(self):"
        if inject_point not in cc_src:
            print("[PATCH] FAILED: could not find on_command_step in command_controller.py")
            return False

        # Add the toggle queue and handler
        toggle_queue_init = f"""
        {MARKER}
        # Dynamic grasp toggle: queue of (prim_path, is_dynamic) tuples
        if not hasattr(self, '_bp_dynamic_toggle_queue'):
            self._bp_dynamic_toggle_queue = []
"""

        toggle_handler = f"""
        {MARKER}
        # Process dynamic grasp toggle requests (runs on sim thread — safe for USD)
        if hasattr(self, '_bp_dynamic_toggle_queue') and self._bp_dynamic_toggle_queue:
            _toggles = list(self._bp_dynamic_toggle_queue)
            self._bp_dynamic_toggle_queue.clear()
            for _prim_path, _is_dynamic in _toggles:
                try:
                    from pxr import UsdPhysics, Gf
                    _stage = None
                    if hasattr(self, 'sim_app') and self.sim_app:
                        _ctx = self.sim_app.get_active_context()
                        if _ctx:
                            _stage = _ctx.get_stage()
                    if _stage is None and hasattr(self, '_stage'):
                        _stage = self._stage
                    if _stage is None:
                        import omni.usd
                        _stage = omni.usd.get_context().get_stage()
                    if _stage is None:
                        print(f"[TOGGLE] FAILED: no USD stage for {{_prim_path}}")
                        continue
                    _prim = _stage.GetPrimAtPath(_prim_path)
                    if not _prim or not _prim.IsValid():
                        print(f"[TOGGLE] FAILED: invalid prim {{_prim_path}}")
                        continue
                    _rb_api = UsdPhysics.RigidBodyAPI(_prim)
                    if _rb_api:
                        _kin_attr = _rb_api.GetKinematicEnabledAttr()
                        if _kin_attr:
                            _kin_attr.Set(not _is_dynamic)
                        # Zero velocity to prevent explosive forces on toggle
                        if _is_dynamic:
                            _vel_attr = _rb_api.GetVelocityAttr()
                            if _vel_attr:
                                _vel_attr.Set(Gf.Vec3f(0, 0, 0))
                            _ang_attr = _rb_api.GetAngularVelocityAttr()
                            if _ang_attr:
                                _ang_attr.Set(Gf.Vec3f(0, 0, 0))
                        # Physics settle: run 3 sim steps to let collision mesh stabilize
                        # before grasp force is applied. Prevents initial-frame interpenetration.
                        try:
                            import omni.kit.app
                            _app = omni.kit.app.get_app()
                            if _app:
                                for _settle_i in range(3):
                                    _app.update()
                                print(f"[TOGGLE] {{_prim_path}} settled (3 physics steps)")
                        except Exception as _settle_err:
                            print(f"[TOGGLE] WARNING: physics settle failed: {{_settle_err}}")
                        print(f"[TOGGLE] {{_prim_path}} -> {{'dynamic' if _is_dynamic else 'kinematic'}}")
                except Exception as _e:
                    print(f"[TOGGLE] ERROR for {{_prim_path}}: {{_e}}")
"""

        # Inject queue init after class __init__ or first method
        # Find the on_physics_step method to inject before the command dispatch
        if "def on_physics_step(self" in cc_src:
            cc_src = cc_src.replace(
                "def on_physics_step(self",
                toggle_queue_init + "    def on_physics_step(self",
                1,
            )
        else:
            # Fallback: inject before on_command_step
            cc_src = cc_src.replace(
                inject_point,
                toggle_queue_init + "    " + inject_point,
                1,
            )

        # Inject the toggle handler at the start of on_command_step
        cc_src = cc_src.replace(
            inject_point,
            inject_point + toggle_handler,
            1,
        )

        with open(CC_PATH, "w") as f:
            f.write(cc_src)
        print("[PATCH] dynamic_grasp_toggle: APPLIED to command_controller.py")

    # --- Patch grpc_server.py: add toggle RPC handler + set_task_metric dispatch ---
    with open(GRPC_PATH) as f:
        grpc_src = f.read()

    grpc_changed = False

    # Add a handler for the toggle command via the generic handler mechanism
    toggle_rpc = f"""
    {MARKER}
    def _bp_handle_dynamic_toggle(self, prim_path, is_dynamic):
        \"\"\"Queue a dynamic/kinematic toggle for the sim thread.\"\"\"
        _cc = getattr(self, 'command_controller', None)
        if _cc is None:
            _cc = getattr(self, '_command_controller', None)
        if _cc is None:
            # Try server_function (GenieSim convention)
            _sf = getattr(self, 'server_function', None)
            if _sf:
                _cc = getattr(_sf, 'command_controller', None)
                if _cc is None:
                    _cc = getattr(_sf, '_command_controller', None)
        if _cc is None:
            print("[TOGGLE-RPC] No command_controller reference")
            return False
        if not hasattr(_cc, '_bp_dynamic_toggle_queue'):
            _cc._bp_dynamic_toggle_queue = []
        _cc._bp_dynamic_toggle_queue.append((str(prim_path), bool(is_dynamic)))
        print(f"[TOGGLE-RPC] Queued: {{prim_path}} -> {{'dynamic' if is_dynamic else 'kinematic'}}")
        return True
"""

    if "def _bp_handle_dynamic_toggle(self, prim_path, is_dynamic):" not in grpc_src:
        # Find a good injection point in grpc_server.py
        if "class " in grpc_src:
            # Insert before the last method definition in the class.
            last_def = list(re.finditer(r"\n    def ", grpc_src))
            if last_def:
                insert_pos = last_def[-1].start()
                grpc_src = grpc_src[:insert_pos] + toggle_rpc + grpc_src[insert_pos:]
                grpc_changed = True
            else:
                grpc_src += toggle_rpc
                grpc_changed = True
        print("[PATCH] dynamic_grasp_toggle: injected _bp_handle_dynamic_toggle")

    # --- CRITICAL: Patch set_task_metric to dispatch bp::set_object_dynamic:: commands ---
    # Without this, the client sends set_task_metric("bp::set_object_dynamic::...") but
    # the server just returns "metric set" without processing the toggle command.
    _stm_dispatch = f"""
        {DISPATCH_MARKER}
        _metric_str = str(getattr(req, 'metric', '') or '')
        if _metric_str.startswith('bp::set_object_dynamic::'):
            try:
                import json as _bp_json
                _bp_payload_str = _metric_str.split('::', 2)[2]
                _bp_payload = _bp_json.loads(_bp_payload_str)
                _bp_prim = _bp_payload.get('prim_path', '')
                _bp_dyn = bool(_bp_payload.get('is_dynamic', True))
                if hasattr(self, '_bp_handle_dynamic_toggle'):
                    _bp_ok = self._bp_handle_dynamic_toggle(_bp_prim, _bp_dyn)
                    if _bp_ok:
                        return SetTaskMetricRsp(msg=f"bp::dynamic_toggle::ok::{{_bp_prim}}")
                    else:
                        return SetTaskMetricRsp(msg="bp::dynamic_toggle::failed::no_command_controller")
                else:
                    print("[TOGGLE-RPC] _bp_handle_dynamic_toggle not found on server instance")
                    return SetTaskMetricRsp(msg="bp::dynamic_toggle::unsupported")
            except Exception as _bp_err:
                print(f"[TOGGLE-RPC] set_task_metric dispatch error: {{_bp_err}}")
                return SetTaskMetricRsp(msg=f"bp::dynamic_toggle::error::{{_bp_err}}")
"""

    if DISPATCH_MARKER not in grpc_src and LEGACY_DISPATCH_MARKER not in grpc_src:
        grpc_src, _dispatch_added = _inject_into_set_task_metric(
            grpc_src,
            _stm_dispatch,
            DISPATCH_MARKER,
        )
        if _dispatch_added:
            print("[PATCH] dynamic_grasp_toggle: injected set_task_metric dispatch")
            grpc_changed = True
        else:
            print("[PATCH] WARNING: could not find set_task_metric method in grpc_server.py")
            print("[PATCH] The dynamic toggle will NOT work without manual set_task_metric patching!")

    # Compatibility upgrade: older patch revisions used SetTaskMetricRsp directly
    # but some runtime grpc_server.py variants do not import the symbol. Resolve
    # the response class from the request module at method entry.
    _stm_rsp_guard = f"""
        {RSP_GUARD_MARKER}
        SetTaskMetricRsp = globals().get('SetTaskMetricRsp')
        if SetTaskMetricRsp is None:
            try:
                import importlib as _bp_importlib
                _bp_pb2 = _bp_importlib.import_module(req.__class__.__module__)
                SetTaskMetricRsp = getattr(_bp_pb2, 'SetTaskMetricRsp', None)
            except Exception as _bp_rsp_err:
                print(f"[TOGGLE-RPC] SetTaskMetricRsp resolution error: {{_bp_rsp_err}}")
        if SetTaskMetricRsp is None:
            raise RuntimeError('SetTaskMetricRsp unavailable for set_task_metric response')
"""
    grpc_src, _guard_added = _inject_into_set_task_metric(
        grpc_src,
        _stm_rsp_guard,
        RSP_GUARD_MARKER,
    )
    if _guard_added:
        print("[PATCH] dynamic_grasp_toggle: injected SetTaskMetricRsp compatibility guard")
        grpc_changed = True

    if grpc_changed:
        with open(GRPC_PATH, "w") as f:
            f.write(grpc_src)
        print("[PATCH] dynamic_grasp_toggle: APPLIED to grpc_server.py")
    else:
        print("[PATCH] dynamic_grasp_toggle: already up-to-date on grpc_server.py")

    return True


if __name__ == "__main__":
    sys.exit(0 if apply() else 1)
