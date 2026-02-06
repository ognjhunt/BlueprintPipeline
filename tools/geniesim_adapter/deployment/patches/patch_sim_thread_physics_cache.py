#!/usr/bin/env python3
"""
Patch: Cache real joint efforts + contact data on the SIM THREAD.

Problem:
  Joint efforts returned by `get_joint_position` are stale (cached from init).
  Contact data from `get_contact_report` is empty because it runs on the gRPC
  thread where PhysX APIs may deadlock or return stale results.

Solution:
  Hook into `on_physics_step()` in command_controller.py (which runs on the sim
  thread every physics step) to:
    1. Query real joint efforts from the robot articulation
    2. Query PhysX contact reports
    3. Cache both in attributes on the CommandController
  Then the gRPC handlers read from these caches instead of querying PhysX directly.

Architecture:
  data_collector_server.py → my_world.step() → on_physics_step() → on_command_step()
  The on_physics_step callback runs on the sim/main thread, so USD and PhysX
  APIs are safe to call here.

Usage:
    python3 /tmp/patches/patch_sim_thread_physics_cache.py

Idempotent — re-running is a no-op.
Must run AFTER patch_joint_efforts_handler.py and patch_contact_report.py
(those inject the gRPC-side code; this patch replaces the query with cache reads).
"""
import os
import re
import sys

GENIESIM_ROOT = os.environ.get("GENIESIM_ROOT", "/opt/geniesim")
CC_FILE = os.path.join(
    GENIESIM_ROOT,
    "source", "data_collection", "server", "command_controller.py",
)
GRPC_FILE = os.path.join(
    GENIESIM_ROOT,
    "source", "data_collection", "server", "grpc_server.py",
)

PATCH_MARKER = "BlueprintPipeline sim_thread_physics_cache patch"


def patch_command_controller():
    """Inject per-step effort + contact caching into on_physics_step."""
    if not os.path.isfile(CC_FILE):
        print(f"[PATCH] command_controller.py not found at {CC_FILE}")
        return False

    with open(CC_FILE, "r") as f:
        content = f.read()

    if PATCH_MARKER in content:
        print("[PATCH] sim_thread_physics_cache already applied to command_controller.py — skipping")
        return True

    # Find on_physics_step method. We inject our cache-update code at the START
    # of the method body (before any existing logic) so it runs every step.
    pattern = re.compile(
        r"^([ \t]+)def on_physics_step\(self.*?\):\s*\n",
        re.MULTILINE,
    )
    match = pattern.search(content)
    if not match:
        print("[PATCH] on_physics_step not found in command_controller.py — skipping")
        return False

    method_indent = match.group(1)
    body_indent = method_indent + "    "
    insert_pos = match.end()

    # Skip past docstring if present
    doc_pattern = re.compile(
        r'^[ \t]*(?:"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')[ \t]*\n',
        re.MULTILINE,
    )
    doc_match = doc_pattern.match(content[insert_pos:])
    if doc_match:
        insert_pos += doc_match.end()

    injection = f"""{body_indent}# {PATCH_MARKER}
{body_indent}# --- Cache real joint efforts + contacts every physics step ---
{body_indent}try:
{body_indent}    self._bp_cache_physics_data()
{body_indent}except Exception as _bp_cache_err:
{body_indent}    if not getattr(self, '_bp_cache_warned', False):
{body_indent}        print(f'[SIM_CACHE] Physics cache update failed: {{_bp_cache_err}}')
{body_indent}        self._bp_cache_warned = True
"""

    content = content[:insert_pos] + injection + content[insert_pos:]

    # Now inject the _bp_cache_physics_data method itself.
    # Find the end of the class or before the next class-level definition.
    # We'll insert it right after the on_physics_step method.
    # Safer: insert it at the end of the class, before any standalone code.
    # Find a good insertion point: after the last method in the class.
    # Strategy: insert after the PATCH_MARKER in the on_physics_step we just added,
    # at the class level (same indent as on_physics_step).

    # Find the end of on_physics_step body - look for next method at same indent
    # Since we modified content, re-search
    on_phys_match = re.search(
        rf"^{re.escape(method_indent)}def on_physics_step\(self.*?\):",
        content, re.MULTILINE,
    )
    if not on_phys_match:
        print("[PATCH] Could not re-find on_physics_step after injection")
        return False

    # Find next method at same indent level after on_physics_step
    next_method = re.compile(
        rf"^{re.escape(method_indent)}def (?!on_physics_step)\w+",
        re.MULTILINE,
    )
    nm_match = next_method.search(content, on_phys_match.end())

    if nm_match:
        helper_insert_pos = nm_match.start()
    else:
        # Insert at end of file
        helper_insert_pos = len(content)

    helper_method = f"""
{method_indent}def _bp_cache_physics_data(self):
{method_indent}    \"\"\"Cache joint efforts and contact data on the sim thread.
{method_indent}    Called every physics step from on_physics_step.\"\"\"
{method_indent}    # --- Joint efforts ---
{method_indent}    try:
{method_indent}        _articulation = None
{method_indent}        # Find articulation via ui_builder (primary path for GenieSim)
{method_indent}        _ui = getattr(self, 'ui_builder', None)
{method_indent}        if _ui is not None:
{method_indent}            _articulation = getattr(_ui, 'articulation', None)
{method_indent}        if _articulation is None:
{method_indent}            _articulation = getattr(self, 'articulation', None)
{method_indent}        if _articulation is None:
{method_indent}            _robot = getattr(self, 'robot', None)
{method_indent}            if _robot is not None:
{method_indent}                _articulation = getattr(_robot, 'articulation', None) or _robot
{method_indent}
{method_indent}        if _articulation is not None:
{method_indent}            _efforts = None
{method_indent}            _source = 'unavailable'
{method_indent}
{method_indent}            def _nontrivial(vals):
{method_indent}                try:
{method_indent}                    _v = vals.flatten().tolist() if hasattr(vals, 'flatten') else list(vals)
{method_indent}                    return any(abs(float(x)) > 1e-6 for x in _v)
{method_indent}                except Exception:
{method_indent}                    return False
{method_indent}
{method_indent}            # 1. Measured joint forces (6D wrench → torque magnitude)
{method_indent}            if hasattr(_articulation, 'get_measured_joint_forces'):
{method_indent}                try:
{method_indent}                    import numpy as _np
{method_indent}                    _forces = _articulation.get_measured_joint_forces()
{method_indent}                    if _forces is not None:
{method_indent}                        if hasattr(_forces, 'shape') and len(_forces.shape) >= 2 and _forces.shape[-1] >= 6:
{method_indent}                            _efforts = _np.linalg.norm(_forces[..., 3:6], axis=-1)
{method_indent}                        else:
{method_indent}                            _efforts = _forces
{method_indent}                        if _efforts is not None and _nontrivial(_efforts):
{method_indent}                            _source = 'physx_measured_forces'
{method_indent}                        else:
{method_indent}                            _efforts = None
{method_indent}                except Exception:
{method_indent}                    _efforts = None
{method_indent}
{method_indent}            # 2. Measured joint efforts
{method_indent}            if _efforts is None and hasattr(_articulation, 'get_measured_joint_efforts'):
{method_indent}                try:
{method_indent}                    _efforts = _articulation.get_measured_joint_efforts()
{method_indent}                    if _efforts is not None and _nontrivial(_efforts):
{method_indent}                        _source = 'physx_measured'
{method_indent}                    else:
{method_indent}                        _efforts = None
{method_indent}                except Exception:
{method_indent}                    _efforts = None
{method_indent}
{method_indent}            # 3. Applied joint efforts
{method_indent}            if _efforts is None and hasattr(_articulation, 'get_applied_joint_efforts'):
{method_indent}                try:
{method_indent}                    _efforts = _articulation.get_applied_joint_efforts()
{method_indent}                    if _efforts is not None and _nontrivial(_efforts):
{method_indent}                        _source = 'physx_applied'
{method_indent}                    else:
{method_indent}                        _efforts = None
{method_indent}                except Exception:
{method_indent}                    _efforts = None
{method_indent}
{method_indent}            # 4. Direct joint efforts
{method_indent}            if _efforts is None and hasattr(_articulation, 'get_joint_efforts'):
{method_indent}                try:
{method_indent}                    _efforts = _articulation.get_joint_efforts()
{method_indent}                    if _efforts is not None and _nontrivial(_efforts):
{method_indent}                        _source = 'physx_joint'
{method_indent}                    else:
{method_indent}                        _efforts = None
{method_indent}                except Exception:
{method_indent}                    _efforts = None
{method_indent}
{method_indent}            # 5. Dynamic control fallback
{method_indent}            if _efforts is None:
{method_indent}                try:
{method_indent}                    from omni.isaac.dynamic_control import _dynamic_control
{method_indent}                    _dc = _dynamic_control.acquire_dynamic_control_interface()
{method_indent}                    _art_handle = None
{method_indent}                    for _attr in ('_articulation', 'articulation', 'handle'):
{method_indent}                        _art_handle = getattr(_articulation, _attr, None)
{method_indent}                        if _art_handle:
{method_indent}                            break
{method_indent}                    if _art_handle is None:
{method_indent}                        _prim_path = getattr(_articulation, 'prim_path', None)
{method_indent}                        if _prim_path:
{method_indent}                            _art_handle = _dc.get_articulation(str(_prim_path))
{method_indent}                    if _art_handle:
{method_indent}                        _dof_states = _dc.get_articulation_dof_states(
{method_indent}                            _art_handle, _dynamic_control.STATE_ALL
{method_indent}                        )
{method_indent}                        _eff_list = []
{method_indent}                        if hasattr(_dof_states, 'dtype') and getattr(_dof_states.dtype, 'names', None):
{method_indent}                            if 'effort' in _dof_states.dtype.names:
{method_indent}                                _eff_list = [float(x) for x in _dof_states['effort'].tolist()]
{method_indent}                        else:
{method_indent}                            for _ds in _dof_states:
{method_indent}                                if hasattr(_ds, 'effort'):
{method_indent}                                    _eff_list.append(float(_ds.effort))
{method_indent}                                elif isinstance(_ds, dict) and 'effort' in _ds:
{method_indent}                                    _eff_list.append(float(_ds['effort']))
{method_indent}                                else:
{method_indent}                                    try:
{method_indent}                                        _eff_list.append(float(_ds[2]))
{method_indent}                                    except Exception:
{method_indent}                                        _eff_list.append(0.0)
{method_indent}                        if _eff_list and _nontrivial(_eff_list):
{method_indent}                            _efforts = _eff_list
{method_indent}                            _source = 'dynamic_control'
{method_indent}                except Exception:
{method_indent}                    _efforts = None
{method_indent}
{method_indent}            # 6. Gravity compensation fallback
{method_indent}            if _efforts is None:
{method_indent}                try:
{method_indent}                    if hasattr(_articulation, 'get_coriolis_and_centrifugal_forces'):
{method_indent}                        _coriolis = _articulation.get_coriolis_and_centrifugal_forces()
{method_indent}                        if _coriolis is not None and _nontrivial(_coriolis):
{method_indent}                            _efforts = _coriolis
{method_indent}                            _source = 'coriolis_and_centrifugal'
{method_indent}                    elif hasattr(_articulation, 'get_generalized_gravity_forces'):
{method_indent}                        _gravity = _articulation.get_generalized_gravity_forces()
{method_indent}                        if _gravity is not None and _nontrivial(_gravity):
{method_indent}                            _efforts = _gravity
{method_indent}                            _source = 'generalized_gravity'
{method_indent}                except Exception:
{method_indent}                    _efforts = None
{method_indent}
{method_indent}            if _efforts is not None:
{method_indent}                if hasattr(_efforts, 'flatten'):
{method_indent}                    self._bp_cached_efforts = _efforts.flatten().tolist()
{method_indent}                else:
{method_indent}                    self._bp_cached_efforts = list(_efforts)
{method_indent}                self._bp_cached_efforts_source = _source
{method_indent}                if not getattr(self, '_bp_efforts_logged', False):
{method_indent}                    _sample = [round(float(e), 4) for e in self._bp_cached_efforts[:5]]
{method_indent}                    print(f'[SIM_CACHE] Joint efforts cached: source={{_source}}, sample={{_sample}}')
{method_indent}                    self._bp_efforts_logged = True
{method_indent}    except Exception as _eff_err:
{method_indent}        if not getattr(self, '_bp_effort_err_logged', False):
{method_indent}            print(f'[SIM_CACHE] Effort cache error: {{_eff_err}}')
{method_indent}            self._bp_effort_err_logged = True
{method_indent}
{method_indent}    # --- Contact data ---
{method_indent}    try:
{method_indent}        _contact_data = None
{method_indent}        try:
{method_indent}            from omni.physx import get_physx_simulation_interface
{method_indent}            _sim = get_physx_simulation_interface()
{method_indent}            if _sim is not None and hasattr(_sim, 'get_contact_report'):
{method_indent}                _contact_data = _sim.get_contact_report()
{method_indent}        except Exception:
{method_indent}            pass
{method_indent}
{method_indent}        if _contact_data is None:
{method_indent}            try:
{method_indent}                from omni.physx import get_physx_interface
{method_indent}                _physx = get_physx_interface()
{method_indent}                if _physx is not None and hasattr(_physx, 'get_contact_report'):
{method_indent}                    _contact_data = _physx.get_contact_report()
{method_indent}            except Exception:
{method_indent}                pass
{method_indent}
{method_indent}        _contacts = []
{method_indent}        _total_force = 0.0
{method_indent}        _max_pen = 0.0
{method_indent}        if _contact_data:
{method_indent}            for _c in _contact_data:
{method_indent}                if isinstance(_c, dict):
{method_indent}                    _ba = str(_c.get('actor0', ''))
{method_indent}                    _bb = str(_c.get('actor1', ''))
{method_indent}                    _imp = float(_c.get('impulse', 0.0))
{method_indent}                    _sep = float(_c.get('separation', 0.0))
{method_indent}                    _pos = _c.get('position', [0, 0, 0])
{method_indent}                    _nrm = _c.get('normal', [0, 0, 1])
{method_indent}                else:
{method_indent}                    _ba = str(getattr(_c, 'actor0', ''))
{method_indent}                    _bb = str(getattr(_c, 'actor1', ''))
{method_indent}                    _imp = float(getattr(_c, 'impulse', 0.0))
{method_indent}                    _sep = float(getattr(_c, 'separation', 0.0))
{method_indent}                    _pos = getattr(_c, 'position', [0, 0, 0])
{method_indent}                    _nrm = getattr(_c, 'normal', [0, 0, 1])
{method_indent}                _pen = abs(_sep)
{method_indent}                _total_force += _imp
{method_indent}                _max_pen = max(_max_pen, _pen)
{method_indent}                _contacts.append({{
{method_indent}                    'body_a': _ba, 'body_b': _bb,
{method_indent}                    'impulse': _imp, 'penetration': _pen,
{method_indent}                    'position': list(_pos), 'normal': list(_nrm),
{method_indent}                }})
{method_indent}
{method_indent}        self._bp_cached_contacts = _contacts
{method_indent}        self._bp_cached_contact_total_force = _total_force
{method_indent}        self._bp_cached_contact_max_penetration = _max_pen
{method_indent}        if _contacts and not getattr(self, '_bp_contacts_logged', False):
{method_indent}            print(f'[SIM_CACHE] Contacts cached: count={{len(_contacts)}}, '
{method_indent}                  f'total_force={{_total_force:.4f}}, bodies={{_contacts[0]["body_a"]}} vs {{_contacts[0]["body_b"]}}')
{method_indent}            self._bp_contacts_logged = True
{method_indent}    except Exception as _ct_err:
{method_indent}        if not getattr(self, '_bp_contact_err_logged', False):
{method_indent}            print(f'[SIM_CACHE] Contact cache error: {{_ct_err}}')
{method_indent}            self._bp_contact_err_logged = True

"""

    content = content[:helper_insert_pos] + helper_method + content[helper_insert_pos:]

    with open(CC_FILE, "w") as f:
        f.write(content)

    print(f"[PATCH] Injected sim-thread physics cache into command_controller.py")
    return True


def patch_grpc_server_effort_cache():
    """Modify the gRPC get_joint_position handler to read cached efforts
    from the CommandController instead of using stale state.effort values."""
    if not os.path.isfile(GRPC_FILE):
        print(f"[PATCH] grpc_server.py not found at {GRPC_FILE}")
        return False

    with open(GRPC_FILE, "r") as f:
        content = f.read()

    EFFORT_CACHE_MARKER = "BlueprintPipeline effort_cache_read"
    if EFFORT_CACHE_MARKER in content:
        print("[PATCH] effort_cache_read already applied to grpc_server.py — skipping")
        return True

    # Find the 'return rsp' in get_joint_position that's preceded by
    # the joint_efforts_capture patch or the original states.append.
    # We want to inject AFTER the existing effort capture tries (which may
    # fail on the gRPC thread) but BEFORE the return.
    #
    # Strategy: look for the END marker of the joint_efforts_capture patch,
    # and inject our cache-read override right before the 'return rsp'.

    # Check if the joint_efforts_capture patch is present
    jec_end_marker = "# --- END BlueprintPipeline joint_efforts_capture patch ---"
    if jec_end_marker in content:
        # Find the 'return rsp' that follows the end marker
        jec_end_pos = content.index(jec_end_marker)
        return_match = re.search(r"^(\s+)return rsp\b", content[jec_end_pos:], re.MULTILINE)
        if return_match:
            insert_pos = jec_end_pos + return_match.start()
            indent = return_match.group(1)
        else:
            print("[PATCH] Could not find 'return rsp' after joint_efforts_capture — skipping")
            return False
    else:
        # No joint_efforts_capture patch — find return rsp in get_joint_position
        gjp_match = re.search(r"def get_joint_position\(self", content)
        if not gjp_match:
            print("[PATCH] get_joint_position not found — skipping")
            return False
        # Find next 'return rsp' after get_joint_position
        return_match = re.search(r"^(\s+)return rsp\b", content[gjp_match.end():], re.MULTILINE)
        if return_match:
            insert_pos = gjp_match.end() + return_match.start()
            indent = return_match.group(1)
        else:
            print("[PATCH] Could not find 'return rsp' in get_joint_position — skipping")
            return False

    cache_read_code = f"""{indent}# {EFFORT_CACHE_MARKER}
{indent}# Override stale efforts with sim-thread cached values
{indent}try:
{indent}    _cmd = getattr(self, 'server_function', None)
{indent}    _cached_eff = getattr(_cmd, '_bp_cached_efforts', None) if _cmd else None
{indent}    if _cached_eff and len(_cached_eff) >= len(rsp.states):
{indent}        _eff_src = getattr(_cmd, '_bp_cached_efforts_source', 'sim_cache')
{indent}        for _si, _state in enumerate(rsp.states):
{indent}            if _si < len(_cached_eff):
{indent}                _state.effort = float(_cached_eff[_si])
{indent}        if not getattr(self, '_bp_effort_cache_logged', False):
{indent}            _sample = [round(float(_cached_eff[i]), 4) for i in range(min(5, len(_cached_eff)))]
{indent}            print(f'[EFFORT_CACHE] Overriding gRPC efforts with sim-thread cache: source={{_eff_src}}, sample={{_sample}}')
{indent}            self._bp_effort_cache_logged = True
{indent}except Exception as _ecr_err:
{indent}    pass  # Fall through to original efforts
"""

    content = content[:insert_pos] + cache_read_code + content[insert_pos:]

    with open(GRPC_FILE, "w") as f:
        f.write(content)

    print("[PATCH] Injected effort cache read into grpc_server.py get_joint_position")
    return True


def patch_grpc_server_contact_cache():
    """Modify the gRPC get_contact_report handler to read cached contact data
    from the CommandController instead of querying PhysX from gRPC thread."""
    if not os.path.isfile(GRPC_FILE):
        print(f"[PATCH] grpc_server.py not found at {GRPC_FILE}")
        return False

    with open(GRPC_FILE, "r") as f:
        content = f.read()

    CONTACT_CACHE_MARKER = "BlueprintPipeline contact_cache_read"
    if CONTACT_CACHE_MARKER in content:
        print("[PATCH] contact_cache_read already applied to grpc_server.py — skipping")
        return True

    # Find the get_contact_report method and inject cache-read logic at the
    # START of the method body, before the existing PhysX query code.
    # If cached data is available, return it immediately.
    cr_match = re.search(
        r"^(\s+)def get_contact_report\(self, request, context\):\s*\n",
        content, re.MULTILINE,
    )
    if not cr_match:
        print("[PATCH] get_contact_report method not found in grpc_server.py — skipping")
        return False

    method_indent = cr_match.group(1)
    body_indent = method_indent + "    "
    insert_pos = cr_match.end()

    # Skip past docstring if present
    doc_pattern = re.compile(
        r'^[ \t]*(?:"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')[ \t]*\n',
        re.MULTILINE,
    )
    doc_match = doc_pattern.match(content[insert_pos:])
    if doc_match:
        insert_pos += doc_match.end()

    cache_read_code = f"""{body_indent}# {CONTACT_CACHE_MARKER}
{body_indent}# Try to return sim-thread cached contact data (avoids gRPC thread deadlock)
{body_indent}try:
{body_indent}    _cmd = getattr(self, 'server_function', None)
{body_indent}    _cached_contacts = getattr(_cmd, '_bp_cached_contacts', None) if _cmd else None
{body_indent}    if _cached_contacts is not None:
{body_indent}        import importlib.util
{body_indent}        import os as _os
{body_indent}        _pb2_path = "/opt/geniesim/source/data_collection/common/aimdk/protocol/contact_report_pb2.py"
{body_indent}        if _os.path.exists(_pb2_path):
{body_indent}            _spec = importlib.util.spec_from_file_location("_bp_contact_pb2", _pb2_path)
{body_indent}            if _spec and _spec.loader:
{body_indent}                _pb2 = importlib.util.module_from_spec(_spec)
{body_indent}                _spec.loader.exec_module(_pb2)
{body_indent}                _ContactReportRsp = _pb2.ContactReportRsp
{body_indent}                _ContactPoint = _pb2.ContactPoint
{body_indent}                _include_points = getattr(request, 'include_points', False)
{body_indent}                _contacts = []
{body_indent}                for _cc in _cached_contacts:
{body_indent}                    _contacts.append(_ContactPoint(
{body_indent}                        body_a=_cc.get('body_a', ''),
{body_indent}                        body_b=_cc.get('body_b', ''),
{body_indent}                        normal_force=float(_cc.get('impulse', 0.0)),
{body_indent}                        penetration_depth=float(_cc.get('penetration', 0.0)),
{body_indent}                        position=list(_cc.get('position', [])) if _include_points else [],
{body_indent}                        normal=list(_cc.get('normal', [])) if _include_points else [],
{body_indent}                    ))
{body_indent}                return _ContactReportRsp(
{body_indent}                    contacts=_contacts,
{body_indent}                    total_normal_force=float(getattr(_cmd, '_bp_cached_contact_total_force', 0.0)),
{body_indent}                    max_penetration_depth=float(getattr(_cmd, '_bp_cached_contact_max_penetration', 0.0)),
{body_indent}                )
{body_indent}except Exception as _ccr_err:
{body_indent}    pass  # Fall through to original PhysX query
"""

    content = content[:insert_pos] + cache_read_code + content[insert_pos:]

    with open(GRPC_FILE, "w") as f:
        f.write(content)

    print("[PATCH] Injected contact cache read into grpc_server.py get_contact_report")
    return True


if __name__ == "__main__":
    success = True
    success = patch_command_controller() and success
    success = patch_grpc_server_effort_cache() and success
    success = patch_grpc_server_contact_cache() and success
    if success:
        print("[PATCH] sim_thread_physics_cache fully applied")
    else:
        print("[PATCH] sim_thread_physics_cache partially applied (some steps failed)")
    sys.exit(0)
