#!/usr/bin/env python3
"""Patch: Fix dynamic objects causing physics view invalidation during init.

Injects into _init_robot_cfg:
- BEFORE self._play(): make dynamic objects kinematic (so physics stays stable)
- AFTER articulation init completes: restore them to dynamic

Also patches _set_object_pose to use DEFERRED kinematic restore for dynamic body
teleport. Instead of toggling kinematic=False immediately (which causes PhysX to
snap the object back to its pre-teleport position), the restore is deferred for
several physics steps so PhysX fully registers the new xformOp position while
the object is still kinematic.
"""
import os
import re
import sys

CC_PATH = os.path.join(
    os.environ.get("GENIESIM_ROOT", "/opt/geniesim"),
    "source", "data_collection", "server", "command_controller.py",
)

MARKER = "# BPv3_pre_play_kinematic"
MARKER_POSE = "# BPv3_dynamic_teleport"
MARKER_DEFERRED = "# BPv3_deferred_dynamic_restore"

# Number of physics steps to keep object kinematic after teleport.
# At 60 Hz physics, 5 steps = ~83 ms — enough for PhysX to latch the new pose.
DEFERRED_STEPS = 5


def _strip_old_stripped_blocks(src: str) -> str:
    """Remove any old pre-play blocks left behind by marker-stripping sed commands.

    When we `sed 's/BPv3_pre_play_kinematic/BPv3_STRIPPED/g'`, the old code block
    stays in the file with the new marker name. This causes DUPLICATE pre-play
    blocks on the next patch application. Remove any block that starts with
    '# BPv3_STRIPPED' and ends before the next '# BPv3_' or 'self._play()'.
    """
    stripped_marker = "# BPv3_STRIPPED"
    if stripped_marker not in src:
        return src

    # Find and remove each stripped block
    # A stripped block starts with the marker line and ends before the
    # next non-blank line at the same or lower indentation that is NOT
    # part of the block (try/except, print, etc.)
    import re as _re_strip
    lines = src.split('\n')
    result_lines = []
    skip_until_outdent = False
    block_indent = 0

    for line in lines:
        if stripped_marker in line and not skip_until_outdent:
            # Start skipping this block
            skip_until_outdent = True
            block_indent = len(line) - len(line.lstrip())
            continue

        if skip_until_outdent:
            stripped = line.lstrip()
            if stripped == '':
                continue  # skip blank lines inside block
            line_indent = len(line) - len(stripped)
            if line_indent <= block_indent and not stripped.startswith('#'):
                # Check if this is the start of a new significant block
                # (like '# BPv3_pre_play_kinematic' or 'self._play()')
                if (MARKER in line or 'self._play()' in line
                        or stripped.startswith('def ')
                        or stripped.startswith('class ')):
                    skip_until_outdent = False
                    result_lines.append(line)
                    continue
            # Still inside the stripped block — keep skipping
            continue

        result_lines.append(line)

    new_src = '\n'.join(result_lines)
    if stripped_marker not in new_src:
        print("[PATCH] v3: cleaned up old STRIPPED block(s)")
    return new_src


def apply():
    with open(CC_PATH) as f:
        src = f.read()

    if MARKER in src:
        print("[PATCH] v3 scene_objects: already applied")
        return True

    # Clean up any old STRIPPED blocks from previous marker-stripping
    src = _strip_old_stripped_blocks(src)

    # ── Part 1: Before self._play(), make dynamic objects kinematic ──
    # Find the exact line: "            self._play()" inside _init_robot_cfg
    play_target = '\n            self._play()\n'
    if play_target not in src:
        # Try with different whitespace
        play_target = '\n            self._play()'
        if play_target not in src:
            print("[PATCH] FAILED - cannot find self._play() in _init_robot_cfg")
            return False
        play_target_end = play_target + '\n'
        if play_target_end in src:
            play_target = play_target_end

    pre_play_code = f'''
            {MARKER}
            # Make dynamic scene objects kinematic before _play() to prevent
            # PhysX tensor view invalidation during articulation init.
            self._bp_dynamic_to_restore = []
            self._bp_deferred_dynamic_restore = {{}}  # prim_path -> countdown
            self._bp_kinematic_static = []  # prims forced kinematic permanently
            self._bp_collision_applied_pre_play = False
            try:
                _sp_pre = stage.GetPrimAtPath("/World/Scene")
                if _sp_pre and _sp_pre.IsValid():
                    for _ch_pre in _sp_pre.GetChildren():
                        if not _ch_pre.HasAPI(UsdPhysics.RigidBodyAPI):
                            continue
                        _ka_pre = _ch_pre.GetAttribute("physics:kinematicEnabled")
                        if _ka_pre and _ka_pre.Get() == False:
                            _ka_pre.Set(True)
                            self._bp_dynamic_to_restore.append(str(_ch_pre.GetPath()))
                if self._bp_dynamic_to_restore:
                    print(f"[PATCH] Pre-play: {{len(self._bp_dynamic_to_restore)}} objects set to kinematic")
                # Ensure GroundPlane and other structural prims stay kinematic
                for _gp_path in ["/World/GroundPlane", "/World/groundPlane", "/World/Ground"]:
                    _gp = stage.GetPrimAtPath(_gp_path)
                    if _gp and _gp.IsValid() and _gp.HasAPI(UsdPhysics.RigidBodyAPI):
                        _gp_ka = _gp.GetAttribute("physics:kinematicEnabled")
                        if _gp_ka and _gp_ka.Get() == False:
                            _gp_ka.Set(True)
                            self._bp_kinematic_static.append(_gp_path)
                            print(f"[PATCH] Pre-play: forced {{_gp_path}} kinematic (ground plane)")
            except Exception as _e_pp:
                print(f"[PATCH] Pre-play kinematic toggle failed: {{_e_pp}}")

            # Runtime collision mutation is intentionally disabled.
            # Collision correctness must be baked into scene USD assets offline.
            self._bp_collision_applied_pre_play = False
'''

    src = src.replace(play_target, pre_play_code + play_target)

    # ── Part 2: After articulation init, restore dynamic and register objects ──
    # Find "self.dof_names = articulation.dof_names" which is right after _initialize_articulation
    restore_target = '            self.dof_names = articulation.dof_names\n'
    if restore_target not in src:
        print("[PATCH] FAILED - cannot find dof_names assignment")
        return False

    post_init_code = f'''
            # Restore dynamic objects after articulation is stable
            _bp_dyn_paths = set()
            try:
                for _dp_r in getattr(self, '_bp_dynamic_to_restore', []):
                    _pr = stage.GetPrimAtPath(_dp_r)
                    if _pr and _pr.IsValid():
                        _kar = _pr.GetAttribute("physics:kinematicEnabled")
                        if _kar:
                            _kar.Set(False)
                            _bp_dyn_paths.add(_dp_r)
                if _bp_dyn_paths:
                    print(f"[PATCH] Post-init: restored {{len(_bp_dyn_paths)}} objects to dynamic")
                # Register scene objects in usd_objects for pose queries
                _sp_post = stage.GetPrimAtPath("/World/Scene")
                if _sp_post and _sp_post.IsValid():
                    _nreg = 0
                    for _chp in _sp_post.GetChildren():
                        _cpp = str(_chp.GetPath())
                        if _cpp not in self.usd_objects:
                            self.usd_objects[_cpp] = XFormPrim(prim_path=_cpp)
                            _nreg += 1
                    print(f"[PATCH] Registered {{_nreg}} scene objects, {{len(_bp_dyn_paths)}} dynamic")
            except Exception as _epi:
                print(f"[PATCH] Post-init restore failed: {{_epi}}")
            self._bp_dynamic_scene_prims = _bp_dyn_paths
'''

    src = src.replace(restore_target, restore_target + post_init_code)

    # ── Part 3: Patch _set_object_pose for dynamic body teleport ──
    # Uses DEFERRED restore: sets kinematic=True before teleport, then
    # schedules kinematic=False after N physics steps via on_physics_step hook.
    old_pose = '''            else:
                stage = omni.usd.get_context().get_stage()
                if not stage:
                    return
                prim = stage.GetPrimAtPath(pose["prim_path"])
                if not prim.IsValid():
                    continue
                translate_attr = prim.GetAttribute("xformOp:translate")'''

    new_pose = f'''            else:
                {MARKER_POSE}
                stage = omni.usd.get_context().get_stage()
                if not stage:
                    return
                prim = stage.GetPrimAtPath(pose["prim_path"])
                if not prim.IsValid():
                    continue
                _is_dyn_tp = pose["prim_path"] in getattr(self, "_bp_dynamic_scene_prims", set())
                if _is_dyn_tp:
                    try:
                        UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Set(True)
                    except Exception:
                        pass
                translate_attr = prim.GetAttribute("xformOp:translate")'''

    # Also need the deferred restore after rotation is set
    old_pose_end = '''                    orient_attr.Set(quat_type(*rotation_data))'''

    # Count occurrences — should be in the else branch of _set_object_pose
    if old_pose in src:
        src = src.replace(old_pose, new_pose, 1)

        # Now add the DEFERRED restore after orient_attr.Set in that same else block
        # Instead of immediately setting kinematic=False, schedule it for N steps later
        marker_idx = src.index(MARKER_POSE)
        rest_after = src[marker_idx:]
        orient_idx = rest_after.index(old_pose_end)
        abs_orient_end = marker_idx + orient_idx + len(old_pose_end)

        restore_code = f'''
                if _is_dyn_tp:
                    # Deferred restore: keep kinematic for {DEFERRED_STEPS} physics steps
                    # so PhysX fully registers the new xformOp position before
                    # re-enabling dynamic simulation.
                    _deferred = getattr(self, "_bp_deferred_dynamic_restore", None)
                    if _deferred is None:
                        self._bp_deferred_dynamic_restore = {{}}
                        _deferred = self._bp_deferred_dynamic_restore
                    _deferred[pose["prim_path"]] = {DEFERRED_STEPS}'''

        src = src[:abs_orient_end] + restore_code + src[abs_orient_end:]
        print("[PATCH] dynamic_teleport: applied (deferred restore)")
    elif MARKER_POSE in src:
        print("[PATCH] dynamic_teleport: already applied")
    else:
        # The old pose code might have already been patched by a previous version
        print("[PATCH] WARNING: _set_object_pose fallback not found (may be pre-patched)")

    # ── Part 4: Inject deferred restore hook into on_physics_step ──
    # This runs on the sim thread every physics step and restores objects
    # to dynamic after the countdown expires.
    _inject_deferred_restore_hook(src)
    # Re-read since _inject writes directly
    with open(CC_PATH) as f:
        src = f.read()
    if MARKER_DEFERRED not in src:
        # Injection via helper failed; do inline injection
        src = _inject_deferred_restore_inline(src)

    with open(CC_PATH, 'w') as f:
        f.write(src)

    print("[PATCH] v3 scene_objects: all patches applied successfully")
    return True


def _inject_deferred_restore_inline(src: str) -> str:
    """Inject the deferred dynamic restore logic into on_physics_step.

    Must work whether or not patch_sim_thread_physics_cache has already
    added its own on_physics_step hook.
    """
    if MARKER_DEFERRED in src:
        print("[PATCH] deferred_restore hook: already present")
        return src

    # Find on_physics_step method definition
    pattern = re.compile(
        r"^([ \t]+)def on_physics_step\(self.*?\):\s*\n",
        re.MULTILINE,
    )
    match = pattern.search(src)
    if not match:
        print("[PATCH] WARNING: on_physics_step not found — deferred restore hook not injected")
        return src

    method_indent = match.group(1)
    body_indent = method_indent + "    "
    insert_pos = match.end()

    # Skip past docstring if present
    doc_pattern = re.compile(
        r'^[ \t]*(?:"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')[ \t]*\n',
        re.MULTILINE,
    )
    doc_match = doc_pattern.match(src[insert_pos:])
    if doc_match:
        insert_pos += doc_match.end()

    hook_code = f"""{body_indent}{MARKER_DEFERRED}
{body_indent}# Process deferred kinematic -> dynamic restores for teleported objects.
{body_indent}# After N physics steps with kinematic=True, PhysX has registered the
{body_indent}# new xformOp position; safe to re-enable dynamic simulation.
{body_indent}try:
{body_indent}    _bp_ddr = getattr(self, '_bp_deferred_dynamic_restore', None)
{body_indent}    if _bp_ddr:
{body_indent}        _bp_ddr_done = []
{body_indent}        for _bp_ddr_path, _bp_ddr_count in list(_bp_ddr.items()):
{body_indent}            _bp_ddr_count -= 1
{body_indent}            if _bp_ddr_count <= 0:
{body_indent}                try:
{body_indent}                    import omni.usd
{body_indent}                    _bp_ddr_stage = omni.usd.get_context().get_stage()
{body_indent}                    if _bp_ddr_stage:
{body_indent}                        _bp_ddr_prim = _bp_ddr_stage.GetPrimAtPath(_bp_ddr_path)
{body_indent}                        if _bp_ddr_prim and _bp_ddr_prim.IsValid():
{body_indent}                            from pxr import UsdPhysics
{body_indent}                            UsdPhysics.RigidBodyAPI(_bp_ddr_prim).GetKinematicEnabledAttr().Set(False)
{body_indent}                            print(f'[DEFERRED_RESTORE] {{_bp_ddr_path}} restored to dynamic')
{body_indent}                except Exception as _bp_ddr_err:
{body_indent}                    print(f'[DEFERRED_RESTORE] Error restoring {{_bp_ddr_path}}: {{_bp_ddr_err}}')
{body_indent}                _bp_ddr_done.append(_bp_ddr_path)
{body_indent}            else:
{body_indent}                _bp_ddr[_bp_ddr_path] = _bp_ddr_count
{body_indent}        for _bp_ddr_d in _bp_ddr_done:
{body_indent}            _bp_ddr.pop(_bp_ddr_d, None)
{body_indent}except Exception as _bp_ddr_outer_err:
{body_indent}    if not getattr(self, '_bp_ddr_err_logged', False):
{body_indent}        print(f'[DEFERRED_RESTORE] Hook error: {{_bp_ddr_outer_err}}')
{body_indent}        self._bp_ddr_err_logged = True
"""

    src = src[:insert_pos] + hook_code + src[insert_pos:]
    print("[PATCH] deferred_restore hook: injected into on_physics_step")
    return src


def _inject_deferred_restore_hook(src: str) -> bool:
    """Write patched source with deferred restore hook."""
    result = _inject_deferred_restore_inline(src)
    with open(CC_PATH, 'w') as f:
        f.write(result)
    return MARKER_DEFERRED in result


if __name__ == "__main__":
    sys.exit(0 if apply() else 1)
