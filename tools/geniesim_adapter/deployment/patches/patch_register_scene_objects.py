#!/usr/bin/env python3
"""Patch: Fix dynamic objects causing physics view invalidation during init.

Injects into _init_robot_cfg:
- BEFORE self._play(): make dynamic objects kinematic (so physics stays stable)
- AFTER articulation init completes: restore them to dynamic

Also patches _set_object_pose to toggle kinematic for dynamic body teleport.
"""
import sys

CC_PATH = "/opt/geniesim/source/data_collection/server/command_controller.py"

MARKER = "# BPv3_pre_play_kinematic"
MARKER_POSE = "# BPv3_dynamic_teleport"


def apply():
    with open(CC_PATH) as f:
        src = f.read()

    if MARKER in src:
        print("[PATCH] v3 scene_objects: already applied")
        return True

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
            except Exception as _e_pp:
                print(f"[PATCH] Pre-play kinematic toggle failed: {{_e_pp}}")
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

    # Also need the restore after rotation is set
    old_pose_end = '''                    orient_attr.Set(quat_type(*rotation_data))'''

    # Count occurrences — should be in the else branch of _set_object_pose
    if old_pose in src:
        src = src.replace(old_pose, new_pose, 1)

        # Now add the restore-to-dynamic after orient_attr.Set in that same else block
        # Find the pose block we just modified (it has our marker)
        marker_idx = src.index(MARKER_POSE)
        # Find the next orient_attr.Set after our marker
        rest_after = src[marker_idx:]
        orient_idx = rest_after.index(old_pose_end)
        abs_orient_end = marker_idx + orient_idx + len(old_pose_end)

        restore_code = '''
                if _is_dyn_tp:
                    try:
                        UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Set(False)
                    except Exception:
                        pass'''

        src = src[:abs_orient_end] + restore_code + src[abs_orient_end:]
        print("[PATCH] dynamic_teleport: applied")
    elif MARKER_POSE in src:
        print("[PATCH] dynamic_teleport: already applied")
    else:
        # The old pose code might have already been patched by a previous version
        print("[PATCH] WARNING: _set_object_pose fallback not found (may be pre-patched)")

    with open(CC_PATH, 'w') as f:
        f.write(src)

    print("[PATCH] v3 scene_objects: all patches applied successfully")
    return True


if __name__ == "__main__":
    sys.exit(0 if apply() else 1)
