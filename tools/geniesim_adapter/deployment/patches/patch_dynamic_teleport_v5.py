#!/usr/bin/env python3
"""Patch v5: Fix dynamic object teleport in the usd_objects branch.

ROOT CAUSE: _set_object_pose has two branches:
  1. `if pose["prim_path"] in self.usd_objects:` → XFormPrim.set_world_pose()
  2. `else:` → manual xformOp + kinematic toggle + deferred restore

Dynamic objects ARE registered in usd_objects (post-init code), so branch 1
is always taken. XFormPrim.set_world_pose() only updates xformOps, but PhysX
overwrites xformOps for dynamic bodies every physics step → teleport has no effect.

FIX: In branch 1, detect dynamic objects and wrap set_world_pose() with
kinematic toggle + deferred restore.
"""
import sys
import os

CC_PATH = os.environ.get(
    "GENIESIM_CC_PATH",
    "/opt/geniesim/source/data_collection/server/command_controller.py",
)

MARKER = "# BPv5_dynamic_teleport_usd_objects"
DEFERRED_STEPS = 5


def apply():
    with open(CC_PATH) as f:
        src = f.read()

    if MARKER in src:
        print("[PATCH-v5] dynamic_teleport_usd_objects: already applied")
        return True

    # Find the exact pattern in _set_object_pose
    old_code = """            if pose["prim_path"] in self.usd_objects:
                object = self.usd_objects[pose["prim_path"]]
                object.set_world_pose(pose["position"], pose["rotation"])"""

    new_code = f"""            if pose["prim_path"] in self.usd_objects:
                {MARKER}
                object = self.usd_objects[pose["prim_path"]]
                _is_dyn_usd = pose["prim_path"] in getattr(self, "_bp_dynamic_scene_prims", set())
                if _is_dyn_usd:
                    # Dynamic rigid body: must toggle kinematic before teleport
                    # so PhysX doesn't override the new xformOp position.
                    try:
                        import omni.usd as _omu5
                        from pxr import UsdPhysics as _UP5
                        _st5 = _omu5.get_context().get_stage()
                        _pr5 = _st5.GetPrimAtPath(pose["prim_path"]) if _st5 else None
                        if _pr5 and _pr5.IsValid():
                            _UP5.RigidBodyAPI(_pr5).GetKinematicEnabledAttr().Set(True)
                    except Exception as _e5:
                        print(f'[PATCH-v5] kinematic toggle failed: {{_e5}}')
                object.set_world_pose(pose["position"], pose["rotation"])
                if _is_dyn_usd:
                    # Schedule deferred restore to dynamic after {DEFERRED_STEPS} physics steps
                    _deferred5 = getattr(self, "_bp_deferred_dynamic_restore", None)
                    if _deferred5 is None:
                        self._bp_deferred_dynamic_restore = {{}}
                        _deferred5 = self._bp_deferred_dynamic_restore
                    _deferred5[pose["prim_path"]] = {DEFERRED_STEPS}
                    print(f'[DEFERRED-v5] Scheduled {{pose["prim_path"]}} for restore in {DEFERRED_STEPS} steps')"""

    if old_code not in src:
        print("[PATCH-v5] FAILED: Could not find usd_objects branch in _set_object_pose")
        print("[PATCH-v5] Looking for alternate pattern...")

        # Try with slightly different formatting
        alt_old = '            if pose["prim_path"] in self.usd_objects:'
        if alt_old in src:
            # Find the 3-line block
            idx = src.index(alt_old)
            # Read next 2 lines
            lines = src[idx:].split('\n')[:3]
            actual_pattern = '\n'.join(lines)
            print(f"[PATCH-v5] Found block:\n{actual_pattern}")
            print("[PATCH-v5] Attempting match with actual content...")

            # Replace just the block
            src = src.replace(actual_pattern, new_code, 1)
            with open(CC_PATH, "w") as f:
                f.write(src)
            print("[PATCH-v5] dynamic_teleport_usd_objects: APPLIED (alt pattern)")
            return True

        print("[PATCH-v5] FAILED: Cannot locate _set_object_pose usd_objects branch")
        return False

    src = src.replace(old_code, new_code, 1)
    with open(CC_PATH, "w") as f:
        f.write(src)

    print("[PATCH-v5] dynamic_teleport_usd_objects: APPLIED")
    return True


if __name__ == "__main__":
    sys.exit(0 if apply() else 1)
