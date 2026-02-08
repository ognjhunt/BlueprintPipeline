#!/usr/bin/env python3
"""Patch v6: Fix _bp_dynamic_scene_prims being overwritten by old register code.

ROOT CAUSE: The old register_scene_objects code (from a previous patch version)
runs inside handle_init_robot AFTER the post-init restore. It rebuilds
_bp_dynamic_scene_prims but skips all objects that are already in usd_objects.
Since GenieSim registers objects during its own init, ALL objects are already in
usd_objects, so _dyn_paths is always empty, and _bp_dynamic_scene_prims = set().

This makes the v5 kinematic toggle in _set_object_pose never fire.

FIX: Modify the old register code to NOT skip objects already in usd_objects
when checking for dynamic status. It should still track which objects are dynamic
regardless of whether they're registered.
"""
import sys
import os

CC_PATH = os.environ.get(
    "GENIESIM_CC_PATH",
    "/opt/geniesim/source/data_collection/server/command_controller.py",
)

MARKER = "# BPv6_fix_dynamic_prims"


def apply():
    with open(CC_PATH) as f:
        src = f.read()

    if MARKER in src:
        print("[PATCH-v6] fix_dynamic_prims: already applied")
        return True

    # Find the old register code that overwrites _bp_dynamic_scene_prims
    # The problematic pattern: iterates children, skips if in usd_objects, only
    # adds to _dyn_paths for objects NOT in usd_objects.
    old_code = """                for _ch in _sp.GetChildren():
                    _cp = str(_ch.GetPath())
                    if _cp in self.usd_objects:
                        continue
                    _hr = _ch.HasAPI(_UsdPhy.RigidBodyAPI)
                    _ka = _ch.GetAttribute("physics:kinematicEnabled")
                    _ik = _ka.Get() if _ka else True
                    self.usd_objects[_cp] = XFormPrim(prim_path=_cp)
                    if _hr and not _ik:
                        _dyn_paths.add(_cp)
                    _reg += 1
                    print(f"[PATCH] Registered: {_cp} (rigid={_hr}, kin={_ik})")
                self._bp_dynamic_scene_prims = _dyn_paths"""

    new_code = f"""                {MARKER}
                for _ch in _sp.GetChildren():
                    _cp = str(_ch.GetPath())
                    _hr = _ch.HasAPI(_UsdPhy.RigidBodyAPI)
                    _ka = _ch.GetAttribute("physics:kinematicEnabled")
                    _ik = _ka.Get() if _ka else True
                    # Always track dynamic status, even for already-registered objects
                    if _hr and not _ik:
                        _dyn_paths.add(_cp)
                    if _cp not in self.usd_objects:
                        self.usd_objects[_cp] = XFormPrim(prim_path=_cp)
                        _reg += 1
                    print(f"[PATCH] Scene object: {{_cp}} (rigid={{_hr}}, kin={{_ik}}, dyn={{_hr and not _ik}})")
                # Merge with existing dynamic prims (don't overwrite post-init result)
                _existing_dyn = getattr(self, '_bp_dynamic_scene_prims', set())
                self._bp_dynamic_scene_prims = _existing_dyn | _dyn_paths
                print(f"[PATCH-v6] dynamic_scene_prims: {{len(self._bp_dynamic_scene_prims)}} total")"""

    if old_code in src:
        src = src.replace(old_code, new_code, 1)
    else:
        print("[PATCH-v6] FAILED: Could not find old register code pattern")
        # Try to at least fix the overwrite by finding the assignment
        overwrite_line = "                self._bp_dynamic_scene_prims = _dyn_paths"
        merge_line = f"""                # BPv6: Merge instead of overwrite
                _existing_dyn = getattr(self, '_bp_dynamic_scene_prims', set())
                self._bp_dynamic_scene_prims = _existing_dyn | _dyn_paths
                print(f"[PATCH-v6] dynamic_scene_prims: {{len(self._bp_dynamic_scene_prims)}} total (merged)")"""
        if overwrite_line in src:
            src = src.replace(overwrite_line, merge_line, 1)
            print("[PATCH-v6] Applied merge-only fix")
        else:
            print("[PATCH-v6] FAILED: Cannot find overwrite line either")
            return False

    with open(CC_PATH, "w") as f:
        f.write(src)

    print("[PATCH-v6] fix_dynamic_prims: APPLIED")
    return True


if __name__ == "__main__":
    sys.exit(0 if apply() else 1)
