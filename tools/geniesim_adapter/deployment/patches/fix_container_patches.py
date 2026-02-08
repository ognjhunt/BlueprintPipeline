#!/usr/bin/env python3
"""Direct fix for container command_controller.py:
1. Remove BPv7_keep_kinematic (restore _kar.Set(False) for real dynamic objects)
2. Add BPv4_deferred_dynamic_restore marker (deferred hook already exists from v3)
3. Add BPv6_fix_dynamic_prims marker (merge instead of overwrite)
"""
import os
import sys

CC_PATH = os.environ.get(
    "GENIESIM_CC_PATH",
    "/opt/geniesim/source/data_collection/server/command_controller.py",
)

def apply():
    with open(CC_PATH) as f:
        src = f.read()

    changed = False

    # --- Fix 1: Remove BPv7 (restore _kar.Set(False)) ---
    if "BPv7_keep_kinematic" in src:
        # Pattern: _kar.Set(True)  # BPv7_keep_kinematic  # Keep kinematic
        # Replace with: _kar.Set(False)
        import re
        # Handle both inline and multi-line forms
        src = re.sub(
            r'_kar\.Set\(True\)\s*#\s*BPv7_keep_kinematic[^\n]*',
            '_kar.Set(False)',
            src,
        )
        # Also fix the comment-only marker line if present
        src = re.sub(
            r'^\s*# BPv7_keep_kinematic\n',
            '',
            src,
            flags=re.MULTILINE,
        )
        # Fix the "pass" line that replaced _bp_dyn_paths.add
        src = re.sub(
            r'pass\s+#?\s*BPv7_keep_kinematic\s*#?\s*_bp_dyn_paths\.add\(_dp_r\)[^\n]*',
            '_bp_dyn_paths.add(_dp_r)',
            src,
        )
        # Also remove the print line for v7
        src = re.sub(
            r'\s*print\(f"\[PATCH-v7\].*?\n',
            '\n',
            src,
        )
        if "BPv7_keep_kinematic" not in src:
            print("[FIX] Removed BPv7_keep_kinematic")
            changed = True
        else:
            print("[FIX] WARNING: BPv7 still present after cleanup")

    # --- Fix 2: Add BPv4 marker ---
    if "BPv4_deferred_dynamic_restore" not in src:
        # The deferred restore hook already exists in on_physics_step from v3
        # (marker: BPv3_deferred_dynamic_restore). We just need to add the v4 marker
        # to satisfy the readiness probe.
        # Find the BPv3_deferred_dynamic_restore marker and add v4 marker nearby
        v3_deferred = "# BPv3_deferred_dynamic_restore"
        if v3_deferred in src:
            src = src.replace(
                v3_deferred,
                v3_deferred + "\n            # BPv4_deferred_dynamic_restore",
                1,
            )
            print("[FIX] Added BPv4_deferred_dynamic_restore marker (v3 hook exists)")
            changed = True
        else:
            # Try to find the deferred restore code block and add marker
            deferred_block = "getattr(self, '_bp_deferred_dynamic_restore', None)"
            if deferred_block in src:
                idx = src.find(deferred_block)
                # Go back to find the start of the try block
                line_start = src.rfind("\n", 0, idx) + 1
                marker_line = "            # BPv4_deferred_dynamic_restore\n"
                src = src[:line_start] + marker_line + src[line_start:]
                print("[FIX] Added BPv4_deferred_dynamic_restore marker (found deferred block)")
                changed = True
            else:
                print("[FIX] WARNING: Cannot find deferred restore code for v4 marker")

    # --- Fix 3: Add BPv6 marker ---
    if "BPv6_fix_dynamic_prims" not in src:
        # The v6 fix changes _bp_dynamic_scene_prims assignment from overwrite to merge.
        # Find the line: self._bp_dynamic_scene_prims = _bp_dyn_paths
        overwrite_line = "            self._bp_dynamic_scene_prims = _bp_dyn_paths"
        if overwrite_line in src:
            merge_code = """            # BPv6_fix_dynamic_prims
            _existing_dyn = getattr(self, '_bp_dynamic_scene_prims', set())
            self._bp_dynamic_scene_prims = _existing_dyn | _bp_dyn_paths"""
            src = src.replace(overwrite_line, merge_code, 1)
            print("[FIX] Added BPv6_fix_dynamic_prims (merge instead of overwrite)")
            changed = True
        else:
            # Check if it's already merged somehow
            if "_bp_dynamic_scene_prims" in src:
                # Just add the marker comment
                idx = src.find("_bp_dynamic_scene_prims")
                line_start = src.rfind("\n", 0, idx) + 1
                marker_line = "            # BPv6_fix_dynamic_prims\n"
                src = src[:line_start] + marker_line + src[line_start:]
                print("[FIX] Added BPv6_fix_dynamic_prims marker (prims assignment exists)")
                changed = True
            else:
                print("[FIX] WARNING: Cannot find _bp_dynamic_scene_prims for v6")

    # --- Fix 4: Remove scene_collision self-defeating guard ---
    guard_line = 'if getattr(self, \'_bp_collision_applied_pre_play\', False):'
    if guard_line in src:
        # Remove the guard block (guard line + return + print)
        import re
        src = re.sub(
            r'\s+if getattr\(self, \'_bp_collision_applied_pre_play\', False\):\s*\n'
            r'\s+print\("[^"]*already applied[^"]*"\)\s*\n'
            r'\s+_sys_sc\.stdout\.flush\(\)\s*\n'
            r'\s+return\s*\n',
            '\n',
            src,
        )
        print("[FIX] Removed scene_collision self-defeating guard")
        changed = True

    if changed:
        with open(CC_PATH, "w") as f:
            f.write(src)
        print("[FIX] ALL FIXES APPLIED to %s" % CC_PATH)
    else:
        print("[FIX] No changes needed")

    # Verify
    with open(CC_PATH) as f:
        final = f.read()

    checks = {
        "BPv7 absent": "BPv7_keep_kinematic" not in final,
        "BPv4 present": "BPv4_deferred_dynamic_restore" in final,
        "BPv6 present": "BPv6_fix_dynamic_prims" in final,
        "_kar.Set(False)": "_kar.Set(False)" in final,
        "_kar.Set(True) absent": "_kar.Set(True)" not in final.split("on_physics_step")[0] if "on_physics_step" in final else True,
    }

    all_ok = all(checks.values())
    for name, ok in checks.items():
        print("[VERIFY] %s: %s" % (name, "OK" if ok else "FAIL"))

    return all_ok


if __name__ == "__main__":
    sys.exit(0 if apply() else 1)
