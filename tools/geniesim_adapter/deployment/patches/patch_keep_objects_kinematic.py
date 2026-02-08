#!/usr/bin/env python3
"""Patch v7: Keep scene objects kinematic — prevent v3 dynamic restore.

ROOT CAUSE: The v3 patch (patch_register_scene_objects.py) restores scene
objects to dynamic (kinematicEnabled=False) after articulation init. Dynamic
objects then fall through surfaces that lack collision geometry, causing their
positions to drift to unreachable coordinates (y=5-10m). This makes ALL IK
methods fail because the objects are far outside the robot's workspace.

FIX: Override the v3 post-init restore so objects STAY kinematic. They keep
their original USD positions and are reachable by the robot. Object motion
during pick-place is handled via kinematic teleportation (set_object_pose).

This patch must run AFTER v3 (patch_register_scene_objects.py).
"""
import sys
import os

CC_PATH = os.environ.get(
    "GENIESIM_CC_PATH",
    "/opt/geniesim/source/data_collection/server/command_controller.py",
)

MARKER = "# BPv7_keep_kinematic"


def apply():
    with open(CC_PATH) as f:
        src = f.read()

    if MARKER in src:
        print("[PATCH-v7] keep_kinematic: already applied")
        return True

    # The v3 patch injects this block after dof_names assignment:
    #   for _dp_r in getattr(self, '_bp_dynamic_to_restore', []):
    #       ...
    #       _kar.Set(False)        ← THIS restores to dynamic
    #       _bp_dyn_paths.add(_dp_r)
    #
    # We replace _kar.Set(False) with _kar.Set(True) (keep kinematic)
    # and don't add to _bp_dyn_paths (no objects tracked as dynamic).

    old_restore = """                        if _kar:
                            _kar.Set(False)
                            _bp_dyn_paths.add(_dp_r)"""

    new_restore = f"""                        if _kar:
                            {MARKER}
                            _kar.Set(True)  # Keep kinematic — surfaces lack collision
                            # Don't add to _bp_dyn_paths — objects stay kinematic
                            print(f"[PATCH-v7] Keeping {{_dp_r}} kinematic")"""

    if old_restore in src:
        src = src.replace(old_restore, new_restore, 1)
        with open(CC_PATH, "w") as f:
            f.write(src)
        print("[PATCH-v7] keep_kinematic: APPLIED")
        return True

    # Fallback: try to find the restore pattern even if whitespace differs
    # The critical line is `_kar.Set(False)` inside the `_bp_dynamic_to_restore` loop
    import re
    pattern = re.compile(
        r"(for _dp_r in getattr\(self, '_bp_dynamic_to_restore'.*?\n)"
        r"(.*?)"
        r"(_kar\.Set\(False\))",
        re.DOTALL,
    )
    m = pattern.search(src)
    if m:
        # Replace _kar.Set(False) with _kar.Set(True) + marker
        new_line = f"_kar.Set(True)  {MARKER} # Keep kinematic"
        src = src[:m.start(3)] + new_line + src[m.end(3):]

        # Also prevent adding to _bp_dyn_paths
        src = src.replace(
            "_bp_dyn_paths.add(_dp_r)",
            f"pass  {MARKER}  # _bp_dyn_paths.add(_dp_r) — kept kinematic",
            1,
        )

        with open(CC_PATH, "w") as f:
            f.write(src)
        print("[PATCH-v7] keep_kinematic: APPLIED (regex fallback)")
        return True

    # Check if v3 patch is even present
    if "BPv3_pre_play_kinematic" not in src:
        print("[PATCH-v7] SKIPPED: v3 patch not found — nothing to override")
        return True  # Not an error, just no v3 to fix

    print("[PATCH-v7] FAILED: Could not find dynamic restore pattern")
    return False


if __name__ == "__main__":
    sys.exit(0 if apply() else 1)
