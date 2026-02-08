#!/usr/bin/env python3
"""Patch: Fix dynamic object teleport snapback by deferring kinematic→dynamic restore.

The current _set_object_pose does:
  kinematic=True → set xformOp → kinematic=False (immediately)

This causes PhysX to snap the object back to its pre-teleport position because
PhysX hasn't had a chance to register the new xformOp while kinematic.

Fix: Remove the immediate kinematic=False and instead schedule a deferred
restore that runs after N physics steps via on_physics_step.

This patch is designed to work on the ALREADY-PATCHED server file (not fresh).
"""
import re
import sys
import os

CC_PATH = os.environ.get(
    "GENIESIM_CC_PATH",
    "/opt/geniesim/source/data_collection/server/command_controller.py",
)

MARKER = "# BPv4_deferred_dynamic_restore"
DEFERRED_STEPS = 5  # ~83ms at 60Hz physics


def apply():
    with open(CC_PATH) as f:
        src = f.read()

    if MARKER in src:
        print("[PATCH-v4] deferred_dynamic_restore: already applied")
        return True

    changed = False

    # ── Fix 1: Replace immediate kinematic=False in _set_object_pose ──
    # Current code has either:
    #   A) _kin_a.Set(False)  # back to dynamic
    #   B) Some variation with try/except
    #
    # We need to find the pattern within _set_object_pose and replace it.

    # Pattern A: direct immediate restore
    old_restore_a = """                if _is_dyn:
                    # Restore dynamic after a few physics steps
                    try:
                        _kin_a.Set(False)  # back to dynamic
                    except Exception:
                        pass"""

    new_restore = f"""                if _is_dyn:
                    {MARKER}
                    # Deferred restore: keep kinematic for {DEFERRED_STEPS} physics steps
                    # so PhysX fully registers the new xformOp position.
                    _deferred = getattr(self, "_bp_deferred_dynamic_restore", None)
                    if _deferred is None:
                        self._bp_deferred_dynamic_restore = {{}}
                        _deferred = self._bp_deferred_dynamic_restore
                    _deferred[pose["prim_path"]] = {DEFERRED_STEPS}
                    print(f'[DEFERRED] Scheduled {{pose["prim_path"]}} for restore in {DEFERRED_STEPS} steps')"""

    if old_restore_a in src:
        src = src.replace(old_restore_a, new_restore, 1)
        changed = True
        print("[PATCH-v4] _set_object_pose: replaced immediate restore with deferred")
    else:
        # Try alternative patterns
        # Pattern B: just the _kin_a.Set(False) line within an if _is_dyn block
        # Use regex to be more flexible
        pat = re.compile(
            r"(                if _is_dyn:\s*\n)"
            r".*?"  # any comment/code
            r"(_kin_a\.Set\(False\).*?\n)"
            r"(.*?(?:except|pass).*?\n)?",
            re.DOTALL,
        )
        m = pat.search(src)
        if m:
            src = src[:m.start()] + new_restore + "\n" + src[m.end():]
            changed = True
            print("[PATCH-v4] _set_object_pose: replaced immediate restore (regex match)")
        else:
            print("[PATCH-v4] WARNING: Could not find immediate kinematic restore to replace")
            print("[PATCH-v4] Attempting to locate _set_object_pose for manual inspection")
            # Try to at least add deferred scheduling after orient_attr.Set
            orient_pat = re.compile(
                r"(orient_attr\.Set\(quat_type\(\*rotation_data\)\))\s*\n"
                r"(\s+if _is_dyn:\s*\n)",
            )
            m2 = orient_pat.search(src)
            if m2:
                # Found the pattern - just need to ensure the deferred code replaces the immediate restore
                print("[PATCH-v4] Found orient_attr + _is_dyn block - please inspect manually")

    # ── Fix 2: Inject deferred restore hook into on_physics_step ──
    if "_bp_deferred_dynamic_restore" not in src.split("on_physics_step")[1] if "on_physics_step" in src else True:
        # Need to inject the deferred restore processor into on_physics_step
        # Find "def on_physics_step(self" and inject after the first line
        ops_pat = re.compile(
            r"^([ \t]+)(def on_physics_step\(self.*?\):\s*\n)",
            re.MULTILINE,
        )
        ops_match = ops_pat.search(src)
        if ops_match:
            method_indent = ops_match.group(1)
            bi = method_indent + "    "  # body indent

            # Find insertion point - after the method def line
            insert_pos = ops_match.end()

            # Skip past any "with self._timing_context" wrapper
            with_pat = re.compile(r"^[ \t]+with self\._timing_context.*?:\s*\n", re.MULTILINE)
            with_match = with_pat.match(src[insert_pos:])
            if with_match:
                insert_pos += with_match.end()
                bi = method_indent + "        "  # extra indent inside with block

            hook_code = f"""{bi}{MARKER}_hook
{bi}# Process deferred kinematic -> dynamic restores for teleported objects.
{bi}try:
{bi}    _bp_ddr = getattr(self, '_bp_deferred_dynamic_restore', None)
{bi}    if _bp_ddr:
{bi}        _bp_ddr_done = []
{bi}        for _bp_ddr_path, _bp_ddr_count in list(_bp_ddr.items()):
{bi}            _bp_ddr_count -= 1
{bi}            if _bp_ddr_count <= 0:
{bi}                try:
{bi}                    import omni.usd
{bi}                    from pxr import UsdPhysics
{bi}                    _bp_ddr_stage = omni.usd.get_context().get_stage()
{bi}                    if _bp_ddr_stage:
{bi}                        _bp_ddr_prim = _bp_ddr_stage.GetPrimAtPath(_bp_ddr_path)
{bi}                        if _bp_ddr_prim and _bp_ddr_prim.IsValid():
{bi}                            UsdPhysics.RigidBodyAPI(_bp_ddr_prim).GetKinematicEnabledAttr().Set(False)
{bi}                            print(f'[DEFERRED_RESTORE] {{_bp_ddr_path}} restored to dynamic')
{bi}                except Exception as _bp_ddr_err:
{bi}                    print(f'[DEFERRED_RESTORE] Error restoring {{_bp_ddr_path}}: {{_bp_ddr_err}}')
{bi}                _bp_ddr_done.append(_bp_ddr_path)
{bi}            else:
{bi}                _bp_ddr[_bp_ddr_path] = _bp_ddr_count
{bi}        for _bp_ddr_d in _bp_ddr_done:
{bi}            _bp_ddr.pop(_bp_ddr_d, None)
{bi}except Exception as _bp_ddr_outer_err:
{bi}    if not getattr(self, '_bp_ddr_err_logged', False):
{bi}        print(f'[DEFERRED_RESTORE] Hook error: {{_bp_ddr_outer_err}}')
{bi}        self._bp_ddr_err_logged = True
"""

            src = src[:insert_pos] + hook_code + src[insert_pos:]
            changed = True
            print("[PATCH-v4] on_physics_step: deferred restore hook injected")
        else:
            print("[PATCH-v4] WARNING: on_physics_step not found")

    if changed:
        with open(CC_PATH, "w") as f:
            f.write(src)
        print("[PATCH-v4] deferred_dynamic_restore: ALL FIXES APPLIED")
    else:
        print("[PATCH-v4] WARNING: no changes made")

    return changed


if __name__ == "__main__":
    sys.exit(0 if apply() else 1)
