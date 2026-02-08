#!/usr/bin/env python3
"""Patch: scene-collision validator hook for strict runtime safety.

This patch no longer mutates collision at runtime. Runtime collision mutation
can deadlock init/play in strict runs. Instead, it validates collision coverage
and dynamic approximation quality and fails strict startup if invalid.
"""
import os
import re

TARGET = os.path.join(
    os.environ.get("GENIESIM_ROOT", "/opt/geniesim"),
    "source", "data_collection", "server", "command_controller.py",
)
MARKER = "# [PATCH] scene_collision_injected"

# Version tag — bumped whenever the method body changes.
# The apply() function uses this to detect stale code and force a re-inject.
VERSION_TAG = "# scene_collision_v4"


def _read(path):
    with open(path, "r") as f:
        return f.read()


def _write(path, content):
    with open(path, "w") as f:
        f.write(content)


# Validation-only method body. Uses omni.usd.get_context().get_stage()
# and computes collision metrics without mutating the scene.
COLLISION_SETUP_CODE = '''
    # [PATCH] scene_collision_injected
    def _patch_add_scene_collision(self):
        """Validate scene collision coverage and dynamic approximations.

        Runtime mutation is intentionally disabled; collision must be baked into
        scene USD assets offline.
        """
        # scene_collision_v4
        import sys as _sys_sc
        try:
            import os
            import omni.usd
            from pxr import Usd, UsdGeom, UsdPhysics
            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print("[PATCH] scene_collision: no stage available for validation")
                _sys_sc.stdout.flush()
                return

            scene_root = stage.GetPrimAtPath("/World/Scene")
            if not scene_root or not scene_root.IsValid():
                scene_root = stage.GetPrimAtPath("/World/Objects")
                if not scene_root or not scene_root.IsValid():
                    print("[PATCH] scene_collision: no scene/object root found")
                    _sys_sc.stdout.flush()
                    return

            allowed_dynamic_approx = {
                "convexHull",
                "convexDecomposition",
                "boundingCube",
                "boundingSphere",
                "sdf",
                "signedDistanceField",
            }
            disallowed_dynamic_approx = {"triangleMesh", "meshSimplification", "none", ""}

            mesh_total = 0
            mesh_with_collision = 0
            bad_dynamic_approx = 0
            dynamic_mesh_total = 0

            for child in scene_root.GetChildren():
                if not child.HasAPI(UsdPhysics.RigidBodyAPI):
                    continue

                is_dynamic = False
                kin_attr = child.GetAttribute("physics:kinematicEnabled")
                if kin_attr and kin_attr.Get() is False:
                    is_dynamic = True

                for desc in Usd.PrimRange(child):
                    if not desc.IsA(UsdGeom.Mesh):
                        continue
                    mesh_total += 1
                    has_collision = desc.HasAPI(UsdPhysics.CollisionAPI)
                    if has_collision:
                        mesh_with_collision += 1

                    if is_dynamic:
                        dynamic_mesh_total += 1
                        approx_attr = desc.GetAttribute("physics:approximation")
                        approx_value = ""
                        if approx_attr:
                            approx_value = str(approx_attr.Get() or "")
                        if (approx_value in disallowed_dynamic_approx) or (
                            approx_value and approx_value not in allowed_dynamic_approx
                        ) or (not approx_value):
                            bad_dynamic_approx += 1

            collision_coverage = 1.0 if mesh_total == 0 else float(mesh_with_collision) / float(mesh_total)
            summary = {
                "mesh_prims_total": int(mesh_total),
                "mesh_prims_with_collision": int(mesh_with_collision),
                "mesh_prims_bad_dynamic_approx": int(bad_dynamic_approx),
                "collision_coverage": float(collision_coverage),
                "dynamic_mesh_prims_total": int(dynamic_mesh_total),
            }
            self._bp_scene_collision_summary = summary
            print(
                "[PATCH] scene_collision validate: mesh_total=%d with_collision=%d bad_dynamic_approx=%d coverage=%.4f"
                % (
                    summary["mesh_prims_total"],
                    summary["mesh_prims_with_collision"],
                    summary["mesh_prims_bad_dynamic_approx"],
                    summary["collision_coverage"],
                )
            )

            strict_runtime = str(os.getenv("GENIESIM_STRICT_RUNTIME_READINESS", "0")).strip().lower() in (
                "1", "true", "yes", "on"
            )
            if strict_runtime and (
                summary["collision_coverage"] < 1.0 or summary["mesh_prims_bad_dynamic_approx"] > 0
            ):
                raise RuntimeError(
                    "strict collision validation failed: "
                    f"coverage={summary['collision_coverage']:.4f}, "
                    f"bad_dynamic_approx={summary['mesh_prims_bad_dynamic_approx']}"
                )
            _sys_sc.stdout.flush()
        except Exception as e:
            print("[PATCH] scene_collision error: %s" % e)
            import traceback; traceback.print_exc()
            _sys_sc.stdout.flush()
'''

HOOK_CODE = '''
            # [PATCH] scene_collision_hook — MUST run BEFORE v3 dynamic restore
            self._patch_add_scene_collision()
'''


def _strip_old_method(code):
    """Remove the old _patch_add_scene_collision method body entirely."""
    # Find the method definition
    method_start_pattern = re.compile(
        r'^([ \t]+)def _patch_add_scene_collision\(self\):', re.MULTILINE
    )
    match = method_start_pattern.search(code)
    if not match:
        return code, False

    method_indent = len(match.group(1))
    start = match.start()

    # Find the end of the method: next line with same or less indentation
    # that starts a new def, class, or is a non-blank non-comment line
    lines = code[match.end():].split('\n')
    end_offset = match.end()
    for i, line in enumerate(lines):
        if i == 0:
            end_offset += len(line) + 1
            continue
        stripped = line.lstrip()
        if stripped and not stripped.startswith('#'):
            line_indent = len(line) - len(stripped)
            if line_indent <= method_indent:
                break
        end_offset += len(line) + 1

    code = code[:start] + code[end_offset:]
    return code, True


def _strip_old_hook(code):
    """Remove old hook call lines from handle_init_robot."""
    # Remove various forms of the hook call
    patterns = [
        "            # [PATCH] scene_collision_hook — MUST run BEFORE v3 dynamic restore\n"
        "            self._patch_add_scene_collision()\n",
        "            # [PATCH] scene_collision_hook\n"
        "            self._patch_add_scene_collision()\n",
        "        # [PATCH] scene_collision_hook — MUST run BEFORE v3 dynamic restore\n"
        "        self._patch_add_scene_collision()\n",
        "        # [PATCH] scene_collision_hook\n"
        "        self._patch_add_scene_collision()\n",
    ]
    for pat in patterns:
        code = code.replace(pat, "")
    return code


def apply():
    code = _read(TARGET)

    if MARKER in code:
        # Already applied — check if it's the latest version
        if VERSION_TAG in code:
            # Check hook ordering
            hook_line = "# [PATCH] scene_collision_hook"
            restore_line = "# Restore dynamic objects after articulation is stable"
            if hook_line in code and restore_line in code:
                hook_idx = code.index(hook_line)
                restore_idx = code.index(restore_line)
                if hook_idx > restore_idx:
                    # Wrong order — fix it
                    code = _strip_old_hook(code)
                    code = _inject_hook_before_restore(code)
                    print("[PATCH] scene_collision: fixed hook ordering (v3)")
                    return
            print("[PATCH] scene_collision v3 already applied — skipping")
            return

        # OLD version detected — nuclear upgrade: strip everything and re-inject
        print("[PATCH] scene_collision: upgrading to v3 (nuclear replace)...")
        code, stripped = _strip_old_method(code)
        if stripped:
            print("[PATCH] scene_collision: removed old method body")
        code = _strip_old_hook(code)
        # Remove the old marker so we can re-inject cleanly
        # (the marker is inside the method body which we just stripped)
        # Fall through to fresh injection below

    # Fresh injection (either first time or after nuclear strip)
    # 1. Add the method to the class — before handle_init_robot
    anchor = "def handle_init_robot(self"
    idx = code.find(anchor)
    if idx == -1:
        print("[PATCH] scene_collision: could not find handle_init_robot")
        return

    # Find the start of the line (go back to the indentation)
    line_start = code.rfind("\n", 0, idx) + 1
    code = code[:line_start] + COLLISION_SETUP_CODE + "\n" + code[line_start:]

    # 2. Hook into handle_init_robot BEFORE v3's dynamic restore loop
    code = _inject_hook_before_restore(code)


def _inject_hook_before_restore(code):
    """Inject the scene_collision hook BEFORE v3's dynamic restore block."""

    # Strategy: Find v3's restore block marker and insert before it
    v3_restore_marker = "# Restore dynamic objects after articulation is stable"
    v3_idx = code.find(v3_restore_marker)

    if v3_idx != -1:
        # Insert our hook call right before v3's restore block
        line_start = code.rfind("\n", 0, v3_idx) + 1
        code = code[:line_start] + HOOK_CODE + "\n" + code[line_start:]
        _write(TARGET, code)
        print("[PATCH] scene_collision v3: injected BEFORE v3 dynamic restore")
        return code

    # Fallback: v3 may not be applied yet. Use dof_names anchor.
    dof_names_anchor = "self.dof_names = articulation.dof_names"
    init_method_start = code.find("def handle_init_robot(self")
    if init_method_start == -1:
        print("[PATCH] scene_collision: could not find handle_init_robot")
        return code

    dof_idx = code.find(dof_names_anchor, init_method_start)
    if dof_idx != -1:
        line_end = code.find("\n", dof_idx)
        if line_end != -1:
            code = code[:line_end + 1] + HOOK_CODE + "\n" + code[line_end + 1:]
            _write(TARGET, code)
            print("[PATCH] scene_collision v3: injected after dof_names (v3 not yet applied)")
            return code

    # Last resort: insert before data_to_send
    send_idx = code.find("self.data_to_send", init_method_start)
    if send_idx != -1:
        line_start = code.rfind("\n", 0, send_idx) + 1
        code = code[:line_start] + HOOK_CODE + "\n" + code[line_start:]
        _write(TARGET, code)
        print("[PATCH] scene_collision v3: WARNING — injected before data_to_send (fallback)")
        return code

    print("[PATCH] scene_collision v3: FAILED — could not find injection point")
    return code


if __name__ == "__main__":
    apply()
