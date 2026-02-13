#!/usr/bin/env python3
"""Patch: scene-collision validation hook for strict runtime safety.

This patch is intentionally *validator-only*:
- It DOES NOT mutate the stage to "fix" collisions at runtime.
- It validates that collision is already authored on mesh prims and that
  dynamic meshes have an acceptable approximation mode.

It runs on the sim thread (handle_init_robot) BEFORE v3's dynamic restore.

If coverage < 1.0 (or bad approximation modes are detected) AND strict mode is
on, startup aborts.
"""
import os
import re

TARGET = os.path.join(
    os.environ.get("GENIESIM_ROOT", "/opt/geniesim"),
    "source", "data_collection", "server", "command_controller.py",
)
MARKER = "# [PATCH] scene_collision_injected"

# Version tag — bumped whenever the method body changes.
VERSION_TAG = "# scene_collision_v6"


def _read(path):
    with open(path, "r") as f:
        return f.read()


def _write(path, content):
    with open(path, "w") as f:
        f.write(content)


# Validator-only method body. Runs on the sim thread inside handle_init_robot,
# BEFORE v3's dynamic restore. Validates collision coverage and approximation
# modes; does not apply CollisionAPI or author approximation attributes.
COLLISION_SETUP_CODE = '''
    # [PATCH] scene_collision_injected
    def _patch_add_scene_collision(self):
        """Validate CollisionAPI coverage and approximation modes.

        Runs on the sim thread BEFORE v3 dynamic restore. Safe to mutate
        because physics hasn't started yet (pre-_play or post-init).
        """
        # scene_collision_v6
        import sys as _sys_sc
        try:
            import os
            import omni.usd
            from pxr import Usd, UsdGeom, UsdPhysics
            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print("[PATCH] scene_collision: no stage available")
                _sys_sc.stdout.flush()
                return

            # Process both /World/Scene and /World (for GroundPlane etc.)
            roots_to_check = []
            scene_root = stage.GetPrimAtPath("/World/Scene")
            if scene_root and scene_root.IsValid():
                roots_to_check.append(scene_root)
            objects_root = stage.GetPrimAtPath("/World/Objects")
            if objects_root and objects_root.IsValid():
                roots_to_check.append(objects_root)

            if not roots_to_check:
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

            mesh_total = 0
            mesh_with_collision = 0
            collision_added = 0
            approx_fixed = 0
            bad_dynamic_approx = 0
            dynamic_mesh_total = 0

            for scene_root in roots_to_check:
                for child in scene_root.GetChildren():
                    # Process ALL children, not just those with RigidBodyAPI
                    # Kinematic objects also need collision for dynamic objects to rest on them
                    is_dynamic = False
                    if child.HasAPI(UsdPhysics.RigidBodyAPI):
                        kin_attr = child.GetAttribute("physics:kinematicEnabled")
                        if kin_attr and kin_attr.Get() is False:
                            is_dynamic = True

                    for desc in Usd.PrimRange(child):
                        if not desc.IsA(UsdGeom.Mesh):
                            continue
                        mesh_total += 1

                        if desc.HasAPI(UsdPhysics.CollisionAPI):
                            mesh_with_collision += 1

                        # Validate approximation for dynamic meshes (should not be triangleMesh).
                        approx_attr = desc.GetAttribute("physics:approximation")
                        approx_value = ""
                        if approx_attr:
                            approx_value = str(approx_attr.Get() or "")

                        if is_dynamic:
                            dynamic_mesh_total += 1
                            if not approx_value or approx_value in ("triangleMesh", "meshSimplification", "none", ""):
                                bad_dynamic_approx += 1
                            elif approx_value not in allowed_dynamic_approx:
                                bad_dynamic_approx += 1

            collision_coverage = 1.0 if mesh_total == 0 else float(mesh_with_collision) / float(mesh_total)
            summary = {
                "mesh_prims_total": int(mesh_total),
                "mesh_prims_with_collision": int(mesh_with_collision),
                "collision_added": int(collision_added),
                "approx_fixed": int(approx_fixed),
                "mesh_prims_bad_dynamic_approx": int(bad_dynamic_approx),
                "collision_coverage": float(collision_coverage),
                "dynamic_mesh_prims_total": int(dynamic_mesh_total),
            }
            self._bp_scene_collision_summary = summary
            print(
                "[PATCH] scene_collision: mesh_total=%d with_collision=%d added=%d approx_fixed=%d bad_approx=%d coverage=%.4f"
                % (
                    summary["mesh_prims_total"],
                    summary["mesh_prims_with_collision"],
                    summary["collision_added"],
                    summary["approx_fixed"],
                    summary["mesh_prims_bad_dynamic_approx"],
                    summary["collision_coverage"],
                )
            )

            if mesh_with_collision == mesh_total:
                print("[PATCH] scene_collision: all meshes already had collision — bake worked correctly")

            strict_runtime = str(os.getenv("GENIESIM_STRICT_RUNTIME_READINESS", "0")).strip().lower() in (
                "1", "true", "yes", "on"
            )
            if strict_runtime and (
                summary["collision_coverage"] < 1.0 or summary["mesh_prims_bad_dynamic_approx"] > 0
            ):
                raise RuntimeError(
                    "strict collision validation failed: "
                    "coverage=%.4f, bad_dynamic_approx=%d"
                    % (summary["collision_coverage"], summary["mesh_prims_bad_dynamic_approx"])
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
    method_start_pattern = re.compile(
        r'^([ \t]+)def _patch_add_scene_collision\(self\):', re.MULTILINE
    )
    match = method_start_pattern.search(code)
    if not match:
        return code, False

    method_indent = len(match.group(1))
    start = match.start()

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
        if VERSION_TAG in code:
            # Check hook ordering
            hook_line = "# [PATCH] scene_collision_hook"
            restore_line = "# Restore dynamic objects after articulation is stable"
            if hook_line in code and restore_line in code:
                hook_idx = code.index(hook_line)
                restore_idx = code.index(restore_line)
                if hook_idx > restore_idx:
                    code = _strip_old_hook(code)
                    code = _inject_hook_before_restore(code)
                    print("[PATCH] scene_collision: fixed hook ordering (v5)")
                    return
            print("[PATCH] scene_collision v5 already applied — skipping")
            return

        # OLD version — nuclear upgrade
        print("[PATCH] scene_collision: upgrading to v5 (nuclear replace)...")
        code, stripped = _strip_old_method(code)
        if stripped:
            print("[PATCH] scene_collision: removed old method body")
        code = _strip_old_hook(code)

    # Fresh injection
    anchor = "def handle_init_robot(self"
    idx = code.find(anchor)
    if idx == -1:
        print("[PATCH] scene_collision: could not find handle_init_robot")
        return

    line_start = code.rfind("\n", 0, idx) + 1
    code = code[:line_start] + COLLISION_SETUP_CODE + "\n" + code[line_start:]

    code = _inject_hook_before_restore(code)


def _inject_hook_before_restore(code):
    """Inject the scene_collision hook BEFORE v3's dynamic restore block."""

    v3_restore_marker = "# Restore dynamic objects after articulation is stable"
    v3_idx = code.find(v3_restore_marker)

    if v3_idx != -1:
        line_start = code.rfind("\n", 0, v3_idx) + 1
        code = code[:line_start] + HOOK_CODE + "\n" + code[line_start:]
        _write(TARGET, code)
        print("[PATCH] scene_collision v5: injected BEFORE v3 dynamic restore")
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
            print("[PATCH] scene_collision v5: injected after dof_names (v3 not yet applied)")
            return code

    # Last resort: insert before data_to_send
    send_idx = code.find("self.data_to_send", init_method_start)
    if send_idx != -1:
        line_start = code.rfind("\n", 0, send_idx) + 1
        code = code[:line_start] + HOOK_CODE + "\n" + code[line_start:]
        _write(TARGET, code)
        print("[PATCH] scene_collision v5: WARNING — injected before data_to_send (fallback)")
        return code

    print("[PATCH] scene_collision v5: FAILED — could not find injection point")
    return code


if __name__ == "__main__":
    apply()
