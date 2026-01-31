#!/usr/bin/env python3
"""
Patch the Genie Sim server to log USD stage contents after init_robot.

This helps diagnose object pose issues (objects returning zeros) by showing
exactly which prims are loaded in the server's USD stage after scene loading.

Usage (inside Docker build):
    python3 /tmp/patches/patch_stage_diagnostics.py

The script is idempotent — re-running it on an already-patched file is a no-op.
"""
import os
import re
import sys
import textwrap

GENIESIM_ROOT = os.environ.get("GENIESIM_ROOT", "/opt/geniesim")
COMMAND_CONTROLLER = os.path.join(
    GENIESIM_ROOT,
    "source", "data_collection", "server", "command_controller.py",
)

DIAGNOSTICS_CODE = textwrap.dedent("""\

    # --- BEGIN BlueprintPipeline stage diagnostics patch ---
    def _bp_log_stage_contents(self):
        \"\"\"Log all prims in the USD stage for debugging object pose issues.\"\"\"
        try:
            import omni.usd
            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print("[DIAG] USD stage is None — scene not loaded")
                return
            prim_count = 0
            camera_prims = []
            mesh_prims = []
            xform_prims = []
            from pxr import UsdGeom
            for prim in stage.Traverse():
                prim_count += 1
                path = str(prim.GetPath())
                if prim.IsA(UsdGeom.Camera):
                    camera_prims.append(path)
                elif prim.IsA(UsdGeom.Mesh):
                    # Only log top-level meshes (depth <= 4) to avoid noise
                    if path.count("/") <= 4:
                        mesh_prims.append(path)
                elif prim.IsA(UsdGeom.Xform):
                    if path.count("/") <= 3:
                        xform_prims.append(path)
            print(f"[DIAG] USD stage: {prim_count} total prims")
            print(f"[DIAG] Cameras ({len(camera_prims)}): {camera_prims}")
            print(f"[DIAG] Top-level Xforms ({len(xform_prims)}): {xform_prims}")
            if mesh_prims:
                print(f"[DIAG] Top-level Meshes ({len(mesh_prims)}): {mesh_prims[:20]}")
        except Exception as e:
            print(f"[DIAG] Stage diagnostics failed: {e}")
    # --- END BlueprintPipeline stage diagnostics patch ---
""")

# Call the diagnostics after init_robot completes
INIT_HOOK = textwrap.dedent("""\
        # BlueprintPipeline stage diagnostics hook
        try:
            self._bp_log_stage_contents()
        except Exception as _diag_e:
            print(f"[DIAG] Stage diagnostics hook failed: {_diag_e}")
""")

PATCH_MARKER = "BlueprintPipeline stage diagnostics patch"


def patch_file():
    if not os.path.isfile(COMMAND_CONTROLLER):
        print(f"[PATCH] command_controller.py not found at {COMMAND_CONTROLLER}")
        print("[PATCH] Skipping stage diagnostics patch (server source not available)")
        sys.exit(0)

    with open(COMMAND_CONTROLLER, "r") as f:
        content = f.read()

    if PATCH_MARKER in content:
        print("[PATCH] Stage diagnostics already patched — skipping")
        sys.exit(0)

    # 1. Add the diagnostics method to the class
    method_indent = "    "
    m = re.search(r"^([ \t]+)def \w+\(self", content, re.MULTILINE)
    if m:
        method_indent = m.group(1)

    indented = "\n".join(
        (method_indent + line) if line.strip() else line
        for line in DIAGNOSTICS_CODE.splitlines()
    ) + "\n"

    patched = content.rstrip() + "\n\n" + indented

    # 2. Hook into init_robot handler to call diagnostics after robot init.
    # Look for the method that handles INIT_ROBOT command completion.
    # The init_robot handler typically ends with notify_all() — inject right before.
    init_robot_pattern = re.compile(
        r"(Command\.INIT_ROBOT.*?)(self\.condition\.notify_all\(\))",
        re.DOTALL,
    )
    match = init_robot_pattern.search(patched)
    if match:
        insert_pos = match.start(2)
        patched = patched[:insert_pos] + INIT_HOOK + patched[insert_pos:]
        print("[PATCH] Injected stage diagnostics hook after INIT_ROBOT")
    else:
        print("[PATCH] WARNING: Could not find INIT_ROBOT handler — diagnostics method added but not auto-called")
        print("[PATCH] Server will still log stage contents if _bp_log_stage_contents() is called manually")

    with open(COMMAND_CONTROLLER, "w") as f:
        f.write(patched)

    print(f"[PATCH] Successfully patched {COMMAND_CONTROLLER} with stage diagnostics")


if __name__ == "__main__":
    patch_file()
