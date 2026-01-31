#!/usr/bin/env python3
"""
Patch the Genie Sim server's command_controller.py to handle GET_CAMERA_DATA.

The upstream server dispatches commands via on_command_step() but does not
handle Command value 1 (GET_CAMERA_DATA), causing a ValueError that crashes
the gRPC server thread.  This patch adds a handler that renders the current
frame using Isaac Sim's Replicator API and returns RGB + depth images.

Usage (inside Docker build):
    python3 /tmp/patches/patch_camera_handler.py

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

# The camera handler code to inject.  It captures the current frame via
# Isaac Sim's Replicator (omni.replicator) or the lower-level SyntheticData
# helper, whichever is available.
CAMERA_HANDLER = textwrap.dedent("""\

    # --- BEGIN BlueprintPipeline camera patch ---
    def handle_get_camera_data(self):
        \"\"\"Handle GET_CAMERA_DATA (Command=1) — render current frame.\"\"\"
        import numpy as np

        command_data = self.data if self.data else {}
        camera_prim_path = ""
        render_depth = True
        if isinstance(command_data, dict):
            camera_prim_path = command_data.get("camera_prim", "")
            render_depth = command_data.get("render_depth", True)
        elif hasattr(command_data, "serial_no"):
            camera_prim_path = command_data.serial_no

        result = {}

        try:
            # Try Replicator first (Isaac Sim 4.x preferred API)
            import omni.replicator.core as rep
            from omni.isaac.core.utils.stage import get_stage

            # If no specific camera requested, use the first available
            if not camera_prim_path:
                stage = get_stage()
                from pxr import UsdGeom
                for prim in stage.Traverse():
                    if prim.IsA(UsdGeom.Camera):
                        camera_prim_path = str(prim.GetPath())
                        break

            if not camera_prim_path:
                camera_prim_path = "/OmniverseKit_Persp"

            # Create a render product for this camera
            import os as _os
            _cam_res_str = _os.environ.get("CAMERA_RESOLUTION", "1280x720")
            try:
                _cam_w, _cam_h = (int(x) for x in _cam_res_str.split("x"))
            except (ValueError, TypeError):
                _cam_w, _cam_h = 1280, 720
            rp = rep.create.render_product(camera_prim_path, (_cam_w, _cam_h))

            # Attach annotators
            rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
            rgb_annot.attach([rp])

            depth_annot = None
            if render_depth:
                depth_annot = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
                depth_annot.attach([rp])

            # Trigger a single render
            rep.orchestrator.step()

            # Read back data
            rgb_data = rgb_annot.get_data()
            if rgb_data is not None:
                if hasattr(rgb_data, "numpy"):
                    rgb_data = rgb_data.numpy()
                rgb_array = np.asarray(rgb_data, dtype=np.uint8)
                h, w = rgb_array.shape[:2]
                result["rgb"] = {
                    "width": int(w),
                    "height": int(h),
                    "data": rgb_array.tobytes(),
                    "encoding": "raw_rgb",
                    "channels": int(rgb_array.shape[2]) if rgb_array.ndim == 3 else 1,
                }
                result["camera_info"] = {"width": int(w), "height": int(h)}

            if depth_annot is not None:
                depth_data = depth_annot.get_data()
                if depth_data is not None:
                    if hasattr(depth_data, "numpy"):
                        depth_data = depth_data.numpy()
                    depth_array = np.asarray(depth_data, dtype=np.float32)
                    h, w = depth_array.shape[:2]
                    result["depth"] = {
                        "width": int(w),
                        "height": int(h),
                        "data": depth_array.tobytes(),
                        "encoding": "raw_float32",
                    }

            # Clean up render product
            rgb_annot.detach()
            if depth_annot:
                depth_annot.detach()
            rp.destroy()

        except ImportError:
            # Fallback: try SyntheticData helper (older Isaac Sim versions)
            try:
                from omni.isaac.synthetic_utils import SyntheticDataHelper
                sd = SyntheticDataHelper()
                gt = sd.get_groundtruth(
                    ["rgb", "depthLinear"] if render_depth else ["rgb"],
                    viewport=None,
                )
                if "rgb" in gt:
                    rgb_array = np.asarray(gt["rgb"], dtype=np.uint8)
                    h, w = rgb_array.shape[:2]
                    result["rgb"] = {
                        "width": int(w),
                        "height": int(h),
                        "data": rgb_array.tobytes(),
                        "encoding": "raw_rgb",
                        "channels": int(rgb_array.shape[2]) if rgb_array.ndim == 3 else 1,
                    }
                    result["camera_info"] = {"width": int(w), "height": int(h)}
                if render_depth and "depthLinear" in gt:
                    depth_array = np.asarray(gt["depthLinear"], dtype=np.float32)
                    h, w = depth_array.shape[:2]
                    result["depth"] = {
                        "width": int(w),
                        "height": int(h),
                        "data": depth_array.tobytes(),
                        "encoding": "raw_float32",
                    }
            except Exception as e:
                print(f"[PATCH] Camera fallback also failed: {e}")

        except Exception as e:
            print(f"[PATCH] Camera capture failed: {e}")

        self.data_to_send = result
    # --- END BlueprintPipeline camera patch ---
""")

PATCH_MARKER = "BlueprintPipeline camera patch"


def patch_file():
    if not os.path.isfile(COMMAND_CONTROLLER):
        print(f"[PATCH] command_controller.py not found at {COMMAND_CONTROLLER}")
        print("[PATCH] Skipping camera patch (server source not available)")
        sys.exit(0)

    with open(COMMAND_CONTROLLER, "r") as f:
        content = f.read()

    if PATCH_MARKER in content:
        print("[PATCH] Camera handler already patched — skipping")
        sys.exit(0)

    # 1. Add the handler method to the class
    # Find the last method definition in the class and append after it
    # We'll add it before the final lines of the file
    patched = content.rstrip() + "\n" + CAMERA_HANDLER

    # 2. Add dispatch in on_command_step
    # We inject our handler at the top of the dispatch chain.
    # Detect the actual indentation used in the method body so we match it.
    on_cmd_pattern = re.compile(
        r"(def on_command_step\s*\(self.*?\):\s*\n(?:[ \t]*#[^\n]*\n|[ \t]*\"\"\"[\s\S]*?\"\"\"[ \t]*\n)?)",
        re.MULTILINE,
    )
    match = on_cmd_pattern.search(patched)
    if match:
        insert_pos = match.end()
        # Detect indentation of the next non-empty line (the method body)
        rest = patched[insert_pos:]
        body_indent = "        "  # fallback: 8 spaces (2-level)
        for line in rest.split("\n"):
            stripped = line.lstrip()
            if stripped:
                body_indent = line[: len(line) - len(stripped)]
                break
        deeper = body_indent + "    "
        dispatch = (
            f"{body_indent}# BlueprintPipeline camera patch\n"
            f"{body_indent}if self.Command == Command.GET_CAMERA_DATA:\n"
            f"{deeper}self.handle_get_camera_data()\n"
            f"{deeper}return\n"
        )
        patched = patched[:insert_pos] + dispatch + patched[insert_pos:]
        print("[PATCH] Injected GET_CAMERA_DATA dispatch in on_command_step")
    else:
        # Fallback: just prepend a note that manual wiring is needed
        print("[PATCH] WARNING: Could not find on_command_step — handler added but dispatch not wired")
        print("[PATCH] You must manually add: if self.Command == Command.GET_CAMERA_DATA: self.handle_get_camera_data(); return")

    with open(COMMAND_CONTROLLER, "w") as f:
        f.write(patched)

    print(f"[PATCH] Successfully patched {COMMAND_CONTROLLER}")


if __name__ == "__main__":
    patch_file()
