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
    _bp_render_products = {}   # cached per camera_prim_path
    _bp_rgb_annotators = {}
    _bp_depth_annotators = {}
    _bp_warmup_done = set()
    _bp_cameras_logged = False

    def handle_get_camera_data(self):
        \"\"\"Handle GET_CAMERA_DATA (Command=1) — render current frame.

        Returns a dict matching grpc_server.py expectations:
          - camera_info: dict with width, height, ppx, ppy, fx, fy
          - rgb: numpy uint8 array (H, W, 3/4)
          - depth: numpy float32 array (H, W)
        \"\"\"
        import numpy as np

        command_data = self.data if self.data else {}
        camera_prim_path = ""
        if isinstance(command_data, dict):
            camera_prim_path = command_data.get("Cam_prim_path", "")
        elif hasattr(command_data, "serial_no"):
            camera_prim_path = command_data.serial_no

        # Default resolution
        _w, _h = 1280, 720
        import os as _os
        _cam_res_str = _os.environ.get("CAMERA_RESOLUTION", "1280x720")
        try:
            _w, _h = (int(x) for x in _cam_res_str.split("x"))
        except (ValueError, TypeError):
            pass

        # Default camera intrinsics (approximate)
        _fx = _fy = float(_w)
        _ppx, _ppy = float(_w) / 2.0, float(_h) / 2.0

        # Fallback: black frame so gRPC never crashes
        result = {
            "camera_info": {
                "width": _w, "height": _h,
                "ppx": _ppx, "ppy": _ppy, "fx": _fx, "fy": _fy,
            },
            "rgb": np.zeros((_h, _w, 3), dtype=np.uint8),
            "depth": np.zeros((_h, _w), dtype=np.float32),
        }

        try:
            import omni.replicator.core as rep

            from pxr import UsdGeom
            import omni.usd
            stage = omni.usd.get_context().get_stage()

            # Log all available cameras on first call
            cls = type(self)
            if not cls._bp_cameras_logged and stage:
                _all_cams = [str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdGeom.Camera)]
                print(f"[PATCH] Available cameras in stage: {_all_cams}")
                cls._bp_cameras_logged = True

            if not camera_prim_path:
                for prim in stage.Traverse():
                    if prim.IsA(UsdGeom.Camera):
                        camera_prim_path = str(prim.GetPath())
                        break
            if not camera_prim_path:
                camera_prim_path = "/OmniverseKit_Persp"

            # Compute camera extrinsics from USD (camera-to-world)
            try:
                cam_prim = stage.GetPrimAtPath(camera_prim_path)
                if cam_prim and cam_prim.IsValid() and cam_prim.IsA(UsdGeom.Camera):
                    xformable = UsdGeom.Xformable(cam_prim)
                    if xformable:
                        from pxr import Usd
                        cam_xform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                        result["camera_info"]["extrinsic"] = np.array(cam_xform, dtype=np.float64).reshape(4, 4).tolist()
                        result["camera_info"]["calibration_id"] = f"{camera_prim_path}_calib"
            except Exception as _ext_err:
                print(f"[PATCH] Failed to compute camera extrinsic: {_ext_err}")

            # Cache render products and annotators across calls
            cls = type(self)
            if camera_prim_path not in cls._bp_render_products:
                rp = rep.create.render_product(camera_prim_path, (_w, _h))
                rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
                rgb_annot.attach([rp])
                depth_annot = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
                depth_annot.attach([rp])
                cls._bp_render_products[camera_prim_path] = rp
                cls._bp_rgb_annotators[camera_prim_path] = rgb_annot
                cls._bp_depth_annotators[camera_prim_path] = depth_annot
                print(f"[PATCH] Created render product for {camera_prim_path}")

            rp = cls._bp_render_products[camera_prim_path]
            rgb_annot = cls._bp_rgb_annotators[camera_prim_path]
            depth_annot = cls._bp_depth_annotators[camera_prim_path]

            # Warm-up: Replicator annotators need several frames before
            # returning valid data.  Run extra steps on first use.
            # When CAMERA_REWARMUP_ON_RESET=1, re-warm cameras every call
            # to handle render pipeline staleness after physics resets.
            if _os.environ.get("CAMERA_REWARMUP_ON_RESET", "0") == "1":
                cls._bp_warmup_done.discard(camera_prim_path)
            if camera_prim_path not in cls._bp_warmup_done:
                _warmup_steps = int(_os.environ.get("CAMERA_WARMUP_STEPS", "5"))
                print(f"[PATCH] Warming up camera {camera_prim_path} ({_warmup_steps} frames)...")
                for _ in range(_warmup_steps):
                    rep.orchestrator.step()
                cls._bp_warmup_done.add(camera_prim_path)
                print(f"[PATCH] Camera warmup complete for {camera_prim_path}")

            rep.orchestrator.step()

            rgb_data = rgb_annot.get_data()
            if rgb_data is not None:
                if hasattr(rgb_data, "numpy"):
                    rgb_data = rgb_data.numpy()
                result["rgb"] = np.asarray(rgb_data, dtype=np.uint8)
                h, w = result["rgb"].shape[:2]
                result["camera_info"]["width"] = w
                result["camera_info"]["height"] = h

            depth_data = depth_annot.get_data()
            if depth_data is not None:
                if hasattr(depth_data, "numpy"):
                    depth_data = depth_data.numpy()
                result["depth"] = np.asarray(depth_data, dtype=np.float32)

            # Log data quality
            _nonzero = int(np.count_nonzero(result["rgb"]))
            _total = result["rgb"].size
            if _nonzero == 0:
                print(f"[PATCH] WARNING: Camera {camera_prim_path} returned all-zero RGB frame")
            else:
                print(f"[PATCH] Camera {camera_prim_path}: {result['rgb'].shape}, non-zero pixels: {_nonzero}/{_total}")

        except Exception as e:
            print(f"[PATCH] Camera capture failed (returning black frame): {e}")

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

    # 1. Add the handler method inside the class.
    # Detect the indentation used for method definitions (e.g. "    def ").
    method_indent = "    "  # default 4 spaces
    m = re.search(r"^([ \t]+)def \w+\(self", content, re.MULTILINE)
    if m:
        method_indent = m.group(1)

    # Indent each line of CAMERA_HANDLER to match class method level.
    indented_handler = "\n".join(
        (method_indent + line) if line.strip() else line
        for line in CAMERA_HANDLER.splitlines()
    ) + "\n"

    # Append inside the class (the class runs to end of file).
    patched = content.rstrip() + "\n\n" + indented_handler

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
            f"{deeper}with self.condition:\n"
            f"{deeper}    self.condition.notify_all()\n"
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
