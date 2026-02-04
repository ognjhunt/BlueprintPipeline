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
    _bp_total_frames_rendered = {}  # Track total frames per camera since creation

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

            # Compute camera intrinsics and extrinsics from USD camera
            try:
                cam_prim = stage.GetPrimAtPath(camera_prim_path)
                if cam_prim and cam_prim.IsValid() and cam_prim.IsA(UsdGeom.Camera):
                    camera = UsdGeom.Camera(cam_prim)

                    # Extract intrinsics from USD camera attributes
                    # Focal length is in scene units (usually mm for cameras)
                    focal_length = camera.GetFocalLengthAttr().Get()
                    h_aperture = camera.GetHorizontalApertureAttr().Get()
                    v_aperture = camera.GetVerticalApertureAttr().Get()

                    _intrinsics_source = "default_fov"
                    if focal_length and h_aperture and h_aperture > 0:
                        # fx = focal_length_mm * width_pixels / aperture_mm
                        _fx = float(focal_length) * float(_w) / float(h_aperture)
                        if v_aperture and v_aperture > 0:
                            _fy = float(focal_length) * float(_h) / float(v_aperture)
                        else:
                            _fy = _fx  # Assume square pixels
                        _intrinsics_source = "usd_camera"
                        result["camera_info"]["fx"] = _fx
                        result["camera_info"]["fy"] = _fy
                        result["camera_info"]["ppx"] = float(_w) / 2.0
                        result["camera_info"]["ppy"] = float(_h) / 2.0
                        print(f"[PATCH] Camera {camera_prim_path}: focal_length={focal_length}, aperture=({h_aperture}, {v_aperture}), fx={_fx:.1f}, fy={_fy:.1f}")

                    result["camera_info"]["intrinsics_source"] = _intrinsics_source
                    result["camera_info"]["focal_length_mm"] = float(focal_length) if focal_length else None
                    result["camera_info"]["h_aperture_mm"] = float(h_aperture) if h_aperture else None
                    result["camera_info"]["v_aperture_mm"] = float(v_aperture) if v_aperture else None

                    # Compute extrinsics (camera-to-world transform)
                    xformable = UsdGeom.Xformable(cam_prim)
                    if xformable:
                        from pxr import Usd
                        cam_xform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                        result["camera_info"]["extrinsic"] = np.array(cam_xform, dtype=np.float64).reshape(4, 4).tolist()
                        result["camera_info"]["calibration_id"] = f"{camera_prim_path}_calib"
            except Exception as _ext_err:
                print(f"[PATCH] Failed to compute camera intrinsics/extrinsics: {_ext_err}")
                result["camera_info"]["intrinsics_source"] = "error"

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

                # Initial prime: run extra frames immediately after render product creation
                # to bootstrap the Replicator pipeline (shaders, textures, lighting)
                _initial_prime = int(_os.environ.get("CAMERA_INITIAL_PRIME_STEPS", "15"))
                if _initial_prime > 0:
                    print(f"[PATCH] Initial priming camera {camera_prim_path} ({_initial_prime} frames)...")
                    for _ in range(_initial_prime):
                        rep.orchestrator.step()
                    cls._bp_total_frames_rendered[camera_prim_path] = _initial_prime
                    print(f"[PATCH] Initial prime complete for {camera_prim_path}")
                else:
                    cls._bp_total_frames_rendered[camera_prim_path] = 0

            rp = cls._bp_render_products[camera_prim_path]
            rgb_annot = cls._bp_rgb_annotators[camera_prim_path]
            depth_annot = cls._bp_depth_annotators[camera_prim_path]

            # Warm-up: Replicator annotators need several frames before
            # returning valid data.  Run extra steps on first use.
            # Warmup logic with cumulative frame tracking to ensure render pipeline
            # gets enough total frames across retries.
            _rewarmup_on_reset = _os.environ.get("CAMERA_REWARMUP_ON_RESET", "0") == "1"
            _cumulative_target = int(_os.environ.get("CAMERA_CUMULATIVE_WARMUP_TARGET", "30"))
            _warmup_steps = int(_os.environ.get("CAMERA_WARMUP_STEPS", "5"))

            _total_rendered = cls._bp_total_frames_rendered.get(camera_prim_path, 0)
            _needs_warmup = False

            if _rewarmup_on_reset:
                # Always do warmup on rewarm mode, but track cumulative total
                _needs_warmup = True
            elif camera_prim_path not in cls._bp_warmup_done:
                # First-time warmup (non-rewarm mode)
                _needs_warmup = True
            elif _total_rendered < _cumulative_target:
                # Haven't reached cumulative target yet
                _needs_warmup = True

            if _needs_warmup:
                print(f"[PATCH] Warming up camera {camera_prim_path} ({_warmup_steps} frames, total={_total_rendered})...")
                for _ in range(_warmup_steps):
                    rep.orchestrator.step()
                cls._bp_total_frames_rendered[camera_prim_path] = _total_rendered + _warmup_steps
                cls._bp_warmup_done.add(camera_prim_path)
                print(f"[PATCH] Camera warmup complete for {camera_prim_path} (total={cls._bp_total_frames_rendered[camera_prim_path]})")

            _min_colors = int(_os.environ.get("CAMERA_QUALITY_MIN_COLORS", "100"))
            _min_std = float(_os.environ.get("CAMERA_QUALITY_MIN_STD", "10"))
            _max_retries = int(_os.environ.get("CAMERA_QUALITY_MAX_RETRIES", "3"))
            _retry_steps = int(_os.environ.get("CAMERA_QUALITY_RETRY_STEPS", "2"))

            def _bp_rgb_quality(_arr):
                if _arr is None:
                    return 0, 0.0
                try:
                    _h, _w = _arr.shape[:2]
                    _step_h = max(1, _h // 64)
                    _step_w = max(1, _w // 64)
                    _small = _arr[::_step_h, ::_step_w]
                    if _small.ndim >= 3 and _small.shape[-1] >= 3:
                        _flat = _small.reshape(-1, _small.shape[-1])[:, :3]
                        _uniq = np.unique(_flat, axis=0)
                        _unique_colors = len(_uniq)
                    else:
                        _unique_colors = len(np.unique(_small))
                    _std = float(np.std(_small.astype(float)))
                    return _unique_colors, _std
                except Exception:
                    return 0, 0.0

            rep.orchestrator.step()
            cls._bp_total_frames_rendered[camera_prim_path] = cls._bp_total_frames_rendered.get(camera_prim_path, 0) + 1

            rgb_data = rgb_annot.get_data()
            if rgb_data is not None:
                if hasattr(rgb_data, "numpy"):
                    rgb_data = rgb_data.numpy()
                result["rgb"] = np.asarray(rgb_data, dtype=np.uint8)
                h, w = result["rgb"].shape[:2]
                result["camera_info"]["width"] = w
                result["camera_info"]["height"] = h
                _uniq, _std = _bp_rgb_quality(result["rgb"])
                _retry = 0
                while (_uniq < _min_colors or _std < _min_std) and _retry < _max_retries:
                    for _ in range(_retry_steps):
                        rep.orchestrator.step()
                    cls._bp_total_frames_rendered[camera_prim_path] = cls._bp_total_frames_rendered.get(camera_prim_path, 0) + _retry_steps
                    _retry += 1
                    rgb_data = rgb_annot.get_data()
                    if rgb_data is None:
                        continue
                    if hasattr(rgb_data, "numpy"):
                        rgb_data = rgb_data.numpy()
                    result["rgb"] = np.asarray(rgb_data, dtype=np.uint8)
                    h, w = result["rgb"].shape[:2]
                    result["camera_info"]["width"] = w
                    result["camera_info"]["height"] = h
                    _uniq, _std = _bp_rgb_quality(result["rgb"])
                if _retry:
                    print(f"[PATCH] Camera {camera_prim_path} quality retry={_retry} unique={_uniq} std={_std:.2f}")

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

        # Serialize numpy arrays to bytes for gRPC transport and flatten structure
        _rgb_arr = result.get("rgb")
        _depth_arr = result.get("depth")
        _cam_info = result.get("camera_info", {})

        # Flatten camera_info to top level and serialize images to bytes
        flat_result = {
            "width": _cam_info.get("width", _w),
            "height": _cam_info.get("height", _h),
            "fx": _cam_info.get("fx", _fx),
            "fy": _cam_info.get("fy", _fy),
            "ppx": _cam_info.get("ppx", _ppx),
            "ppy": _cam_info.get("ppy", _ppy),
            "intrinsics_source": _cam_info.get("intrinsics_source", "default"),
            "extrinsic": _cam_info.get("extrinsic"),
            "calibration_id": _cam_info.get("calibration_id", ""),
            "camera_prim_path": camera_prim_path,
            # Keep nested camera_info for backward compatibility
            "camera_info": _cam_info,
        }

        # Serialize RGB to bytes
        if _rgb_arr is not None and hasattr(_rgb_arr, "tobytes"):
            flat_result["rgb"] = _rgb_arr.tobytes()
            flat_result["rgb_shape"] = list(_rgb_arr.shape)
            flat_result["rgb_dtype"] = str(_rgb_arr.dtype)
            flat_result["rgb_encoding"] = "raw_rgb_uint8"
        else:
            flat_result["rgb"] = bytes(_h * _w * 3)
            flat_result["rgb_shape"] = [_h, _w, 3]
            flat_result["rgb_dtype"] = "uint8"
            flat_result["rgb_encoding"] = "raw_rgb_uint8"

        # Serialize depth to bytes
        if _depth_arr is not None and hasattr(_depth_arr, "tobytes"):
            flat_result["depth"] = _depth_arr.tobytes()
            flat_result["depth_shape"] = list(_depth_arr.shape)
            flat_result["depth_dtype"] = str(_depth_arr.dtype)
        else:
            flat_result["depth"] = bytes(_h * _w * 4)  # float32 = 4 bytes
            flat_result["depth_shape"] = [_h, _w]
            flat_result["depth_dtype"] = "float32"

        _depth_dtype = str(flat_result.get("depth_dtype", "float32")).lower()
        if "float32" in _depth_dtype:
            flat_result["depth_encoding"] = "raw_depth_float32"
        elif "float16" in _depth_dtype:
            flat_result["depth_encoding"] = "raw_depth_float16"
        elif "uint16" in _depth_dtype:
            flat_result["depth_encoding"] = "raw_depth_uint16"
        else:
            flat_result["depth_encoding"] = "raw_depth"

        print(f"[PATCH] Camera data prepared: rgb={len(flat_result['rgb'])} bytes, depth={len(flat_result['depth'])} bytes, fx={flat_result['fx']:.1f}, fy={flat_result['fy']:.1f}")
        self.data_to_send = flat_result
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

    force_repatch = "--force" in sys.argv or os.environ.get("FORCE_REPATCH", "0") == "1"
    if PATCH_MARKER in content:
        if force_repatch:
            # Remove old patch to re-apply updated version
            print("[PATCH] Force re-patching: removing old camera handler...")
            # Remove everything from BEGIN marker to END marker (inclusive)
            begin_marker = "# --- BEGIN BlueprintPipeline camera patch ---"
            end_marker = "# --- END BlueprintPipeline camera patch ---"
            begin_idx = content.find(begin_marker)
            end_idx = content.find(end_marker)
            if begin_idx != -1 and end_idx != -1:
                # Find start of line containing begin marker
                line_start = content.rfind("\n", 0, begin_idx)
                if line_start == -1:
                    line_start = 0
                else:
                    line_start += 1
                # Find end of line containing end marker
                line_end = content.find("\n", end_idx)
                if line_end == -1:
                    line_end = len(content)
                content = content[:line_start] + content[line_end + 1:]
                print("[PATCH] Old camera handler removed")
            # Also remove the dispatch code in on_command_step
            dispatch_marker = "# BlueprintPipeline camera patch"
            if dispatch_marker in content:
                lines = content.split("\n")
                filtered = []
                skip_until_return = False
                for line in lines:
                    if dispatch_marker in line:
                        skip_until_return = True
                        continue
                    if skip_until_return:
                        if line.strip() == "return":
                            skip_until_return = False
                            continue
                    filtered.append(line)
                content = "\n".join(filtered)
                print("[PATCH] Old dispatch code removed")
        else:
            print("[PATCH] Camera handler already patched — skipping (use --force to re-apply)")
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
