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
    _bp_lighting_ensured = False  # Track whether we've ensured lighting exists
    _bp_stage_id = None  # Track stage identity to re-check lighting on stage change
    _bp_async_disabled = False  # Track whether we've disabled async rendering

    @classmethod
    def _bp_disable_async_rendering(cls):
        \"\"\"Disable async rendering and ensure the post-processing pipeline
        is active for reliable RGB camera capture in headless mode.

        The LdrColor/rgb annotator requires the full post-processing pipeline
        (tonemapping, exposure, AA) to convert HDR output to LDR uint8 RGB.
        Normals/depth work without this because they are raw render variables.
        \"\"\"
        if cls._bp_async_disabled:
            return
        cls._bp_async_disabled = True
        try:
            import carb.settings
            settings = carb.settings.get_settings()

            # Step 1: Enable critical extensions for RGB post-processing pipeline
            # The rgb/LdrColor annotator needs the viewport rendering pipeline
            # and post-processing (tonemapping, exposure) to produce color output.
            try:
                import omni.kit.app
                _mgr = omni.kit.app.get_app().get_extension_manager()
                _critical_exts = [
                    "omni.kit.viewport.window",
                    "omni.graph.image.nodes",
                    "omni.kit.hydra_texture",
                    "omni.kit.viewport.utility",
                    "omni.kit.viewport.bundle",
                    "omni.kit.viewport.rtx",
                    "omni.hydra.rtx",
                ]
                for _ext in _critical_exts:
                    try:
                        _mgr.set_extension_enabled(_ext, True)
                        print(f"[PATCH] Enabled extension: {_ext}")
                    except Exception as _ext_err:
                        print(f"[PATCH] Could not enable {_ext}: {_ext_err}")
            except Exception as _mgr_err:
                print(f"[PATCH] Extension manager error: {_mgr_err}")

            # Let extensions initialize
            try:
                import omni.kit.app
                _app = omni.kit.app.get_app()
                for _ in range(5):
                    _app.update()
            except Exception:
                pass

            # Log current render settings BEFORE changes
            _diag_keys = [
                "/app/asyncRendering", "/rtx/rendermode",
                "/rtx/pathtracing/spp", "/rtx/pathtracing/totalSpp",
                "/persistent/rtx/modes/rt2/enabled",
                "/rtx/directLighting/sampledLighting/enabled",
                "/app/renderer/skipWhileMinimized",
                "/rtx/post/tonemap/op", "/rtx/post/aa/op",
                "/rtx/hydra/mdlMaterialWarmup",
            ]
            for _dk in _diag_keys:
                try:
                    print(f"[PATCH-DIAG] BEFORE {_dk} = {settings.get(_dk)}")
                except Exception:
                    pass

            # Step 2: Disable async rendering
            settings.set("/app/asyncRendering", False)
            settings.set("/app/asyncRenderingLowLatency", False)
            settings.set("/rtx/materialDb/syncLoads", True)
            settings.set("/rtx/hydra/materialSyncLoads", True)
            settings.set("/app/renderer/skipWhileMinimized", False)
            settings.set("/app/renderer/sleepMsOnFocus", 0)
            settings.set("/app/renderer/sleepMsOutOfFocus", 0)

            # Step 3: Enable post-processing pipeline (CRITICAL for RGB)
            # Without tonemapping, HDR output won't be converted to LDR uint8
            settings.set("/rtx/post/tonemap/op", 1)  # 1=Aces
            settings.set("/rtx/post/histogram/enabled", True)
            settings.set("/rtx/post/lensFlare/enabled", False)
            settings.set("/rtx/post/aa/op", 2)  # FXAA (simpler than DLSS for headless)
            # Exposure
            settings.set("/rtx/post/tonemap/autoExposure", True)

            # Step 4: Renderer settings
            settings.set("/rtx/pathtracing/spp", 64)
            settings.set("/rtx/pathtracing/totalSpp", 64)
            settings.set("/rtx/post/denoiser/enabled", True)
            settings.set("/rtx/pathtracing/optixDenoiser/enabled", True)
            settings.set("/rtx/hydra/mdlMaterialWarmup", True)
            settings.set("/rtx/directLighting/sampledLighting/enabled", True)
            settings.set("/app/hydraEngine/waitIdle", True)
            settings.set("/renderer/enabled", "rtx")
            settings.set("/app/usd/useFabricSceneDelegate", True)

            # Enable render product rendering (critical for headless mode)
            settings.set("/omni/replicator/asyncRendering", False)
            # Ensure RTX renders sub-frames for material loading
            settings.set("/rtx/rendermode/subframes", 2)

            # Log current render settings AFTER changes
            for _dk in _diag_keys:
                try:
                    print(f"[PATCH-DIAG] AFTER  {_dk} = {settings.get(_dk)}")
                except Exception:
                    pass

            # Step 5: Create or enable viewport for post-processing pipeline
            try:
                from omni.kit.viewport.utility import get_active_viewport
                _vp = get_active_viewport()
                if _vp:
                    _vp.updates_enabled = True
                    print(f"[PATCH] Viewport found and enabled: {_vp}")
                else:
                    print("[PATCH] No active viewport — attempting to create one...")
                    try:
                        from omni.kit.viewport.window import ViewportWindow
                        _vp_win = ViewportWindow("headless_cam", width=1280, height=720, visible=False)
                        print(f"[PATCH] Created hidden ViewportWindow: {_vp_win}")
                        # Re-check for active viewport
                        import omni.kit.app
                        for _ in range(5):
                            omni.kit.app.get_app().update()
                        _vp2 = get_active_viewport()
                        if _vp2:
                            _vp2.updates_enabled = True
                            print(f"[PATCH] Viewport now active: {_vp2}")
                    except Exception as _vw_err:
                        print(f"[PATCH] Could not create ViewportWindow: {_vw_err}")
            except Exception as _vp_err:
                print(f"[PATCH] Viewport utility: {_vp_err}")

            try:
                from omni.kit.viewport.utility import get_num_viewports
                _nv = get_num_viewports()
                print(f"[PATCH] Number of viewports: {_nv}")
            except Exception:
                pass

            print("[PATCH] Disabled async rendering + enabled post-processing for headless capture")
        except Exception as _e:
            print(f"[PATCH] WARNING: Could not configure rendering: {_e}")
            import traceback
            traceback.print_exc()

    @classmethod
    def _bp_ensure_lighting(cls):
        \"\"\"Ensure lights exist in the stage for proper camera rendering.

        Without any light, all cameras render black (RGB=0, Alpha=255).
        Creates both a DomeLight (ambient) and DistantLight (directional)
        if no lights exist.  Re-checks when stage identity changes.
        \"\"\"
        try:
            import omni.usd
            from pxr import UsdLux, Sdf
            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print("[PATCH] Cannot ensure lighting: no stage available")
                return

            # Re-check lighting if the stage changed (e.g. after scene reload)
            _current_stage_id = id(stage)
            if cls._bp_lighting_ensured and cls._bp_stage_id == _current_stage_id:
                return
            cls._bp_stage_id = _current_stage_id
            cls._bp_lighting_ensured = True

            # Check if any light exists
            _existing_lights = []
            for prim in stage.Traverse():
                if prim.IsA(UsdLux.DomeLight) or prim.IsA(UsdLux.DistantLight) or prim.IsA(UsdLux.SphereLight) or prim.IsA(UsdLux.RectLight):
                    _existing_lights.append(str(prim.GetPath()))
            if _existing_lights:
                print(f"[PATCH] Found existing lights: {_existing_lights}")
                return

            import os as _light_os
            _intensity = float(_light_os.environ.get("BP_DOME_LIGHT_INTENSITY", "3000.0"))

            # Create DomeLight for ambient illumination
            _dome_path = "/World/BPDefaultDomeLight"
            dome_light = UsdLux.DomeLight.Define(stage, Sdf.Path(_dome_path))
            dome_light.GetIntensityAttr().Set(_intensity)
            dome_light.GetColorTemperatureAttr().Set(6500.0)
            try:
                dome_light.CreateEnableColorTemperatureAttr(True)
            except Exception:
                pass
            print(f"[PATCH] Created DomeLight at {_dome_path} intensity={_intensity}")

            # Also create a DistantLight for directional shadows/fill
            # DomeLights alone sometimes don't work in headless RTX mode
            _dist_path = "/World/BPDefaultDistantLight"
            dist_light = UsdLux.DistantLight.Define(stage, Sdf.Path(_dist_path))
            dist_light.GetIntensityAttr().Set(_intensity * 0.5)
            dist_light.GetAngleAttr().Set(1.0)  # Sun-like angle
            # Point the light downward
            from pxr import UsdGeom, Gf
            xformable = UsdGeom.Xformable(dist_light.GetPrim())
            xformable.AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 30.0, 0.0))
            print(f"[PATCH] Created DistantLight at {_dist_path} intensity={_intensity * 0.5}")

        except Exception as _light_err:
            print(f"[PATCH] Failed to ensure lighting: {_light_err}")
            import traceback
            traceback.print_exc()

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

            # Disable async rendering for reliable capture in headless mode
            cls = type(self)
            cls._bp_disable_async_rendering()

            # Ensure lighting exists before any rendering
            cls._bp_ensure_lighting()

            # Helper: synchronous render step — drives the actual rendering pipeline.
            # The key insight is that we need to use the same rendering path as the
            # server's main loop: World.render() or app.update().
            # rep.orchestrator.step() alone does NOT trigger the GPU render in
            # RealTimePathTracing mode — it only advances the Replicator schedule.
            def _sync_render_step():
                try:
                    import omni.kit.app
                    app = omni.kit.app.get_app()
                    # app.update() runs the full frame: physics + rendering + extensions
                    # This is what actually submits GPU work and produces pixel output
                    app.update()
                    app.update()  # Double update for pipeline flush
                except Exception as _update_err:
                    print(f"[PATCH] app.update() error: {_update_err}")
                # rep.orchestrator.step() advances Replicator annotators to read
                # from the most recently rendered frame
                try:
                    rep.orchestrator.step()
                except Exception as _rep_err:
                    print(f"[PATCH] rep.orchestrator.step() error: {_rep_err}")

            # Cache render products and annotators across calls
            if camera_prim_path not in cls._bp_render_products:
                # Run several app.update() frames BEFORE creating render product
                # to let the renderer fully initialize after our settings changes
                print(f"[PATCH] Pre-warming renderer before creating render product...")
                try:
                    import omni.kit.app
                    _pre_app = omni.kit.app.get_app()
                    for _i in range(10):
                        _pre_app.update()
                except Exception as _pre_err:
                    print(f"[PATCH] Pre-warm app.update() error: {_pre_err}")

                rp = rep.create.render_product(camera_prim_path, (_w, _h))

                # Try both rgb and LdrColor annotators
                rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
                rgb_annot.attach([rp])
                depth_annot = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
                depth_annot.attach([rp])

                # Also try alternative annotators for diagnostics
                try:
                    _ldr_annot = rep.AnnotatorRegistry.get_annotator("LdrColor")
                    _ldr_annot.attach([rp])
                    cls._bp_render_products[camera_prim_path + "_ldr"] = _ldr_annot
                    print(f"[PATCH] Also attached LdrColor annotator for diagnostics")
                except Exception as _ldr_err:
                    print(f"[PATCH] LdrColor annotator not available: {_ldr_err}")

                # HdrColor gives raw HDR float data BEFORE tonemapping
                # If this has data but LdrColor doesn't, we can manually tonemap
                try:
                    _hdr_annot = rep.AnnotatorRegistry.get_annotator("HdrColor")
                    _hdr_annot.attach([rp])
                    cls._bp_render_products[camera_prim_path + "_hdr"] = _hdr_annot
                    print(f"[PATCH] Also attached HdrColor annotator for diagnostics")
                except Exception as _hdr_err:
                    print(f"[PATCH] HdrColor annotator not available: {_hdr_err}")

                try:
                    _normals_annot = rep.AnnotatorRegistry.get_annotator("normals")
                    _normals_annot.attach([rp])
                    cls._bp_render_products[camera_prim_path + "_normals"] = _normals_annot
                    print(f"[PATCH] Also attached normals annotator for diagnostics")
                except Exception as _norm_err:
                    print(f"[PATCH] normals annotator not available: {_norm_err}")

                # Try using Camera helper class from isaacsim
                try:
                    from isaacsim.sensors.camera import Camera as IsaacCamera
                    _test_cam = IsaacCamera(prim_path=camera_prim_path, resolution=(_w, _h))
                    _test_cam.initialize()
                    cls._bp_render_products[camera_prim_path + "_isaccam"] = _test_cam
                    print(f"[PATCH] Also created IsaacCamera helper for diagnostics")
                except Exception as _cam_err:
                    print(f"[PATCH] IsaacCamera helper not available: {_cam_err}")

                cls._bp_render_products[camera_prim_path] = rp
                cls._bp_rgb_annotators[camera_prim_path] = rgb_annot
                cls._bp_depth_annotators[camera_prim_path] = depth_annot
                print(f"[PATCH] Created render product for {camera_prim_path}")

                # Initial prime: run extra frames immediately after render product creation
                # to bootstrap the Replicator pipeline (shaders, textures, lighting)
                _initial_prime = int(_os.environ.get("CAMERA_INITIAL_PRIME_STEPS", "30"))
                if _initial_prime > 0:
                    print(f"[PATCH] Initial priming camera {camera_prim_path} ({_initial_prime} frames)...")
                    for _pi in range(_initial_prime):
                        _sync_render_step()
                        # Check data periodically during prime
                        if _pi in (5, 10, 15, 20, 25, 29):
                            try:
                                _prime_data = rgb_annot.get_data()
                                if _prime_data is not None:
                                    _pd = np.asarray(_prime_data, dtype=np.uint8)
                                    _pd_nz = int(np.count_nonzero(_pd))
                                    _pd_shape = _pd.shape
                                    # Check per-channel stats
                                    if _pd.ndim == 3 and _pd.shape[-1] >= 3:
                                        _ch_means = [float(np.mean(_pd[:,:,c])) for c in range(min(4, _pd.shape[-1]))]
                                        print(f"[PATCH] Prime frame {_pi}: shape={_pd_shape}, nonzero={_pd_nz}/{_pd.size}, ch_means={_ch_means}")
                                    else:
                                        print(f"[PATCH] Prime frame {_pi}: shape={_pd_shape}, nonzero={_pd_nz}/{_pd.size}")
                                else:
                                    print(f"[PATCH] Prime frame {_pi}: data is None")
                            except Exception as _pe:
                                print(f"[PATCH] Prime frame {_pi}: error checking data: {_pe}")
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
                    _sync_render_step()
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

            _sync_render_step()
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
                        _sync_render_step()
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

            # Log data quality with per-channel diagnostics
            _rgb_result = result["rgb"]
            _nonzero = int(np.count_nonzero(_rgb_result))
            _total = _rgb_result.size
            if _rgb_result.ndim == 3:
                _nch = _rgb_result.shape[-1]
                _ch_stats = []
                for _ci in range(min(4, _nch)):
                    _ch = _rgb_result[:, :, _ci]
                    _ch_stats.append(f"ch{_ci}:min={int(_ch.min())} max={int(_ch.max())} mean={float(_ch.mean()):.1f}")
                print(f"[PATCH] Camera {camera_prim_path}: {_rgb_result.shape}, nonzero={_nonzero}/{_total}, {', '.join(_ch_stats)}")
            else:
                print(f"[PATCH] Camera {camera_prim_path}: {_rgb_result.shape}, nonzero={_nonzero}/{_total}")

            # If RGB is all-black but has alpha, try extracting RGB only
            if _rgb_result.ndim == 3 and _rgb_result.shape[-1] == 4:
                _rgb_only = _rgb_result[:, :, :3]
                _alpha = _rgb_result[:, :, 3]
                _rgb_nz = int(np.count_nonzero(_rgb_only))
                _alpha_nz = int(np.count_nonzero(_alpha))
                print(f"[PATCH] RGBA breakdown: RGB nonzero={_rgb_nz}, Alpha nonzero={_alpha_nz}")
                if _rgb_nz == 0 and _alpha_nz > 0:
                    print(f"[PATCH] WARNING: RGB channels all zero but Alpha is set — renderer not producing color data")
                # Return RGB only (strip alpha) to fix downstream RGBA->RGB misinterpretation
                result["rgb"] = _rgb_only

            # Check LdrColor annotator as alternative
            _ldr_key = camera_prim_path + "_ldr"
            if _ldr_key in cls._bp_render_products:
                try:
                    _ldr_annot = cls._bp_render_products[_ldr_key]
                    _ldr_data = _ldr_annot.get_data()
                    if _ldr_data is not None:
                        _ldr_arr = np.asarray(_ldr_data, dtype=np.uint8)
                        _ldr_nz = int(np.count_nonzero(_ldr_arr))
                        if _ldr_arr.ndim == 3:
                            _ldr_ch_stats = [f"ch{_ci}:mean={float(_ldr_arr[:,:,_ci].mean()):.1f}" for _ci in range(min(4, _ldr_arr.shape[-1]))]
                            print(f"[PATCH] LdrColor annotator: {_ldr_arr.shape}, nonzero={_ldr_nz}/{_ldr_arr.size}, {', '.join(_ldr_ch_stats)}")
                            # If LdrColor has actual RGB data, use it instead
                            if _ldr_arr.shape[-1] >= 3:
                                _ldr_rgb_nz = int(np.count_nonzero(_ldr_arr[:,:,:3]))
                                if _ldr_rgb_nz > 0 and _rgb_nz == 0:
                                    print(f"[PATCH] Using LdrColor data instead of rgb annotator (has actual colors!)")
                                    result["rgb"] = _ldr_arr[:, :, :3] if _ldr_arr.shape[-1] == 4 else _ldr_arr
                        else:
                            print(f"[PATCH] LdrColor annotator: {_ldr_arr.shape}, nonzero={_ldr_nz}/{_ldr_arr.size}")
                    else:
                        print(f"[PATCH] LdrColor annotator returned None")
                except Exception as _ldr_err:
                    print(f"[PATCH] LdrColor check error: {_ldr_err}")

            # Check HdrColor annotator — raw HDR before tonemapping
            _hdr_key = camera_prim_path + "_hdr"
            if _hdr_key in cls._bp_render_products:
                try:
                    _hdr_annot = cls._bp_render_products[_hdr_key]
                    _hdr_data = _hdr_annot.get_data()
                    if _hdr_data is not None:
                        _ha = np.asarray(_hdr_data, dtype=np.float32)
                        _ha_nz = int(np.count_nonzero(_ha))
                        if _ha.ndim == 3:
                            _ha_ch = [f"ch{_ci}:min={float(_ha[:,:,_ci].min()):.4f} max={float(_ha[:,:,_ci].max()):.4f} mean={float(_ha[:,:,_ci].mean()):.4f}" for _ci in range(min(4, _ha.shape[-1]))]
                            print(f"[PATCH] HdrColor annotator: {_ha.shape}, dtype={_ha.dtype}, nonzero={_ha_nz}/{_ha.size}, {', '.join(_ha_ch)}")
                            # If HdrColor has actual data but LdrColor/rgb is black,
                            # manually tonemap HDR -> LDR and use as RGB output
                            if _ha.shape[-1] >= 3:
                                _hdr_rgb_nz = int(np.count_nonzero(_ha[:,:,:3]))
                                if _hdr_rgb_nz > 0 and _rgb_nz == 0:
                                    print(f"[PATCH] HdrColor has color data! Manually tonemapping HDR->LDR...")
                                    # Simple Reinhard tonemapping: L / (1 + L)
                                    _hdr_rgb = _ha[:, :, :3].copy()
                                    _hdr_rgb = np.clip(_hdr_rgb, 0, None)  # Remove negatives
                                    # Reinhard: x / (1 + x)
                                    _tonemapped = _hdr_rgb / (1.0 + _hdr_rgb)
                                    # Gamma correction (linear -> sRGB)
                                    _tonemapped = np.power(np.clip(_tonemapped, 0, 1), 1.0/2.2)
                                    # Convert to uint8
                                    _ldr_from_hdr = (np.clip(_tonemapped, 0, 1) * 255).astype(np.uint8)
                                    result["rgb"] = _ldr_from_hdr
                                    _uniq2, _std2 = _bp_rgb_quality(_ldr_from_hdr)
                                    print(f"[PATCH] Tonemapped HDR->RGB: unique={_uniq2} std={_std2:.2f}")
                        else:
                            print(f"[PATCH] HdrColor annotator: {_ha.shape}, dtype={_ha.dtype}, nonzero={_ha_nz}/{_ha.size}")
                    else:
                        print(f"[PATCH] HdrColor annotator returned None")
                except Exception as _he:
                    print(f"[PATCH] HdrColor check error: {_he}")

            # Check normals annotator
            _normals_key = camera_prim_path + "_normals"
            if _normals_key in cls._bp_render_products:
                try:
                    _normals_annot = cls._bp_render_products[_normals_key]
                    _normals_data = _normals_annot.get_data()
                    if _normals_data is not None:
                        _na = np.asarray(_normals_data)
                        _na_nz = int(np.count_nonzero(_na))
                        print(f"[PATCH] Normals annotator: {_na.shape}, dtype={_na.dtype}, nonzero={_na_nz}/{_na.size}, range=[{float(np.min(_na)):.3f}, {float(np.max(_na)):.3f}]")
                    else:
                        print(f"[PATCH] Normals annotator returned None")
                except Exception as _ne:
                    print(f"[PATCH] Normals check error: {_ne}")

            # Check IsaacCamera helper
            _isaccam_key = camera_prim_path + "_isaccam"
            if _isaccam_key in cls._bp_render_products:
                try:
                    _test_cam = cls._bp_render_products[_isaccam_key]
                    _test_rgba = _test_cam.get_rgba()
                    if _test_rgba is not None:
                        _ta = np.asarray(_test_rgba, dtype=np.uint8)
                        _ta_nz = int(np.count_nonzero(_ta))
                        if _ta.ndim == 3:
                            _ta_ch = [f"ch{i}:mean={float(_ta[:,:,i].mean()):.1f}" for i in range(min(4, _ta.shape[-1]))]
                            print(f"[PATCH] IsaacCamera.get_rgba(): {_ta.shape}, nonzero={_ta_nz}/{_ta.size}, {', '.join(_ta_ch)}")
                        else:
                            print(f"[PATCH] IsaacCamera.get_rgba(): {_ta.shape}, nonzero={_ta_nz}/{_ta.size}")
                    else:
                        print(f"[PATCH] IsaacCamera.get_rgba() returned None")
                except Exception as _ce:
                    print(f"[PATCH] IsaacCamera check error: {_ce}")

            # Check depth data for diagnostic info
            if result["depth"] is not None and hasattr(result["depth"], "shape"):
                _depth_nz = int(np.count_nonzero(result["depth"]))
                _depth_min = float(np.min(result["depth"]))
                _depth_max = float(np.max(result["depth"]))
                print(f"[PATCH] Depth: {result['depth'].shape}, nonzero={_depth_nz}, range=[{_depth_min:.3f}, {_depth_max:.3f}]")

        except Exception as e:
            import traceback
            print(f"[PATCH] Camera capture failed (returning black frame): {e}")
            traceback.print_exc()

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
