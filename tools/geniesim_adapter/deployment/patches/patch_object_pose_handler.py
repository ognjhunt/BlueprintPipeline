#!/usr/bin/env python3
"""
Patch the Genie Sim server's command_controller.py to fix GET_OBJECT_POSE.

The upstream server's handle_get_object_pose returns empty/identity poses
because the requested prim paths don't match the actual USD stage hierarchy.
This patch adds fuzzy path matching with multi-candidate scoring: if the
exact path isn't found, it searches the stage for prims whose name suffix
matches, then scores candidates preferring geometric prims under /World/.

Usage (inside Docker build):
    python3 /tmp/patches/patch_object_pose_handler.py

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

# The object pose handler helper to inject.
OBJECT_POSE_HELPER = textwrap.dedent("""\

    # --- BEGIN BlueprintPipeline object_pose patch ---
    # Class-level transform cache and scene stage reference
    _bp_transform_cache = {}     # {prim_path: {"position": [...], "rotation": [...], "timestamp": float}}
    _bp_cache_ttl_s = 0.05       # 50ms TTL for dynamic objects (20Hz refresh rate)
    _bp_scene_stage = None       # Cached reference to the scene stage
    _bp_scene_stage_id = None    # Stage identifier to detect context switches

    def _bp_capture_scene_stage(self):
        \"\"\"Capture and cache a reference to the scene stage after init_robot.

        This stage reference is used for all object pose queries to avoid
        context switching issues where omni.usd.get_context().get_stage()
        returns a different stage (e.g., the render stage).
        \"\"\"
        try:
            import omni.usd
            stage = omni.usd.get_context().get_stage()
            if stage is not None:
                cls = type(self)
                cls._bp_scene_stage = stage
                try:
                    cls._bp_scene_stage_id = str(stage.GetRootLayer().identifier)
                except Exception:
                    cls._bp_scene_stage_id = "unknown"
                # Log available scene prims for diagnostics
                world_prims = [str(p.GetPath()) for p in stage.Traverse()
                              if str(p.GetPath()).startswith("/World/")]
                print(f"[PATCH] Captured scene stage: {cls._bp_scene_stage_id}")
                print(f"[PATCH] World prims available: {len(world_prims)}")
                if world_prims[:10]:
                    print(f"[PATCH] First 10 World prims: {world_prims[:10]}")
        except Exception as e:
            print(f"[PATCH] Failed to capture scene stage: {e}")

    def _bp_get_scene_stage(self):
        \"\"\"Get the scene stage, using cached reference if available and valid.\"\"\"
        cls = type(self)
        cached_stage = getattr(cls, '_bp_scene_stage', None)
        cached_id = getattr(cls, '_bp_scene_stage_id', None)

        if cached_stage is not None:
            try:
                # Verify the stage is still valid by checking layer identifier
                current_id = str(cached_stage.GetRootLayer().identifier)
                if current_id == cached_id:
                    return cached_stage
            except Exception:
                pass  # Stage invalid, fall through to get fresh one

        # Fall back to current context stage
        try:
            import omni.usd
            return omni.usd.get_context().get_stage()
        except Exception:
            return None

    def _bp_extract_transform(self, prim, prim_path):
        \"\"\"Extract world transform from a prim - prefer live physics state.

        Query order:
        1. Try dynamic_control for live PhysX rigid body pose
        2. Fall back to USD ComputeLocalToWorldTransform for kinematic objects

        Returns (position, rotation) tuple or (None, None) on failure.
        \"\"\"
        try:
            import time
            from pxr import UsdGeom, Usd, Gf

            if not prim.IsA(UsdGeom.Xformable):
                return None, None

            # PRIORITY 1: Try to get live physics pose for rigid bodies
            physics_pos, physics_rot = self._bp_get_physics_pose(prim_path)
            if physics_pos is not None and physics_rot is not None:
                # Cache the transform from physics
                cls = type(self)
                cls._bp_transform_cache[prim_path] = {
                    "position": physics_pos,
                    "rotation": physics_rot,
                    "timestamp": time.time(),
                    "source": "physx",
                }
                return physics_pos, physics_rot

            # PRIORITY 2: Fall back to USD transform (for kinematic/non-physics objects)
            xformable = UsdGeom.Xformable(prim)
            timecode = self._bp_live_timecode() or Usd.TimeCode.Default()
            world_xform = xformable.ComputeLocalToWorldTransform(timecode)

            # Extract position (translation) from the matrix
            translation = world_xform.ExtractTranslation()
            position = [float(translation[0]), float(translation[1]), float(translation[2])]

            # Extract rotation as quaternion from the matrix
            # Get the 3x3 rotation matrix and convert to quaternion
            rotation_matrix = Gf.Matrix3d(
                world_xform[0][0], world_xform[0][1], world_xform[0][2],
                world_xform[1][0], world_xform[1][1], world_xform[1][2],
                world_xform[2][0], world_xform[2][1], world_xform[2][2],
            )
            # Orthonormalize to handle scale/shear
            rotation = rotation_matrix.GetOrthonormalized()
            quat = Gf.Rotation(rotation).GetQuaternion()
            quat_normalized = quat.GetNormalized()
            rotation = [
                float(quat_normalized.GetReal()),
                float(quat_normalized.GetImaginary()[0]),
                float(quat_normalized.GetImaginary()[1]),
                float(quat_normalized.GetImaginary()[2]),
            ]

            # Cache the transform
            cls = type(self)
            cls._bp_transform_cache[prim_path] = {
                "position": position,
                "rotation": rotation,
                "timestamp": time.time(),
                "timecode": float(timecode.GetValue()) if hasattr(timecode, 'GetValue') else 0.0,
                "source": "usd",
            }

            return position, rotation
        except Exception as e:
            print(f"[PATCH] Transform extraction failed for {prim_path}: {e}")
            return None, None

    def _bp_get_cached_transform(self, prim_path):
        \"\"\"Get cached transform for a prim path, refreshing if stale.

        Returns (position, rotation) tuple or (None, None) if not cached.
        \"\"\"
        import time
        cls = type(self)
        cache = cls._bp_transform_cache.get(prim_path)
        if cache is None:
            return None, None

        # Check TTL for dynamic objects
        age = time.time() - cache["timestamp"]
        if age > cls._bp_cache_ttl_s:
            # Stale cache - try to refresh
            self._bp_refresh_transform(prim_path)
            cache = cls._bp_transform_cache.get(prim_path)
            if cache is None:
                return None, None

        return cache["position"], cache["rotation"]

    def _bp_refresh_transform(self, prim_path):
        \"\"\"Refresh the cached transform for a prim using the scene stage.\"\"\"
        try:
            stage = self._bp_get_scene_stage()
            if stage is None:
                return

            prim = stage.GetPrimAtPath(prim_path)
            if prim and prim.IsValid():
                self._bp_extract_transform(prim, prim_path)
        except Exception as e:
            print(f"[PATCH] Transform refresh failed for {prim_path}: {e}")

    def _bp_resolve_prim_path(self, requested_path):
        \"\"\"Resolve a prim path by fuzzy matching and cache its transform.

        CRITICAL: This function extracts the world transform at resolution time
        because the USD context may change before the transform is used.
        The cached transform is stored in _bp_transform_cache.

        If the exact path exists, return it.  Otherwise, search for prims
        whose name (last path component) matches the requested path's name,
        then score candidates: prefer Xformable prims under /World/ with
        shorter paths.  Falls back to substring matching if no exact name
        match is found.
        \"\"\"
        try:
            from pxr import UsdGeom

            # Use cached scene stage if available
            stage = self._bp_get_scene_stage()
            if stage is None:
                return requested_path

            # Exact match
            prim = stage.GetPrimAtPath(requested_path)
            if prim and prim.IsValid():
                # CRITICAL: Extract and cache transform NOW while stage is valid
                self._bp_extract_transform(prim, requested_path)
                return requested_path

            # Extract target name (last path component)
            target_name = requested_path.rstrip("/").rsplit("/", 1)[-1]
            if not target_name:
                return requested_path

            target_lower = target_name.lower()

            # Strip scene prefix for matching: "obj_lightwheel_kitchen_obj_Table049" -> "Table049"
            # Pattern: obj_{scene_id}_obj_{base_name} OR {scene_id}_obj_{base_name}
            base_name = target_name
            if "_obj_" in target_name:
                # Take everything after the LAST "_obj_" as the base name
                base_name = target_name.rsplit("_obj_", 1)[-1]

            # USD uses obj_BaseName format for scene objects
            target_with_obj = f"obj_{base_name}" if not base_name.startswith("obj_") else base_name

            # Create set of names to match against (case-sensitive and lowercase)
            match_names = {target_name, target_lower, base_name, base_name.lower(),
                           target_with_obj, target_with_obj.lower()}

            # Collect candidates: exact name match and substring match
            exact_candidates = []
            substring_candidates = []

            for prim in stage.Traverse():
                prim_path = str(prim.GetPath())
                prim_name = prim_path.rstrip("/").rsplit("/", 1)[-1]
                prim_name_lower = prim_name.lower()

                # Check if prim name matches any of our target variants
                if prim_name in match_names or prim_name_lower in match_names:
                    exact_candidates.append((prim_path, prim))
                elif any(m in prim_name_lower for m in [base_name.lower(), target_with_obj.lower()]):
                    substring_candidates.append((prim_path, prim))

            # Score and pick best from exact matches first, then substring
            def _score(path, p):
                s = 0
                # Prefer geometric prims (Xformable includes Mesh, Xform, etc.)
                try:
                    if p.IsA(UsdGeom.Xformable):
                        s += 100
                except Exception:
                    pass
                # Prefer prims under /World/Scene/ (where scene objects live)
                if "/World/Scene/" in path:
                    s += 75
                elif path.startswith("/World/"):
                    s += 50
                # Prefer paths containing "obj_" (scene objects follow this convention)
                if "/obj_" in path:
                    s += 25
                # Prefer shorter paths (less deeply nested)
                s -= path.count("/")
                return s

            for candidates, label in [(exact_candidates, "exact"), (substring_candidates, "substring")]:
                if not candidates:
                    continue
                scored = sorted(candidates, key=lambda c: _score(c[0], c[1]), reverse=True)
                best_path, best_prim = scored[0]
                if len(scored) > 1:
                    print(f"[PATCH] Prim path {label} candidates for '{target_name}' (base='{base_name}'): "
                          f"{[c[0] for c in scored[:5]]}")
                print(f"[PATCH] Resolved prim path ({label}): {requested_path} -> {best_path} (via base_name='{base_name}')")

                # CRITICAL: Extract and cache transform NOW while stage is valid
                pos, rot = self._bp_extract_transform(best_prim, best_path)
                if pos is not None:
                    print(f"[PATCH] Cached transform for {best_path}: pos={pos[:2]}...")

                return best_path

            # Dump stage hierarchy for debugging (first call only)
            if not getattr(self, '_bp_stage_dumped', False):
                self._bp_stage_dumped = True
                all_prims = []
                for p in stage.Traverse():
                    all_prims.append(str(p.GetPath()))
                print(f"[PATCH] DIAGNOSTIC: Full USD stage has {len(all_prims)} prims. First 50:")
                for pp in all_prims[:50]:
                    print(f"[PATCH]   {pp}")
                if len(all_prims) > 50:
                    print(f"[PATCH]   ... and {len(all_prims) - 50} more")
            print(f"[PATCH] WARNING: Could not resolve prim path {requested_path} in stage")
        except Exception as e:
            print(f"[PATCH] Prim path resolution failed: {e}")

        return requested_path
    # --- END BlueprintPipeline object_pose patch ---
""")

PATCH_MARKER = "BlueprintPipeline object_pose patch"
LIVE_TIME_MARKER = "BlueprintPipeline object_pose live_timecode patch"

LIVE_TIMECODE_HELPER = textwrap.dedent("""\

    # --- BEGIN BlueprintPipeline object_pose live_timecode patch ---
    def _bp_live_timecode(self):
        \"\"\"Return USD timecode for the current sim time (avoid time=0).\"\"\"
        try:
            import omni.usd
            from pxr import Usd
            ctx = omni.usd.get_context()
            if hasattr(ctx, "get_time"):
                _t = ctx.get_time()
                if _t is not None:
                    return Usd.TimeCode(_t)
        except Exception:
            pass
        try:
            import omni.timeline
            from pxr import Usd
            timeline = omni.timeline.get_timeline_interface()
            if timeline is not None:
                return Usd.TimeCode(timeline.get_current_time())
        except Exception:
            pass
        try:
            from pxr import Usd
            return Usd.TimeCode.Default()
        except Exception:
            return None
    # --- END BlueprintPipeline object_pose live_timecode patch ---
""")

PHYSICS_POSE_MARKER = "BlueprintPipeline object_pose physics_pose patch"

PHYSICS_POSE_HELPER = textwrap.dedent("""\

    # --- BEGIN BlueprintPipeline object_pose physics_pose patch ---
    def _bp_get_physics_pose(self, prim_path):
        \"\"\"Get live physics pose for a rigid body using dynamic_control.

        Returns (position, rotation) tuple or (None, None) if not a physics body
        or if physics is not running.

        position: [x, y, z] in meters
        rotation: [w, x, y, z] quaternion (scalar-first)
        \"\"\"
        try:
            from omni.isaac.dynamic_control import _dynamic_control
            from pxr import UsdPhysics

            stage = self._bp_get_scene_stage()
            if stage is None:
                return None, None

            prim = stage.GetPrimAtPath(prim_path)
            if not prim or not prim.IsValid():
                return None, None

            # Check if this prim has RigidBodyAPI (is a physics-enabled rigid body)
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                # Not a rigid body - fall back to USD transform
                return None, None

            # Get the dynamic control interface
            dc = _dynamic_control.acquire_dynamic_control_interface()
            if dc is None:
                return None, None

            # Get the rigid body handle
            rb_handle = dc.get_rigid_body(prim_path)
            if rb_handle == _dynamic_control.INVALID_HANDLE:
                # Physics may not be running or prim not registered
                return None, None

            # Get the live pose from PhysX
            pose = dc.get_rigid_body_pose(rb_handle)
            if pose is None:
                return None, None

            # pose.p is position (carb.Float3), pose.r is rotation (carb.Float4 quaternion)
            position = [float(pose.p.x), float(pose.p.y), float(pose.p.z)]
            # dynamic_control returns quaternion as (x, y, z, w) - convert to (w, x, y, z)
            rotation = [float(pose.r.w), float(pose.r.x), float(pose.r.y), float(pose.r.z)]

            return position, rotation

        except ImportError:
            # dynamic_control not available - this is expected outside Isaac Sim
            return None, None
        except Exception as e:
            # Don't spam logs - physics queries fail often for non-physics objects
            return None, None
    # --- END BlueprintPipeline object_pose physics_pose patch ---
""")


def patch_file():
    if not os.path.isfile(COMMAND_CONTROLLER):
        print(f"[PATCH] command_controller.py not found at {COMMAND_CONTROLLER}")
        print("[PATCH] Skipping object_pose patch (server source not available)")
        sys.exit(0)

    with open(COMMAND_CONTROLLER, "r") as f:
        content = f.read()

    has_patch_marker = PATCH_MARKER in content
    has_live_time = "def _bp_live_timecode" in content

    # 1. Add the helper method to the class (append like camera patch)
    method_indent = "    "  # default 4 spaces
    m = re.search(r"^([ \t]+)def \w+\(self", content, re.MULTILINE)
    if m:
        method_indent = m.group(1)

    patched = content.rstrip()
    if not has_patch_marker:
        indented_helper = "\n".join(
            (method_indent + line) if line.strip() else line
            for line in OBJECT_POSE_HELPER.splitlines()
        ) + "\n"
        patched = patched + "\n\n" + indented_helper

    if not has_live_time:
        indented_live = "\n".join(
            (method_indent + line) if line.strip() else line
            for line in LIVE_TIMECODE_HELPER.splitlines()
        ) + "\n"
        patched = patched + "\n\n" + indented_live

    # 1c. Add physics pose helper for live PhysX queries
    has_physics_pose = "def _bp_get_physics_pose" in content
    if not has_physics_pose:
        indented_physics = "\n".join(
            (method_indent + line) if line.strip() else line
            for line in PHYSICS_POSE_HELPER.splitlines()
        ) + "\n"
        patched = patched + "\n\n" + indented_physics
        print("[PATCH] Added _bp_get_physics_pose helper for live PhysX pose queries")

    # 1b. Inject diagnostic logging after scene_usd_path assignment
    # This tells us whether the server's scene_usd_path resolves to a real file
    scene_path_pattern = re.compile(
        r"((\s+)(self\.scene_usd_path\s*=\s*os\.path\.join\(self\.sim_assets_root,\s*scene_usd\)))\n",
        re.MULTILINE,
    )
    scene_match = scene_path_pattern.search(patched)
    if scene_match:
        indent_s = scene_match.group(2)
        diag = (
            f"\n{indent_s}print(f'[PATCH-DIAG] scene_usd_path={{self.scene_usd_path}}, "
            f"exists={{os.path.exists(self.scene_usd_path)}}, "
            f"sim_assets_root={{self.sim_assets_root}}, scene_usd={{scene_usd}}')"
            f"  # {PATCH_MARKER}"
        )
        patched = patched[:scene_match.end(1)] + diag + patched[scene_match.end(1):]
        print("[PATCH] Injected scene_usd_path diagnostic logging")
    else:
        print("[PATCH] WARNING: Could not find scene_usd_path assignment for diagnostics")

    # 2. Find get_object_pose handling and inject path resolution.
    # Look for patterns like:
    #   prim_path = request.prim_path
    #   prim_path = data.get("prim_path", ...)
    #   prim = stage.GetPrimAtPath(prim_path)
    # and add self._bp_resolve_prim_path() call after prim_path assignment.
    # Match standalone prim_path assignments (not keyword args ending with comma)
    prim_path_pattern = re.compile(
        r"((\s+)(prim_path|object_prim_path|_prim_path)\s*=\s*(?:.*(?:prim_path|Prim_path|object_id).*))\n",
        re.MULTILINE,
    )
    # Filter out matches that are keyword arguments (line ends with comma)
    all_matches = list(prim_path_pattern.finditer(patched))
    match = None
    for m in all_matches:
        line_content = m.group(1).rstrip()
        if not line_content.endswith(","):
            match = m
            break
    resolver_injected = False
    if "_bp_resolve_prim_path" not in patched:
        if match:
            indent = match.group(2)
            var_name = match.group(3)
            original_line = match.group(1)
            resolution_line = f"\n{indent}{var_name} = self._bp_resolve_prim_path({var_name})  # {PATCH_MARKER}"
            patched = patched[:match.end(1)] + resolution_line + patched[match.end(1):]
            print(f"[PATCH] Injected prim path resolution after {var_name} assignment")
            resolver_injected = True
        else:
            print("[PATCH] WARNING: Could not find prim_path assignment in get_object_pose handler")
            print("[PATCH] Falling back to wrapping stage.GetPrimAtPath(prim_path) calls")
            fallback_pattern = re.compile(r"stage\.GetPrimAtPath\(\s*(prim_path)\s*\)")
            patched, fallback_count = fallback_pattern.subn(
                rf"stage.GetPrimAtPath(self._bp_resolve_prim_path(\1))  # {PATCH_MARKER}",
                patched,
            )
            if fallback_count:
                resolver_injected = True
                print(f"[PATCH] Wrapped {fallback_count} stage.GetPrimAtPath(prim_path) call(s)")
            else:
                print("[PATCH] ERROR: No prim_path assignment or stage.GetPrimAtPath(prim_path) call found")
                print("[PATCH] Cannot wire prim path resolution automatically")
                sys.exit(1)
    else:
        resolver_injected = True

    # 3. Ensure live timecode is used for world transforms (avoid time=0).
    timecode_pattern = re.compile(
        r"(ComputeLocalToWorldTransform\()\s*Usd\.TimeCode\([^\)]*\)\s*(\))"
    )
    patched, timecode_count = timecode_pattern.subn(
        rf"\1self._bp_live_timecode()\2  # {LIVE_TIME_MARKER}",
        patched,
    )
    if timecode_count:
        print(f"[PATCH] Updated {timecode_count} ComputeLocalToWorldTransform timecodes")

    if PATCH_MARKER not in patched:
        print("[PATCH] ERROR: PATCH_MARKER not found after patching")
        sys.exit(1)

    if not resolver_injected or re.search(r"self\._bp_resolve_prim_path\(", patched) is None:
        # Don't fail - the helper method is still added and patch_grpc_server.py
        # will use it via the object_pose handler in grpc_server.py
        print("[PATCH] WARNING: No resolver call was inserted into command_controller.py")
        print("[PATCH] The _bp_resolve_prim_path helper was added; grpc_server.py patch will use it")

    with open(COMMAND_CONTROLLER, "w") as f:
        f.write(patched)

    print(f"[PATCH] Successfully patched {COMMAND_CONTROLLER}")


if __name__ == "__main__":
    patch_file()
