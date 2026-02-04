#!/usr/bin/env python3
"""
Patch the Genie Sim server's grpc_server.py to fix several runtime bugs.

Fixes:
1. get_object_pose: "too many values to unpack (expected 2)" — safe unpacking.
2. get_observation recordingState: expects string but receives dict.
3. ALL rsp.msg = self.server_function.blocking_start_server(...) calls:
   blocking_start_server() often returns dicts but rsp.msg is a protobuf
   string field. Wrap ALL occurrences with str().
4. get_observation pose loop: same tuple-unpacking issue as (1).
5. set literal bugs: {"reset", Reset} → {"reset": Reset} etc.
6. Gripper handler: string concatenation with dict return value.
7. get_joint_position: joint_positions[name] may be dict instead of float.
8. get_ee_pose: "too many values to unpack (expected 2)" — safe unpacking.

Usage (inside Docker build or at runtime):
    python3 /tmp/patches/patch_grpc_server.py

The script is idempotent — re-running it on an already-patched file is a no-op.
"""
import os
import re
import sys

GENIESIM_ROOT = os.environ.get("GENIESIM_ROOT", "/opt/geniesim")
GRPC_SERVER = os.path.join(
    GENIESIM_ROOT,
    "source", "data_collection", "server", "grpc_server.py",
)

PATCH_MARKER = "BlueprintPipeline grpc_server patch"
EE_POSE_MARKER = "BlueprintPipeline ee_pose grpc patch"
EE_RSP_MARKER = "BlueprintPipeline ee_rsp grpc patch"
CAMERA_BYTES_MARKER = "BlueprintPipeline camera_bytes grpc patch"
CAMERA_INTRINSICS_MARKER = "BlueprintPipeline camera_intrinsics grpc patch"
CAMERA_ENCODING_MARKER = "BlueprintPipeline camera_encoding grpc patch"


def _find_matching_paren(text, start):
    """Find the index of the closing paren matching the open paren at `start`."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == '(':
            depth += 1
        elif text[i] == ')':
            depth -= 1
            if depth == 0:
                return i
    return -1


def patch_file():
    if not os.path.isfile(GRPC_SERVER):
        print(f"[PATCH] grpc_server.py not found at {GRPC_SERVER}")
        print("[PATCH] Skipping grpc_server patch (server source not available)")
        sys.exit(0)

    with open(GRPC_SERVER, "r") as f:
        content = f.read()

    # Check if fully patched (base + EE pose)
    if PATCH_MARKER in content and EE_POSE_MARKER in content:
        print("[PATCH] grpc_server.py already fully patched — skipping")
        sys.exit(0)

    already_base_patched = PATCH_MARKER in content
    changes = 0

    # ── Base patches (skip if already applied) ──

    if not already_base_patched:
        # Fix 0: Inject _safe_float helper at top of file (after imports)
        safe_float_helper = (
            "\n# " + PATCH_MARKER + " — safe float helper for unit-suffixed values\n"
            "def _bp_safe_float(val, default=0.0):\n"
            "    try:\n"
            "        return float(val)\n"
            "    except (ValueError, TypeError):\n"
            "        if isinstance(val, str):\n"
            "            import re as _re\n"
            "            _m = _re.match(r'([+-]?\\d*\\.?\\d+)', val.strip())\n"
            "            if _m:\n"
            "                return float(_m.group(1))\n"
            "        return default\n\n"
        )
        import_end = 0
        for m in re.finditer(r'^(?:import |from )\S+.*$', content, re.MULTILINE):
            import_end = m.end()
        if import_end > 0:
            content = content[:import_end] + "\n" + safe_float_helper + content[import_end:]
            changes += 1
            print("[PATCH] Injected _bp_safe_float helper")

        # Fix 1: object_pose unpacking
        old = "position, rotation = object_pose"
        new = (
            "# " + PATCH_MARKER + " — safe object_pose unpacking\n"
            "        _op = object_pose if isinstance(object_pose, (list, tuple)) else (object_pose, None)\n"
            "        position = _op[0] if len(_op) >= 1 else (0, 0, 0)\n"
            "        rotation = _op[1] if len(_op) >= 2 else (1, 0, 0, 0)"
        )
        if old in content:
            content = content.replace(old, new, 1)
            changes += 1
            print("[PATCH] Fixed object_pose unpacking in get_object_pose")

        # Fix 1b: position tuple unpacking in get_object_pose rsp
        old_pos = (
            "(\n"
            "            rsp.object_pose.position.x,\n"
            "            rsp.object_pose.position.y,\n"
            "            rsp.object_pose.position.z,\n"
            "        ) = position"
        )
        new_pos = (
            "# " + PATCH_MARKER + " — safe position assignment\n"
            "        _pos = list(position) if hasattr(position, '__iter__') else [0, 0, 0]\n"
            "        rsp.object_pose.position.x = _bp_safe_float(_pos[0]) if len(_pos) > 0 else 0.0\n"
            "        rsp.object_pose.position.y = _bp_safe_float(_pos[1]) if len(_pos) > 1 else 0.0\n"
            "        rsp.object_pose.position.z = _bp_safe_float(_pos[2]) if len(_pos) > 2 else 0.0"
        )
        if old_pos in content:
            content = content.replace(old_pos, new_pos, 1)
            changes += 1
            print("[PATCH] Fixed position tuple unpacking in get_object_pose")

        old_rot = (
            "(\n"
            "            rsp.object_pose.rpy.rw,\n"
            "            rsp.object_pose.rpy.rx,\n"
            "            rsp.object_pose.rpy.ry,\n"
            "            rsp.object_pose.rpy.rz,\n"
            "        ) = rotation"
        )
        new_rot = (
            "# " + PATCH_MARKER + " — safe rotation assignment\n"
            "        _rot = list(rotation) if hasattr(rotation, '__iter__') and rotation is not None else [1, 0, 0, 0]\n"
            "        rsp.object_pose.rpy.rw = _bp_safe_float(_rot[0], 1.0) if len(_rot) > 0 else 1.0\n"
            "        rsp.object_pose.rpy.rx = _bp_safe_float(_rot[1]) if len(_rot) > 1 else 0.0\n"
            "        rsp.object_pose.rpy.ry = _bp_safe_float(_rot[2]) if len(_rot) > 2 else 0.0\n"
            "        rsp.object_pose.rpy.rz = _bp_safe_float(_rot[3]) if len(_rot) > 3 else 0.0"
        )
        if old_rot in content:
            content = content.replace(old_rot, new_rot, 1)
            changes += 1
            print("[PATCH] Fixed rotation tuple unpacking in get_object_pose")

        # Fix 2: recordingState
        old = "rsp.recordingState = result"
        new = 'rsp.recordingState = str(result) if result is not None else ""'
        if old in content:
            content = content.replace(old, new)
            changes += 1
            print("[PATCH] Fixed recordingState assignment")

        # Fix 3: wrap ALL rsp.msg = blocking_start_server() with str()
        rsp_msg_pattern = "rsp.msg = self.server_function.blocking_start_server("
        fix3_count = 0
        search_start = 0
        while True:
            idx = content.find(rsp_msg_pattern, search_start)
            if idx == -1:
                break
            paren_start = idx + len(rsp_msg_pattern) - 1
            paren_end = _find_matching_paren(content, paren_start)
            if paren_end == -1:
                search_start = idx + 1
                continue
            old_expr = content[idx:paren_end + 1]
            call_expr = content[idx + len("rsp.msg = "):paren_end + 1]
            new_expr = f'rsp.msg = str({call_expr} or "")'
            content = content[:idx] + new_expr + content[paren_end + 1:]
            fix3_count += 1
            search_start = idx + len(new_expr)
        if fix3_count:
            changes += fix3_count
            print(f"[PATCH] Wrapped {fix3_count} rsp.msg = blocking_start_server() calls with str()")

        # Fix 4: observation pose loop unpacking
        old = "position, rotation = _pose"
        new = (
            "_p = _pose if isinstance(_pose, (list, tuple)) else (_pose, None)\n"
            "                position = _p[0] if len(_p) >= 1 else (0, 0, 0)\n"
            "                rotation = _p[1] if len(_p) >= 2 else (1, 0, 0, 0)"
        )
        if old in content:
            content = content.replace(old, new)
            changes += 1
            print("[PATCH] Fixed _pose unpacking in get_observation")

        # Fix 5: set literal bugs
        if 'data={"reset", Reset}' in content:
            content = content.replace('data={"reset", Reset}', 'data={"reset": Reset}')
            changes += 1
            print("[PATCH] Fixed reset set literal -> dict")
        if 'data={"detach", detach}' in content:
            content = content.replace('data={"detach", detach}', 'data={"detach": detach}')
            changes += 1
            print("[PATCH] Fixed detach set literal -> dict")

        # Fix 6: Gripper handler string concatenation
        old = 'rsp.msg = front_msg + " " + msg'
        new = 'rsp.msg = front_msg + " " + str(msg or "")'
        if old in content:
            content = content.replace(old, new)
            changes += 1
            print("[PATCH] Fixed gripper handler string concatenation")

        # Fix 7: get_joint_position — guard against non-dict return
        old_get_jp = "        for joint_name in joint_positions:"
        new_get_jp = (
            "        if not isinstance(joint_positions, dict):\n"
            "            _jp_type = type(joint_positions)\n"
            "            print(f'[PATCH] get_joint_position got non-dict: {_jp_type}')\n"
            "            return rsp\n"
            "        for joint_name in joint_positions:"
        )
        if old_get_jp in content:
            content = content.replace(old_get_jp, new_get_jp, 1)
            changes += 1
            print("[PATCH] Fixed get_joint_position string guard")

        # Fix 7b: get_joint_position — dict-as-value
        old_joint_12 = "            joint_state.position = joint_positions[joint_name]"
        new_joint_12 = (
            "            _jval = joint_positions[joint_name]\n"
            "            if isinstance(_jval, str):\n"
            "                if joint_name in {\"error\", \"err\", \"errors\"}:\n"
            "                    print(f'[PATCH] get_joint_position error key: {_jval}')\n"
            "                    continue\n"
            "                joint_state.position = _bp_safe_float(_jval)\n"
            "            elif isinstance(_jval, dict):\n"
            "                continue\n"
            "            else:\n"
            "                _jscalar = np.asarray(_jval).flat[0] if _jval is not None else 0.0\n"
            "                joint_state.position = _bp_safe_float(_jscalar)"
        )
        old_joint_16 = "                joint_state.position = joint_positions[joint_name]"
        new_joint_16 = (
            "                _jval = joint_positions[joint_name]\n"
            "                if isinstance(_jval, str):\n"
            "                    if joint_name in {\"error\", \"err\", \"errors\"}:\n"
            "                        print(f'[PATCH] get_joint_position error key: {_jval}')\n"
            "                        continue\n"
            "                    joint_state.position = _bp_safe_float(_jval)\n"
            "                elif isinstance(_jval, dict):\n"
            "                    continue\n"
            "                else:\n"
            "                    _jscalar = np.asarray(_jval).flat[0] if _jval is not None else 0.0\n"
            "                    joint_state.position = _bp_safe_float(_jscalar)"
        )
        if old_joint_16 in content:
            content = content.replace(old_joint_16, new_joint_16)
            changes += 1
            print("[PATCH] Fixed get_joint_position dict-as-value (16-space)")
        if old_joint_12 in content:
            content = content.replace(old_joint_12, new_joint_12)
            changes += 1
            print("[PATCH] Fixed get_joint_position dict-as-value (12-space)")

    # ── EE pose patches (always check, even on already-base-patched files) ──

    # Fix 8: get_ee_pose — safe unpacking of blocking_start_server result
    if EE_POSE_MARKER not in content:
        old_ee = "position, rotation = self.server_function.blocking_start_server(\n            data={\"isRight\": is_right}, Command=Command.GET_EE_POSE\n        )"
        new_ee = (
            "# " + EE_POSE_MARKER + "\n"
            "        _ee_result = self.server_function.blocking_start_server(\n"
            "            data={\"isRight\": is_right}, Command=Command.GET_EE_POSE\n"
            "        )\n"
            "        _ee = _ee_result if isinstance(_ee_result, (list, tuple)) else (_ee_result, None)\n"
            "        position = _ee[0] if len(_ee) >= 1 else (0, 0, 0)\n"
            "        rotation = _ee[1] if len(_ee) >= 2 else (1, 0, 0, 0)"
        )
        if old_ee in content:
            content = content.replace(old_ee, new_ee, 1)
            changes += 1
            print("[PATCH] Fixed get_ee_pose unpacking in grpc_server.py")
        else:
            ee_pattern = re.compile(
                r'([ \t]*)position, rotation = self\.server_function\.blocking_start_server\(\s*'
                r'data=\{"isRight": is_right\},\s*Command=Command\.GET_EE_POSE\s*\)',
                re.DOTALL
            )
            m = ee_pattern.search(content)
            if m:
                indent = m.group(1)
                replacement = (
                    f"{indent}# {EE_POSE_MARKER}\n"
                    f"{indent}_ee_result = self.server_function.blocking_start_server(\n"
                    f"{indent}    data={{\"isRight\": is_right}}, Command=Command.GET_EE_POSE\n"
                    f"{indent})\n"
                    f"{indent}_ee = _ee_result if isinstance(_ee_result, (list, tuple)) else (_ee_result, None)\n"
                    f"{indent}position = _ee[0] if len(_ee) >= 1 else (0, 0, 0)\n"
                    f"{indent}rotation = _ee[1] if len(_ee) >= 2 else (1, 0, 0, 0)"
                )
                content = content[:m.start()] + replacement + content[m.end():]
                changes += 1
                print("[PATCH] Fixed get_ee_pose unpacking (regex)")
            else:
                print("[PATCH] WARNING: Could not find get_ee_pose unpacking pattern")

    # Fix 9: get_ee_pose rsp assignment — safe array access
    if EE_RSP_MARKER not in content:
        old_ee_rsp = "rsp.ee_pose.position.x, rsp.ee_pose.position.y, rsp.ee_pose.position.z = position"
        new_ee_rsp = (
            "# " + EE_RSP_MARKER + "\n"
            "        _epos = list(position) if hasattr(position, '__iter__') else [0, 0, 0]\n"
            "        rsp.ee_pose.position.x = float(_epos[0]) if len(_epos) > 0 else 0.0\n"
            "        rsp.ee_pose.position.y = float(_epos[1]) if len(_epos) > 1 else 0.0\n"
            "        rsp.ee_pose.position.z = float(_epos[2]) if len(_epos) > 2 else 0.0"
        )
        if old_ee_rsp in content:
            content = content.replace(old_ee_rsp, new_ee_rsp, 1)
            changes += 1
            print("[PATCH] Fixed ee_pose position rsp assignment")

        old_ee_rot_rsp = (
            "(\n"
            "            rsp.ee_pose.rpy.rw,\n"
            "            rsp.ee_pose.rpy.rx,\n"
            "            rsp.ee_pose.rpy.ry,\n"
            "            rsp.ee_pose.rpy.rz,\n"
            "        ) = rotation"
        )
        new_ee_rot_rsp = (
            "# " + EE_RSP_MARKER + " — rotation\n"
            "        _erot = list(rotation) if hasattr(rotation, '__iter__') and rotation is not None else [1, 0, 0, 0]\n"
            "        rsp.ee_pose.rpy.rw = float(_erot[0]) if len(_erot) > 0 else 1.0\n"
            "        rsp.ee_pose.rpy.rx = float(_erot[1]) if len(_erot) > 1 else 0.0\n"
            "        rsp.ee_pose.rpy.ry = float(_erot[2]) if len(_erot) > 2 else 0.0\n"
            "        rsp.ee_pose.rpy.rz = float(_erot[3]) if len(_erot) > 3 else 0.0"
        )
        if old_ee_rot_rsp in content:
            content = content.replace(old_ee_rot_rsp, new_ee_rot_rsp, 1)
            changes += 1
            print("[PATCH] Fixed ee_pose rotation rsp assignment")

    # ── Camera bytes fix (handle numpy arrays OR already-bytes) ──

    if CAMERA_BYTES_MARKER not in content:
        # Fix 10: get_camera_data — rgb_camera and depth_camera might already be bytes
        # The original code is:
        #     if rgb_camera is not None:
        #         rsp.color_image.data = rgb_camera.tobytes()
        # We replace the entire if block to handle bytes or numpy array
        old_rgb_block = "if rgb_camera is not None:\n            rsp.color_image.data = rgb_camera.tobytes()"
        new_rgb_block = (
            "# " + CAMERA_BYTES_MARKER + " — handle bytes or numpy array\n"
            "        if isinstance(rgb_camera, bytes):\n"
            "            rsp.color_image.data = rgb_camera\n"
            "        elif rgb_camera is not None:\n"
            "            rsp.color_image.data = rgb_camera.tobytes()"
        )
        if old_rgb_block in content:
            content = content.replace(old_rgb_block, new_rgb_block, 1)
            changes += 1
            print("[PATCH] Fixed rgb_camera block to handle bytes")

        old_depth_block = "if depth_camera is not None:\n            rsp.depth_image.data = depth_camera.tobytes()"
        new_depth_block = (
            "# " + CAMERA_BYTES_MARKER + " — handle bytes or numpy array\n"
            "        if isinstance(depth_camera, bytes):\n"
            "            rsp.depth_image.data = depth_camera\n"
            "        elif depth_camera is not None:\n"
            "            rsp.depth_image.data = depth_camera.tobytes()"
        )
        if old_depth_block in content:
            content = content.replace(old_depth_block, new_depth_block, 1)
            changes += 1
            print("[PATCH] Fixed depth_camera block to handle bytes")

    # ── Camera intrinsics fix (populate fx, fy, ppx, ppy in response) ──

    if CAMERA_INTRINSICS_MARKER not in content:
        # Find where camera_info.width and camera_info.height are set and add intrinsics after
        # Pattern: rsp.camera_info.height = camera_info["height"]
        # or: rsp.camera_info.height = camera_info.get("height", ...)
        intrinsics_patterns = [
            # Pattern 1: Direct dict access
            (
                'rsp.camera_info.height = camera_info["height"]',
                'rsp.camera_info.height = camera_info["height"]\n'
                '        # ' + CAMERA_INTRINSICS_MARKER + '\n'
                '        rsp.camera_info.fx = float(camera_info.get("fx", camera_info.get("width", 1280)))\n'
                '        rsp.camera_info.fy = float(camera_info.get("fy", camera_info.get("height", 720)))\n'
                '        rsp.camera_info.ppx = float(camera_info.get("ppx", camera_info.get("width", 1280) / 2.0))\n'
                '        rsp.camera_info.ppy = float(camera_info.get("ppy", camera_info.get("height", 720) / 2.0))'
            ),
            # Pattern 2: Using .get() method
            (
                'rsp.camera_info.height = camera_info.get("height"',
                None  # Will use regex for this
            ),
        ]

        # Try direct pattern replacement first
        old_height, new_height = intrinsics_patterns[0]
        if old_height in content:
            content = content.replace(old_height, new_height, 1)
            changes += 1
            print("[PATCH] Added camera intrinsics (fx, fy, ppx, ppy) after height assignment")
        else:
            # Try regex pattern for .get() style
            intrinsics_regex = re.compile(
                r'([ \t]*)(rsp\.camera_info\.height\s*=\s*camera_info\.get\([^)]+\))',
                re.MULTILINE
            )
            m = intrinsics_regex.search(content)
            if m:
                indent = m.group(1)
                replacement = (
                    m.group(0) + '\n'
                    f'{indent}# {CAMERA_INTRINSICS_MARKER}\n'
                    f'{indent}rsp.camera_info.fx = float(camera_info.get("fx", camera_info.get("width", 1280)))\n'
                    f'{indent}rsp.camera_info.fy = float(camera_info.get("fy", camera_info.get("height", 720)))\n'
                    f'{indent}rsp.camera_info.ppx = float(camera_info.get("ppx", camera_info.get("width", 1280) / 2.0))\n'
                    f'{indent}rsp.camera_info.ppy = float(camera_info.get("ppy", camera_info.get("height", 720) / 2.0))'
                )
                content = content[:m.start()] + replacement + content[m.end():]
                changes += 1
                print("[PATCH] Added camera intrinsics (fx, fy, ppx, ppy) via regex")
            else:
                # Try finding after color_info (alternative field name)
                color_info_regex = re.compile(
                    r'([ \t]*)(rsp\.color_info\.height\s*=\s*[^\n]+)',
                    re.MULTILINE
                )
                m = color_info_regex.search(content)
                if m:
                    indent = m.group(1)
                    replacement = (
                        m.group(0) + '\n'
                        f'{indent}# {CAMERA_INTRINSICS_MARKER}\n'
                        f'{indent}rsp.color_info.fx = float(camera_info.get("fx", camera_info.get("width", 1280)) if isinstance(camera_info, dict) else 1280)\n'
                        f'{indent}rsp.color_info.fy = float(camera_info.get("fy", camera_info.get("height", 720)) if isinstance(camera_info, dict) else 720)\n'
                        f'{indent}rsp.color_info.ppx = float(camera_info.get("ppx", camera_info.get("width", 1280) / 2.0) if isinstance(camera_info, dict) else 640.0)\n'
                        f'{indent}rsp.color_info.ppy = float(camera_info.get("ppy", camera_info.get("height", 720) / 2.0) if isinstance(camera_info, dict) else 360.0)'
                    )
                    content = content[:m.start()] + replacement + content[m.end():]
                    changes += 1
                    print("[PATCH] Added camera intrinsics to color_info via regex")
                else:
                    print("[PATCH] WARNING: Could not find camera_info/color_info height assignment pattern for intrinsics")

    # ── Camera encoding fix (set color_image.format and depth_image.format) ──

    if CAMERA_ENCODING_MARKER not in content:
        # Find where color_image.data is set and add format before it
        # Pattern: rsp.color_image.data = ...
        encoding_patterns = [
            # Pattern 1: After the bytes marker patch (most likely)
            (
                '# ' + CAMERA_BYTES_MARKER + ' — handle bytes or numpy array\n'
                '        if isinstance(rgb_camera, bytes):\n'
                '            rsp.color_image.data = rgb_camera',
                '# ' + CAMERA_BYTES_MARKER + ' — handle bytes or numpy array\n'
                '        # ' + CAMERA_ENCODING_MARKER + '\n'
                '        _rgb_enc = current_camera.get("rgb_encoding", "") if isinstance(current_camera, dict) else ""\n'
                '        if not _rgb_enc:\n'
                '            _rgb_enc = current_camera.get("encoding", "raw_rgb_uint8") if isinstance(current_camera, dict) else "raw_rgb_uint8"\n'
                '        rsp.color_image.format = _rgb_enc\n'
                '        if isinstance(rgb_camera, bytes):\n'
                '            rsp.color_image.data = rgb_camera'
            ),
        ]

        old_enc, new_enc = encoding_patterns[0]
        if old_enc in content:
            content = content.replace(old_enc, new_enc, 1)
            changes += 1
            print("[PATCH] Added camera encoding format assignment")
        else:
            # Try regex for original pattern (before bytes marker patch)
            enc_regex = re.compile(
                r'([ \t]*)(if rgb_camera is not None:\s*\n\s*rsp\.color_image\.data\s*=)',
                re.MULTILINE
            )
            m = enc_regex.search(content)
            if m:
                indent = m.group(1)
                replacement = (
                    f'{indent}# {CAMERA_ENCODING_MARKER}\n'
                    f'{indent}_rgb_enc = current_camera.get("rgb_encoding", "") if isinstance(current_camera, dict) else ""\n'
                    f'{indent}if not _rgb_enc:\n'
                    f'{indent}    _rgb_enc = current_camera.get("encoding", "raw_rgb_uint8") if isinstance(current_camera, dict) else "raw_rgb_uint8"\n'
                    f'{indent}rsp.color_image.format = _rgb_enc\n'
                    f'{m.group(0)}'
                )
                content = content[:m.start()] + replacement + content[m.end():]
                changes += 1
                print("[PATCH] Added camera encoding format via regex")
            else:
                print("[PATCH] WARNING: Could not find color_image.data assignment pattern for encoding")

    # ── Write result ──

    if not already_base_patched:
        content = f"# {PATCH_MARKER} applied\n" + content

    if changes == 0:
        print("[PATCH] No matching patterns found — file may already be fixed")
        sys.exit(0)

    with open(GRPC_SERVER, "w") as f:
        f.write(content)

    print(f"[PATCH] Successfully patched {GRPC_SERVER} ({changes} fixes)")


if __name__ == "__main__":
    patch_file()
