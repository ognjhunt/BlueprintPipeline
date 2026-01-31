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

    if PATCH_MARKER in content:
        print("[PATCH] grpc_server.py already patched — skipping")
        sys.exit(0)

    changes = 0

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
    # Insert after the last import line
    import_end = 0
    for m in re.finditer(r'^(?:import |from )\S+.*$', content, re.MULTILINE):
        import_end = m.end()
    if import_end > 0:
        content = content[:import_end] + "\n" + safe_float_helper + content[import_end:]
        changes += 1
        print("[PATCH] Injected _bp_safe_float helper")

    # Fix 1: object_pose unpacking in get_object_pose
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

    # Fix 1b: position/rotation tuple unpacking in rsp assignment
    # position might be a numpy array or list with >3 elements
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

    # Fix 2: recordingState assignment
    old = "rsp.recordingState = result"
    new = 'rsp.recordingState = str(result) if result is not None else ""'
    if old in content:
        content = content.replace(old, new)
        changes += 1
        print("[PATCH] Fixed recordingState assignment in get_observation")

    # Fix 3: COMPREHENSIVE — wrap ALL rsp.msg = self.server_function.blocking_start_server(...)
    # Handle multi-line calls by finding balanced parentheses.
    rsp_msg_pattern = "rsp.msg = self.server_function.blocking_start_server("
    fix3_count = 0
    search_start = 0
    while True:
        idx = content.find(rsp_msg_pattern, search_start)
        if idx == -1:
            break
        # Find the opening paren of blocking_start_server(
        paren_start = idx + len(rsp_msg_pattern) - 1  # index of '('
        paren_end = _find_matching_paren(content, paren_start)
        if paren_end == -1:
            search_start = idx + 1
            continue
        # Replace: rsp.msg = self.server_function.blocking_start_server(...)
        # With:    rsp.msg = str(self.server_function.blocking_start_server(...) or "")
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

    # Fix 5: set literal bugs (idempotent — only matches set syntax)
    if 'data={"reset", Reset}' in content:
        content = content.replace('data={"reset", Reset}', 'data={"reset": Reset}')
        changes += 1
        print("[PATCH] Fixed reset set literal -> dict")

    if 'data={"detach", detach}' in content:
        content = content.replace('data={"detach", detach}', 'data={"detach": detach}')
        changes += 1
        print("[PATCH] Fixed detach set literal -> dict")

    # Fix 6: Gripper handler — msg is used in string concatenation
    # rsp.msg = front_msg + " " + msg  where msg may be a dict
    old = 'rsp.msg = front_msg + " " + msg'
    new = 'rsp.msg = front_msg + " " + str(msg or "")'
    if old in content:
        content = content.replace(old, new)
        changes += 1
        print("[PATCH] Fixed gripper handler string concatenation")

    # Fix 8: get_joint_position — guard against string return (error message)
    # When robot articulation init fails, blocking_start_server returns a
    # string error. Iterating over it treats each char as a "joint name".
    old_get_jp = "        for joint_name in joint_positions:"
    new_get_jp = (
        "        if not isinstance(joint_positions, dict):\n"
        "            print(f'[PATCH] get_joint_position got non-dict: {type(joint_positions)}')\n"
        "            return rsp\n"
        "        for joint_name in joint_positions:"
    )
    if old_get_jp in content:
        content = content.replace(old_get_jp, new_get_jp, 1)
        changes += 1
        print("[PATCH] Fixed get_joint_position string guard")

    # Fix 7: get_joint_position — joint_positions[joint_name] may be dict
    # Appears in JointService (12-space indent) and ObservationService (16-space indent)
    old_joint_12 = "            joint_state.position = joint_positions[joint_name]"
    new_joint_12 = (
        "            _jval = joint_positions[joint_name]\n"
        "            joint_state.position = float(np.asarray(_jval).flat[0]) if not isinstance(_jval, dict) else 0.0"
    )
    old_joint_16 = "                joint_state.position = joint_positions[joint_name]"
    new_joint_16 = (
        "                _jval = joint_positions[joint_name]\n"
        "                joint_state.position = float(np.asarray(_jval).flat[0]) if not isinstance(_jval, dict) else 0.0"
    )
    if old_joint_16 in content:
        content = content.replace(old_joint_16, new_joint_16)
        changes += 1
        print("[PATCH] Fixed get_joint_position dict-as-value (16-space indent)")
    if old_joint_12 in content:
        content = content.replace(old_joint_12, new_joint_12)
        changes += 1
        print("[PATCH] Fixed get_joint_position dict-as-value (12-space indent)")

    # Add patch marker as a comment at the top of the file
    content = f"# {PATCH_MARKER} applied\n" + content

    if changes == 0:
        print("[PATCH] No matching patterns found — file may already be fixed")
        sys.exit(0)

    with open(GRPC_SERVER, "w") as f:
        f.write(content)

    print(f"[PATCH] Successfully patched {GRPC_SERVER} ({changes} fixes)")


if __name__ == "__main__":
    patch_file()
