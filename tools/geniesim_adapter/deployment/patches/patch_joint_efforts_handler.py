#!/usr/bin/env python3
"""
Patch Genie Sim server to populate JointState.effort with real PhysX efforts.

This patch searches for the get_joint_position handler in the server sources
and injects effort values from the robot articulation (not inverse dynamics).
It is idempotent and safe to run multiple times.
"""
import os
import re
import sys

GENIESIM_ROOT = os.environ.get("GENIESIM_ROOT", "/opt/geniesim")
SERVER_DIR = os.path.join(GENIESIM_ROOT, "source", "data_collection", "server")
PATCH_MARKER = "BlueprintPipeline joint_efforts patch"
LEGACY_MARKER = "BlueprintPipeline joint_efforts handler patch"


def _find_target_file() -> str | None:
    if not os.path.isdir(SERVER_DIR):
        return None
    for root, _dirs, files in os.walk(SERVER_DIR):
        for fname in files:
            if not fname.endswith(".py"):
                continue
            path = os.path.join(root, fname)
            try:
                content = open(path, "r").read()
            except Exception:
                continue
            if "def get_joint_position" in content and "JointState" in content:
                return path
    return None


def _inject_helper(content: str, method_indent: str) -> str:
    helper = f"""
{method_indent}# --- BEGIN {PATCH_MARKER} ---
{method_indent}def _bp_get_real_joint_efforts(self):
{method_indent}    \"\"\"Get joint efforts from robot articulation.\"\"\"
{method_indent}    try:
{method_indent}        robot = None
{method_indent}        server_fn = getattr(self, "server_function", None)
{method_indent}        cmd = getattr(server_fn, "command_controller", None) or server_fn
{method_indent}        robot = getattr(cmd, "robot", None)
{method_indent}        if robot is None:
{method_indent}            return None
{method_indent}        efforts = None
{method_indent}        if hasattr(robot, "get_applied_joint_efforts"):
{method_indent}            efforts = robot.get_applied_joint_efforts()
{method_indent}        elif hasattr(robot, "get_measured_joint_efforts"):
{method_indent}            efforts = robot.get_measured_joint_efforts()
{method_indent}        elif hasattr(robot, "get_joint_efforts"):
{method_indent}            efforts = robot.get_joint_efforts()
{method_indent}        if efforts is None:
{method_indent}            return None
{method_indent}        try:
{method_indent}            return [float(x) for x in efforts]
{method_indent}        except Exception:
{method_indent}            return list(efforts)
{method_indent}    except Exception:
{method_indent}        return None
{method_indent}# --- END {PATCH_MARKER} ---
"""
    return content + "\n" + helper + "\n"


def _patch_file(path: str) -> None:
    with open(path, "r") as f:
        content = f.read()

    if PATCH_MARKER in content or LEGACY_MARKER in content:
        print("[PATCH] joint_efforts handler already patched — skipping")
        return

    # Locate class definition around get_joint_position to determine indent
    class_match = None
    class_pattern = re.compile(r"^class\\s+(\\w+).*?:", re.MULTILINE)
    for m in class_pattern.finditer(content):
        if "def get_joint_position" in content[m.end():]:
            class_match = m
            break
    method_indent = "    "
    if class_match:
        # Find method indentation within the class
        class_body = content[class_match.end():]
        method_match = re.search(r"^([ \\t]+)def\\s+\\w+\\(self", class_body, re.MULTILINE)
        if method_match:
            method_indent = method_match.group(1)

    # Inject helper at end of file (safe for idempotency)
    patched = _inject_helper(content.rstrip(), method_indent)

    # Inject effort mapping inside get_joint_position
    # 1) After joint_positions assignment, build effort map.
    jp_pattern = re.compile(r"(\\n[ \\t]+joint_positions\\s*=.*\\n)")
    effort_block = (
        "\n        # " + PATCH_MARKER + " — effort mapping\n"
        "        _bp_efforts = None\n"
        "        try:\n"
        "            _bp_efforts = self._bp_get_real_joint_efforts()\n"
        "        except Exception:\n"
        "            _bp_efforts = None\n"
        "        _bp_effort_map = {}\n"
        "        if isinstance(_bp_efforts, dict):\n"
        "            _bp_effort_map = _bp_efforts\n"
        "        elif isinstance(_bp_efforts, (list, tuple)):\n"
        "            try:\n"
        "                if isinstance(joint_positions, dict):\n"
        "                    _bp_effort_map = {name: _bp_efforts[i] for i, name in enumerate(joint_positions)}\n"
        "                elif isinstance(joint_names, (list, tuple)):\n"
        "                    _bp_effort_map = {name: _bp_efforts[i] for i, name in enumerate(joint_names)}\n"
        "            except Exception:\n"
        "                _bp_effort_map = {}\n"
    )
    if jp_pattern.search(patched):
        patched = jp_pattern.sub(r"\\1" + effort_block, patched, count=1)

    # 2) Before return rsp, inject effort assignment for any states list
    return_pattern = re.compile(r"(\\n[ \\t]+return\\s+rsp\\s*\\n)")
    effort_apply = (
        "\n        # " + PATCH_MARKER + " — apply efforts to response\n"
        "        try:\n"
        "            if hasattr(rsp, 'states') and _bp_effort_map:\n"
        "                for _st in rsp.states:\n"
        "                    if hasattr(_st, 'name') and _st.name in _bp_effort_map:\n"
        "                        _st.effort = float(_bp_effort_map[_st.name])\n"
        "        except Exception:\n"
        "            pass\n"
    )
    if return_pattern.search(patched):
        patched = return_pattern.sub(effort_apply + r"\\1", patched, count=1)

    with open(path, "w") as f:
        f.write(patched)
    print(f"[PATCH] joint_efforts handler injected into {path}")


def main() -> None:
    target = _find_target_file()
    if not target:
        print(f"[PATCH] No server file with get_joint_position found under {SERVER_DIR}")
        sys.exit(0)
    _patch_file(target)


if __name__ == "__main__":
    main()
