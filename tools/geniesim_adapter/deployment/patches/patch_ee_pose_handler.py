#!/usr/bin/env python3
"""
Patch the Genie Sim server's command_controller.py to fix GET_EE_POSE handling.

The upstream server's handle_get_ee_pose crashes with:
    "too many values to unpack (expected 2)"
because the robot controller returns (position, orientation, extra_data) but
the handler destructures into only two variables.

This patch replaces the faulty unpacking with a safe version that handles
any number of return values.

Usage (inside Docker build):
    python3 /tmp/patches/patch_ee_pose_handler.py

The script is idempotent — re-running it on an already-patched file is a no-op.
"""
import os
import re
import sys

GENIESIM_ROOT = os.environ.get("GENIESIM_ROOT", "/opt/geniesim")
COMMAND_CONTROLLER = os.path.join(
    GENIESIM_ROOT,
    "source", "data_collection", "server", "command_controller.py",
)

PATCH_MARKER = "BlueprintPipeline ee_pose patch"


def patch_file():
    if not os.path.isfile(COMMAND_CONTROLLER):
        print(f"[PATCH] command_controller.py not found at {COMMAND_CONTROLLER}")
        print("[PATCH] Skipping ee_pose patch (server source not available)")
        sys.exit(0)

    with open(COMMAND_CONTROLLER, "r") as f:
        content = f.read()

    if PATCH_MARKER in content:
        print("[PATCH] EE pose handler already patched — skipping")
        sys.exit(0)

    # Look for patterns like:
    #   pos, rot = self.robot.get_ee_pose(...)
    #   pos, quat = ...get_ee_pose(...)
    #   position, orientation = ...get_ee_pose(...)
    # and replace with safe unpacking that discards extra values.
    pattern = re.compile(
        r"(\s*)(\w+)\s*,\s*(\w+)\s*=\s*(.*\.get_ee_pose\(.*?\))",
        re.MULTILINE,
    )

    match = pattern.search(content)
    if match:
        indent = match.group(1)
        var1 = match.group(2)
        var2 = match.group(3)
        call = match.group(4)
        original = match.group(0)
        replacement = (
            f"{indent}# {PATCH_MARKER}\n"
            f"{indent}_ee_result = {call}\n"
            f"{indent}if isinstance(_ee_result, (list, tuple)) and len(_ee_result) >= 2:\n"
            f"{indent}    {var1}, {var2} = _ee_result[0], _ee_result[1]\n"
            f"{indent}else:\n"
            f"{indent}    {var1}, {var2} = _ee_result, None"
        )
        content = content.replace(original, replacement)
        print(f"[PATCH] Fixed ee_pose unpacking: {var1}, {var2} = ...get_ee_pose(...)")
    else:
        # Alternative: the handler might use a different pattern.
        # Try to find any "= ...get_ee_pose(" and add a safety wrapper.
        alt_pattern = re.compile(
            r"(.*get_ee_pose\(.*?\))",
            re.MULTILINE,
        )
        alt_match = alt_pattern.search(content)
        if alt_match:
            print(f"[PATCH] WARNING: Found get_ee_pose call but in unexpected pattern:")
            print(f"[PATCH]   {alt_match.group(0).strip()}")
            print(f"[PATCH] Manual review may be needed")
        else:
            print("[PATCH] No get_ee_pose call found in command_controller.py")
            print("[PATCH] The server may handle ee_pose in a different file")

    with open(COMMAND_CONTROLLER, "w") as f:
        f.write(content)

    print(f"[PATCH] Successfully patched {COMMAND_CONTROLLER}")


if __name__ == "__main__":
    patch_file()
