#!/usr/bin/env python3
"""
Patch the Genie Sim server's command_controller.py to fix GET_EE_POSE handling.

The upstream server's handle_get_ee_pose crashes with:
    "too many values to unpack (expected 2)"
because the robot controller returns (position, orientation, extra_data) but
the handler destructures into only two variables.

This patch:
1. Replaces the faulty unpacking with a safe version that handles any number
   of return values (broadened regex handles multi-line and N-variable cases).
2. Injects a monkey-patch wrapper on the robot's get_ee_pose method to always
   return exactly 2 values, as a safety net for unmatched code paths.

Usage (inside Docker build):
    python3 /tmp/patches/patch_ee_pose_handler.py

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

PATCH_MARKER = "BlueprintPipeline ee_pose patch"

# Monkey-patch wrapper injected at class level to ensure get_ee_pose always
# returns exactly 2 values regardless of what the robot controller does.
MONKEY_PATCH_SNIPPET = textwrap.dedent("""\

    # --- BEGIN {marker} (monkey-patch) ---
    @staticmethod
    def _bp_wrap_ee_pose(robot):
        \"\"\"Wrap robot.get_ee_pose to always return exactly (pos, rot).\"\"\"
        _orig_fn = getattr(robot, 'get_ee_pose', None)
        if _orig_fn is None or getattr(_orig_fn, '_bp_wrapped', False):
            return
        def _safe_get_ee_pose(*args, **kwargs):
            try:
                result = _orig_fn(*args, **kwargs)
            except Exception as _e:
                print(f'[PATCH] get_ee_pose call failed: {{_e}}')
                return None, None
            if isinstance(result, (list, tuple)):
                if len(result) >= 2:
                    return result[0], result[1]
                elif len(result) == 1:
                    return result[0], None
                else:
                    return None, None
            return result, None
        _safe_get_ee_pose._bp_wrapped = True
        robot.get_ee_pose = _safe_get_ee_pose
        print('[PATCH] Wrapped robot.get_ee_pose for safe 2-value unpacking')
    # --- END {marker} (monkey-patch) ---
""")


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

    patched = False

    # --- Primary pattern: N-variable unpacking from get_ee_pose ---
    # Handles: pos, rot = ...get_ee_pose(...)
    #          pos, rot, extra = ...get_ee_pose(...)
    #          position, orientation = ...get_ee_pose(some_arg)
    # Uses .*? with re.DOTALL to cross newlines inside parens.
    pattern = re.compile(
        r"(\s*)((\w+)(?:\s*,\s*(\w+))+)\s*=\s*(.*?\.get_ee_pose\(.*?\))",
        re.MULTILINE | re.DOTALL,
    )

    match = pattern.search(content)
    if match:
        indent = match.group(1)
        full_vars = match.group(2)  # e.g. "pos, rot" or "pos, rot, extra"
        call = match.group(5)
        original = match.group(0)

        # Parse variable names, take first two
        var_names = [v.strip() for v in full_vars.split(",")]
        var1 = var_names[0]
        var2 = var_names[1] if len(var_names) >= 2 else "None"

        replacement = (
            f"{indent}# {PATCH_MARKER}\n"
            f"{indent}_ee_result = {call}\n"
            f"{indent}if isinstance(_ee_result, (list, tuple)) and len(_ee_result) >= 2:\n"
            f"{indent}    {var1}, {var2} = _ee_result[0], _ee_result[1]\n"
            f"{indent}elif isinstance(_ee_result, (list, tuple)) and len(_ee_result) == 1:\n"
            f"{indent}    {var1}, {var2} = _ee_result[0], None\n"
            f"{indent}else:\n"
            f"{indent}    {var1}, {var2} = _ee_result, None"
        )
        # Assign remaining vars to None
        for extra_var in var_names[2:]:
            replacement += f"\n{indent}{extra_var} = None"

        content = content.replace(original, replacement)
        print(f"[PATCH] Fixed ee_pose unpacking: {', '.join(var_names)} = ...get_ee_pose(...)")
        patched = True
    else:
        # Fallback: find any "= ...get_ee_pose(" on a single line and wrap in try/except
        alt_pattern = re.compile(
            r"([ \t]+)(.*get_ee_pose\([^)]*\).*)",
            re.MULTILINE,
        )
        alt_match = alt_pattern.search(content)
        if alt_match:
            indent = alt_match.group(1)
            line_text = alt_match.group(0)
            print(f"[PATCH] Found get_ee_pose call in unexpected pattern:")
            print(f"[PATCH]   {line_text.strip()[:120]}")
            replacement = (
                f"{indent}# {PATCH_MARKER} - safety wrapper\n"
                f"{indent}try:\n"
                f"    {line_text}\n"
                f"{indent}except Exception as _ee_err:\n"
                f"{indent}    print(f'[PATCH] get_ee_pose unpacking failed: {{_ee_err}}')\n"
                f"{indent}    pos, rot = None, None"
            )
            content = content.replace(line_text, replacement)
            print(f"[PATCH] Applied safety wrapper to get_ee_pose call")
            patched = True
        else:
            print("[PATCH] No get_ee_pose call found in command_controller.py")
            print("[PATCH] The server may handle ee_pose in a different file")

    # --- Inject monkey-patch wrapper at class level ---
    # Find the class definition and append the wrapper method.
    # Also inject a call to _bp_wrap_ee_pose(self.robot) after robot init.
    snippet = MONKEY_PATCH_SNIPPET.format(marker=PATCH_MARKER)

    # Find method indent
    method_indent = "    "
    m = re.search(r"^([ \t]+)def \w+\(self", content, re.MULTILINE)
    if m:
        method_indent = m.group(1)

    indented_snippet = "\n".join(
        (method_indent + line) if line.strip() else line
        for line in snippet.splitlines()
    ) + "\n"

    content = content.rstrip() + "\n" + indented_snippet

    # Inject the wrapper call after robot assignment in init_robot handler.
    # Look for: self.robot = ... (in INIT_ROBOT handler)
    robot_assign = re.search(
        r"([ \t]+)(self\.robot\s*=\s*.+)\n",
        content,
    )
    if robot_assign:
        indent_ra = robot_assign.group(1)
        end_pos = robot_assign.end()
        wrap_call = f"{indent_ra}self._bp_wrap_ee_pose(self.robot)  # {PATCH_MARKER}\n"
        # Only inject if not already present
        if "_bp_wrap_ee_pose" not in content[end_pos:end_pos + 200]:
            content = content[:end_pos] + wrap_call + content[end_pos:]
            print("[PATCH] Injected _bp_wrap_ee_pose(self.robot) after robot assignment")
    else:
        print("[PATCH] WARNING: Could not find self.robot assignment to inject wrapper call")
        print("[PATCH] Monkey-patch wrapper added but not auto-wired — call _bp_wrap_ee_pose(robot) manually")

    with open(COMMAND_CONTROLLER, "w") as f:
        f.write(content)

    print(f"[PATCH] Successfully patched {COMMAND_CONTROLLER}")


if __name__ == "__main__":
    patch_file()
