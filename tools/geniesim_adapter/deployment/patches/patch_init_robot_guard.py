#!/usr/bin/env python3
"""Patch handle_init_robot to catch any unhandled exception from _init_robot_cfg.

When _init_robot_cfg raises (e.g. RuntimeError from physics backend timeout),
the handler must still set data_to_send so the gRPC thread is unblocked.
Without this, an unhandled exception leaves data_to_send=None, causing the
gRPC thread to block forever.

Idempotent — re-running is a no-op.
"""
import os
import re
import sys

GENIESIM_ROOT = os.environ.get("GENIESIM_ROOT", "/opt/geniesim")
CMD_CTRL = os.path.join(
    GENIESIM_ROOT,
    "source", "data_collection", "server", "command_controller.py",
)

PATCH_MARKER = "BlueprintPipeline init_robot_guard patch"


def patch_file():
    if not os.path.isfile(CMD_CTRL):
        print(f"[PATCH] command_controller.py not found at {CMD_CTRL}")
        sys.exit(0)

    with open(CMD_CTRL, "r") as f:
        content = f.read()

    if PATCH_MARKER in content:
        print("[PATCH] init_robot_guard already applied -- skipping")
        sys.exit(0)

    # Find the _init_robot_cfg call inside handle_init_robot
    # It looks like:
    #         self._init_robot_cfg(
    #             robot_cfg=robot_cfg_file,
    #             ...
    #         )
    # We want to wrap it in a try/except that sets data_to_send on failure.

    # Look for the pattern starting with self._init_robot_cfg( inside handle_init_robot
    handler_re = re.compile(
        r"(    def handle_init_robot\(self\):.*?\n)"
        r"(.*?)"  # everything before _init_robot_cfg
        r"(        self\._init_robot_cfg\(\n"
        r"(?:.*?\n)*?"  # all the kwargs
        r"        \)\n)",
        re.DOTALL,
    )
    match = handler_re.search(content)
    if not match:
        print("[PATCH] Could not find _init_robot_cfg call pattern -- skipping")
        sys.exit(0)

    init_cfg_start = match.start(3)
    init_cfg_end = match.end(3)
    init_cfg_text = match.group(3)

    # Re-indent the _init_robot_cfg call by 4 spaces (add one try level)
    indented_lines = []
    for line in init_cfg_text.split("\n"):
        if line.strip():
            indented_lines.append("    " + line)
        else:
            indented_lines.append(line)
    indented_cfg = "\n".join(indented_lines)

    replacement = (
        f"        # {PATCH_MARKER}\n"
        f"        try:\n"
        f"{indented_cfg}"
        f"        except Exception as _init_err:\n"
        f"            print(f'[PATCH] handle_init_robot: _init_robot_cfg failed: {{_init_err}}')\n"
        f"            import traceback as _tb\n"
        f"            _tb.print_exc()\n"
        f"            self.data_to_send = f'error: init_robot_cfg failed: {{_init_err}}'\n"
        f"            return\n"
    )

    content = content[:init_cfg_start] + replacement + content[init_cfg_end:]

    with open(CMD_CTRL, "w") as f:
        f.write(content)

    print(f"[PATCH] Wrapped _init_robot_cfg in try/except in handle_init_robot")
    print(f"[PATCH] Successfully patched {CMD_CTRL}")


if __name__ == "__main__":
    patch_file()
