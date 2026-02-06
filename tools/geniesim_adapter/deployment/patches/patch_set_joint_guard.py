#!/usr/bin/env python3
"""
Patch the Genie Sim server's command_controller.py to guard
handle_set_joint_position against articulation being None.

The server crashes at the line:
    target_joint_indices = [articulation.get_dof_index(name) for name in target_joint_names]
when articulation is None (e.g. before async initialization completes).

This patch injects a None check at the top of the list-comprehension block
so that when articulation is None, it sets self.data_to_send = "error" and
returns instead of crashing.

Usage (inside Docker build or at runtime):
    python3 /tmp/patches/patch_set_joint_guard.py

Idempotent -- re-running is a no-op.
"""
import os
import re
import sys

GENIESIM_ROOT = os.environ.get("GENIESIM_ROOT", "/opt/geniesim")
CMD_CTRL = os.path.join(
    GENIESIM_ROOT,
    "source", "data_collection", "server", "command_controller.py",
)

PATCH_MARKER = "BlueprintPipeline set_joint_guard patch"


def patch_file():
    if not os.path.isfile(CMD_CTRL):
        print(f"[PATCH] command_controller.py not found at {CMD_CTRL}")
        sys.exit(0)

    with open(CMD_CTRL, "r") as f:
        content = f.read()

    if PATCH_MARKER in content:
        print("[PATCH] set_joint_guard already applied -- skipping")
        sys.exit(0)

    # ------------------------------------------------------------------ #
    # Strategy: find the line that does
    #   target_joint_indices = [articulation.get_dof_index(name) ...
    # inside handle_set_joint_position, and inject a guard block
    # immediately before it that checks whether articulation is None.
    # ------------------------------------------------------------------ #

    # Locate the handler method
    handler_match = re.search(
        r"^([ \t]+)def handle_set_joint_position\(self.*?\):",
        content,
        re.MULTILINE,
    )
    if not handler_match:
        print("[PATCH] handle_set_joint_position not found -- skipping")
        sys.exit(0)

    handler_start = handler_match.start()
    handler_indent = handler_match.group(1)
    body_indent = handler_indent + "    "  # one level deeper

    # Find the crash line within the handler body
    # Pattern:  target_joint_indices = [articulation.get_dof_index(name) for name in target_joint_names]
    crash_pattern = re.compile(
        r"^([ \t]+)(target_joint_indices\s*=\s*\[articulation\.get_dof_index\(name\)\s+for\s+name\s+in\s+target_joint_names\])",
        re.MULTILINE,
    )
    crash_match = crash_pattern.search(content, handler_start)
    if not crash_match:
        # Fallback: look for any articulation.get_dof_index call in the handler
        crash_pattern_alt = re.compile(
            r"^([ \t]+)(.*articulation\.get_dof_index.*)",
            re.MULTILINE,
        )
        crash_match = crash_pattern_alt.search(content, handler_start)

    if not crash_match:
        print("[PATCH] Could not find articulation.get_dof_index call in handle_set_joint_position -- skipping")
        sys.exit(0)

    crash_indent = crash_match.group(1)

    # We also need to find where 'articulation' is assigned so we inject
    # the guard after the assignment but before the crash line.
    # Typical pattern:
    #     articulation = self.ui_builder.articulation
    # If we can find it, inject right after.  Otherwise inject right before
    # the crash line.
    assign_pattern = re.compile(
        r"^([ \t]+)(articulation\s*=\s*self\.ui_builder\.articulation)\s*\n",
        re.MULTILINE,
    )
    assign_match = assign_pattern.search(content, handler_start)

    # Make sure the assignment is inside the same handler (before the crash line)
    if assign_match and assign_match.start() < crash_match.start():
        # Inject guard immediately after the assignment line
        insert_pos = assign_match.end()
        guard_indent = crash_indent
    else:
        # Inject guard right before the crash line
        insert_pos = crash_match.start()
        guard_indent = crash_indent

    guard_code = (
        f"{guard_indent}# {PATCH_MARKER}\n"
        f"{guard_indent}if articulation is None:\n"
        f"{guard_indent}    print('[PATCH] handle_set_joint_position: articulation is None -- returning error')\n"
        f"{guard_indent}    self.data_to_send = \"error: articulation not initialized\"\n"
        f"{guard_indent}    return\n"
    )

    content = content[:insert_pos] + guard_code + content[insert_pos:]

    with open(CMD_CTRL, "w") as f:
        f.write(content)

    print(f"[PATCH] Injected articulation None guard into handle_set_joint_position")
    print(f"[PATCH] Successfully patched {CMD_CTRL}")


if __name__ == "__main__":
    patch_file()
