#!/usr/bin/env python3
"""
Patch the Genie Sim server's command_controller.py to guard against
calling articulation methods when articulation is None.

The server's _get_joint_positions() calls self.ui_builder.articulation
which can be None if the async initialization (from _on_reset or
init_robot) hasn't completed yet. This crashes the server process,
causing gRPC UNAVAILABLE errors.

Usage (inside Docker build or at runtime):
    python3 /tmp/patches/patch_articulation_guard.py

Idempotent — re-running is a no-op.
"""
import os
import sys

GENIESIM_ROOT = os.environ.get("GENIESIM_ROOT", "/opt/geniesim")
CMD_CTRL = os.path.join(
    GENIESIM_ROOT,
    "source", "data_collection", "server", "command_controller.py",
)

PATCH_MARKER = "BlueprintPipeline articulation_guard patch"


def patch_file():
    if not os.path.isfile(CMD_CTRL):
        print(f"[PATCH] command_controller.py not found at {CMD_CTRL}")
        sys.exit(0)

    with open(CMD_CTRL, "r") as f:
        content = f.read()

    if PATCH_MARKER in content:
        print("[PATCH] command_controller.py articulation guard already applied — skipping")
        sys.exit(0)

    changes = 0

    # Guard _get_joint_positions: return structured error if articulation is None,
    # and catch exceptions from get_joint_positions.
    old = (
        "    def _get_joint_positions(self):\n"
        "        self._initialize_articulation()\n"
        "        articulation = self.ui_builder.articulation\n"
        "        joint_positions = articulation.get_joint_positions()"
    )
    new = (
        "    def _get_joint_positions(self):\n"
        f"        # {PATCH_MARKER}\n"
        "        self._initialize_articulation()\n"
        "        articulation = self.ui_builder.articulation\n"
        "        if articulation is None:\n"
        "            print('[PATCH] articulation is None — returning empty joint positions')\n"
        "            self._articulation_needs_reinit = True\n"
        "            return {}\n"
        "        try:\n"
        "            joint_positions = articulation.get_joint_positions()\n"
        "        except Exception as exc:\n"
        "            print(f\"[PATCH] get_joint_positions exception: {exc}\")\n"
        "            self._articulation_needs_reinit = True\n"
        "            return {\"error\": f\"get_joint_positions exception: {exc}\"}\n"
        "        if not isinstance(joint_positions, dict):\n"
        "            print(f\"[PATCH] get_joint_positions non-dict: {type(joint_positions)}\")\n"
        "            self._articulation_needs_reinit = True\n"
        "            return {\"error\": \"get_joint_positions returned non-dict\"}"
    )
    if old in content:
        content = content.replace(old, new, 1)
        changes += 1
        print("[PATCH] Added articulation None guard to _get_joint_positions")

    # Guard handle_get_ee_pose / _get_ee_pose: return default if articulation is None
    # The _get_ee_pose calls ui_builder._get_ee_pose which accesses articulation
    old_ee = "    def _get_ee_pose(self, is_right: bool)"
    if old_ee in content and "articulation is None" not in content.split(old_ee)[1][:200]:
        # Add guard at the start of _get_ee_pose
        old_ee_full = "    def _get_ee_pose(self, is_right: bool) -> Tuple[np.ndarray, np.ndarray]:\n"
        # Find the next line after the def
        idx = content.find(old_ee_full)
        if idx >= 0:
            insert_point = idx + len(old_ee_full)
            guard = (
                f"        # {PATCH_MARKER}\n"
                "        if self.ui_builder.articulation is None:\n"
                "            print('[PATCH] articulation is None — returning default ee_pose')\n"
                "            return np.zeros(3), np.eye(3)\n"
            )
            content = content[:insert_point] + guard + content[insert_point:]
            changes += 1
            print("[PATCH] Added articulation None guard to _get_ee_pose")

    if changes == 0:
        print("[PATCH] No matching patterns found — command_controller.py may have different structure")
        sys.exit(0)

    with open(CMD_CTRL, "w") as f:
        f.write(content)

    print(f"[PATCH] Successfully patched {CMD_CTRL} ({changes} fixes)")


if __name__ == "__main__":
    patch_file()
