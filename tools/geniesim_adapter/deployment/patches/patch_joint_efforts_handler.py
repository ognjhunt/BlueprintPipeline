#!/usr/bin/env python3
"""
Patch Genie Sim server to populate JointState.effort with real PhysX efforts.

This patch injects effort capture into the get_joint_position handler in grpc_server.py.
It is idempotent and safe to run multiple times.
"""
import os
import re
import sys

GENIESIM_ROOT = os.environ.get("GENIESIM_ROOT", "/opt/geniesim")
GRPC_SERVER = os.path.join(
    GENIESIM_ROOT,
    "source", "data_collection", "server", "grpc_server.py",
)
PATCH_MARKER = "BlueprintPipeline joint_efforts_capture patch"


def patch_file():
    if not os.path.isfile(GRPC_SERVER):
        print(f"[PATCH] grpc_server.py not found at {GRPC_SERVER}")
        sys.exit(0)

    with open(GRPC_SERVER, "r") as f:
        content = f.read()

    if PATCH_MARKER in content:
        print("[PATCH] joint_efforts_capture already patched — skipping")
        sys.exit(0)

    # Find the get_joint_position method and inject effort capture
    # Look for the pattern:  rsp.states.append(joint_state)
    #                        return rsp
    # and inject effort capture before the return

    # First, verify we're in the right method by finding it
    if "def get_joint_position(self, req, rsp):" not in content:
        print("[PATCH] get_joint_position method not found — skipping")
        sys.exit(0)

    # Pattern: find the end of get_joint_position before return rsp
    # We want to inject effort capture after all states are built

    # Look for the specific pattern in get_joint_position
    old_pattern = (
        "            rsp.states.append(joint_state)\n"
        "        return rsp\n"
        "\n"
        "    def set_joint_position"
    )

    if old_pattern not in content:
        # Try alternative pattern
        old_pattern = (
            "            rsp.states.append(joint_state)\n"
            "        return rsp"
        )
        if old_pattern not in content:
            print("[PATCH] Could not find insertion point in get_joint_position — skipping")
            sys.exit(0)

    # Inject effort capture
    effort_capture = '''            rsp.states.append(joint_state)
        # --- BEGIN BlueprintPipeline joint_efforts_capture patch ---
        # Capture real joint efforts from robot articulation
        try:
            _efforts = None
            _articulation = None
            # Try multiple access paths to find the articulation object
            _server_fn = getattr(self, "server_function", None)
            _cmd = getattr(_server_fn, "command_controller", None) if _server_fn else None

            # Path 1: command_controller.ui_builder.articulation (most common)
            if _cmd is not None:
                _ui_builder = getattr(_cmd, "ui_builder", None)
                if _ui_builder is not None:
                    _articulation = getattr(_ui_builder, "articulation", None)

            # Path 2: command_controller.articulation (fallback)
            if _articulation is None and _cmd is not None:
                _articulation = getattr(_cmd, "articulation", None)

            # Path 3: server_function.articulation (fallback)
            if _articulation is None and _server_fn is not None:
                _articulation = getattr(_server_fn, "articulation", None)

            # Path 4: direct self.articulation (fallback)
            if _articulation is None:
                _articulation = getattr(self, "articulation", None)

            if _articulation is not None:
                # Try different effort APIs (Isaac Sim version dependent)
                if hasattr(_articulation, "get_applied_joint_efforts"):
                    _efforts = _articulation.get_applied_joint_efforts()
                elif hasattr(_articulation, "get_measured_joint_efforts"):
                    _efforts = _articulation.get_measured_joint_efforts()
                elif hasattr(_articulation, "get_joint_efforts"):
                    _efforts = _articulation.get_joint_efforts()

                if _efforts is not None:
                    _efforts_list = list(_efforts)
                    _effort_map = {}
                    for _idx, _state in enumerate(rsp.states):
                        if _idx < len(_efforts_list):
                            _effort_map[_state.name] = float(_efforts_list[_idx])
                    _populated = 0
                    for _state in rsp.states:
                        if _state.name in _effort_map:
                            _state.effort = _effort_map[_state.name]
                            _populated += 1
                    print(f"[JOINT_EFFORTS] Populated efforts for {_populated}/{len(rsp.states)} joints")
                else:
                    print("[JOINT_EFFORTS] articulation found but effort APIs returned None")
            else:
                print("[JOINT_EFFORTS] articulation not found via any access path")
        except Exception as _eff_err:
            print(f"[JOINT_EFFORTS] Failed to capture efforts: {_eff_err}")
        # --- END BlueprintPipeline joint_efforts_capture patch ---
        return rsp'''

    if "def set_joint_position" in old_pattern:
        effort_capture += "\n\n    def set_joint_position"
        content = content.replace(old_pattern, effort_capture)
    else:
        content = content.replace(old_pattern, effort_capture)

    with open(GRPC_SERVER, "w") as f:
        f.write(content)

    print("[PATCH] Injected joint_efforts_capture into grpc_server.py")


if __name__ == "__main__":
    patch_file()
