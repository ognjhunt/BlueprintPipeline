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
            _robot = None
            _server_fn = getattr(self, "server_function", None)
            _cmd = getattr(_server_fn, "command_controller", None) or _server_fn
            _robot = getattr(_cmd, "robot", None)
            if _robot is not None:
                if hasattr(_robot, "get_applied_joint_efforts"):
                    _efforts = _robot.get_applied_joint_efforts()
                elif hasattr(_robot, "get_measured_joint_efforts"):
                    _efforts = _robot.get_measured_joint_efforts()
                elif hasattr(_robot, "get_joint_efforts"):
                    _efforts = _robot.get_joint_efforts()
            if _efforts is not None:
                _efforts_list = list(_efforts)
                _effort_map = {}
                for _idx, _state in enumerate(rsp.states):
                    if _idx < len(_efforts_list):
                        _effort_map[_state.name] = float(_efforts_list[_idx])
                for _state in rsp.states:
                    if _state.name in _effort_map:
                        _state.effort = _effort_map[_state.name]
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
