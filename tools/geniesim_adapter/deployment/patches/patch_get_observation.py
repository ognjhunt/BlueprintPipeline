#!/usr/bin/env python3
"""
Patch the Genie Sim server's command_controller.py to support non-recording GET_OBSERVATION.

The server's handle_get_observation only supports recording mode. When neither
startRecording nor stopRecording is set, it raises ValueError. This patch adds
actual observation capture (cameras, joints, poses) for real-time observation requests.

Usage (inside Docker build):
    python3 /tmp/patches/patch_get_observation.py

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

PATCH_MARKER = "BlueprintPipeline get_observation patch"

# The observation capture code to inject (with 12-space indentation to match else block)
OBSERVATION_CAPTURE = """\
            # --- BEGIN BlueprintPipeline get_observation patch ---
            # Non-recording observation: capture camera, joints, and poses
            import numpy as np
            result = {"camera": [], "object": [], "joint": {}}

            # Capture cameras
            if self.data.get("isCam"):
                camera_list = self.data.get("camera_prim_list", [])
                render_depth = self.data.get("render_depth", True)
                render_semantic = self.data.get("render_semantic", False)
                for cam_prim in camera_list:
                    try:
                        img = self._capture_camera(
                            prim_path=cam_prim,
                            isRGB=True,
                            isDepth=render_depth,
                            isSemantic=render_semantic,
                            isGN=False,
                        )
                        if img is None:
                            img = {}
                        # Ensure camera_info is present
                        if "camera_info" not in img:
                            _dw, _dh = 1280, 720
                            img["camera_info"] = {
                                "width": _dw, "height": _dh,
                                "ppx": float(_dw)/2.0, "ppy": float(_dh)/2.0,
                                "fx": float(_dw), "fy": float(_dh),
                            }
                        # Ensure rgb/depth arrays
                        if "rgb" not in img or img["rgb"] is None:
                            img["rgb"] = None
                        elif isinstance(img["rgb"], list) and len(img["rgb"]) == 0:
                            img["rgb"] = None
                        if "depth" not in img or img["depth"] is None:
                            img["depth"] = None
                        elif isinstance(img["depth"], list) and len(img["depth"]) == 0:
                            img["depth"] = None
                        if "semantic" not in img:
                            img["semantic"] = None
                        result["camera"].append(img)
                    except Exception as _cam_err:
                        print(f"[PATCH] Camera capture failed for {cam_prim}: {_cam_err}")
                        _dw, _dh = 1280, 720
                        result["camera"].append({
                            "camera_info": {
                                "width": _dw, "height": _dh,
                                "ppx": float(_dw)/2.0, "ppy": float(_dh)/2.0,
                                "fx": float(_dw), "fy": float(_dh),
                            },
                            "rgb": None,
                            "depth": None,
                            "semantic": None,
                        })

            # Capture joints
            if self.data.get("isJoint"):
                try:
                    result["joint"] = self._get_joint_positions()
                except Exception as _joint_err:
                    print(f"[PATCH] Joint capture failed: {_joint_err}")
                    result["joint"] = {}

            # Capture object poses
            if self.data.get("isPose"):
                object_prims = self.data.get("objectPrims", [])
                for obj_prim in object_prims:
                    try:
                        pose = self._get_object_pose(obj_prim)
                        if pose is not None:
                            result["object"].append(pose)
                        else:
                            result["object"].append(([0,0,0], [1,0,0,0]))
                    except Exception as _pose_err:
                        print(f"[PATCH] Object pose failed for {obj_prim}: {_pose_err}")
                        result["object"].append(([0,0,0], [1,0,0,0]))

            # Capture gripper poses
            if self.data.get("isGripper"):
                try:
                    left_pose = self._get_ee_pose(is_right=False)
                    right_pose = self._get_ee_pose(is_right=True)
                    result["gripper"] = {"left": left_pose, "right": right_pose}
                except Exception as _grip_err:
                    print(f"[PATCH] Gripper pose failed: {_grip_err}")
                    result["gripper"] = {"left": None, "right": None}

            self.data_to_send = result
            # --- END BlueprintPipeline get_observation patch ---
"""


def patch_file():
    if not os.path.isfile(COMMAND_CONTROLLER):
        print(f"[PATCH] command_controller.py not found at {COMMAND_CONTROLLER}")
        print("[PATCH] Skipping get_observation patch (server source not available)")
        sys.exit(0)

    with open(COMMAND_CONTROLLER, "r") as f:
        content = f.read()

    if PATCH_MARKER in content:
        print("[PATCH] get_observation already patched — skipping")
        sys.exit(0)

    # Replace the ValueError with actual observation capture
    old_error = 'raise ValueError("Invalid command: GetObservation is not supported")'

    if old_error not in content:
        print("[PATCH] Could not find ValueError for GetObservation — may already be patched differently")
        sys.exit(0)

    # Replace the error with the observation capture code
    content = content.replace(old_error, OBSERVATION_CAPTURE)

    with open(COMMAND_CONTROLLER, "w") as f:
        f.write(content)

    print("[PATCH] Injected non-recording GET_OBSERVATION support")


if __name__ == "__main__":
    patch_file()
