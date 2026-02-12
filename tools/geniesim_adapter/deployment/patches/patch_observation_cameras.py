#!/usr/bin/env python3
"""
Patch the Genie Sim server's command_controller.py to handle unknown cameras
in handle_get_observation.

The server crashes with KeyError when a camera prim path (e.g.
'/Franka/panda_hand/hand_camera') is not in self.cameras dict (populated
from the active robot config at init time).

This patch wraps self.cameras[camera] lookups with .get() and a default
resolution, so unknown cameras don't crash the server.

Usage (inside Docker build or at runtime):
    python3 /tmp/patches/patch_observation_cameras.py

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

PATCH_MARKER = "BlueprintPipeline observation_cameras patch"


def patch_file():
    if not os.path.isfile(COMMAND_CONTROLLER):
        print(f"[PATCH] command_controller.py not found at {COMMAND_CONTROLLER}")
        print("[PATCH] Skipping observation_cameras patch (server source not available)")
        sys.exit(0)

    with open(COMMAND_CONTROLLER, "r") as f:
        content = f.read()

    if PATCH_MARKER in content:
        print("[PATCH] observation_cameras already patched — skipping")
        sys.exit(0)

    changes = 0

    # Pattern 1: self.cameras[camera][0] and self.cameras[camera][1]
    # Replace with .get() that returns a default [1280, 720]
    old1 = "self.cameras[camera][0]"
    new1 = (
        "self.cameras.get(camera, "
        "{\"camera_info\": {\"width\": 1280, \"height\": 720}})"
        "[\"camera_info\"][\"width\"]"
    )
    if old1 in content:
        content = content.replace(old1, new1)
        changes += 1
        print("[PATCH] Fixed self.cameras[camera][0] -> .get() with default")

    old2 = "self.cameras[camera][1]"
    new2 = (
        "self.cameras.get(camera, "
        "{\"camera_info\": {\"width\": 1280, \"height\": 720}})"
        "[\"camera_info\"][\"height\"]"
    )
    if old2 in content:
        content = content.replace(old2, new2)
        changes += 1
        print("[PATCH] Fixed self.cameras[camera][1] -> .get() with default")

    # Pattern 2: bare self.cameras[camera] used as a whole value
    # (only replace remaining instances not already caught above)
    # We also want to register unknown cameras when first seen.
    # Inject a camera auto-registration block at the start of handle_get_observation.
    obs_method = "def handle_get_observation(self"
    if obs_method in content:
        idx = content.find(obs_method)
        # Find the end of the def line (the colon + newline)
        colon_idx = content.find(":", idx + len(obs_method))
        newline_idx = content.find("\n", colon_idx)
        # Detect body indentation
        rest = content[newline_idx + 1:]
        body_indent = "        "
        for line in rest.split("\n"):
            stripped = line.lstrip()
            if stripped and not stripped.startswith("#") and not stripped.startswith('"""') and not stripped.startswith("'''"):
                body_indent = line[:len(line) - len(stripped)]
                break

        registration_block = (
            f"\n{body_indent}# {PATCH_MARKER} — auto-register unknown cameras\n"
            f"{body_indent}if hasattr(self, 'camera_prim_list') and hasattr(self, 'cameras'):\n"
            f"{body_indent}    import os as _obs_os\n"
            f"{body_indent}    _default_res = _obs_os.environ.get('CAMERA_RESOLUTION', '1280x720')\n"
            f"{body_indent}    try:\n"
            f"{body_indent}        _dw, _dh = (int(x) for x in _default_res.split('x'))\n"
            f"{body_indent}    except (ValueError, TypeError):\n"
            f"{body_indent}        _dw, _dh = 1280, 720\n"
            f"{body_indent}    _default_info = {{\n"
            f"{body_indent}        \"width\": _dw, \"height\": _dh,\n"
            f"{body_indent}        \"ppx\": float(_dw) / 2.0, \"ppy\": float(_dh) / 2.0,\n"
            f"{body_indent}        \"fx\": float(_dw), \"fy\": float(_dh),\n"
            f"{body_indent}    }}\n"
            f"{body_indent}    for _cam_key, _cam_val in list(self.cameras.items()):\n"
            f"{body_indent}        if isinstance(_cam_val, dict):\n"
            f"{body_indent}            _info = _cam_val.get(\"camera_info\")\n"
            f"{body_indent}            if not isinstance(_info, dict):\n"
            f"{body_indent}                _cam_val[\"camera_info\"] = dict(_default_info)\n"
            f"{body_indent}            _cam_val.setdefault(\"prim_path\", _cam_key)\n"
            f"{body_indent}            continue\n"
            f"{body_indent}        if isinstance(_cam_val, (list, tuple)) and len(_cam_val) >= 2:\n"
            f"{body_indent}            try:\n"
            f"{body_indent}                _lw, _lh = int(_cam_val[0]), int(_cam_val[1])\n"
            f"{body_indent}            except (ValueError, TypeError):\n"
            f"{body_indent}                _lw, _lh = _dw, _dh\n"
            f"{body_indent}            _info = dict(_default_info)\n"
            f"{body_indent}            _info.update({{\"width\": _lw, \"height\": _lh}})\n"
            f"{body_indent}            self.cameras[_cam_key] = {{\"camera_info\": _info, \"prim_path\": _cam_key}}\n"
            f"{body_indent}            continue\n"
            f"{body_indent}        if isinstance(_cam_val, str):\n"
            f"{body_indent}            self.cameras[_cam_key] = {{\"camera_info\": dict(_default_info), \"prim_path\": _cam_val}}\n"
            f"{body_indent}    for _cam in (self.camera_prim_list or []):\n"
            f"{body_indent}        if _cam not in self.cameras:\n"
            f"{body_indent}            self.cameras[_cam] = {{\"camera_info\": dict(_default_info), \"prim_path\": _cam}}\n"
            f'{body_indent}            print(f"[PATCH] Auto-registered camera {{_cam}} with resolution {{_dw}}x{{_dh}}")\n'
        )

        # Insert after the docstring if present, or right after def line
        # Find position after def line
        insert_pos = newline_idx + 1
        # Skip docstring if present
        check = content[insert_pos:].lstrip()
        if check.startswith('"""') or check.startswith("'''"):
            quote = check[:3]
            # Find closing triple-quote
            doc_start = content.find(quote, insert_pos)
            doc_end = content.find(quote, doc_start + 3)
            if doc_end != -1:
                insert_pos = content.find("\n", doc_end) + 1

        content = content[:insert_pos] + registration_block + content[insert_pos:]
        changes += 1
        print("[PATCH] Injected camera auto-registration in handle_get_observation")

    # Pattern 4: publish_ros ValueError crash
    # handle_get_observation raises ValueError("publish ros is not enabled")
    # when startRecording is requested but --publish_ros was not passed.
    # Replace with a warning so the server doesn't crash.
    old_raise = 'raise ValueError("publish ros is not enabled")'
    new_raise = 'print("[PATCH] publish_ros not enabled — skipping ROS recording"); self.data_to_send = "Start"'
    if old_raise in content:
        content = content.replace(old_raise, new_raise)
        changes += 1
        print("[PATCH] Replaced publish_ros ValueError with warning")

    if changes == 0:
        print("[PATCH] No matching patterns found — file may already be fixed")
        sys.exit(0)

    with open(COMMAND_CONTROLLER, "w") as f:
        f.write(content)

    print(f"[PATCH] Successfully patched {COMMAND_CONTROLLER} ({changes} fixes)")


def patch_grpc_server():
    """Patch grpc_server.py to handle missing camera_info in get_camera_data."""
    grpc_server = os.path.join(
        GENIESIM_ROOT,
        "source", "data_collection", "server", "grpc_server.py",
    )
    if not os.path.isfile(grpc_server):
        print(f"[PATCH] grpc_server.py not found at {grpc_server}")
        return

    with open(grpc_server, "r") as f:
        content = f.read()

    grpc_marker = "BlueprintPipeline grpc_camera_info patch"
    if grpc_marker in content:
        print("[PATCH] grpc_server.py camera_info already patched — skipping")
        return

    # Replace bare camera_info = current_camera["camera_info"] with safe access
    pattern = re.compile(
        r'^(?P<indent>[ \t]*)camera_info\s*=\s*current_camera\["camera_info"\]\s*$',
        re.MULTILINE,
    )

    def replace_camera_info(match: re.Match) -> str:
        indent = match.group("indent")
        return (
            f"{indent}# {grpc_marker}\n"
            f'{indent}_default_ci = {{"width": 1280, "height": 720, "ppx": 640.0, "ppy": 360.0, "fx": 1280.0, "fy": 720.0}}\n'
            f"{indent}if isinstance(current_camera, dict):\n"
            f'{indent}    camera_info = current_camera.get("camera_info", _default_ci)\n'
            f"{indent}    if not isinstance(camera_info, dict):\n"
            f"{indent}        camera_info = _default_ci\n"
            f"{indent}else:\n"
            f"{indent}    camera_info = _default_ci"
        )

    content, replacements = pattern.subn(replace_camera_info, content, count=1)
    if replacements:
        with open(grpc_server, "w") as f:
            f.write(content)
        print("[PATCH] Patched grpc_server.py get_camera_data camera_info access")
    else:
        print(
            "[PATCH] WARNING: Could not find camera_info assignment pattern in "
            "grpc_server.py get_camera_data; no changes applied."
        )


if __name__ == "__main__":
    patch_file()
    patch_grpc_server()
