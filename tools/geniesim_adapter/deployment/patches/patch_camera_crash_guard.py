#!/usr/bin/env python3
"""
Patch the Genie Sim server's command_controller.py to guard
handle_get_camera_data against ValueError from omni.replicator.core.

The server crashes when render_product() raises:
    ValueError: No valid sensor paths provided
This happens when the camera prim path is stale or the stage has not
finished loading.

This patch wraps the Replicator render_product call inside
handle_get_camera_data in a try/except ValueError so that when the
"No valid sensor paths" error occurs, the handler returns empty camera
data (black frame + default intrinsics) instead of crashing.

Usage (inside Docker build or at runtime):
    python3 /tmp/patches/patch_camera_crash_guard.py

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

PATCH_MARKER = "BlueprintPipeline camera_crash_guard patch"


def patch_file():
    if not os.path.isfile(CMD_CTRL):
        print(f"[PATCH] command_controller.py not found at {CMD_CTRL}")
        sys.exit(0)

    with open(CMD_CTRL, "r") as f:
        content = f.read()

    if PATCH_MARKER in content:
        print("[PATCH] camera_crash_guard already applied -- skipping")
        sys.exit(0)

    # ------------------------------------------------------------------ #
    # Strategy: find the handle_get_camera_data method and wrap its body
    # in a try/except ValueError that catches the "No valid sensor paths"
    # crash from omni.replicator.core's render_product().
    #
    # We locate the method def line, then find the first non-blank,
    # non-docstring line of the body.  We inject a try: before the body
    # and an except ValueError block after the body (before the next
    # method at the same indent level or end-of-file).
    # ------------------------------------------------------------------ #

    handler_pattern = re.compile(
        r"^([ \t]+)def handle_get_camera_data\(self.*?\):\s*\n",
        re.MULTILINE,
    )
    handler_match = handler_pattern.search(content)
    if not handler_match:
        print("[PATCH] handle_get_camera_data not found -- skipping")
        sys.exit(0)

    method_indent = handler_match.group(1)  # e.g. "    "
    body_indent = method_indent + "    "     # e.g. "        "
    extra_indent = "    "                    # one additional level for try body

    body_start = handler_match.end()

    # Skip past the docstring (if any) and blank lines to find true body start
    docstring_pattern = re.compile(
        r'^([ \t]*)(?:"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')\s*\n',
        re.MULTILINE,
    )
    doc_match = docstring_pattern.match(content[body_start:])
    if doc_match:
        body_start += doc_match.end()

    # Find the end of this method body.
    # First, look for the "END BlueprintPipeline camera patch" marker which
    # the object_pose_handler patch inserts after the camera body — this is
    # the most reliable boundary.  If not found, fall back to the next def/
    # class/@ at the same indent level, or any non-blank line at the method
    # indent level that is NOT deeper-indented (i.e. class-level code like
    # variable assignments from other patches).
    end_marker = re.compile(
        rf"^{re.escape(body_indent)}# --- END BlueprintPipeline camera patch ---",
        re.MULTILINE,
    )
    end_marker_match = end_marker.search(content, body_start)

    if end_marker_match:
        # Include the END marker comment line inside the body, end right after it
        eol = content.find("\n", end_marker_match.end())
        body_end = (eol + 1) if eol != -1 else end_marker_match.end()
    else:
        # Fallback: next def/class/@ at the method indent level
        next_method = re.compile(
            rf"^{re.escape(method_indent)}(?:def |class |@)",
            re.MULTILINE,
        )
        end_match = next_method.search(content, body_start)
        body_end = end_match.start() if end_match else len(content)

    # Extract body lines
    body_text = content[body_start:body_end]

    # Re-indent the body by one extra level (add 4 spaces to each non-empty line)
    indented_body_lines = []
    for line in body_text.split("\n"):
        if line.strip():
            indented_body_lines.append(extra_indent + line)
        else:
            indented_body_lines.append(line)
    indented_body = "\n".join(indented_body_lines)

    # Build the replacement: try + original body + except ValueError
    replacement = (
        f"{body_indent}# {PATCH_MARKER}\n"
        f"{body_indent}try:\n"
        f"{indented_body}"
        f"{body_indent}except ValueError as _cam_val_err:\n"
        f"{body_indent}    if \"No valid sensor paths\" in str(_cam_val_err):\n"
        f"{body_indent}        print(f'[PATCH] handle_get_camera_data: caught ValueError: {{_cam_val_err}}')\n"
        f"{body_indent}        print('[PATCH] Returning empty camera data instead of crashing')\n"
        f"{body_indent}        import numpy as _np\n"
        f"{body_indent}        _w, _h = 1280, 720\n"
        f"{body_indent}        self.data_to_send = {{\n"
        f"{body_indent}            \"width\": _w, \"height\": _h,\n"
        f"{body_indent}            \"fx\": float(_w), \"fy\": float(_w),\n"
        f"{body_indent}            \"ppx\": float(_w) / 2.0, \"ppy\": float(_h) / 2.0,\n"
        f"{body_indent}            \"rgb\": bytes(_h * _w * 3),\n"
        f"{body_indent}            \"rgb_shape\": [_h, _w, 3],\n"
        f"{body_indent}            \"rgb_dtype\": \"uint8\",\n"
        f"{body_indent}            \"rgb_encoding\": \"raw_rgb_uint8\",\n"
        f"{body_indent}            \"depth\": bytes(_h * _w * 4),\n"
        f"{body_indent}            \"depth_shape\": [_h, _w],\n"
        f"{body_indent}            \"depth_dtype\": \"float32\",\n"
        f"{body_indent}            \"depth_encoding\": \"raw_depth_float32\",\n"
        f"{body_indent}            \"camera_info\": {{\n"
        f"{body_indent}                \"width\": _w, \"height\": _h,\n"
        f"{body_indent}                \"fx\": float(_w), \"fy\": float(_w),\n"
        f"{body_indent}                \"ppx\": float(_w) / 2.0, \"ppy\": float(_h) / 2.0,\n"
        f"{body_indent}            }},\n"
        f"{body_indent}        }}\n"
        f"{body_indent}        return\n"
        f"{body_indent}    else:\n"
        f"{body_indent}        raise\n"
    )

    content = content[:body_start] + replacement + content[body_end:]

    with open(CMD_CTRL, "w") as f:
        f.write(content)

    print(f"[PATCH] Wrapped handle_get_camera_data body in try/except ValueError")
    print(f"[PATCH] Successfully patched {CMD_CTRL}")


if __name__ == "__main__":
    patch_file()
