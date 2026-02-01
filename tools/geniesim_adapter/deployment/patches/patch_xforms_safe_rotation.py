#!/usr/bin/env python3
"""Patch Isaac Sim's xforms.py to handle degenerate rotation matrices.

During init_robot, get_world_pose() can receive a rotation matrix with
all-zero rows (prim not yet initialized by physics). scipy's
Rotation.from_matrix() raises ValueError on such matrices. This patch
wraps every such call to return identity quaternion when the matrix
is degenerate.

Target: /isaac-sim/exts/isaacsim.core.utils/isaacsim/core/utils/xforms.py
"""
import os
import re
import sys

ISAAC_SIM_PATH = os.environ.get("ISAAC_SIM_PATH", "/isaac-sim")
XFORMS_PY = os.path.join(
    ISAAC_SIM_PATH,
    "exts", "isaacsim.core.utils", "isaacsim", "core", "utils", "xforms.py",
)

PATCH_MARKER = "# BlueprintPipeline: safe rotation patch"

if not os.path.isfile(XFORMS_PY):
    print(f"[XFORMS-PATCH] xforms.py not found at {XFORMS_PY}")
    sys.exit(0)

with open(XFORMS_PY, "r") as f:
    content = f.read()

if PATCH_MARKER in content:
    print("[XFORMS-PATCH] Already patched — skipping")
    sys.exit(0)

# Match lines like:
#     r = Rotation.from_matrix(result_transform[:3, :3])
#         r = Rotation.from_matrix(...)
# Capture the leading whitespace and the matrix argument.
pattern = re.compile(
    r'^([ \t]*)(r\s*=\s*Rotation\.from_matrix\(([^)]+)\))',
    re.MULTILINE,
)

count = 0
def _replace(m):
    global count
    count += 1
    indent = m.group(1)
    mat_arg = m.group(3)
    return (
        f"{indent}{PATCH_MARKER}\n"
        f"{indent}try:\n"
        f"{indent}    r = Rotation.from_matrix({mat_arg})\n"
        f"{indent}except ValueError:\n"
        f"{indent}    r = Rotation.from_quat([0.0, 0.0, 0.0, 1.0])"
    )

content = pattern.sub(_replace, content)

if count == 0:
    print(f"[XFORMS-PATCH] No Rotation.from_matrix calls found in {XFORMS_PY}")
    sys.exit(0)

with open(XFORMS_PY, "w") as f:
    f.write(content)

print(f"[XFORMS-PATCH] Patched {count} Rotation.from_matrix call(s) in xforms.py")
