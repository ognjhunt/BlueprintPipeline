#!/usr/bin/env python3
"""Patch motion_gen_reacher.py to handle missing cuRobo module gracefully.

When GENIESIM_SERVER_CUROBO_MODE=off and cuRobo is not installed, the top-level
`from curobo.cuda_robot_model...` import crashes the server. This patch wraps
the import in a try/except so the server can start without cuRobo.
"""
import os
import sys

GENIESIM_ROOT = os.environ.get("GENIESIM_ROOT", "/opt/geniesim")
TARGET = os.path.join(
    GENIESIM_ROOT,
    "source/data_collection/server/motion_generator/motion_gen_reacher.py",
)

MARKER = "# BlueprintPipeline curobo_import_guard patch"

if not os.path.exists(TARGET):
    print(f"[PATCH] curobo_import_guard: target not found: {TARGET}")
    sys.exit(0)

with open(TARGET) as f:
    src = f.read()

if MARKER in src:
    print("[PATCH] curobo_import_guard: already applied")
    sys.exit(0)

# Wrap the hard curobo imports in try/except
old_import = "from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig"
if old_import not in src:
    print("[PATCH] curobo_import_guard: could not find target import line")
    sys.exit(1)

# Find ALL curobo import lines at the top of the file
lines = src.split("\n")
new_lines = []
curobo_imports = []
in_curobo_block = False

for i, line in enumerate(lines):
    stripped = line.strip()
    if stripped.startswith("from curobo.") or stripped.startswith("import curobo"):
        if not in_curobo_block:
            new_lines.append(f"{MARKER}")
            new_lines.append("try:")
            in_curobo_block = True
        new_lines.append(f"    {line}")
        curobo_imports.append(stripped)
    else:
        if in_curobo_block:
            new_lines.append("except ImportError:")
            new_lines.append("    CudaRobotModel = None")
            new_lines.append("    CudaRobotModelConfig = None")
            new_lines.append('    print("[motion_gen_reacher] cuRobo not installed — running in curobo-off mode")')
            in_curobo_block = False
        new_lines.append(line)

if in_curobo_block:
    new_lines.append("except ImportError:")
    new_lines.append("    CudaRobotModel = None")
    new_lines.append("    CudaRobotModelConfig = None")
    new_lines.append('    print("[motion_gen_reacher] cuRobo not installed — running in curobo-off mode")')

with open(TARGET, "w") as f:
    f.write("\n".join(new_lines))

print(f"[PATCH] curobo_import_guard: wrapped {len(curobo_imports)} curobo imports in try/except")
