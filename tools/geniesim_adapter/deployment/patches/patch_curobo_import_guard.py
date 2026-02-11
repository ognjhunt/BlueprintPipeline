#!/usr/bin/env python3
"""Patch GenieSim server files to handle missing cuRobo module gracefully.

When GENIESIM_SERVER_CUROBO_MODE=off and cuRobo is not installed, top-level
`from curobo...` imports crash the server. This patch wraps them in try/except.
Only modifies TOP-LEVEL imports (no leading whitespace), not in-function imports.
"""
import os
import sys

GENIESIM_ROOT = os.environ.get("GENIESIM_ROOT", "/opt/geniesim")
MARKER = "# BlueprintPipeline curobo_import_guard patch"

TARGETS = [
    os.path.join(GENIESIM_ROOT, "source/data_collection/server/motion_generator/motion_gen_reacher.py"),
    os.path.join(GENIESIM_ROOT, "source/data_collection/server/motion_generator/mesh_utils.py"),
]

total_patched = 0

for target in TARGETS:
    basename = os.path.basename(target)
    if not os.path.exists(target):
        print(f"[PATCH] curobo_import_guard: {basename} not found, skipping")
        continue

    with open(target) as f:
        src = f.read()

    if MARKER in src:
        print(f"[PATCH] curobo_import_guard: {basename} already applied")
        continue

    if "from curobo." not in src and "import curobo" not in src:
        print(f"[PATCH] curobo_import_guard: {basename} has no curobo imports, skipping")
        continue

    lines = src.split("\n")
    new_lines = []
    curobo_imports = []
    in_curobo_block = False

    for line in lines:
        stripped = line.strip()
        is_toplevel_curobo = (
            line == line.lstrip()
            and stripped
            and (stripped.startswith("from curobo.") or stripped.startswith("import curobo"))
        )
        if is_toplevel_curobo:
            if not in_curobo_block:
                new_lines.append(MARKER)
                new_lines.append("try:")
                in_curobo_block = True
            new_lines.append(f"    {line}")
            curobo_imports.append(stripped)
        else:
            if in_curobo_block:
                new_lines.append("except ImportError:")
                new_lines.append(f'    print("[{basename}] cuRobo not available")')
                in_curobo_block = False
            new_lines.append(line)

    if in_curobo_block:
        new_lines.append("except ImportError:")
        new_lines.append(f'    print("[{basename}] cuRobo not available")')

    with open(target, "w") as f:
        f.write("\n".join(new_lines))

    total_patched += 1
    print(f"[PATCH] curobo_import_guard: {basename} — wrapped {len(curobo_imports)} top-level imports")

if total_patched == 0:
    print("[PATCH] curobo_import_guard: no files needed patching")
else:
    print(f"[PATCH] curobo_import_guard: patched {total_patched} files")
