#!/usr/bin/env python3
"""Patch GenieSim server files to handle missing cuRobo module gracefully.

When GENIESIM_SERVER_CUROBO_MODE=off and cuRobo is not installed, top-level
`from curobo...` imports crash the server. This patch wraps them in try/except
and defines all imported names as None/object fallbacks.
Only modifies TOP-LEVEL imports (no leading whitespace).
"""
import os
import re
import sys

GENIESIM_ROOT = os.environ.get("GENIESIM_ROOT", "/opt/geniesim")
MARKER = "# BlueprintPipeline curobo_import_guard patch"

TARGETS = [
    os.path.join(GENIESIM_ROOT, "source/data_collection/server/motion_generator/motion_gen_reacher.py"),
    os.path.join(GENIESIM_ROOT, "source/data_collection/server/motion_generator/mesh_utils.py"),
]

# Names known to be subclassed — need `object` as fallback, not `None`
SUBCLASSED_NAMES = {"UsdHelper"}


def extract_imported_names(import_line):
    """Extract the imported names from a from...import statement."""
    stripped = import_line.strip()
    # Handle: from foo.bar import A, B, C
    m = re.match(r"from\s+\S+\s+import\s+(.+)", stripped)
    if m:
        names_str = m.group(1)
        # Handle parenthesized imports
        names_str = names_str.strip("()")
        return [n.strip().split(" as ")[-1].strip() for n in names_str.split(",") if n.strip()]
    # Handle: import foo.bar
    m = re.match(r"import\s+(\S+)", stripped)
    if m:
        return [m.group(1).split(".")[-1]]
    return []


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
    all_imported_names = []
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
            all_imported_names.extend(extract_imported_names(stripped))
        else:
            if in_curobo_block:
                new_lines.append("except ImportError:")
                # Define fallback for every imported name
                for name in all_imported_names:
                    fallback = "object" if name in SUBCLASSED_NAMES else "None"
                    new_lines.append(f"    {name} = {fallback}")
                new_lines.append(f'    print("[{basename}] cuRobo not available")')
                in_curobo_block = False
            new_lines.append(line)

    if in_curobo_block:
        new_lines.append("except ImportError:")
        for name in all_imported_names:
            fallback = "object" if name in SUBCLASSED_NAMES else "None"
            new_lines.append(f"    {name} = {fallback}")
        new_lines.append(f'    print("[{basename}] cuRobo not available")')

    with open(target, "w") as f:
        f.write("\n".join(new_lines))

    total_patched += 1
    print(f"[PATCH] curobo_import_guard: {basename} — wrapped {len(curobo_imports)} imports, "
          f"defined {len(all_imported_names)} fallback names")

if total_patched == 0:
    print("[PATCH] curobo_import_guard: no files needed patching")
else:
    print(f"[PATCH] curobo_import_guard: patched {total_patched} files")
