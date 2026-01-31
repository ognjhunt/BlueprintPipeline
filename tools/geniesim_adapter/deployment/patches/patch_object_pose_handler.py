#!/usr/bin/env python3
"""
Patch the Genie Sim server's command_controller.py to fix GET_OBJECT_POSE.

The upstream server's handle_get_object_pose returns empty/identity poses
because the requested prim paths don't match the actual USD stage hierarchy.
This patch adds fuzzy path matching with multi-candidate scoring: if the
exact path isn't found, it searches the stage for prims whose name suffix
matches, then scores candidates preferring geometric prims under /World/.

Usage (inside Docker build):
    python3 /tmp/patches/patch_object_pose_handler.py

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

# The object pose handler helper to inject.
OBJECT_POSE_HELPER = textwrap.dedent("""\

    # --- BEGIN BlueprintPipeline object_pose patch ---
    def _bp_resolve_prim_path(self, requested_path):
        \"\"\"Resolve a prim path by fuzzy matching against the USD stage.

        If the exact path exists, return it.  Otherwise, search for prims
        whose name (last path component) matches the requested path's name,
        then score candidates: prefer Xformable prims under /World/ with
        shorter paths.  Falls back to substring matching if no exact name
        match is found.
        \"\"\"
        try:
            import omni.usd
            from pxr import UsdGeom
            stage = omni.usd.get_context().get_stage()
            if stage is None:
                return requested_path

            # Exact match
            prim = stage.GetPrimAtPath(requested_path)
            if prim and prim.IsValid():
                return requested_path

            # Extract target name (last path component)
            target_name = requested_path.rstrip("/").rsplit("/", 1)[-1]
            if not target_name:
                return requested_path

            target_lower = target_name.lower()

            # Collect candidates: exact name match and substring match
            exact_candidates = []
            substring_candidates = []

            for prim in stage.Traverse():
                prim_path = str(prim.GetPath())
                prim_name = prim_path.rstrip("/").rsplit("/", 1)[-1]

                if prim_name == target_name:
                    exact_candidates.append((prim_path, prim))
                elif target_lower in prim_name.lower():
                    substring_candidates.append((prim_path, prim))

            # Score and pick best from exact matches first, then substring
            def _score(path, p):
                s = 0
                # Prefer geometric prims (Xformable includes Mesh, Xform, etc.)
                try:
                    if p.IsA(UsdGeom.Xformable):
                        s += 100
                except Exception:
                    pass
                # Prefer prims under /World/
                if path.startswith("/World/"):
                    s += 50
                # Prefer shorter paths (less deeply nested)
                s -= path.count("/")
                return s

            for candidates, label in [(exact_candidates, "exact"), (substring_candidates, "substring")]:
                if not candidates:
                    continue
                scored = sorted(candidates, key=lambda c: _score(c[0], c[1]), reverse=True)
                best_path = scored[0][0]
                if len(scored) > 1:
                    print(f"[PATCH] Prim path {label} candidates for '{target_name}': "
                          f"{[c[0] for c in scored[:5]]}")
                print(f"[PATCH] Resolved prim path ({label}): {requested_path} -> {best_path}")
                return best_path

            print(f"[PATCH] WARNING: Could not resolve prim path {requested_path} in stage")
        except Exception as e:
            print(f"[PATCH] Prim path resolution failed: {e}")

        return requested_path
    # --- END BlueprintPipeline object_pose patch ---
""")

PATCH_MARKER = "BlueprintPipeline object_pose patch"


def patch_file():
    if not os.path.isfile(COMMAND_CONTROLLER):
        print(f"[PATCH] command_controller.py not found at {COMMAND_CONTROLLER}")
        print("[PATCH] Skipping object_pose patch (server source not available)")
        sys.exit(0)

    with open(COMMAND_CONTROLLER, "r") as f:
        content = f.read()

    if PATCH_MARKER in content:
        print("[PATCH] Object pose handler already patched — skipping")
        sys.exit(0)

    # 1. Add the helper method to the class (append like camera patch)
    method_indent = "    "  # default 4 spaces
    m = re.search(r"^([ \t]+)def \w+\(self", content, re.MULTILINE)
    if m:
        method_indent = m.group(1)

    indented_helper = "\n".join(
        (method_indent + line) if line.strip() else line
        for line in OBJECT_POSE_HELPER.splitlines()
    ) + "\n"

    patched = content.rstrip() + "\n\n" + indented_helper

    # 2. Find get_object_pose handling and inject path resolution.
    # Look for patterns like:
    #   prim_path = request.prim_path
    #   prim_path = data.get("prim_path", ...)
    #   prim = stage.GetPrimAtPath(prim_path)
    # and add self._bp_resolve_prim_path() call after prim_path assignment.
    # Match standalone prim_path assignments (not keyword args ending with comma)
    prim_path_pattern = re.compile(
        r"((\s+)(prim_path|object_prim_path|_prim_path)\s*=\s*(?:.*(?:prim_path|Prim_path|object_id).*))\n",
        re.MULTILINE,
    )
    # Filter out matches that are keyword arguments (line ends with comma)
    all_matches = list(prim_path_pattern.finditer(patched))
    match = None
    for m in all_matches:
        line_content = m.group(1).rstrip()
        if not line_content.endswith(","):
            match = m
            break
    if match:
        indent = match.group(2)
        var_name = match.group(3)
        original_line = match.group(1)
        resolution_line = f"\n{indent}{var_name} = self._bp_resolve_prim_path({var_name})  # {PATCH_MARKER}"
        patched = patched[:match.end(1)] + resolution_line + patched[match.end(1):]
        print(f"[PATCH] Injected prim path resolution after {var_name} assignment")
    else:
        print("[PATCH] WARNING: Could not find prim_path assignment in get_object_pose handler")
        print("[PATCH] Helper method added but automatic resolution not wired")
        print("[PATCH] You may need to manually call self._bp_resolve_prim_path(prim_path) in the handler")

    with open(COMMAND_CONTROLLER, "w") as f:
        f.write(patched)

    print(f"[PATCH] Successfully patched {COMMAND_CONTROLLER}")


if __name__ == "__main__":
    patch_file()
