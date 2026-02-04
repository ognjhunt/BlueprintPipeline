#!/usr/bin/env python3
"""
Patch the Genie Sim server's ui_builder.py to wait for the physics backend
to be ready before creating an Articulation.

The error:
  AttributeError: 'NoneType' object has no attribute 'is_homogeneous'

Occurs because Articulation._on_physics_ready is called before the physics
backend (self._physics_view._backend) is initialized. This happens during
server startup when init_robot is called too quickly.

This patch wraps the Articulation() call with retry logic that waits for
the physics simulation to be ready.

Usage (inside Docker build or at runtime):
    python3 /tmp/patches/patch_articulation_physics_wait.py

Idempotent - re-running is a no-op.
"""
import os
import sys
import re

GENIESIM_ROOT = os.environ.get("GENIESIM_ROOT", "/opt/geniesim")
UI_BUILDER = os.path.join(
    GENIESIM_ROOT,
    "source", "data_collection", "server", "ui_builder.py",
)

PATCH_MARKER = "BlueprintPipeline articulation_physics_wait patch"


def patch_file():
    if not os.path.isfile(UI_BUILDER):
        print(f"[PATCH] ui_builder.py not found at {UI_BUILDER}")
        sys.exit(0)

    with open(UI_BUILDER, "r") as f:
        content = f.read()

    if PATCH_MARKER in content:
        print("[PATCH] ui_builder.py articulation physics wait already applied - skipping")
        sys.exit(0)

    changes = 0

    # Pattern: Find the initialize_articulation method and wrap the Articulation() call
    # Look for: self.articulation = Articulation(prim_path=self.robot_prim_path, name=self.robot_name)
    pattern = r'(\s+)(self\.articulation = Articulation\(prim_path=self\.robot_prim_path, name=self\.robot_name\))'

    if re.search(pattern, content):
        replacement = r'''\1# ''' + PATCH_MARKER + r'''
\1# Wait for physics backend to be ready before creating Articulation
\1_max_physics_wait_retries = 10
\1_physics_wait_delay = 0.5
\1for _attempt in range(_max_physics_wait_retries):
\1    try:
\1        self.articulation = Articulation(prim_path=self.robot_prim_path, name=self.robot_name)
\1        print(f"[PATCH] Articulation created successfully on attempt {_attempt + 1}")
\1        break
\1    except AttributeError as e:
\1        if "is_homogeneous" in str(e) or "_backend" in str(e):
\1            print(f"[PATCH] Physics backend not ready (attempt {_attempt + 1}/{_max_physics_wait_retries}), waiting {_physics_wait_delay}s...")
\1            import time
\1            time.sleep(_physics_wait_delay)
\1            # Try to advance the simulation a few frames
\1            try:
\1                import omni.timeline
\1                timeline = omni.timeline.get_timeline_interface()
\1                if not timeline.is_playing():
\1                    timeline.play()
\1                for _ in range(10):
\1                    import omni.kit.app
\1                    omni.kit.app.get_app().update()
\1            except Exception as setup_err:
\1                print(f"[PATCH] Error during physics warmup: {setup_err}")
\1        else:
\1            raise
\1else:
\1    print("[PATCH] CRITICAL: Physics backend failed to initialize after all retries")
\1    raise RuntimeError("Physics backend not ready after maximum retries")'''

        content = re.sub(pattern, replacement, content, count=1)
        changes += 1
        print("[PATCH] Added physics wait retry loop around Articulation creation")
    else:
        # Try alternate pattern with slightly different formatting
        alt_pattern = r'self\.articulation\s*=\s*Articulation\s*\(\s*prim_path\s*=\s*self\.robot_prim_path\s*,\s*name\s*=\s*self\.robot_name\s*\)'
        match = re.search(alt_pattern, content)
        if match:
            # Get the indentation
            line_start = content.rfind('\n', 0, match.start()) + 1
            indent = content[line_start:match.start()]

            replacement = f'''# {PATCH_MARKER}
{indent}# Wait for physics backend to be ready before creating Articulation
{indent}_max_physics_wait_retries = 10
{indent}_physics_wait_delay = 0.5
{indent}for _attempt in range(_max_physics_wait_retries):
{indent}    try:
{indent}        self.articulation = Articulation(prim_path=self.robot_prim_path, name=self.robot_name)
{indent}        print(f"[PATCH] Articulation created successfully on attempt {{_attempt + 1}}")
{indent}        break
{indent}    except AttributeError as e:
{indent}        if "is_homogeneous" in str(e) or "_backend" in str(e):
{indent}            print(f"[PATCH] Physics backend not ready (attempt {{_attempt + 1}}/{{_max_physics_wait_retries}}), waiting {{_physics_wait_delay}}s...")
{indent}            import time
{indent}            time.sleep(_physics_wait_delay)
{indent}            try:
{indent}                import omni.timeline
{indent}                timeline = omni.timeline.get_timeline_interface()
{indent}                if not timeline.is_playing():
{indent}                    timeline.play()
{indent}                for _ in range(10):
{indent}                    import omni.kit.app
{indent}                    omni.kit.app.get_app().update()
{indent}            except Exception as setup_err:
{indent}                print(f"[PATCH] Error during physics warmup: {{setup_err}}")
{indent}        else:
{indent}            raise
{indent}else:
{indent}    print("[PATCH] CRITICAL: Physics backend failed to initialize after all retries")
{indent}    raise RuntimeError("Physics backend not ready after maximum retries")'''

            content = content[:match.start()] + replacement + content[match.end():]
            changes += 1
            print("[PATCH] Added physics wait retry loop (alternate pattern)")

    if changes == 0:
        print("[PATCH] No matching Articulation creation pattern found in ui_builder.py")
        # Print some context to help debug
        if "Articulation(" in content:
            print("[PATCH] Found 'Articulation(' in file, but pattern didn't match")
            # Find and print the line
            for i, line in enumerate(content.split('\n')):
                if 'self.articulation = Articulation(' in line:
                    print(f"[PATCH] Line {i+1}: {line.strip()}")
        sys.exit(0)

    with open(UI_BUILDER, "w") as f:
        f.write(content)

    print(f"[PATCH] Successfully patched {UI_BUILDER} ({changes} fixes)")


if __name__ == "__main__":
    patch_file()
