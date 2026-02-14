#!/usr/bin/env python3
"""
Patch SAGE's object_placement_planner.py to handle Isaac Sim being unavailable.

Bug: When Isaac Sim is not running, simulate_the_scene() returns a string (not dict).
The code then tries result_sim["unstable_objects"] on a string → TypeError: string indices must be integers.
This causes place_objects_in_room to return success: false even though all objects were placed fine.

Fix: Wrap each physics check block in try/except. If Isaac Sim is down, skip stability
filtering and keep all placed objects. Log a warning instead of crashing.

Also fixes simulate_the_scene() in isaac_mcp/server.py to return dict instead of str(dict).

This patch is idempotent — safe to run multiple times.
"""

import re
import sys
import os

def patch_object_placement_planner(filepath):
    """Patch the 3 physics check blocks in object_placement_planner.py"""

    with open(filepath, 'r') as f:
        content = f.read()

    if 'PATCH:physics_crash_guard' in content:
        print(f"  [SKIP] {filepath} — already patched")
        return False

    # The buggy pattern appears 3 times — identical blocks:
    #
    #   result_sim = simulate_the_scene()
    #   if not isinstance(result_sim, dict) or result_sim.get("status") != "success":
    #       # raise exception
    #       pass
    #
    #   unstable_object_ids = result_sim["unstable_objects"]  ← CRASH HERE
    #   ...
    #
    # Fix: wrap from result_create = ... through the unstable removal in try/except

    # Pattern to match each physics check block
    old_block = '''        result_create = create_single_room_layout_scene_from_room(
            scene_save_dir,
            room_dict_save_path
        )
        if not isinstance(result_create, dict) or result_create.get("status") != "success":
            # raise exception
            pass

        result_sim = simulate_the_scene()
        if not isinstance(result_sim, dict) or result_sim.get("status") != "success":
            # raise exception
            pass

        unstable_object_ids = result_sim["unstable_objects"]
        print(f"number of unstable objects: ", len(unstable_object_ids), file=sys.stderr)
        print(f"room.objects: ", len(room.objects), file=sys.stderr)
        if len(unstable_object_ids) > 0:
            print(f"unstable_object_ids: ", unstable_object_ids, file=sys.stderr)
            room.objects = [obj for obj in room.objects if obj.id not in unstable_object_ids]
            print(f"after removing unstable objects, room.objects: ", len(room.objects), file=sys.stderr)'''

    new_block = '''        try:  # PATCH:physics_crash_guard — handle Isaac Sim unavailable
            result_create = create_single_room_layout_scene_from_room(
                scene_save_dir,
                room_dict_save_path
            )
            if not isinstance(result_create, dict) or result_create.get("status") != "success":
                print(f"⚠️ Isaac Sim scene creation failed (result={type(result_create).__name__}), skipping physics check", file=sys.stderr)
                raise RuntimeError("Scene creation failed")

            result_sim = simulate_the_scene()
            if not isinstance(result_sim, dict):
                print(f"⚠️ Isaac Sim simulation returned {type(result_sim).__name__} instead of dict, skipping physics check", file=sys.stderr)
                raise RuntimeError("Simulation returned non-dict")
            if result_sim.get("status") != "success":
                print(f"⚠️ Isaac Sim simulation failed (status={result_sim.get('status')}), skipping physics check", file=sys.stderr)
                raise RuntimeError("Simulation failed")

            unstable_object_ids = result_sim.get("unstable_objects", [])
            print(f"number of unstable objects: ", len(unstable_object_ids), file=sys.stderr)
            print(f"room.objects: ", len(room.objects), file=sys.stderr)
            if len(unstable_object_ids) > 0:
                print(f"unstable_object_ids: ", unstable_object_ids, file=sys.stderr)
                room.objects = [obj for obj in room.objects if obj.id not in unstable_object_ids]
                print(f"after removing unstable objects, room.objects: ", len(room.objects), file=sys.stderr)
        except Exception as physics_err:
            print(f"⚠️ Physics stability check skipped — Isaac Sim unavailable or error: {physics_err}", file=sys.stderr)
            print(f"  Objects remain placed (skipping stability filtering)", file=sys.stderr)'''

    count = content.count(old_block)
    if count == 0:
        print(f"  [WARN] {filepath} — could not find expected physics block pattern. Trying fallback...")
        # Fallback: use regex for more flexible matching
        pattern = re.compile(
            r'        result_create = create_single_room_layout_scene_from_room\(\s*\n'
            r'            scene_save_dir,\s*\n'
            r'            room_dict_save_path\s*\n'
            r'        \)\s*\n'
            r'        if not isinstance\(result_create, dict\).*?pass\s*\n'
            r'\s*\n'
            r'        result_sim = simulate_the_scene\(\)\s*\n'
            r'        if not isinstance\(result_sim, dict\).*?pass\s*\n'
            r'\s*\n'
            r'        unstable_object_ids = result_sim\["unstable_objects"\]\s*\n'
            r'.*?print\(f"after removing unstable objects.*?\n',
            re.DOTALL
        )
        matches = list(pattern.finditer(content))
        if not matches:
            print(f"  [ERROR] {filepath} — fallback regex also failed. Manual patching needed.")
            return False
        print(f"  Found {len(matches)} blocks via regex fallback")
        # Apply from last to first to preserve positions
        for match in reversed(matches):
            content = content[:match.start()] + new_block + content[match.end():]
    else:
        print(f"  Found {count} physics check blocks to patch")
        content = content.replace(old_block, new_block)

    with open(filepath, 'w') as f:
        f.write(content)

    final_count = content.count('PATCH:physics_crash_guard')
    print(f"  [OK] Patched {final_count} physics check blocks in {filepath}")
    return True


def patch_isaac_mcp_server(filepath):
    """Fix simulate_the_scene() to return dict instead of str(dict) on error."""

    with open(filepath, 'r') as f:
        content = f.read()

    if 'PATCH:simulate_return_dict' in content:
        print(f"  [SKIP] {filepath} — already patched")
        return False

    # Fix 1: simulate_the_scene returns str(dict) instead of dict
    old_sim = '        return str({"status": "error", "error": str(e), "message": "Error simulate_the_scene"})'
    new_sim = '        return {"status": "error", "error": str(e), "message": "Error simulate_the_scene", "unstable_objects": []}  # PATCH:simulate_return_dict'

    if old_sim in content:
        content = content.replace(old_sim, new_sim)
        print(f"  [OK] Fixed simulate_the_scene() return type")
    else:
        print(f"  [SKIP] simulate_the_scene str() pattern not found (may already be fixed)")

    # Fix 2: simulate_the_scene_groups has the same bug
    old_sim_groups = '        return str({"status": "error", "error": str(e), "message": "Error simulate_the_scene_groups"})'
    new_sim_groups = '        return {"status": "error", "error": str(e), "message": "Error simulate_the_scene_groups", "unstable_objects": []}  # PATCH:simulate_return_dict'

    if old_sim_groups in content:
        content = content.replace(old_sim_groups, new_sim_groups)
        print(f"  [OK] Fixed simulate_the_scene_groups() return type")

    with open(filepath, 'w') as f:
        f.write(content)

    return True


def main():
    sage_dir = os.environ.get('SAGE_DIR', '/workspace/SAGE')

    print("=" * 60)
    print("SAGE Physics Crash Guard Patch")
    print("Fixes 'string indices must be integers' in place_objects_in_room")
    print("=" * 60)
    print()

    # Patch 1: object_placement_planner.py
    opp_path = os.path.join(sage_dir, 'server/objects/object_placement_planner.py')
    if os.path.exists(opp_path):
        print(f"Patching {opp_path}...")
        patch_object_placement_planner(opp_path)
    else:
        print(f"[ERROR] Not found: {opp_path}")
        sys.exit(1)

    print()

    # Patch 2: isaac_mcp/server.py
    mcp_path = os.path.join(sage_dir, 'server/isaacsim/isaac_mcp/server.py')
    if os.path.exists(mcp_path):
        print(f"Patching {mcp_path}...")
        patch_isaac_mcp_server(mcp_path)
    else:
        print(f"[WARN] Not found: {mcp_path} — skipping MCP server fix")

    print()
    print("Done. Physics checks will now gracefully skip when Isaac Sim is unavailable.")
    print("Objects will remain placed instead of being removed by a failed physics check.")


if __name__ == '__main__':
    main()
