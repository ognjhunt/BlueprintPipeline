#!/usr/bin/env python3
"""
Patch the Genie Sim server's ros_publisher/base.py to avoid re-creating
OmniGraph graphs that already exist.

The upstream publish_tf() calls og.Controller.edit() with a graph_path that
may already exist (e.g. /World/RobotTFActionGraph), causing:

    OmniGraphError: Failed to wrap graph in node given
    {'graph_path': '/World/RobotTFActionGraph', ...}

This patch wraps the og.Controller.edit() call in publish_tf to skip
graph creation if the graph already exists at the target path.

Usage (inside Docker container):
    python3 /workspace/BlueprintPipeline/tools/geniesim_adapter/deployment/patches/patch_omnigraph_dedup.py

The script is idempotent — re-running it on an already-patched file is a no-op.
"""
import os
import sys

GENIESIM_ROOT = os.environ.get("GENIESIM_ROOT", "/opt/geniesim")
BASE_PY = os.path.join(
    GENIESIM_ROOT,
    "source", "data_collection", "server", "ros_publisher", "base.py",
)

PATCH_MARKER = "BlueprintPipeline omnigraph dedup patch"


def patch_file():
    if not os.path.isfile(BASE_PY):
        print(f"[PATCH] base.py not found at {BASE_PY}")
        print("[PATCH] Skipping omnigraph dedup patch (server source not available)")
        sys.exit(0)

    with open(BASE_PY, "r") as f:
        content = f.read()

    if PATCH_MARKER in content:
        print("[PATCH] OmniGraph dedup already patched — skipping")
        sys.exit(0)

    # Strategy: Find og.Controller.edit calls in publish_tf and wrap them
    # with a check for existing graph.
    # The pattern is:  og.Controller.edit({"graph_path": ..., "evaluator_name": ...}, {...})
    # We need to add a guard before this call.

    # Look for the publish_tf method and add a graph-existence check
    old = "og.Controller.edit("
    if old not in content:
        print("[PATCH] Could not find og.Controller.edit( in base.py")
        print("[PATCH] Attempting alternative patch on command_controller.py")
        patch_command_controller()
        return

    # Add an import and helper at the top of the file
    import_patch = (
        "\n# --- BEGIN {marker} ---\n"
        "def _og_controller_edit_safe(graph_spec, *args, **kwargs):\n"
        "    \"\"\"Wrapper around og.Controller.edit that skips if graph exists.\"\"\"\n"
        "    import omni.graph.core as og\n"
        "    if isinstance(graph_spec, dict) and 'graph_path' in graph_spec:\n"
        "        gp = graph_spec['graph_path']\n"
        "        try:\n"
        "            from pxr import Usd\n"
        "            import omni.usd\n"
        "            stage = omni.usd.get_context().get_stage()\n"
        "            if stage and stage.GetPrimAtPath(gp).IsValid():\n"
        "                print(f'[PATCH] Graph already exists at {{gp}}, skipping creation')\n"
        "                return og.get_graph_by_path(gp)\n"
        "        except Exception:\n"
        "            pass\n"
        "    return og.Controller.edit(graph_spec, *args, **kwargs)\n"
        "# --- END {marker} ---\n"
    ).format(marker=PATCH_MARKER)

    # Insert the helper after the last top-level import
    # Find the publish_tf method and replace og.Controller.edit with our safe version
    patched = content.replace(
        "og.Controller.edit(",
        "_og_controller_edit_safe(",
    )

    # Add our helper function before the class definition
    # Find "class " to insert before it
    class_pos = patched.find("\nclass ")
    if class_pos >= 0:
        patched = patched[:class_pos] + import_patch + patched[class_pos:]
    else:
        # Fallback: add at the end of imports
        patched = import_patch + patched

    with open(BASE_PY, "w") as f:
        f.write(patched)

    print(f"[PATCH] Successfully patched {BASE_PY}")


def patch_command_controller():
    """Alternative: patch command_controller.py directly."""
    cc_path = os.path.join(
        GENIESIM_ROOT,
        "source", "data_collection", "server", "command_controller.py",
    )
    if not os.path.isfile(cc_path):
        print(f"[PATCH] command_controller.py not found at {cc_path}")
        sys.exit(1)

    with open(cc_path, "r") as f:
        content = f.read()

    if PATCH_MARKER in content:
        print("[PATCH] OmniGraph dedup already patched in command_controller.py — skipping")
        sys.exit(0)

    old = "og.Controller.edit("
    if old not in content:
        print("[PATCH] Could not find og.Controller.edit in command_controller.py either")
        sys.exit(1)

    import_patch = (
        "\n# --- BEGIN {marker} ---\n"
        "def _og_controller_edit_safe(graph_spec, *args, **kwargs):\n"
        "    import omni.graph.core as og\n"
        "    if isinstance(graph_spec, dict) and 'graph_path' in graph_spec:\n"
        "        gp = graph_spec['graph_path']\n"
        "        try:\n"
        "            from pxr import Usd\n"
        "            import omni.usd\n"
        "            stage = omni.usd.get_context().get_stage()\n"
        "            if stage and stage.GetPrimAtPath(gp).IsValid():\n"
        "                print(f'[PATCH] Graph already exists at {{gp}}, skipping creation')\n"
        "                return og.get_graph_by_path(gp)\n"
        "        except Exception:\n"
        "            pass\n"
        "    return og.Controller.edit(graph_spec, *args, **kwargs)\n"
        "# --- END {marker} ---\n"
    ).format(marker=PATCH_MARKER)

    patched = content.replace(old, "_og_controller_edit_safe(")
    class_pos = patched.find("\nclass ")
    if class_pos >= 0:
        patched = patched[:class_pos] + import_patch + patched[class_pos:]
    else:
        patched = import_patch + patched

    with open(cc_path, "w") as f:
        f.write(patched)

    print(f"[PATCH] Successfully patched {cc_path}")


if __name__ == "__main__":
    patch_file()
