#!/usr/bin/env python3
"""
Patch Genie Sim's data_collector_server.py with a camera-safe render config.

Why this exists:
- In Isaac Sim, launching SimulationApp with headless=True forces --no-window.
- In this deployment, --no-window can produce RGB=0 frames while depth/normals
  still update.
- Runtime edits inside containers are lost when containers are recreated.

This patch codifies the render config so it is re-applied deterministically
from Docker build/bootstrap scripts.

Idempotent: re-running after successful patch application keeps file normalized.
"""

import os
import re
import sys


GENIESIM_ROOT = os.environ.get("GENIESIM_ROOT", "/opt/geniesim")
SERVER_SCRIPT = os.path.join(
    GENIESIM_ROOT,
    "source",
    "data_collection",
    "scripts",
    "data_collector_server.py",
)

PATCH_MARKER = "BlueprintPipeline render config patch"
HEADLESS_LINE = '"headless": False if (os.environ.get("DISPLAY") and os.environ.get("ENABLE_CAMERAS") == "1") else args.headless,'
RENDERER_LINE = '"renderer": "RaytracedLighting",'


def _replace_once(content: str, pattern: str, repl, desc: str, flags: int = 0) -> str:
    updated, count = re.subn(pattern, repl, content, count=1, flags=flags)
    if count != 1:
        raise RuntimeError(f"[PATCH] Could not patch {desc} in data_collector_server.py")
    return updated


def _replace_extra_args_block(content: str) -> str:
    lines = content.splitlines()
    output = []
    replacing = False
    replaced = False
    block_indent = ""

    for line in lines:
        if not replacing and re.match(r'^\s*"extra_args"\s*:\s*\[', line):
            block_indent = line[: line.index('"extra_args"')]
            # NOTE: Do NOT set --/renderer/activeGpu=0 — it breaks RGB output
            # on L4 GPUs.  Let Isaac Sim auto-detect the GPU (activeGpu=-1).
            output.extend(
                [
                    f'{block_indent}"extra_args": [',
                    f'{block_indent}    "--reset-user",',
                    f"{block_indent}],",
                ]
            )
            replaced = True

            # Handle one-line forms like: "extra_args": [],
            if "]" in line:
                continue

            replacing = True
            continue

        if replacing:
            if re.match(r"^\s*\]\s*,?\s*$", line):
                replacing = False
            continue

        output.append(line)

    if replacing:
        raise RuntimeError("[PATCH] Unterminated extra_args block while patching data_collector_server.py")
    if not replaced:
        raise RuntimeError("[PATCH] Could not find extra_args block in data_collector_server.py")

    return "\n".join(output) + "\n"


def _normalize_import_os(content: str) -> str:
    # Remove duplicate import os lines first.
    content = re.sub(r"^\s*import os\s*$\n?", "", content, flags=re.MULTILINE)

    # Re-insert one import os right after "import sys" — must be before
    # root_directory = os.path.dirname(...) which is near the top of the file.
    if re.search(r"^import sys\s*$", content, flags=re.MULTILINE):
        content = re.sub(
            r"^(import sys\s*)$",
            r"\1import os\n",
            content,
            count=1,
            flags=re.MULTILINE,
        )
    elif re.search(r"^import argparse\s*$", content, flags=re.MULTILINE):
        content = re.sub(
            r"^(import argparse\s*)$",
            r"\1import os\n",
            content,
            count=1,
            flags=re.MULTILINE,
        )
    else:
        # Last resort: prepend import os at the very top (after any comments).
        content = "import os\n" + content
    return content


def patch_file() -> None:
    if not os.path.isfile(SERVER_SCRIPT):
        print(f"[PATCH] data_collector_server.py not found at {SERVER_SCRIPT}")
        sys.exit(0)

    with open(SERVER_SCRIPT, "r") as f:
        content = f.read()

    content = _normalize_import_os(content)

    # Force renderer to RaytracedLighting (canonical Isaac Sim renderer string).
    def _renderer_repl(match: re.Match) -> str:
        indent = match.group("indent")
        return f"{indent}{RENDERER_LINE}"

    content = _replace_once(
        content,
        r'^(?P<indent>\s*)"renderer"\s*:\s*"[^"]+"\s*,\s*$',
        _renderer_repl,
        "renderer configuration",
        flags=re.MULTILINE,
    )

    # Avoid --no-window in camera mode by toggling headless based on DISPLAY.
    def _headless_repl(match: re.Match) -> str:
        indent = match.group("indent")
        return (
            f"{indent}# {PATCH_MARKER}: keep camera rendering surface when DISPLAY is available.\n"
            f"{indent}{HEADLESS_LINE}"
        )

    # Accept any existing headless assignment shape and normalize it.
    headless_pattern = r'^(?P<indent>\s*)"headless"\s*:\s*.*,\s*$'
    if re.search(headless_pattern, content, flags=re.MULTILINE):
        content = re.sub(headless_pattern, _headless_repl, content, count=1, flags=re.MULTILINE)
    else:
        raise RuntimeError("[PATCH] Could not patch headless configuration in data_collector_server.py")

    # Canonical extra args for render reliability.
    content = _replace_extra_args_block(content)

    # Remove any stale/duplicate rendermode and rt2 lines before re-inserting canonical block.
    content = re.sub(
        r'^.*simulation_app\._carb_settings\.set\("/persistent/rtx/modes/rt2/enabled",\s*False\)\s*$\n?',
        "",
        content,
        flags=re.MULTILINE,
    )
    content = re.sub(
        r'^.*simulation_app\._carb_settings\.set\("/rtx/rendermode",\s*"[^"]+"\)\s*$\n?',
        "",
        content,
        flags=re.MULTILINE,
    )

    # Re-assert renderer settings immediately after SimulationApp init.
    def _post_init_repl(match: re.Match) -> str:
        indent = match.group("indent")
        return (
            f'{indent}simulation_app._carb_settings.set("/omni/replicator/asyncRendering", False)\n'
            f"{indent}# {PATCH_MARKER}: disable legacy rt2 and pin renderer mode.\n"
            f'{indent}simulation_app._carb_settings.set("/persistent/rtx/modes/rt2/enabled", False)\n'
            f'{indent}simulation_app._carb_settings.set("/rtx/rendermode", "RaytracedLighting")'
        )

    replicator_pattern = r'^(?P<indent>\s*)simulation_app\._carb_settings\.set\("/omni/replicator/asyncRendering",\s*False\)\s*$'
    if re.search(replicator_pattern, content, flags=re.MULTILINE):
        content = re.sub(replicator_pattern, _post_init_repl, content, count=1, flags=re.MULTILINE)
    else:
        raise RuntimeError("[PATCH] Could not patch post-init carb settings in data_collector_server.py")

    with open(SERVER_SCRIPT, "w") as f:
        f.write(content)

    print("[PATCH] Applied data_collector_server render config patch")


if __name__ == "__main__":
    patch_file()
