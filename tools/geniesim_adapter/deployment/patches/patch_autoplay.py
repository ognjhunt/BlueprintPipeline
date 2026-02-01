#!/usr/bin/env python3
"""
Patch the Genie Sim server's data_collector_server.py to auto-play the
simulation when it enters a PAUSED state.

Problem: After `docker restart`, Isaac Sim's World is not playing, so the
main loop just spins printing "**** simulation paused ****" without
processing any gRPC commands. All gRPC calls then time out with
DEADLINE_EXCEEDED, creating a deadlock — the client can't send a RESET
command because the server won't process it.

Fix: When the main loop detects `is_playing() == False`, call
`world.play()` to resume the simulation automatically.

Usage:
    python3 /tmp/patches/patch_autoplay.py

Idempotent.
"""
import os
import re
import sys

GENIESIM_ROOT = os.environ.get("GENIESIM_ROOT", "/opt/geniesim")
SERVER_SCRIPT = os.path.join(
    GENIESIM_ROOT,
    "source", "data_collection", "scripts", "data_collector_server.py",
)

PATCH_MARKER = "BlueprintPipeline autoplay patch"


def patch_file():
    if not os.path.isfile(SERVER_SCRIPT):
        print(f"[PATCH] data_collector_server.py not found at {SERVER_SCRIPT}")
        sys.exit(0)

    with open(SERVER_SCRIPT, "r") as f:
        content = f.read()

    if PATCH_MARKER in content:
        print("[PATCH] data_collector_server.py autoplay already applied — skipping")
        sys.exit(0)

    paused_loop_pattern = re.compile(
        r"(?P<indent>^[ \t]*)if not .*?is_playing\(\):\n"
        r"(?P<body>(?:^[ \t]+.*\n)*?)"
        r"(?P=indent)continue",
        re.MULTILINE,
    )

    paused_match = paused_loop_pattern.search(content)
    if not paused_match:
        print("[PATCH] Could not find paused check pattern in data_collector_server.py")
        sys.exit(1)

    indent = paused_match.group("indent")
    new = (
        f'{indent}if not ui_builder.my_world.is_playing():  # {PATCH_MARKER}\n'
        f'{indent}    if step % 100 == 0:\n'
        f'{indent}        logger.info("**** simulation paused — auto-playing ****")\n'
        f'{indent}    try:\n'
        f'{indent}        ui_builder.my_world.play()\n'
        f'{indent}    except Exception as _play_err:\n'
        f'{indent}        if step % 100 == 0:\n'
        f'{indent}            logger.warning(f"**** auto-play failed: {{_play_err}} ****")\n'
        f'{indent}    step += 1\n'
        f'{indent}    continue'
    )

    content, replace_count = paused_loop_pattern.subn(new, content, count=1)
    if replace_count != 1:
        print("[PATCH] Failed to replace paused check pattern in data_collector_server.py")
        sys.exit(1)

    startup_pattern = re.compile(r"^(?P<indent>[ \t]*)while\s+True\s*:\n", re.MULTILINE)
    startup_match = startup_pattern.search(content)
    if startup_match:
        startup_indent = startup_match.group("indent")
        startup_block = (
            f'{startup_indent}if not ui_builder.my_world.is_playing():  # {PATCH_MARKER} startup\n'
            f'{startup_indent}    try:\n'
            f'{startup_indent}        ui_builder.my_world.play()\n'
            f'{startup_indent}    except Exception as _play_err:\n'
            f'{startup_indent}        logger.warning(f"**** startup auto-play failed: {{_play_err}} ****")\n'
        )
        content = startup_pattern.sub(startup_block + r"\g<0>", content, count=1)

    if PATCH_MARKER not in content:
        raise RuntimeError("[PATCH] Patch marker missing after update; aborting.")

    with open(SERVER_SCRIPT, "w") as f:
        f.write(content)
    print("[PATCH] Added auto-play to data_collector_server.py main loop")


if __name__ == "__main__":
    patch_file()
