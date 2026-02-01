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

    # Replace the paused check that just logs with one that auto-plays
    old = (
        '    if not ui_builder.my_world.is_playing():\n'
        '        if step % 100 == 0:\n'
        '            logger.info("**** simulation paused ****")\n'
        '        step += 1\n'
        '        continue'
    )
    new = (
        f'    if not ui_builder.my_world.is_playing():  # {PATCH_MARKER}\n'
        '        if step % 100 == 0:\n'
        '            logger.info("**** simulation paused — auto-playing ****")\n'
        '        try:\n'
        '            ui_builder.my_world.play()\n'
        '        except Exception as _play_err:\n'
        '            if step % 100 == 0:\n'
        '                logger.warning(f"**** auto-play failed: {_play_err} ****")\n'
        '        step += 1\n'
        '        continue'
    )

    if old in content:
        content = content.replace(old, new, 1)
        with open(SERVER_SCRIPT, "w") as f:
            f.write(content)
        print("[PATCH] Added auto-play to data_collector_server.py main loop")
    else:
        print("[PATCH] Could not find paused check pattern in data_collector_server.py")
        sys.exit(0)


if __name__ == "__main__":
    patch_file()
