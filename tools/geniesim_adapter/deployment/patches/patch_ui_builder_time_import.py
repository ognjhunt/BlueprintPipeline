#!/usr/bin/env python3
"""Patch: Fix 'time' UnboundLocalError in ui_builder.py initialize_articulation.

The articulation_physics_wait patch added `import time` inside an except block,
which makes Python treat 'time' as a local variable throughout the function.
When the except block is not entered, line 437 (`before = time.time()`) fails
with UnboundLocalError because the local 'time' was never assigned.

Fix: Add `import time` at the top of initialize_articulation().
"""
import os

TARGET = os.path.join(
    os.environ.get("GENIESIM_ROOT", "/opt/geniesim"),
    "source", "data_collection", "server", "ui_builder.py",
)
MARKER = "# [PATCH] ui_builder_time_import"


def apply():
    with open(TARGET) as f:
        src = f.read()

    if MARKER in src:
        print("[PATCH] ui_builder_time_import already applied")
        return

    # Find initialize_articulation and add import time at the top
    anchor = "def initialize_articulation(self, batch_num=0):"
    idx = src.find(anchor)
    if idx == -1:
        print("[PATCH] ui_builder_time_import: could not find initialize_articulation")
        return

    line_end = src.find("\n", idx)
    if line_end == -1:
        print("[PATCH] ui_builder_time_import: could not find end of function def line")
        return

    inject = "\n        import time  %s" % MARKER
    src = src[:line_end] + inject + src[line_end:]

    with open(TARGET, "w") as f:
        f.write(src)

    print("[PATCH] ui_builder_time_import: fixed time import in initialize_articulation")


if __name__ == "__main__":
    apply()
