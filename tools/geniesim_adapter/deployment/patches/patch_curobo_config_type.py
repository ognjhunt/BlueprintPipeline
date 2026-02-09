#!/usr/bin/env python3
"""Patch: Fix cuRobo config type error in ui_builder.py.

The G1 robot config provides curobo_config_file as a string (single-arm),
but ui_builder.py:378 calls .items() on it, expecting a dict (dual-arm).

Fix: Wrap string config in a dict with key "right" before iterating.
"""
import os
import re

TARGET = os.path.join(
    os.environ.get("GENIESIM_ROOT", "/opt/geniesim"),
    "source", "data_collection", "server", "ui_builder.py",
)
MARKER = "# [PATCH] curobo_config_type_fix"


def apply():
    with open(TARGET) as f:
        src = f.read()

    if MARKER in src:
        print("[PATCH] curobo_config_type_fix already applied")
        return

    # Find the line: for key, cfg in self.curobo_config_file.items():
    old = "            for key, cfg in self.curobo_config_file.items():"
    if old not in src:
        print("[PATCH] curobo_config_type_fix: could not find target line")
        return

    new = (
        "            # [PATCH] curobo_config_type_fix\n"
        "            _curobo_cfg = self.curobo_config_file\n"
        "            if isinstance(_curobo_cfg, str):\n"
        "                _curobo_cfg = {\"right\": _curobo_cfg}\n"
        "            for key, cfg in _curobo_cfg.items():"
    )

    src = src.replace(old, new, 1)

    with open(TARGET, "w") as f:
        f.write(src)
    print("[PATCH] curobo_config_type_fix: wrapped string config in dict")


if __name__ == "__main__":
    apply()
