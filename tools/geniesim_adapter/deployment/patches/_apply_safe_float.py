#!/usr/bin/env python3
"""Inject _bp_safe_float into already-patched grpc_server.py and update float() calls."""
import os
import re
import sys

GENIESIM_ROOT = os.environ.get("GENIESIM_ROOT", "/opt/geniesim")
GRPC_SERVER = os.path.join(
    GENIESIM_ROOT, "source", "data_collection", "server", "grpc_server.py"
)

if not os.path.isfile(GRPC_SERVER):
    print(f"[SAFE-FLOAT] grpc_server.py not found at {GRPC_SERVER}")
    sys.exit(0)

with open(GRPC_SERVER, "r") as f:
    content = f.read()

if "_bp_safe_float" in content:
    print("[SAFE-FLOAT] _bp_safe_float already present — skipping")
    sys.exit(0)

# Find last import line to inject helper after it
import_end = 0
for m in re.finditer(r'^(?:import |from )\S+.*$', content, re.MULTILINE):
    import_end = m.end()

helper = """

# BlueprintPipeline safe float helper for unit-suffixed values (e.g. "1.5 m")
def _bp_safe_float(val, default=0.0):
    try:
        return float(val)
    except (ValueError, TypeError):
        if isinstance(val, str):
            import re as _re
            _m = _re.match(r'([+-]?\\d*\\.?\\d+)', val.strip())
            if _m:
                return float(_m.group(1))
        return default

"""

if import_end > 0:
    content = content[:import_end] + helper + content[import_end:]

# Replace float(_pos[i]) and float(_rot[i]) with _bp_safe_float
replacements = [
    ("float(_pos[0]) if len(_pos)", "_bp_safe_float(_pos[0]) if len(_pos)"),
    ("float(_pos[1]) if len(_pos)", "_bp_safe_float(_pos[1]) if len(_pos)"),
    ("float(_pos[2]) if len(_pos)", "_bp_safe_float(_pos[2]) if len(_pos)"),
    ("float(_rot[0]) if len(_rot)", "_bp_safe_float(_rot[0], 1.0) if len(_rot)"),
    ("float(_rot[1]) if len(_rot)", "_bp_safe_float(_rot[1]) if len(_rot)"),
    ("float(_rot[2]) if len(_rot)", "_bp_safe_float(_rot[2]) if len(_rot)"),
    ("float(_rot[3]) if len(_rot)", "_bp_safe_float(_rot[3]) if len(_rot)"),
]

count = 0
for old, new in replacements:
    if old in content:
        content = content.replace(old, new)
        count += 1

with open(GRPC_SERVER, "w") as f:
    f.write(content)

print(f"[SAFE-FLOAT] Injected _bp_safe_float and updated {count} float() calls")
