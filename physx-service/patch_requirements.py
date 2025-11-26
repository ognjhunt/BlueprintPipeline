#!/usr/bin/env python3
"""
Patch PhysX-Anything requirements.txt to replace local cluster wheel paths
with standard PyPI package specifications.
"""

import os
import sys
from pathlib import Path


def patch_requirements(req_file: str) -> None:
    """
    Rewrites requirements.txt to replace absolute paths to .whl files
    with standard package==version specifications.
    """
    req = Path(req_file)
    if not req.is_file():
        raise SystemExit(f"requirements.txt not found: {req_file}")

    lines = req.read_text().splitlines()
    new_lines = []

    for line in lines:
        stripped = line.strip()
        
        # Keep comments and blank lines as-is
        if not stripped or stripped.startswith("#"):
            new_lines.append(line)
            continue

        # Check if this line looks like an absolute path to a .whl file
        if ".whl" in stripped and (stripped.startswith("/") or "mnt/" in stripped):
            fname = os.path.basename(stripped)
            if fname.endswith(".whl"):
                # Remove .whl extension
                fname = fname[:-4]

            # Parse wheel filename: package-version-py_version-abi-platform.whl
            # We want to extract package and version
            parts = fname.split("-")
            if len(parts) >= 2:
                pkg = parts[0]
                ver = parts[1]
                print(f"[PATCH-REQ] Rewriting {stripped!r} -> {pkg}=={ver}")
                new_lines.append(f"{pkg}=={ver}")
                continue
            else:
                # Fallback: comment out the line if we can't parse it
                print(f"[PATCH-REQ] Could not parse wheel {stripped!r}, commenting it out")
                new_lines.append(f"# commented-out: {line}")
                continue

        # Default: keep the line unchanged
        new_lines.append(line)

    # Write patched requirements back
    req.write_text("\n".join(new_lines) + "\n")
    print(f"[PATCH-REQ] Successfully patched {req_file}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: patch_requirements.py <requirements.txt>")
        sys.exit(1)
    
    patch_requirements(sys.argv[1])