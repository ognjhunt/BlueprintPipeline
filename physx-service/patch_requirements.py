#!/usr/bin/env python3
"""
Aggressively sanitize PhysX-Anything requirements.txt so that:

  * It never refers to local conda paths (/mnt, /croot, file://).
  * It never pulls torch / torchvision / torchaudio / kaolin from PyPI.
  * It never contains obviously invalid version strings like
      ==v2.2.1-2-g1505ef3-master
    which cause pip's packaging.version to explode.
  * It avoids installing all the big custom CUDA libs that are handled by
    setup.sh (mmcv, mmengine, xformers, flash-attn, spconv, nvdiffrast, etc.).

Usage:
    python patch_requirements.py requirements.txt
"""

import os
import re
import sys
from pathlib import Path

# Packages installed explicitly (Dockerfile) or via setup.sh
BLOCKED_PKGS = {
    # Core torch stack (Dockerfile)
    "torch",
    "torchvision",
    "torchaudio",
    # Kaolin comes from NVIDIA S3 wheels, not PyPI
    "kaolin",
    # Heavy stuff setup.sh should own
    "mmcv",
    "mmcv-full",
    "mmengine",
    "mmdet",
    "mmdet3d",
    "openmim",
    "xformers",
    "flash-attn",
    "flash_attn",
    "spconv",
    "spconv-cu118",
    "diffoctreerast",
    "nvdiffrast",
    "mip-splatting",
    "diff-gaussian-rasterization",
    "simple-knn",
    # Other custom CUDA extension packages we never want pip to
    # try building from a random conda export
    "chamferdist",
    "emd",
    "pointnet2",
    "pointnet2_ops",
    "carafe",
    "cuvoxelization",
    "cuvoxel",
    # Possible future offenders
    "pytorch3d",
    "tiny-cuda-nn",
    "tinycudann",
}

# e.g. "mmcv==v2.2.1-2-g1505ef3-master"
BAD_VERSION_RE = re.compile(r"==\s*v\d", re.IGNORECASE)


def starts_with_pkg(line_lower: str, pkg: str) -> bool:
    """Return True if requirement line starts with the given package name."""
    if not line_lower.startswith(pkg):
        return False
    next_ch = line_lower[len(pkg):len(pkg) + 1]
    return not next_ch or next_ch in (" ", "=", "<", ">", "!", "[", "@", "~")


def patch_requirements(req_file: str) -> None:
    req = Path(req_file)
    if not req.is_file():
        raise SystemExit(f"requirements.txt not found: {req_file}")

    lines = req.read_text().splitlines()
    out_lines = []

    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()

        # Blank lines and comments stay as-is
        if not stripped or stripped.startswith("#"):
            out_lines.append(line)
            continue

        # pip options like -f, -i, --extra-index-url
        if stripped.startswith("-"):
            out_lines.append(line)
            continue

        # 1) Kill obviously invalid git-style versions
        if BAD_VERSION_RE.search(stripped) or ("master" in lower and "==" in stripped):
            print(f"[PATCH-REQ] Commenting invalid version line: {stripped!r}")
            out_lines.append(
                f"# commented-out invalid version (git-style tag): {line}"
            )
            continue

        # 2) Anything we explicitly block (torch stack, kaolin, mmcv, etc.)
        handled_blocked = False
        for pkg in BLOCKED_PKGS:
            if starts_with_pkg(lower, pkg):
                print(
                    f"[PATCH-REQ] Commenting out {pkg} dependency "
                    f"(handled by Dockerfile/setup.sh): {stripped!r}"
                )
                out_lines.append(
                    f"# commented-out {pkg} (handled by Dockerfile/setup.sh): {line}"
                )
                handled_blocked = True
                break
        if handled_blocked:
            continue

        # 3) Local wheels / file:// / conda paths
        if "file://" in lower or "mnt/" in lower or "croot/" in lower:
            # Try to salvage xxx.whl -> pkg==version when possible
            if ".whl" in stripped:
                fname = os.path.basename(stripped.split("#", 1)[0])
                if fname.endswith(".whl"):
                    fname = fname[:-4]
                parts = fname.split("-")
                if len(parts) >= 2:
                    pkg = parts[0]
                    ver = parts[1]
                    if BAD_VERSION_RE.search("==" + ver):
                        print(
                            f"[PATCH-REQ] Wheel has bad version {ver!r}, "
                            f"commenting out: {stripped!r}"
                        )
                        out_lines.append(
                            f"# commented-out wheel with invalid version: {line}"
                        )
                    else:
                        print(f"[PATCH-REQ] Rewriting {stripped!r} -> {pkg}=={ver}")
                        out_lines.append(f"{pkg}=={ver}")
                    continue
            # Generic local path -> just comment it
            print(
                "[PATCH-REQ] Commenting out local file/path dep "
                f"(not usable in Docker): {stripped!r}"
            )
            out_lines.append(
                f"# commented-out local file/path dep (not usable in Docker): {line}"
            )
            continue

        # 4) Bare filesystem-looking paths with no URL or 'pkg @ url'
        looks_like_path = ("/" in stripped or "\\" in stripped)
        has_urlish = any(proto in lower for proto in ("http://", "https://", "git+"))
        has_at = "@" in stripped  # "pkg @ url" style
        if looks_like_path and not has_urlish and not has_at:
            print(
                f"[PATCH-REQ] Commenting out bare local path dependency: {stripped!r}"
            )
            out_lines.append(f"# commented-out bare local path dep: {line}")
            continue

        # Default: keep line unchanged
        out_lines.append(line)

    req.write_text("\n".join(out_lines) + "\n")
    print(f"[PATCH-REQ] Successfully patched {req_file}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: patch_requirements.py <requirements.txt>")
        raise SystemExit(1)
    patch_requirements(sys.argv[1])
