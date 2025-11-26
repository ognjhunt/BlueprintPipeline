#!/usr/bin/env python3
"""
Patch a cluster-exported requirements.txt into something that can be
installed inside a clean Docker image.

Usage:
    python patch_requirements.py requirements.txt requirements.patched.txt

If only one argument is given, it patches the file in place.
"""

import sys
from pathlib import Path


# Packages we install explicitly in the Dockerfile / via setup.sh
TORCH_PKGS = (
    "torch",
    "torchvision",
    "torchaudio",
    "triton",
)

# Heavy CUDA / geometry packages handled either by setup.sh or
# manually in the Dockerfile.
HEAVY_MANUAL = (
    "chamferdist",
    "emd",
    "pointnet2_ops",
    "pointnet2",
    "flash-attn",
    "flash_attn",
    "kaolin",
    "nvdiffrast",
    "diffoctreerast",
    "spconv",
    "spconv-cu",
    "mip-splatting",
    "diff-gaussian-rasterization",
    "simple-knn",
    "tiny-cuda-nn",
    "tinycudann",
    "xformers",
    "cumm",
    "cumm-cu",
    "pccm",
)

# Split CUDA meta-wheels that came from the original torch install.
NVIDIA_META = (
    "nvidia-cublas-cu11",
    "nvidia-cublas-cu12",
    "nvidia-cuda-cupti-cu11",
    "nvidia-cuda-cupti-cu12",
    "nvidia-cuda-nvrtc-cu11",
    "nvidia-cuda-nvrtc-cu12",
    "nvidia-cuda-runtime-cu11",
    "nvidia-cuda-runtime-cu12",
    "nvidia-cudnn-cu11",
    "nvidia-cudnn-cu12",
    "nvidia-cufft-cu11",
    "nvidia-cufft-cu12",
    "nvidia-curand-cu11",
    "nvidia-curand-cu12",
    "nvidia-cusolver-cu11",
    "nvidia-cusolver-cu12",
    "nvidia-cusparse-cu11",
    "nvidia-cusparse-cu12",
    "nvidia-nccl-cu11",
    "nvidia-nccl-cu12",
    "nvidia-nvjitlink-cu12",
    "nvidia-nvtx-cu11",
    "nvidia-nvtx-cu12",
)


def _matches_pkg(line: str, name: str) -> bool:
    """
    Return True if `line` looks like it starts with the given package name,
    followed by a typical requirement separator (=, <, >, space, [ , @) or EOL.
    """
    s = line.strip().lower()
    if not s:
        return False

    if not s.startswith(name):
        return False

    after = s[len(name):len(name) + 1]
    return (not after) or (after in ("=", "<", ">", " ", "[", "@"))


def patch_requirements(src: str, dst: str) -> None:
    src_path = Path(src)
    if not src_path.is_file():
        raise SystemExit(f"requirements file not found: {src}")

    lines = src_path.read_text().splitlines()
    out_lines = []

    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()

        # Keep comments and blank lines verbatim
        if not stripped or stripped.startswith("#"):
            out_lines.append(line)
            continue

        # Pass through pip options (-f, -i, --find-links, etc.)
        if stripped.startswith("-"):
            out_lines.append(line)
            continue

        # 1) Torch stack is installed in the Dockerfile
        if any(_matches_pkg(stripped, p) for p in TORCH_PKGS):
            out_lines.append(f"# patched-out (torch stack handled in Dockerfile): {line}")
            continue

        # 2) CUDA-heavy packages handled by setup.sh or Dockerfile
        if any(_matches_pkg(stripped, p) for p in HEAVY_MANUAL):
            out_lines.append(f"# patched-out (handled by setup.sh / manual install): {line}")
            continue

        # 3) NVIDIA meta CUDA wheels from original torch install
        if any(_matches_pkg(stripped, p) for p in NVIDIA_META):
            out_lines.append(f"# patched-out (redundant CUDA meta-wheel): {line}")
            continue

        # 4) Cluster-local wheels / conda cache paths
        #    We *do not* try to infer versions from these filenames anymore,
        #    to avoid InvalidVersion errors like 'v2.2.1-2-g1505ef3-master'.
        if "file://" in lower or "croot/" in lower or "/mnt/" in lower:
            out_lines.append(f"# patched-out (unusable file:// / cluster wheel): {line}")
            continue

        if ".whl" in lower and ("/" in lower or "\\" in lower):
            out_lines.append(f"# patched-out (unusable wheel path inside Docker): {line}")
            continue

        # 5) Any other bare filesystem path that is not 'pkg @ url'
        if ("/" in stripped or "\\" in stripped) and "@" not in stripped \
           and "http://" not in lower and "https://" not in lower and "git+" not in lower:
            out_lines.append(f"# patched-out (bare filesystem path): {line}")
            continue

        # Default: keep the requirement line as-is
        out_lines.append(line)

    dst_path = Path(dst)
    dst_path.write_text("\n".join(out_lines) + "\n")
    print(f"[PATCH-REQ] wrote patched requirements to {dst_path}")


if __name__ == "__main__":
    if not (2 <= len(sys.argv) <= 3):
        print("Usage: patch_requirements.py <input-req> [<output-req>]")
        raise SystemExit(1)

    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) == 3 else src
    patch_requirements(src, dst)
