#!/usr/bin/env python3
"""
Patch PhysX-Anything requirements.txt to make it installable in a clean
Docker environment (no access to the original /mnt/... cluster paths).

We do three main things:

  * Rewrite local wheel files (file:///mnt/.../*.whl) into normal
    "package==version" pins so pip can fetch them from PyPI.

  * Strip out CUDA / custom extensions that expect a pre-existing local
    checkout (CARAFE, CuVoxelization, mip-splatting / diff-gaussian-
    rasterization / simple-knn, diffoctreerast, etc.). We install the
    ones we actually need manually in the Dockerfile after torch is
    available.

  * Comment out any other bare filesystem paths so pip doesn't explode
    on missing /mnt/..., /tmp/extensions/..., or ./submodules/... dirs.
"""

import os
import sys
from pathlib import Path


def patch_requirements(req_file: str) -> None:
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

        # Pip options (e.g. "-f https://...") - leave them alone
        if stripped.startswith("-"):
            new_lines.append(line)
            continue

        lowered = stripped.lower()

        # ------------------------------------------------------------
        # 1) CARAFE git dependency
        #
        # Upstream example:
        #   carafe @ git+https://github.com/myownskyW7/CARAFE.git@<commit>
        #
        # CARAFE's setup imports torch; with build isolation pip creates
        # a fresh env that does NOT yet have torch -> ModuleNotFoundError.
        # We comment it out here and install it manually in the Dockerfile.
        # ------------------------------------------------------------
        if "github.com/myownskyw7/carafe" in lowered:
            print(
                f"[PATCH-REQ] Commenting out CARAFE git dependency "
                f"(installed manually in Dockerfile): {stripped!r}"
            )
            new_lines.append(
                f"# commented-out CARAFE (installed manually in Dockerfile): {line}"
            )
            continue

        # ------------------------------------------------------------
        # 2) CuVoxelization local path
        #
        # Upstream contains something like:
        #   cuvoxel @ file:///mnt/.../CuVoxelization
        #
        # That path doesn't exist in Docker. We comment it out and
        # install CuVoxelization from its public GitHub repo manually.
        # ------------------------------------------------------------
        if "cuvoxelization" in lowered or "cuvoxel" in lowered:
            print(
                f"[PATCH-REQ] Commenting out CuVoxelization local path "
                f"(installed manually in Dockerfile): {stripped!r}"
            )
            new_lines.append(
                f"# commented-out CuVoxelization (installed manually in Dockerfile): {line}"
            )
            continue

        # ------------------------------------------------------------
        # 3) Mip-splatting / diff-gaussian-rasterization / simple-knn /
        #    diffoctreerast and any other /tmp/extensions/... local deps
        #
        # These are referenced via local paths like:
        #   diff_gaussian_rasterization @ file:///tmp/extensions/mip-splatting/...
        #   simple-knn @ file:///tmp/extensions/mip-splatting/...
        #   diffoctreerast @ file:///tmp/extensions/diffoctreerast
        #
        # They rely on separate git checkouts. We comment these out and
        # install only what we actually want manually in the Dockerfile.
        # ------------------------------------------------------------
        if (
            any(
                key in lowered
                for key in (
                    "diff-gaussian-rasterization",
                    "mip-splatting",
                    "simple-knn",
                    "diffoctreerast",
                )
            )
            or "file:///tmp/extensions/" in lowered
            or "/tmp/extensions/" in lowered
        ):
            print(
                "[PATCH-REQ] Commenting out mip-splatting / diff-gaussian-rasterization / "
                f"diffoctreerast local dependency (handled manually or not available "
                f"in Docker): {stripped!r}"
            )
            new_lines.append(
                "# commented-out mip-splatting / diff-gaussian-rasterization / diffoctreerast "
                f"(installed manually or not available in Docker): {line}"
            )
            continue

        # ------------------------------------------------------------
        # 4) Local wheel files from /mnt/... etc -> pkg==version
        #
        # Example upstream lines:
        #   boto3 @ file:///mnt/.../boto3-1.34.2-py3-none-any.whl#sha256=...
        #   mmcv  @ file:///mnt/.../mmcv-2.2.0-cp310-cp310-manylinux1_x86_64.whl#...
        #
        # We rewrite these into simple "package==version" pins so pip
        # can install them from PyPI.
        # ------------------------------------------------------------
        if ".whl" in stripped and ("mnt/" in stripped or stripped.startswith("/")):
            fname = os.path.basename(stripped.split("#", 1)[0])
            if fname.endswith(".whl"):
                fname = fname[:-4]

            parts = fname.split("-")
            if len(parts) >= 2:
                pkg = parts[0]
                ver = parts[1]
                print(f"[PATCH-REQ] Rewriting {stripped!r} -> {pkg}=={ver}")
                new_lines.append(f"{pkg}=={ver}")
                continue
            else:
                print(
                    f"[PATCH-REQ] Could not parse wheel {stripped!r}, commenting it out"
                )
                new_lines.append(f"# commented-out wheel path: {line}")
                continue

        # ------------------------------------------------------------
        # 5) Any other bare filesystem path is unusable inside Docker.
        #
        # This is the generic catch-all for lines that are *pure paths*:
        #   ./extensions/...
        #   extensions/mip-splatting/submodules/diff-gaussian-rasterization
        #   /some/absolute/path
        #
        # If the line contains a path separator but *not* an explicit
        # VCS / URL / "pkg @ url" spec, we treat it as a local path and
        # comment it out.
        # ------------------------------------------------------------
        looks_like_path = ("/" in stripped or "\\" in stripped)
        has_urlish = any(proto in lowered for proto in ("http://", "https://", "git+"))
        has_at = "@" in stripped  # "pkg @ url" style
        if looks_like_path and not has_urlish and not has_at:
            print(
                f"[PATCH-REQ] Commenting out unsupported local path dependency: {stripped!r}"
            )
            new_lines.append(f"# commented-out local path dep: {line}")
            continue

        # Default: keep requirement line unchanged
        new_lines.append(line)

    req.write_text("\n".join(new_lines) + "\n")
    print(f"[PATCH-REQ] Successfully patched {req_file}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: patch_requirements.py <requirements.txt>")
        sys.exit(1)

    patch_requirements(sys.argv[1])
