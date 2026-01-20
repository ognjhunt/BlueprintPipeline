#!/usr/bin/env python3
"""Attempt a lightweight GPU allocation to warm up the device."""

import os
import sys
import traceback


def log(message: str) -> None:
    print(message, flush=True)


def parse_preallocate_mb() -> int:
    value = os.getenv("GENIESIM_GPU_PREALLOCATE_MB", "").strip()
    if not value:
        return 0
    try:
        return max(int(value), 0)
    except ValueError:
        log(
            "[gpu-warmup] Invalid GENIESIM_GPU_PREALLOCATE_MB value; "
            "expected integer megabytes."
        )
        return 0


def warmup_with_torch(preallocate_mb: int) -> bool:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - runtime optional dependency
        log(f"[gpu-warmup] Torch not available ({exc}); skipping warmup.")
        return False

    if not torch.cuda.is_available():
        log("[gpu-warmup] Torch CUDA not available; skipping warmup.")
        return False

    bytes_to_alloc = preallocate_mb * 1024 * 1024
    if bytes_to_alloc <= 0:
        log("[gpu-warmup] No allocation requested; skipping warmup.")
        return True

    log(
        f"[gpu-warmup] Allocating ~{preallocate_mb} MB on CUDA device "
        f"{torch.cuda.current_device()}"
    )

    try:
        element_count = max(bytes_to_alloc // 4, 1)
        _tensor = torch.empty(element_count, device="cuda", dtype=torch.float32)
        _tensor.fill_(0)
        torch.cuda.synchronize()
        log("[gpu-warmup] GPU allocation successful.")
        return True
    except Exception as exc:  # pragma: no cover - runtime-specific
        log(f"[gpu-warmup] GPU allocation failed: {exc}")
        return False


def main() -> int:
    strict = os.getenv("GENIESIM_GPU_WARMUP_STRICT", "0").strip() in {"1", "true", "True"}
    preallocate_mb = parse_preallocate_mb()

    if preallocate_mb <= 0:
        log("[gpu-warmup] GENIESIM_GPU_PREALLOCATE_MB not set or zero; skipping warmup.")
        return 0

    try:
        success = warmup_with_torch(preallocate_mb)
    except Exception:  # pragma: no cover - safeguard
        log("[gpu-warmup] Unexpected error during warmup:")
        traceback.print_exc()
        success = False

    if not success and strict:
        log("[gpu-warmup] Strict mode enabled; exiting with failure.")
        return 1

    if not success:
        log("[gpu-warmup] Warmup skipped or failed; continuing.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
