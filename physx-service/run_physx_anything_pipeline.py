#!/usr/bin/env python3
"""
PhysX-Anything Pipeline Runner.

This script orchestrates the 4-stage PhysX-Anything pipeline:
1. VLM Demo (1_vlm_demo.py) - Vision-language model analysis to detect articulated parts
2. Decoder (2_decoder.py) - Generate 3D geometry from VLM output
3. Split (3_split.py) - Segment mesh into individual parts
4. SimReady Gen (4_simready_gen.py) - Generate URDF with joint definitions

The script handles:
- Pipeline serialization via file locking (2_decoder.py and 3_split.py have hardcoded paths)
- Working directory cleanup between runs
- Comprehensive logging for debugging
- Output discovery and copying to output directory

Usage:
    python run_physx_anything_pipeline.py \
        --input_image /path/to/image.png \
        --output_dir /path/to/output \
        --request_id abc123
"""

import argparse
import fcntl
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import List, Optional


# =============================================================================
# Configuration
# =============================================================================

PIPELINE_LOCK_FILE = "/tmp/physx_pipeline.lock"
DEFAULT_TIMEOUT = 600  # 10 minutes per step


# =============================================================================
# Logging
# =============================================================================

def log(msg: str, request_id: str = "", level: str = "INFO") -> None:
    """Log with timestamp and request ID."""
    prefix = f"[PIPELINE] [{request_id}]" if request_id else "[PIPELINE]"
    stream = sys.stderr if level == "ERROR" else sys.stdout
    print(f"{prefix} [{level}] {msg}", file=stream, flush=True)


# =============================================================================
# Pipeline Lock
# =============================================================================

def acquire_lock(timeout: int = 900, request_id: str = "") -> Optional[int]:
    """
    Acquire exclusive lock for pipeline execution.

    Because 2_decoder.py and 3_split.py use hardcoded paths (test_demo),
    only one pipeline can run at a time.
    """
    log(f"Acquiring pipeline lock (timeout={timeout}s)...", request_id)
    start = time.time()

    lock_dir = Path(PIPELINE_LOCK_FILE).parent
    lock_dir.mkdir(parents=True, exist_ok=True)

    fd = os.open(PIPELINE_LOCK_FILE, os.O_CREAT | os.O_RDWR)

    while time.time() - start < timeout:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            log("Lock acquired", request_id)
            return fd
        except BlockingIOError:
            elapsed = int(time.time() - start)
            if elapsed > 0 and elapsed % 30 == 0:
                log(f"Waiting for lock ({elapsed}s)...", request_id)
            time.sleep(1)

    os.close(fd)
    log("Lock timeout", request_id, "ERROR")
    return None


def release_lock(fd: int, request_id: str = "") -> None:
    """Release pipeline lock."""
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
        log("Lock released", request_id)
    except Exception as e:
        log(f"Lock release error: {e}", request_id, "WARNING")


# =============================================================================
# Command Execution
# =============================================================================

def run_step(
    args: List[str],
    cwd: Path,
    request_id: str,
    step_name: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> subprocess.CompletedProcess:
    """
    Run a pipeline step with output capture.

    Raises subprocess.CalledProcessError on failure.
    """
    log(f"Step: {step_name}", request_id)
    log(f"  Command: {' '.join(str(a) for a in args)}", request_id)

    start = time.time()

    try:
        result = subprocess.run(
            args,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        log(f"  Timeout after {timeout}s", request_id, "ERROR")
        if e.stdout:
            log(f"  Partial stdout:\n{e.stdout[-2000:]}", request_id)
        if e.stderr:
            log(f"  Partial stderr:\n{e.stderr[-2000:]}", request_id, "ERROR")
        raise

    elapsed = int(time.time() - start)

    # Log output (truncated for readability)
    if result.stdout:
        lines = result.stdout.strip().split('\n')
        if len(lines) > 30:
            log(f"  stdout ({len(lines)} lines, last 30):", request_id)
            for line in lines[-30:]:
                log(f"    {line}", request_id)
        else:
            log(f"  stdout:", request_id)
            for line in lines:
                log(f"    {line}", request_id)

    if result.stderr:
        for line in result.stderr.strip().split('\n')[-20:]:
            log(f"  stderr: {line}", request_id, "WARNING")

    log(f"  Exit: {result.returncode}, Time: {elapsed}s", request_id)

    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, args,
            output=result.stdout, stderr=result.stderr
        )

    return result


# =============================================================================
# Model Validation
# =============================================================================

def check_models(repo_root: Path, request_id: str) -> bool:
    """Verify required model files exist."""
    vlm_path = repo_root / "pretrain" / "vlm"

    if not vlm_path.is_dir():
        log(f"VLM directory not found: {vlm_path}", request_id, "ERROR")
        return False

    files = list(vlm_path.rglob("*"))
    file_count = len([f for f in files if f.is_file()])

    config_file = vlm_path / "config.json"
    if not config_file.is_file():
        log(f"config.json not found in {vlm_path}", request_id, "ERROR")
        return False

    has_weights = any(f.suffix in ('.safetensors', '.bin') for f in files if f.is_file())
    if not has_weights:
        log(f"No model weights found", request_id, "ERROR")
        return False

    total_size = sum(f.stat().st_size for f in files if f.is_file())
    log(f"VLM: {file_count} files, {total_size / 1e9:.2f} GB", request_id)

    return True


# =============================================================================
# Output Discovery
# =============================================================================

def find_outputs(output_dir: Path, request_id: str) -> tuple:
    """
    Find mesh and URDF in output directory.

    Returns (mesh_path, urdf_path) or (None, None) if not found.
    """
    if not output_dir.is_dir():
        log(f"Output directory not found: {output_dir}", request_id, "ERROR")
        return None, None

    log(f"Searching for outputs in {output_dir}...", request_id)

    # List contents
    for item in sorted(output_dir.rglob("*")):
        if item.is_file():
            log(f"  Found: {item.relative_to(output_dir)} ({item.stat().st_size} bytes)", request_id)

    # Find mesh (prefer GLB > GLTF > OBJ)
    mesh_path = None
    for ext in [".glb", ".gltf", ".obj"]:
        for p in output_dir.rglob(f"*{ext}"):
            if mesh_path is None or p.stat().st_size > mesh_path.stat().st_size:
                mesh_path = p

    # Find URDF
    urdf_path = None
    for p in output_dir.rglob("*.urdf"):
        if urdf_path is None or p.stat().st_size > urdf_path.stat().st_size:
            urdf_path = p

    return mesh_path, urdf_path


# =============================================================================
# Main Pipeline
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="PhysX-Anything Pipeline Runner")
    parser.add_argument("--input_image", required=True, help="Input image path")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--request_id", default="default", help="Request ID for logging")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    input_image = Path(args.input_image).resolve()
    output_dir = Path(args.output_dir).resolve()
    request_id = args.request_id

    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate inputs
    if not input_image.is_file():
        log(f"Input image not found: {input_image}", request_id, "ERROR")
        sys.exit(1)

    log("=" * 60, request_id)
    log("PhysX-Anything Pipeline", request_id)
    log("=" * 60, request_id)
    log(f"Input: {input_image} ({input_image.stat().st_size} bytes)", request_id)
    log(f"Output: {output_dir}", request_id)
    log(f"Repo: {repo_root}", request_id)

    # Validate models
    if not check_models(repo_root, request_id):
        sys.exit(1)

    # Acquire lock
    lock_fd = acquire_lock(timeout=900, request_id=request_id)
    if lock_fd is None:
        log("Failed to acquire lock", request_id, "ERROR")
        sys.exit(1)

    try:
        # Clean working directories
        demo_dir = repo_root / "demo"
        test_demo_dir = repo_root / "test_demo"

        log("Cleaning working directories...", request_id)
        shutil.rmtree(demo_dir, ignore_errors=True)
        shutil.rmtree(test_demo_dir, ignore_errors=True)
        demo_dir.mkdir(parents=True, exist_ok=True)

        # Copy input to demo directory
        target_img = demo_dir / input_image.name
        shutil.copy(input_image, target_img)
        log(f"Copied input to {target_img}", request_id)

        # ======================
        # Step 1: VLM Demo
        # ======================
        run_step(
            [
                sys.executable, "1_vlm_demo.py",
                "--demo_path", str(demo_dir),
                "--save_part_ply", "True",
                "--remove_bg", "False",
                "--ckpt", "./pretrain/vlm",
            ],
            cwd=repo_root,
            request_id=request_id,
            step_name="VLM Analysis",
            timeout=600,
        )

        # ======================
        # Step 2: Decoder
        # ======================
        run_step(
            [sys.executable, "2_decoder.py"],
            cwd=repo_root,
            request_id=request_id,
            step_name="3D Decoder",
            timeout=300,
        )

        # ======================
        # Step 3: Split
        # ======================
        run_step(
            [sys.executable, "3_split.py"],
            cwd=repo_root,
            request_id=request_id,
            step_name="Mesh Split",
            timeout=300,
        )

        # ======================
        # Step 4: SimReady Gen
        # ======================
        run_step(
            [
                sys.executable, "4_simready_gen.py",
                "--voxel_define", "32",
                "--basepath", str(test_demo_dir),
                "--process", "0",
                "--fixed_base", "0",
                "--deformable", "0",
            ],
            cwd=repo_root,
            request_id=request_id,
            step_name="SimReady Generation",
            timeout=300,
        )

        # ======================
        # Find and copy outputs
        # ======================
        log("=" * 60, request_id)
        log("Locating outputs...", request_id)

        if not test_demo_dir.is_dir():
            log(f"Expected output dir not found: {test_demo_dir}", request_id, "ERROR")
            sys.exit(1)

        mesh_path, urdf_path = find_outputs(test_demo_dir, request_id)

        if not mesh_path:
            log("No mesh file found", request_id, "ERROR")
            sys.exit(1)

        if not urdf_path:
            log("No URDF file found", request_id, "ERROR")
            sys.exit(1)

        # Copy to output directory
        out_mesh = output_dir / "part.glb"
        out_urdf = output_dir / "part.urdf"

        shutil.copy(mesh_path, out_mesh)
        shutil.copy(urdf_path, out_urdf)

        log(f"Output mesh: {out_mesh} ({out_mesh.stat().st_size} bytes)", request_id)
        log(f"Output URDF: {out_urdf} ({out_urdf.stat().st_size} bytes)", request_id)

        # Sanity checks
        if out_mesh.stat().st_size < 100:
            log("WARNING: Mesh file suspiciously small!", request_id, "WARNING")
        if out_urdf.stat().st_size < 50:
            log("WARNING: URDF file suspiciously small!", request_id, "WARNING")

        log("=" * 60, request_id)
        log("Pipeline completed successfully!", request_id)
        log("=" * 60, request_id)

    except subprocess.CalledProcessError as e:
        log(f"Pipeline step failed (exit code {e.returncode})", request_id, "ERROR")
        if e.stdout:
            log(f"stdout:\n{e.stdout[-2000:]}", request_id, "ERROR")
        if e.stderr:
            log(f"stderr:\n{e.stderr[-2000:]}", request_id, "ERROR")
        sys.exit(1)
    except Exception as e:
        log(f"Unexpected error: {type(e).__name__}: {e}", request_id, "ERROR")
        log(traceback.format_exc(), request_id, "ERROR")
        sys.exit(1)
    finally:
        # Release lock
        release_lock(lock_fd, request_id)

        # Cleanup
        log("Cleaning up...", request_id)
        try:
            shutil.rmtree(repo_root / "demo", ignore_errors=True)
            shutil.rmtree(repo_root / "test_demo", ignore_errors=True)
        except Exception as e:
            log(f"Cleanup warning: {e}", request_id, "WARNING")


if __name__ == "__main__":
    main()
