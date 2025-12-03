#!/usr/bin/env python
"""
PhysX-Anything pipeline wrapper with proper isolation and error handling.

Key fixes in this version:
1. Full subprocess error capture (stdout AND stderr, no truncation)
2. Better file locking for hardcoded paths
3. Detailed logging at each step
4. shutil.copy() instead of copy2() for container compatibility
5. Model existence validation
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
from typing import Optional, List


# Global lock file for pipeline serialization
PIPELINE_LOCK_FILE = "/tmp/physx_pipeline.lock"


def log(msg: str, request_id: str = "", level: str = "INFO") -> None:
    """Log with timestamp and request ID."""
    prefix = f"[PHYSX-PIPELINE] [{request_id}]" if request_id else "[PHYSX-PIPELINE]"
    print(f"{prefix} [{level}] {msg}", flush=True)


def run_cmd(
    args: List[str], 
    cwd: Path, 
    env: dict = None, 
    request_id: str = "",
    timeout: int = 600,
) -> subprocess.CompletedProcess:
    """
    Run a command with full output capture.
    
    Returns the CompletedProcess result. Raises on non-zero exit.
    """
    log(f"Running: {' '.join(str(a) for a in args)}", request_id)
    log(f"  cwd: {cwd}", request_id)
    log(f"  timeout: {timeout}s", request_id)
    
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            args, 
            cwd=str(cwd),
            env=run_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,  # Capture stderr separately
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        log(f"Command timed out after {timeout}s", request_id, "ERROR")
        # Try to get partial output
        stdout = e.stdout or ""
        stderr = e.stderr or ""
        if stdout:
            log(f"Partial stdout:\n{stdout}", request_id)
        if stderr:
            log(f"Partial stderr:\n{stderr}", request_id, "ERROR")
        raise
    
    elapsed = time.time() - start_time
    
    # Log output
    if result.stdout:
        # Log last 50 lines of stdout
        lines = result.stdout.strip().split('\n')
        if len(lines) > 50:
            log(f"  stdout ({len(lines)} lines, showing last 50):", request_id)
            for line in lines[-50:]:
                log(f"  | {line}", request_id)
        else:
            log(f"  stdout:", request_id)
            for line in lines:
                log(f"  | {line}", request_id)
    
    if result.stderr:
        log(f"  stderr:", request_id, "WARNING")
        for line in result.stderr.strip().split('\n')[-30:]:
            log(f"  ! {line}", request_id, "WARNING")
    
    log(f"  Exit code: {result.returncode}, elapsed: {elapsed:.1f}s", request_id)
    
    if result.returncode != 0:
        error_msg = f"Command failed with exit code {result.returncode}"
        log(error_msg, request_id, "ERROR")
        # Create a custom exception with full output
        raise subprocess.CalledProcessError(
            result.returncode,
            args,
            output=result.stdout,
            stderr=result.stderr,
        )
    
    return result


def acquire_pipeline_lock(timeout: int = 600, request_id: str = "") -> Optional[int]:
    """
    Acquire exclusive lock on the pipeline.
    
    Because 2_decoder.py and 3_split.py have hardcoded paths (test_demo),
    we can only run one pipeline at a time.
    
    Returns: file descriptor if lock acquired, None on timeout
    """
    log(f"Acquiring pipeline lock (timeout={timeout}s)...", request_id)
    
    start_time = time.time()
    
    # Create lock file if it doesn't exist
    lock_dir = Path(PIPELINE_LOCK_FILE).parent
    lock_dir.mkdir(parents=True, exist_ok=True)
    
    fd = os.open(PIPELINE_LOCK_FILE, os.O_CREAT | os.O_RDWR)
    
    while time.time() - start_time < timeout:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            log("Pipeline lock acquired", request_id)
            return fd
        except BlockingIOError:
            # Another process holds the lock
            elapsed = int(time.time() - start_time)
            if elapsed > 0 and elapsed % 30 == 0:  # Log every 30 seconds
                log(f"Waiting for pipeline lock ({elapsed}s elapsed)...", request_id)
            time.sleep(1)
    
    os.close(fd)
    log("ERROR: Failed to acquire pipeline lock within timeout", request_id, "ERROR")
    return None


def release_pipeline_lock(fd: int, request_id: str = "") -> None:
    """Release the pipeline lock."""
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
        log("Pipeline lock released", request_id)
    except Exception as e:
        log(f"Warning: Error releasing lock: {e}", request_id, "WARNING")


def check_models_exist(repo_root: Path, request_id: str = "") -> bool:
    """Check if required model weights exist."""
    vlm_path = repo_root / "pretrain" / "vlm"
    
    if not vlm_path.is_dir():
        log(f"VLM checkpoint directory not found at {vlm_path}", request_id, "ERROR")
        return False
    
    files = list(vlm_path.rglob("*"))
    file_count = len([f for f in files if f.is_file()])
    
    # Check for config.json (essential)
    config_file = vlm_path / "config.json"
    if not config_file.is_file():
        log(f"config.json not found in {vlm_path}", request_id, "ERROR")
        return False
    
    # Check for model weights
    has_weights = any(
        f.suffix in ('.safetensors', '.bin') 
        for f in files if f.is_file()
    )
    
    if not has_weights:
        log(f"No model weights found in {vlm_path}", request_id, "ERROR")
        return False
    
    # Calculate total size
    total_size = sum(f.stat().st_size for f in files if f.is_file())
    size_gb = total_size / 1e9
    
    log(f"VLM checkpoint found at {vlm_path}", request_id)
    log(f"  Files: {file_count}, Size: {size_gb:.2f} GB", request_id)
    
    if size_gb < 1.0:
        log(f"WARNING: Model size ({size_gb:.2f} GB) seems too small!", request_id, "WARNING")
    
    return True


def list_directory(path: Path, request_id: str = "", max_depth: int = 2) -> None:
    """List directory contents for debugging."""
    if not path.exists():
        log(f"Directory does not exist: {path}", request_id, "WARNING")
        return
    
    log(f"Contents of {path}:", request_id)
    
    for item in sorted(path.rglob("*")):
        if item.is_file():
            rel_path = item.relative_to(path)
            depth = len(rel_path.parts)
            if depth <= max_depth:
                size = item.stat().st_size
                log(f"  {rel_path} ({size} bytes)", request_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--request_id", type=str, default="default",
                        help="Unique request ID for logging and isolation")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    input_image = Path(args.input_image).resolve()
    output_dir = Path(args.output_dir).resolve()
    request_id = args.request_id
    
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_image.is_file():
        raise SystemExit(f"Input image not found: {input_image}")

    log(f"Request ID: {request_id}", request_id)
    log(f"Input: {input_image} ({input_image.stat().st_size} bytes)", request_id)
    log(f"Output: {output_dir}", request_id)
    log(f"Repo root: {repo_root}", request_id)

    # Check models exist
    if not check_models_exist(repo_root, request_id):
        raise SystemExit("Required model files not found")

    # Acquire pipeline lock
    lock_fd = acquire_pipeline_lock(timeout=900, request_id=request_id)
    if lock_fd is None:
        raise SystemExit("Failed to acquire pipeline lock - service is busy")

    try:
        # Clean up any previous run artifacts
        demo_dir = repo_root / "demo"
        test_demo_dir = repo_root / "test_demo"
        
        log("Cleaning up previous artifacts...", request_id)
        shutil.rmtree(demo_dir, ignore_errors=True)
        shutil.rmtree(test_demo_dir, ignore_errors=True)
        
        demo_dir.mkdir(parents=True, exist_ok=True)

        # Copy input image to demo directory
        # Use shutil.copy (not copy2) to avoid metadata errors in containers
        target_img = demo_dir / input_image.name
        shutil.copy(input_image, target_img)
        log(f"Copied input to {target_img}", request_id)

        # ======================
        # Step 1: VLM Demo
        # ======================
        log("=" * 60, request_id)
        log("STEP 1: VLM Demo (Vision-Language Model inference)", request_id)
        log("=" * 60, request_id)
        
        cmd1 = [
            sys.executable,
            "1_vlm_demo.py",
            "--demo_path", str(demo_dir),
            "--save_part_ply", "True",
            "--remove_bg", "False",
            "--ckpt", "./pretrain/vlm",
        ]
        
        try:
            run_cmd(cmd1, repo_root, request_id=request_id, timeout=600)
        except subprocess.CalledProcessError as e:
            log(f"VLM demo failed!", request_id, "ERROR")
            log(f"Full stdout:\n{e.stdout}", request_id, "ERROR")
            log(f"Full stderr:\n{e.stderr}", request_id, "ERROR")
            raise

        # ======================
        # Step 2: Decoder
        # ======================
        log("=" * 60, request_id)
        log("STEP 2: Decoder inference", request_id)
        log("=" * 60, request_id)
        
        # Note: 2_decoder.py has hardcoded paths
        cmd2 = [sys.executable, "2_decoder.py"]
        
        try:
            run_cmd(cmd2, repo_root, request_id=request_id, timeout=300)
        except subprocess.CalledProcessError as e:
            log(f"Decoder failed!", request_id, "ERROR")
            log(f"Full stdout:\n{e.stdout}", request_id, "ERROR")
            log(f"Full stderr:\n{e.stderr}", request_id, "ERROR")
            raise

        # ======================
        # Step 3: Split
        # ======================
        log("=" * 60, request_id)
        log("STEP 3: Split mesh", request_id)
        log("=" * 60, request_id)
        
        # Note: 3_split.py has hardcoded paths
        cmd3 = [sys.executable, "3_split.py"]
        
        try:
            run_cmd(cmd3, repo_root, request_id=request_id, timeout=300)
        except subprocess.CalledProcessError as e:
            log(f"Split failed!", request_id, "ERROR")
            log(f"Full stdout:\n{e.stdout}", request_id, "ERROR")
            log(f"Full stderr:\n{e.stderr}", request_id, "ERROR")
            raise

        # ======================
        # Step 4: SimReady Generation
        # ======================
        log("=" * 60, request_id)
        log("STEP 4: SimReady generation (URDF/XML)", request_id)
        log("=" * 60, request_id)
        
        cmd4 = [
            sys.executable,
            "4_simready_gen.py",
            "--voxel_define", "32",
            "--basepath", str(test_demo_dir),
            "--process", "0",
            "--fixed_base", "0",
            "--deformable", "0",
        ]
        
        try:
            run_cmd(cmd4, repo_root, request_id=request_id, timeout=300)
        except subprocess.CalledProcessError as e:
            log(f"SimReady gen failed!", request_id, "ERROR")
            log(f"Full stdout:\n{e.stdout}", request_id, "ERROR")
            log(f"Full stderr:\n{e.stderr}", request_id, "ERROR")
            raise

        # ======================
        # Find and copy outputs
        # ======================
        log("=" * 60, request_id)
        log("Locating outputs...", request_id)
        log("=" * 60, request_id)

        if not test_demo_dir.is_dir():
            log(f"ERROR: Expected output directory {test_demo_dir} does not exist", request_id, "ERROR")
            # List what we DO have for debugging
            log("Listing repo root:", request_id)
            list_directory(repo_root, request_id, max_depth=1)
            raise SystemExit(f"Output directory {test_demo_dir} not created")

        # List all files for debugging
        list_directory(test_demo_dir, request_id, max_depth=3)

        # Find mesh and URDF
        mesh_path = None
        urdf_path = None

        # Look for GLB/GLTF mesh
        for ext in [".glb", ".gltf"]:
            for p in test_demo_dir.rglob(f"*{ext}"):
                if mesh_path is None or p.stat().st_size > mesh_path.stat().st_size:
                    mesh_path = p

        # Look for URDF
        for p in test_demo_dir.rglob("*.urdf"):
            if urdf_path is None or p.stat().st_size > urdf_path.stat().st_size:
                urdf_path = p

        # Fallback to OBJ if no GLB
        if mesh_path is None:
            for p in test_demo_dir.rglob("*.obj"):
                mesh_path = p
                break

        if mesh_path is None:
            log("ERROR: No mesh file found!", request_id, "ERROR")
            raise SystemExit(f"Could not find mesh file in {test_demo_dir}")
        
        if urdf_path is None:
            log("ERROR: No URDF file found!", request_id, "ERROR")
            raise SystemExit(f"Could not find URDF file in {test_demo_dir}")

        log(f"Found mesh: {mesh_path} ({mesh_path.stat().st_size} bytes)", request_id)
        log(f"Found URDF: {urdf_path} ({urdf_path.stat().st_size} bytes)", request_id)

        # Copy to output directory
        out_mesh = output_dir / "part.glb"
        out_urdf = output_dir / "part.urdf"
        
        shutil.copy(mesh_path, out_mesh)
        shutil.copy(urdf_path, out_urdf)

        log(f"Exported mesh -> {out_mesh} ({out_mesh.stat().st_size} bytes)", request_id)
        log(f"Exported URDF -> {out_urdf} ({out_urdf.stat().st_size} bytes)", request_id)

        # Verify outputs are non-trivial
        if out_mesh.stat().st_size < 100:
            log("WARNING: Mesh file is suspiciously small!", request_id, "WARNING")
        if out_urdf.stat().st_size < 50:
            log("WARNING: URDF file is suspiciously small!", request_id, "WARNING")

        log("Pipeline completed successfully!", request_id)

    except subprocess.CalledProcessError as e:
        # Re-raise with full error info preserved
        log(f"Pipeline step failed with exit code {e.returncode}", request_id, "ERROR")
        raise SystemExit(1)
    except Exception as e:
        log(f"Unexpected error: {type(e).__name__}: {e}", request_id, "ERROR")
        log(traceback.format_exc(), request_id, "ERROR")
        raise SystemExit(1)
    finally:
        # Always release lock
        release_pipeline_lock(lock_fd, request_id)
        
        # Clean up working directories
        log("Cleaning up working directories...", request_id)
        try:
            shutil.rmtree(repo_root / "demo", ignore_errors=True)
            shutil.rmtree(repo_root / "test_demo", ignore_errors=True)
            shutil.rmtree(repo_root / "demo_from_service", ignore_errors=True)
        except Exception as e:
            log(f"Warning: Cleanup error: {e}", request_id, "WARNING")


if __name__ == "__main__":
    main()