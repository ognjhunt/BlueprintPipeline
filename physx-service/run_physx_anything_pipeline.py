#!/usr/bin/env python
"""
PhysX-Anything pipeline wrapper with proper isolation and model handling.

This script addresses several issues with the PhysX-Anything pipeline:
1. Hardcoded paths in intermediate scripts (2_decoder.py, 3_split.py)
2. Model download requirements
3. Concurrent request isolation

The PhysX-Anything pipeline flow:
1. 1_vlm_demo.py: VLM inference on input image -> outputs to demo_path
2. 2_decoder.py: Decoder inference -> reads from test_demo (hardcoded!)
3. 3_split.py: Split mesh -> reads/writes to test_demo (hardcoded!)
4. 4_simready_gen.py: Generate URDF/XML -> outputs to basepath

Because 2_decoder.py and 3_split.py have hardcoded paths, we need to:
- Use file locks to ensure only one request uses these paths at a time
- Copy outputs between request-specific and hardcoded directories

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
from pathlib import Path
from typing import Optional


# Global lock file for pipeline serialization
PIPELINE_LOCK_FILE = "/tmp/physx_pipeline.lock"


def log(msg: str, request_id: str = "") -> None:
    """Log with timestamp and request ID."""
    prefix = f"[PHYSX-PIPELINE] [{request_id}]" if request_id else "[PHYSX-PIPELINE]"
    print(f"{prefix} {msg}", flush=True)


def run_cmd(args, cwd: Path, env: dict = None, request_id: str = "") -> str:
    """Run a command with logging and return stdout."""
    log(f"Running: {' '.join(str(a) for a in args)}", request_id)
    log(f"  cwd: {cwd}", request_id)
    
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    
    start_time = time.time()
    result = subprocess.run(
        args, 
        check=True, 
        cwd=str(cwd),
        env=run_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    elapsed = time.time() - start_time
    
    if result.stdout:
        for line in result.stdout.strip().split('\n')[-20:]:  # Last 20 lines
            log(f"  | {line}", request_id)
    
    log(f"  Completed in {elapsed:.1f}s", request_id)
    return result.stdout


def acquire_pipeline_lock(timeout: int = 600) -> Optional[int]:
    """
    Acquire exclusive lock on the pipeline.
    
    Because 2_decoder.py and 3_split.py have hardcoded paths (test_demo),
    we can only run one pipeline at a time.
    
    Returns: file descriptor if lock acquired, None on timeout
    """
    log(f"Acquiring pipeline lock (timeout={timeout}s)...")
    
    start_time = time.time()
    fd = os.open(PIPELINE_LOCK_FILE, os.O_CREAT | os.O_RDWR)
    
    while time.time() - start_time < timeout:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            log("Pipeline lock acquired")
            return fd
        except BlockingIOError:
            # Another process holds the lock
            elapsed = int(time.time() - start_time)
            if elapsed % 30 == 0:  # Log every 30 seconds
                log(f"Waiting for pipeline lock ({elapsed}s elapsed)...")
            time.sleep(1)
    
    os.close(fd)
    log("ERROR: Failed to acquire pipeline lock within timeout")
    return None


def release_pipeline_lock(fd: int) -> None:
    """Release the pipeline lock."""
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
        log("Pipeline lock released")
    except Exception as e:
        log(f"Warning: Error releasing lock: {e}")


def check_models_exist(repo_root: Path, request_id: str) -> bool:
    """Check if required model weights exist."""
    vlm_path = repo_root / "pretrain" / "vlm"
    
    if vlm_path.is_dir():
        files = list(vlm_path.glob("*"))
        log(f"VLM checkpoint found at {vlm_path} ({len(files)} files)", request_id)
        return True
    
    log(f"WARNING: VLM checkpoint not found at {vlm_path}", request_id)
    return False


def download_models_if_needed(repo_root: Path, request_id: str) -> None:
    """Download model weights if not present."""
    vlm_path = repo_root / "pretrain" / "vlm"
    
    if vlm_path.is_dir() and any(vlm_path.glob("*.bin")):
        log("Model weights already present", request_id)
        return
    
    log("Attempting to download model weights...", request_id)
    
    # Try the repo's download script
    download_script = repo_root / "download.py"
    if download_script.is_file():
        try:
            run_cmd([sys.executable, str(download_script)], repo_root, request_id=request_id)
            log("Model download completed", request_id)
            return
        except subprocess.CalledProcessError as e:
            log(f"download.py failed: {e}", request_id)
    
    # Try huggingface-cli
    try:
        log("Trying huggingface-cli download...", request_id)
        # Based on README: Download from huggingface_v1
        run_cmd([
            "huggingface-cli", "download",
            "Caoza/PhysX-Anything",  # or the actual repo name
            "--local-dir", str(repo_root / "pretrain"),
        ], repo_root, request_id=request_id)
    except Exception as e:
        log(f"huggingface-cli download failed: {e}", request_id)
        log("WARNING: Models may not be available, pipeline might fail", request_id)


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
    log(f"Input: {input_image}", request_id)
    log(f"Output: {output_dir}", request_id)
    log(f"Repo root: {repo_root}", request_id)

    # Check/download models
    if not check_models_exist(repo_root, request_id):
        download_models_if_needed(repo_root, request_id)

    # Acquire pipeline lock (only one pipeline can run at a time due to hardcoded paths)
    lock_fd = acquire_pipeline_lock(timeout=900)  # 15 min timeout
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
        log("=" * 50, request_id)
        log("STEP 1: VLM Demo (Vision-Language Model inference)", request_id)
        log("=" * 50, request_id)
        
        cmd1 = [
            sys.executable,
            "1_vlm_demo.py",
            "--demo_path", str(demo_dir),
            "--save_part_ply", "True",
            "--remove_bg", "False",
            "--ckpt", "./pretrain/vlm",
        ]
        run_cmd(cmd1, repo_root, request_id=request_id)

        # ======================
        # Step 2: Decoder
        # ======================
        log("=" * 50, request_id)
        log("STEP 2: Decoder inference", request_id)
        log("=" * 50, request_id)
        
        # Note: 2_decoder.py likely has hardcoded paths
        cmd2 = [sys.executable, "2_decoder.py"]
        run_cmd(cmd2, repo_root, request_id=request_id)

        # ======================
        # Step 3: Split
        # ======================
        log("=" * 50, request_id)
        log("STEP 3: Split mesh", request_id)
        log("=" * 50, request_id)
        
        # Note: 3_split.py likely has hardcoded paths
        cmd3 = [sys.executable, "3_split.py"]
        run_cmd(cmd3, repo_root, request_id=request_id)

        # ======================
        # Step 4: SimReady Generation
        # ======================
        log("=" * 50, request_id)
        log("STEP 4: SimReady generation (URDF/XML)", request_id)
        log("=" * 50, request_id)
        
        cmd4 = [
            sys.executable,
            "4_simready_gen.py",
            "--voxel_define", "32",
            "--basepath", str(test_demo_dir),
            "--process", "0",
            "--fixed_base", "0",
            "--deformable", "0",
        ]
        run_cmd(cmd4, repo_root, request_id=request_id)

        # ======================
        # Find and copy outputs
        # ======================
        log("=" * 50, request_id)
        log("Locating outputs...", request_id)
        log("=" * 50, request_id)

        if not test_demo_dir.is_dir():
            raise SystemExit(f"Expected output directory {test_demo_dir} does not exist")

        # List all files for debugging
        log(f"Contents of {test_demo_dir}:", request_id)
        for item in sorted(test_demo_dir.rglob("*")):
            if item.is_file():
                size = item.stat().st_size
                log(f"  {item.relative_to(test_demo_dir)} ({size} bytes)", request_id)

        # Find mesh and URDF
        mesh_path = None
        urdf_path = None

        for p in test_demo_dir.rglob("*.glb"):
            if mesh_path is None or p.stat().st_size > mesh_path.stat().st_size:
                mesh_path = p
                
        for p in test_demo_dir.rglob("*.gltf"):
            if mesh_path is None or p.stat().st_size > mesh_path.stat().st_size:
                mesh_path = p

        for p in test_demo_dir.rglob("*.urdf"):
            if urdf_path is None or p.stat().st_size > urdf_path.stat().st_size:
                urdf_path = p

        if mesh_path is None:
            # Try .obj as fallback
            for p in test_demo_dir.rglob("*.obj"):
                mesh_path = p
                break

        if mesh_path is None or urdf_path is None:
            raise SystemExit(
                f"Could not find required outputs in {test_demo_dir}.\n"
                f"mesh={mesh_path}, urdf={urdf_path}"
            )

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
            log("WARNING: Mesh file is suspiciously small!", request_id)
        if out_urdf.stat().st_size < 50:
            log("WARNING: URDF file is suspiciously small!", request_id)

        log("Pipeline completed successfully!", request_id)

    finally:
        # Always release lock
        release_pipeline_lock(lock_fd)
        
        # Clean up working directories
        log("Cleaning up working directories...", request_id)
        try:
            shutil.rmtree(repo_root / "demo", ignore_errors=True)
            shutil.rmtree(repo_root / "test_demo", ignore_errors=True)
            shutil.rmtree(repo_root / "demo_from_service", ignore_errors=True)
        except Exception as e:
            log(f"Warning: Cleanup error: {e}", request_id)


if __name__ == "__main__":
    main()