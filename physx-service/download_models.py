#!/usr/bin/env python3
"""
Robust model downloader for PhysX-Anything.

This script downloads the required models with proper authentication,
retry logic, and fallback mechanisms.

Usage:
    python download_models.py [--token HF_TOKEN]

Environment variables:
    HF_TOKEN or HUGGING_FACE_HUB_TOKEN: HuggingFace authentication token
"""
import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

def log(msg: str) -> None:
    """Print with flush for Docker build logs."""
    print(f"[DOWNLOAD] {msg}", flush=True)

def download_with_huggingface_hub(
    repo_id: str,
    local_dir: str,
    token: Optional[str] = None,
    max_retries: int = 3,
) -> bool:
    """
    Download using huggingface_hub library.

    Args:
        repo_id: HuggingFace repo ID (e.g., "Caoza/PhysX-Anything")
        local_dir: Local directory to save files
        token: HF authentication token
        max_retries: Number of retry attempts

    Returns:
        True if successful, False otherwise
    """
    try:
        from huggingface_hub import snapshot_download, login

        log(f"Downloading {repo_id} to {local_dir}")
        log(f"Authentication: {'Yes (token provided)' if token else 'No (public access)'}")

        # Login if token provided
        if token:
            log("Logging in to HuggingFace...")
            login(token=token, add_to_git_credential=False)

        for attempt in range(1, max_retries + 1):
            try:
                log(f"Download attempt {attempt}/{max_retries}...")

                local_dir_result = snapshot_download(
                    repo_id=repo_id,
                    local_dir=local_dir,
                    token=token,
                    resume_download=True,  # Resume partial downloads
                    max_workers=4,  # Parallel downloads
                    local_dir_use_symlinks=False,  # Real copies, not symlinks
                )

                log(f"Download succeeded! Files saved to: {local_dir_result}")
                return True

            except Exception as e:
                log(f"Attempt {attempt} failed: {type(e).__name__}: {e}")
                if attempt < max_retries:
                    wait_time = 5 * attempt  # Exponential backoff
                    log(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    log(f"All {max_retries} attempts failed")
                    return False

        return False

    except ImportError:
        log("ERROR: huggingface_hub not installed")
        return False
    except Exception as e:
        log(f"ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def download_with_git_lfs(
    repo_id: str,
    local_dir: str,
    token: Optional[str] = None,
) -> bool:
    """
    Fallback: Download using git with LFS support.

    Args:
        repo_id: HuggingFace repo ID
        local_dir: Local directory to clone to
        token: HF authentication token

    Returns:
        True if successful, False otherwise
    """
    import subprocess

    try:
        log(f"Attempting git-lfs clone of {repo_id}...")

        # Construct URL with auth if token provided
        if token:
            # Format: https://user:token@huggingface.co/repo/name
            repo_url = f"https://user:{token}@huggingface.co/{repo_id}"
        else:
            repo_url = f"https://huggingface.co/{repo_id}"

        # Check if git-lfs is installed
        result = subprocess.run(
            ["git", "lfs", "version"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            log("ERROR: git-lfs not installed")
            return False

        log(f"git-lfs version: {result.stdout.strip()}")

        # Clone with LFS
        log(f"Cloning repository (this may take 10-30 minutes for large models)...")
        result = subprocess.run(
            [
                "git", "clone",
                "--depth", "1",  # Shallow clone
                repo_url,
                local_dir,
            ],
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )

        if result.returncode != 0:
            log(f"ERROR: git clone failed: {result.stderr}")
            return False

        log("Git clone succeeded!")

        # Verify LFS files were pulled
        result = subprocess.run(
            ["git", "lfs", "ls-files"],
            cwd=local_dir,
            capture_output=True,
            text=True,
        )
        log(f"LFS files: {result.stdout}")

        return True

    except subprocess.TimeoutExpired:
        log("ERROR: git clone timed out after 1 hour")
        return False
    except Exception as e:
        log(f"ERROR: {type(e).__name__}: {e}")
        return False


def verify_download(local_dir: str) -> bool:
    """
    Verify the downloaded model has all required files.

    Args:
        local_dir: Directory containing model files

    Returns:
        True if complete, False otherwise
    """
    vlm_dir = Path(local_dir) / "pretrain" / "vlm"

    log(f"Verifying download in {vlm_dir}...")

    if not vlm_dir.exists():
        log("ERROR: VLM directory not found")
        return False

    # List all files
    files = list(vlm_dir.rglob("*"))
    file_names = [f.name for f in files if f.is_file()]

    log(f"Found {len(file_names)} files:")
    for name in sorted(file_names):
        file_path = next(f for f in files if f.name == name)
        size_mb = file_path.stat().st_size / 1e6
        log(f"  {name}: {size_mb:.1f} MB")

    # Check required files
    required_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    missing = [f for f in required_files if f not in file_names]

    if missing:
        log(f"ERROR: Missing required files: {missing}")
        return False

    # Check for model weights
    has_safetensors = any(f.endswith(".safetensors") for f in file_names)
    has_bin = any(f.endswith(".bin") for f in file_names)

    if not (has_safetensors or has_bin):
        log("ERROR: No model weight files found (*.safetensors or *.bin)")
        return False

    # Check total size
    total_size = sum(f.stat().st_size for f in files if f.is_file())
    total_size_gb = total_size / 1e9

    log(f"Total size: {total_size_gb:.2f} GB")

    if total_size_gb < 1.0:
        log("WARNING: Model size suspiciously small (< 1GB)")
        return False

    log("✓ Verification passed!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download PhysX-Anything models from HuggingFace"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token for authentication",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="Caoza/PhysX-Anything",
        help="HuggingFace repository ID",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory (will create pretrain/vlm subdirs)",
    )
    parser.add_argument(
        "--use-git-lfs",
        action="store_true",
        help="Use git-lfs instead of huggingface_hub (fallback)",
    )
    args = parser.parse_args()

    # Get token from env if not provided
    token = args.token
    if not token:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if not token:
        log("WARNING: No HuggingFace token provided!")
        log("If the repo is private/gated, download will fail.")
        log("Set HF_TOKEN env var or pass --token argument.")

    # Try download
    success = False

    if args.use_git_lfs:
        log("Using git-lfs download method...")
        success = download_with_git_lfs(args.repo_id, args.output_dir, token)
    else:
        log("Using huggingface_hub download method...")
        success = download_with_huggingface_hub(args.repo_id, args.output_dir, token)

        # Fallback to git-lfs if huggingface_hub fails
        if not success:
            log("Trying fallback: git-lfs...")
            success = download_with_git_lfs(args.repo_id, args.output_dir, token)

    if not success:
        log("ERROR: All download methods failed!")
        sys.exit(1)

    # Verify download
    if not verify_download(args.output_dir):
        log("ERROR: Download verification failed!")
        sys.exit(1)

    log("SUCCESS: Model download and verification complete!")
    sys.exit(0)


if __name__ == "__main__":
    main()
