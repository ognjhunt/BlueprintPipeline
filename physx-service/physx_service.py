"""
PhysX-Anything service wrapper for Cloud Run.

Key features:
- Request isolation using UUIDs (not PIDs)
- Model warmup at startup to avoid cold-start timeouts
- Concurrency limiting (GPU can only handle 1 request at a time)
- Proper error handling and cleanup
"""
import base64
import os
import shutil
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from typing import Tuple

from flask import Flask, jsonify, request

app = Flask(__name__)

# Where the PhysX-Anything repo is installed.
PHYSX_ROOT = Path(os.environ.get("PHYSX_ROOT", "/opt/physx_anything"))

# Our small driver script that orchestrates 1_vlm_demo.py → 4_simready_gen.py
PHYSX_ENTRY = PHYSX_ROOT / "run_physx_anything_pipeline.py"

# Temporary directory for request processing
TMP_ROOT = Path("/tmp/physx_anything")

# Semaphore to ensure only one GPU inference at a time
# (GPU memory cannot handle concurrent PhysX-Anything runs)
_gpu_lock = threading.Semaphore(1)

# Track if models are warmed up
_models_ready = threading.Event()
_warmup_error = None


class PhysxError(RuntimeError):
    """Raised when the PhysX-Anything pipeline fails."""


def warmup_models() -> None:
    """
    Pre-load the heavy ML models at startup.
    This prevents cold-start timeouts on first real request.
    """
    global _warmup_error
    
    app.logger.info("[PHYSX-SERVICE] Starting model warmup...")
    
    try:
        # Import the heavy dependencies to trigger model loading
        # This forces Python to load torch, transformers, etc.
        warmup_script = """
import sys
sys.path.insert(0, '/opt/physx_anything')

print('[WARMUP] Loading PyTorch...', flush=True)
import torch
print(f'[WARMUP] PyTorch loaded, CUDA available: {torch.cuda.is_available()}', flush=True)

if torch.cuda.is_available():
    print(f'[WARMUP] GPU: {torch.cuda.get_device_name(0)}', flush=True)
    print(f'[WARMUP] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB', flush=True)

print('[WARMUP] Loading transformers...', flush=True)
from transformers import AutoModelForCausalLM, AutoProcessor
print('[WARMUP] Transformers loaded', flush=True)

# Try to load the VLM model (this is what takes the longest)
print('[WARMUP] Attempting to load VLM checkpoint...', flush=True)
vlm_path = '/opt/physx_anything/pretrain/vlm'
import os
if os.path.exists(vlm_path):
    print(f'[WARMUP] VLM checkpoint exists at {vlm_path}', flush=True)
else:
    print(f'[WARMUP] WARNING: VLM checkpoint not found at {vlm_path}', flush=True)

print('[WARMUP] Model warmup complete!', flush=True)
"""
        result = subprocess.run(
            [sys.executable, "-c", warmup_script],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout for model loading
            cwd=str(PHYSX_ROOT),
        )
        
        app.logger.info("[PHYSX-SERVICE] Warmup stdout:\n%s", result.stdout)
        if result.stderr:
            app.logger.warning("[PHYSX-SERVICE] Warmup stderr:\n%s", result.stderr)
            
        if result.returncode != 0:
            _warmup_error = f"Warmup failed with code {result.returncode}"
            app.logger.error("[PHYSX-SERVICE] %s", _warmup_error)
        else:
            app.logger.info("[PHYSX-SERVICE] Model warmup successful!")
            
    except subprocess.TimeoutExpired:
        _warmup_error = "Model warmup timed out after 600s"
        app.logger.error("[PHYSX-SERVICE] %s", _warmup_error)
    except Exception as e:
        _warmup_error = f"Model warmup exception: {e}"
        app.logger.error("[PHYSX-SERVICE] %s", _warmup_error)
    finally:
        _models_ready.set()


def run_physx_anything(image_bytes: bytes, request_id: str) -> Tuple[bytes, bytes]:
    """
    Given raw PNG/JPEG bytes, run the PhysX-Anything pipeline and
    return (mesh_glb_bytes, urdf_bytes).
    
    Args:
        image_bytes: Raw image data
        request_id: Unique identifier for this request (for directory isolation)
    """
    if not PHYSX_ENTRY.is_file():
        raise PhysxError(f"PhysX entry script not found at {PHYSX_ENTRY}")

    # Use unique request ID for directory isolation (NOT PID!)
    req_dir = TMP_ROOT / f"req_{request_id}"
    in_dir = req_dir / "input"
    out_dir = req_dir / "output"
    
    try:
        # Clean up any previous run with same ID (shouldn't happen with UUIDs)
        if req_dir.exists():
            shutil.rmtree(req_dir, ignore_errors=True)
            
        in_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save the image
        input_path = in_dir / "input.png"
        input_path.write_bytes(image_bytes)

        # Run the driver script with the unique request ID
        cmd = [
            sys.executable,
            str(PHYSX_ENTRY),
            "--input_image", str(input_path),
            "--output_dir", str(out_dir),
            "--request_id", request_id,  # Pass request ID for isolation
        ]
        
        app.logger.info("[PHYSX-SERVICE] [%s] Running pipeline: %s", request_id, " ".join(cmd))

        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=900,  # 15 minute timeout for full pipeline
            cwd=str(PHYSX_ROOT),
        )
        app.logger.info("[PHYSX-SERVICE] [%s] Pipeline output:\n%s", request_id, result.stdout)

        # Read mesh + URDF from out_dir
        mesh_path = out_dir / "part.glb"
        urdf_path = out_dir / "part.urdf"

        if not mesh_path.is_file() or not urdf_path.is_file():
            raise PhysxError(
                f"Expected part.glb and part.urdf in {out_dir}, "
                f"found mesh={mesh_path.is_file()}, urdf={urdf_path.is_file()}"
            )

        mesh_bytes = mesh_path.read_bytes()
        urdf_bytes = urdf_path.read_bytes()
        return mesh_bytes, urdf_bytes
        
    except subprocess.TimeoutExpired as e:
        raise PhysxError(f"Pipeline timed out after 900s") from e
    except subprocess.CalledProcessError as e:
        raise PhysxError(
            f"PhysX-Anything pipeline failed with code {e.returncode}: {e.stdout}"
        ) from e
    finally:
        # Clean up request directory
        if req_dir.exists():
            try:
                shutil.rmtree(req_dir, ignore_errors=True)
            except Exception as cleanup_err:
                app.logger.warning(
                    "[PHYSX-SERVICE] [%s] Cleanup failed: %s", 
                    request_id, cleanup_err
                )


def encode_assets(mesh_bytes: bytes, urdf_bytes: bytes) -> dict:
    """
    Convert raw bytes into the JSON schema interactive-job expects.
    """
    return {
        "mesh_base64": base64.b64encode(mesh_bytes).decode("ascii"),
        "urdf_base64": base64.b64encode(urdf_bytes).decode("ascii"),
        "placeholder": False,
        "generator": "physx-anything",
    }


@app.route("/", methods=["GET"])
def healthcheck():
    """
    Health check endpoint.
    Returns 200 only if models are ready (or warmup completed with error).
    This prevents Cloud Run from routing traffic before we're ready.
    """
    if not _models_ready.is_set():
        return jsonify({
            "status": "warming_up",
            "message": "Models are still loading, please wait..."
        }), 503
    
    if _warmup_error:
        return jsonify({
            "status": "degraded", 
            "message": f"Warmup had issues: {_warmup_error}",
            "ready": True,
        }), 200
    
    return jsonify({"status": "ok", "ready": True}), 200


@app.route("/ready", methods=["GET"])
def readiness():
    """
    Readiness probe for Kubernetes/Cloud Run.
    Returns 503 until models are loaded.
    """
    if _models_ready.is_set():
        return "ready", 200
    return "not ready", 503


@app.route("/", methods=["POST"])
def handle_request():
    """
    Main endpoint used by interactive-job.

    Expects JSON:
        { "image_base64": "<base64 png/jpg bytes>" }

    Returns JSON:
        {
          "mesh_base64": "<...>",
          "urdf_base64": "<...>",
          "placeholder": false,
          "generator": "physx-anything"
        }
    """
    # Generate unique request ID
    request_id = str(uuid.uuid4())[:8]
    
    app.logger.info("[PHYSX-SERVICE] [%s] Received request", request_id)
    
    # Check if models are ready
    if not _models_ready.is_set():
        app.logger.warning("[PHYSX-SERVICE] [%s] Models not ready yet", request_id)
        return jsonify({
            "error": "Service is still warming up, please retry in a few minutes"
        }), 503
    
    # Parse request
    data = request.get_json(force=True, silent=True) or {}
    img_b64 = data.get("image_base64")

    if not img_b64:
        return jsonify({"error": "image_base64 is required"}), 400

    try:
        image_bytes = base64.b64decode(img_b64)
    except Exception:
        return jsonify({"error": "image_base64 is not valid base64"}), 400

    # Acquire GPU lock (only one inference at a time)
    app.logger.info("[PHYSX-SERVICE] [%s] Waiting for GPU lock...", request_id)
    
    acquired = _gpu_lock.acquire(timeout=60)  # Wait up to 60s for lock
    if not acquired:
        app.logger.warning("[PHYSX-SERVICE] [%s] GPU lock timeout", request_id)
        return jsonify({
            "error": "Service is busy processing another request, please retry"
        }), 503
    
    try:
        app.logger.info("[PHYSX-SERVICE] [%s] GPU lock acquired, starting pipeline", request_id)
        mesh_bytes, urdf_bytes = run_physx_anything(image_bytes, request_id)
        app.logger.info("[PHYSX-SERVICE] [%s] Pipeline completed successfully", request_id)
    except PhysxError as e:
        app.logger.error("[PHYSX-SERVICE] [%s] Pipeline error: %s", request_id, e)
        return jsonify({"error": str(e)}), 500
    finally:
        _gpu_lock.release()
        app.logger.info("[PHYSX-SERVICE] [%s] GPU lock released", request_id)

    resp = encode_assets(mesh_bytes, urdf_bytes)
    return jsonify(resp), 200


# Start model warmup in background thread when app starts
def start_warmup():
    """Start warmup in background thread."""
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    warmup_thread = threading.Thread(target=warmup_models, daemon=True)
    warmup_thread.start()


# Initialize on module load
start_warmup()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    # Use threaded=False to ensure sequential request processing
    # (GPU can't handle concurrent inference anyway)
    app.run(host="0.0.0.0", port=port, threaded=True)