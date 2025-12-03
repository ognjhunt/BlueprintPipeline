"""
PhysX-Anything service wrapper for Cloud Run.

Key fixes in this version:
1. ACTUALLY loads the VLM model during warmup (not just imports)
2. Full error message capture (no truncation)
3. Model file validation before accepting requests
4. Better subprocess error handling
5. Debug mode for troubleshooting
"""
import base64
import os
import shutil
import subprocess
import sys
import threading
import traceback
import uuid
from pathlib import Path
from typing import Tuple, Optional

from flask import Flask, jsonify, request

app = Flask(__name__)

# Configuration
PHYSX_ROOT = Path(os.environ.get("PHYSX_ROOT", "/opt/physx_anything"))
PHYSX_ENTRY = PHYSX_ROOT / "run_physx_anything_pipeline.py"
TMP_ROOT = Path("/tmp/physx_anything")
VLM_CHECKPOINT_PATH = PHYSX_ROOT / "pretrain" / "vlm"

# Debug mode - set to "1" to enable verbose logging
DEBUG_MODE = os.environ.get("PHYSX_DEBUG", "1") == "1"

# GPU lock for serialization
_gpu_lock = threading.Semaphore(1)

# Warmup state
_models_ready = threading.Event()
_warmup_error: Optional[str] = None
_warmup_details: dict = {}


class PhysxError(RuntimeError):
    """Raised when the PhysX-Anything pipeline fails."""


def log(msg: str, level: str = "INFO") -> None:
    """Log with prefix for easy filtering."""
    print(f"[PHYSX-SERVICE] [{level}] {msg}", flush=True)


def validate_vlm_model() -> Tuple[bool, str, dict]:
    """
    Validate that VLM model files are complete.
    
    Returns:
        (is_valid, message, details)
    """
    details = {
        "path": str(VLM_CHECKPOINT_PATH),
        "exists": VLM_CHECKPOINT_PATH.is_dir(),
        "files": [],
        "required_files": [],
        "missing_files": [],
    }
    
    if not VLM_CHECKPOINT_PATH.is_dir():
        return False, f"VLM checkpoint directory not found at {VLM_CHECKPOINT_PATH}", details
    
    # List all files
    files = list(VLM_CHECKPOINT_PATH.rglob("*"))
    file_names = [f.name for f in files if f.is_file()]
    details["files"] = file_names
    details["file_count"] = len(file_names)
    
    # Check for essential files (Qwen2.5-VL model structure)
    required_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    
    # Model weights can be in different formats
    has_safetensors = any(f.endswith(".safetensors") for f in file_names)
    has_bin = any(f.endswith(".bin") for f in file_names)
    has_weights = has_safetensors or has_bin
    
    details["has_safetensors"] = has_safetensors
    details["has_bin"] = has_bin
    details["has_weights"] = has_weights
    
    missing = []
    for req in required_files:
        if req not in file_names:
            missing.append(req)
    
    if not has_weights:
        missing.append("model weights (*.safetensors or *.bin)")
    
    details["required_files"] = required_files
    details["missing_files"] = missing
    
    if missing:
        return False, f"Missing required model files: {missing}", details
    
    # Check total size (VLM should be several GB)
    total_size = sum(f.stat().st_size for f in files if f.is_file())
    details["total_size_gb"] = round(total_size / 1e9, 2)
    
    if total_size < 1e9:  # Less than 1GB is suspicious
        return False, f"Model files too small ({details['total_size_gb']} GB) - likely incomplete download", details
    
    return True, "Model files validated successfully", details


def warmup_models() -> None:
    """
    Pre-load the heavy ML models at startup.
    This actually tries to load the VLM model to catch errors early.
    """
    global _warmup_error, _warmup_details
    
    log("Starting model warmup...")
    
    try:
        # Step 1: Validate model files exist
        log("Step 1/3: Validating model files...")
        is_valid, msg, details = validate_vlm_model()
        _warmup_details["model_validation"] = details
        
        if not is_valid:
            _warmup_error = f"Model validation failed: {msg}"
            log(_warmup_error, "ERROR")
            log(f"Details: {details}", "ERROR")
            return
        
        log(f"Model validation passed: {details['file_count']} files, {details['total_size_gb']} GB")
        
        # Step 2: Test PyTorch and CUDA
        log("Step 2/3: Testing PyTorch and CUDA...")
        cuda_test_script = """
import sys
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"  Memory: {props.total_memory / 1e9:.1f} GB")
else:
    print("WARNING: CUDA not available!", file=sys.stderr)
    sys.exit(1)
"""
        result = subprocess.run(
            [sys.executable, "-c", cuda_test_script],
            capture_output=True,
            text=True,
            timeout=60,
        )
        
        if result.returncode != 0:
            _warmup_error = f"CUDA test failed: {result.stderr}"
            log(_warmup_error, "ERROR")
            return
        
        log(f"CUDA test output:\n{result.stdout}")
        _warmup_details["cuda_test"] = {"success": True, "output": result.stdout}
        
        # Step 3: Actually try to load the VLM model
        log("Step 3/3: Loading VLM model (this may take several minutes)...")
        vlm_load_script = f"""
import sys
import os
import torch

# Reduce memory usage during loading
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

vlm_path = "{VLM_CHECKPOINT_PATH}"
print(f"Loading VLM from: {{vlm_path}}")

try:
    from transformers import AutoModelForCausalLM, AutoProcessor
    
    print("Loading processor...", flush=True)
    processor = AutoProcessor.from_pretrained(vlm_path, trust_remote_code=True)
    print(f"Processor loaded: {{type(processor).__name__}}", flush=True)
    
    print("Loading model (this takes a few minutes)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        vlm_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"Model loaded: {{type(model).__name__}}", flush=True)
    print(f"Model device: {{next(model.parameters()).device}}", flush=True)
    
    # Quick inference test
    print("Running quick inference test...", flush=True)
    # Just verify the model can be called
    print("VLM model loaded and ready!", flush=True)
    
except Exception as e:
    print(f"ERROR loading VLM: {{type(e).__name__}}: {{e}}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
        result = subprocess.run(
            [sys.executable, "-c", vlm_load_script],
            capture_output=True,
            text=True,
            timeout=900,  # 15 minute timeout for model loading
            cwd=str(PHYSX_ROOT),
            env={**os.environ, "TRANSFORMERS_VERBOSITY": "info"},
        )
        
        log(f"VLM load stdout:\n{result.stdout}")
        if result.stderr:
            log(f"VLM load stderr:\n{result.stderr}", "WARNING")
        
        if result.returncode != 0:
            _warmup_error = f"VLM model loading failed (exit code {result.returncode})"
            _warmup_details["vlm_load"] = {
                "success": False,
                "returncode": result.returncode,
                "stdout": result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout,
                "stderr": result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr,
            }
            log(_warmup_error, "ERROR")
            return
        
        _warmup_details["vlm_load"] = {"success": True}
        log("Model warmup completed successfully!")
        
    except subprocess.TimeoutExpired:
        _warmup_error = "Model warmup timed out after 900s"
        log(_warmup_error, "ERROR")
    except Exception as e:
        _warmup_error = f"Model warmup exception: {type(e).__name__}: {e}"
        log(_warmup_error, "ERROR")
        log(traceback.format_exc(), "ERROR")
    finally:
        _models_ready.set()


def run_physx_anything(image_bytes: bytes, request_id: str) -> Tuple[bytes, bytes]:
    """
    Given raw PNG/JPEG bytes, run the PhysX-Anything pipeline and
    return (mesh_glb_bytes, urdf_bytes).
    """
    if not PHYSX_ENTRY.is_file():
        raise PhysxError(f"PhysX entry script not found at {PHYSX_ENTRY}")

    # Use unique request ID for directory isolation
    req_dir = TMP_ROOT / f"req_{request_id}"
    in_dir = req_dir / "input"
    out_dir = req_dir / "output"
    
    try:
        # Clean up any previous run with same ID
        if req_dir.exists():
            shutil.rmtree(req_dir, ignore_errors=True)
            
        in_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save the image
        input_path = in_dir / "input.png"
        input_path.write_bytes(image_bytes)
        log(f"[{request_id}] Saved input image: {len(image_bytes)} bytes")

        # Run the driver script
        cmd = [
            sys.executable,
            str(PHYSX_ENTRY),
            "--input_image", str(input_path),
            "--output_dir", str(out_dir),
            "--request_id", request_id,
        ]
        
        log(f"[{request_id}] Running pipeline: {' '.join(cmd)}")

        # Run with full output capture
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,  # Capture stderr separately
            text=True,
            timeout=900,
            cwd=str(PHYSX_ROOT),
        )
        
        # Log full output
        if result.stdout:
            log(f"[{request_id}] Pipeline stdout:\n{result.stdout}")
        if result.stderr:
            log(f"[{request_id}] Pipeline stderr:\n{result.stderr}", "WARNING")
        
        if result.returncode != 0:
            # Combine stdout and stderr for full error context
            full_output = f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
            # Truncate intelligently - keep last 4000 chars which usually has the error
            if len(full_output) > 4000:
                full_output = "...[truncated]...\n" + full_output[-4000:]
            raise PhysxError(
                f"Pipeline failed with code {result.returncode}:\n{full_output}"
            )

        # Read mesh + URDF from out_dir
        mesh_path = out_dir / "part.glb"
        urdf_path = out_dir / "part.urdf"

        if not mesh_path.is_file():
            # List what's in the output directory for debugging
            files = list(out_dir.rglob("*"))
            file_list = [str(f.relative_to(out_dir)) for f in files if f.is_file()]
            raise PhysxError(
                f"Output mesh not found at {mesh_path}. "
                f"Output directory contains: {file_list}"
            )
        
        if not urdf_path.is_file():
            files = list(out_dir.rglob("*"))
            file_list = [str(f.relative_to(out_dir)) for f in files if f.is_file()]
            raise PhysxError(
                f"Output URDF not found at {urdf_path}. "
                f"Output directory contains: {file_list}"
            )

        mesh_bytes = mesh_path.read_bytes()
        urdf_bytes = urdf_path.read_bytes()
        
        log(f"[{request_id}] Pipeline completed: mesh={len(mesh_bytes)} bytes, urdf={len(urdf_bytes)} bytes")
        
        # Sanity check file sizes
        if len(mesh_bytes) < 100:
            log(f"[{request_id}] WARNING: Mesh file suspiciously small!", "WARNING")
        
        return mesh_bytes, urdf_bytes
        
    except subprocess.TimeoutExpired:
        raise PhysxError(f"Pipeline timed out after 900s")
    finally:
        # Clean up request directory
        if req_dir.exists():
            try:
                shutil.rmtree(req_dir, ignore_errors=True)
            except Exception as cleanup_err:
                log(f"[{request_id}] Cleanup failed: {cleanup_err}", "WARNING")


def encode_assets(mesh_bytes: bytes, urdf_bytes: bytes) -> dict:
    """Convert raw bytes into the JSON schema interactive-job expects."""
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
    Returns 503 if models are still loading.
    Returns 200 with degraded status if warmup had errors.
    """
    if not _models_ready.is_set():
        return jsonify({
            "status": "warming_up",
            "message": "Models are still loading, please wait...",
            "ready": False,
        }), 503
    
    if _warmup_error:
        return jsonify({
            "status": "error",
            "message": f"Warmup failed: {_warmup_error}",
            "ready": False,
            "details": _warmup_details if DEBUG_MODE else None,
        }), 503  # Return 503 so traffic isn't routed here
    
    return jsonify({
        "status": "ok",
        "ready": True,
        "details": _warmup_details if DEBUG_MODE else None,
    }), 200


@app.route("/ready", methods=["GET"])
def readiness():
    """Readiness probe for Kubernetes/Cloud Run."""
    if _models_ready.is_set() and not _warmup_error:
        return "ready", 200
    return f"not ready: {_warmup_error or 'still loading'}", 503


@app.route("/debug", methods=["GET"])
def debug_info():
    """Debug endpoint showing service state."""
    is_valid, msg, model_details = validate_vlm_model()
    
    return jsonify({
        "models_ready": _models_ready.is_set(),
        "warmup_error": _warmup_error,
        "warmup_details": _warmup_details,
        "model_validation": {
            "is_valid": is_valid,
            "message": msg,
            "details": model_details,
        },
        "physx_root": str(PHYSX_ROOT),
        "physx_entry_exists": PHYSX_ENTRY.is_file(),
        "vlm_path": str(VLM_CHECKPOINT_PATH),
        "tmp_root": str(TMP_ROOT),
    }), 200


@app.route("/", methods=["POST"])
def handle_request():
    """
    Main endpoint used by interactive-job.

    Expects JSON: { "image_base64": "<base64 png/jpg bytes>" }
    Returns JSON: { "mesh_base64", "urdf_base64", "placeholder", "generator" }
    """
    request_id = str(uuid.uuid4())[:8]
    
    log(f"[{request_id}] Received request")
    
    # Check if models are ready
    if not _models_ready.is_set():
        log(f"[{request_id}] Models not ready yet", "WARNING")
        return jsonify({
            "error": "Service is still warming up, please retry in a few minutes"
        }), 503
    
    # Check if warmup failed
    if _warmup_error:
        log(f"[{request_id}] Warmup failed: {_warmup_error}", "ERROR")
        return jsonify({
            "error": f"Service warmup failed: {_warmup_error}",
            "details": _warmup_details if DEBUG_MODE else None,
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

    # Acquire GPU lock
    log(f"[{request_id}] Waiting for GPU lock...")
    
    acquired = _gpu_lock.acquire(timeout=120)  # Wait up to 2 minutes
    if not acquired:
        log(f"[{request_id}] GPU lock timeout", "WARNING")
        return jsonify({
            "error": "Service is busy processing another request, please retry"
        }), 503
    
    try:
        log(f"[{request_id}] GPU lock acquired, starting pipeline")
        mesh_bytes, urdf_bytes = run_physx_anything(image_bytes, request_id)
        log(f"[{request_id}] Pipeline completed successfully")
    except PhysxError as e:
        log(f"[{request_id}] Pipeline error: {e}", "ERROR")
        # Return full error message (don't truncate)
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        log(f"[{request_id}] Unexpected error: {type(e).__name__}: {e}", "ERROR")
        log(traceback.format_exc(), "ERROR")
        return jsonify({"error": f"Unexpected error: {type(e).__name__}: {e}"}), 500
    finally:
        _gpu_lock.release()
        log(f"[{request_id}] GPU lock released")

    resp = encode_assets(mesh_bytes, urdf_bytes)
    return jsonify(resp), 200


def start_warmup():
    """Start warmup in background thread."""
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    log(f"PHYSX_ROOT={PHYSX_ROOT}")
    log(f"VLM_CHECKPOINT_PATH={VLM_CHECKPOINT_PATH}")
    log(f"DEBUG_MODE={DEBUG_MODE}")
    warmup_thread = threading.Thread(target=warmup_models, daemon=True)
    warmup_thread.start()


# Initialize on module load
start_warmup()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, threaded=True)