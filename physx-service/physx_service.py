#!/usr/bin/env python3
"""
PhysX-Anything Service for Cloud Run.

This service wraps the PhysX-Anything pipeline to provide articulation detection
for 3D assets. It accepts either:
1. Image input (rendered view of an object) - original mode
2. GLB mesh input (from ZeroScene) - new mode for direct mesh processing

The service runs the 4-stage PhysX-Anything pipeline:
1. VLM analysis (Qwen2.5-VL) - detects articulated parts
2. Decoder - generates 3D geometry
3. Split - segments mesh into parts
4. SimReady - generates URDF with joint definitions

Environment Variables:
    PHYSX_ROOT: Path to PhysX-Anything installation (default: /opt/physx_anything)
    PHYSX_DEBUG: Enable verbose logging (default: 1)
    PORT: Service port (default: 8080)

Cloud Run Settings (REQUIRED):
    --memory 32Gi
    --cpu 8
    --gpu 1 --gpu-type nvidia-l4
    --timeout 900
    --concurrency 1
    --min-instances 1
"""

import base64
import io
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import traceback
import uuid
from pathlib import Path
from typing import Optional, Tuple

from flask import Flask, jsonify, request

app = Flask(__name__)

# =============================================================================
# Configuration
# =============================================================================

PHYSX_ROOT = Path(os.environ.get("PHYSX_ROOT", "/opt/physx_anything"))
PHYSX_ENTRY = PHYSX_ROOT / "run_physx_anything_pipeline.py"
TMP_ROOT = Path("/tmp/physx_anything")
VLM_CHECKPOINT_PATH = PHYSX_ROOT / "pretrain" / "vlm"

DEBUG_MODE = os.environ.get("PHYSX_DEBUG", "1") == "1"

# GPU lock for serialization (only one inference at a time)
_gpu_lock = threading.Semaphore(1)

# Warmup state
_models_ready = threading.Event()
_warmup_error: Optional[str] = None
_warmup_details: dict = {}


# =============================================================================
# Exceptions
# =============================================================================

class PhysxError(RuntimeError):
    """Raised when the PhysX-Anything pipeline fails."""


# =============================================================================
# Logging
# =============================================================================

def log(msg: str, level: str = "INFO") -> None:
    """Log with prefix for easy filtering."""
    print(f"[PHYSX-SERVICE] [{level}] {msg}", flush=True)


# =============================================================================
# Model Validation
# =============================================================================

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

    files = list(VLM_CHECKPOINT_PATH.rglob("*"))
    file_names = [f.name for f in files if f.is_file()]
    details["files"] = file_names
    details["file_count"] = len(file_names)

    # Check for essential files
    required_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]

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

    # Check total size
    total_size = sum(f.stat().st_size for f in files if f.is_file())
    details["total_size_gb"] = round(total_size / 1e9, 2)

    if total_size < 1e9:
        return False, f"Model files too small ({details['total_size_gb']} GB)", details

    return True, "Model files validated successfully", details


# =============================================================================
# Model Warmup
# =============================================================================

def warmup_models() -> None:
    """
    Pre-load ML models at startup.

    This actually loads the VLM model to catch errors early and reduce
    first-request latency.
    """
    global _warmup_error, _warmup_details

    log("Starting model warmup...")

    try:
        # Step 1: Validate model files
        log("Step 1/3: Validating model files...")
        is_valid, msg, details = validate_vlm_model()
        _warmup_details["model_validation"] = details

        if not is_valid:
            _warmup_error = f"Model validation failed: {msg}"
            log(_warmup_error, "ERROR")
            return

        log(f"Model validation passed: {details['file_count']} files, {details['total_size_gb']} GB")

        # Step 2: Test PyTorch and CUDA
        log("Step 2/3: Testing PyTorch and CUDA...")
        cuda_test = """
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    import sys
    sys.exit(1)
"""
        result = subprocess.run(
            [sys.executable, "-c", cuda_test],
            capture_output=True, text=True, timeout=60
        )

        if result.returncode != 0:
            _warmup_error = f"CUDA test failed: {result.stderr}"
            log(_warmup_error, "ERROR")
            return

        log(f"CUDA test passed:\n{result.stdout}")
        _warmup_details["cuda_test"] = {"success": True}

        # Step 3: Load VLM model
        log("Step 3/3: Loading VLM model (this takes several minutes)...")
        vlm_load = f"""
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

vlm_path = "{VLM_CHECKPOINT_PATH}"
print(f"Loading from: {{vlm_path}}")

processor = AutoProcessor.from_pretrained(vlm_path, trust_remote_code=True)
print(f"Processor: {{type(processor).__name__}}")

model = AutoModelForCausalLM.from_pretrained(
    vlm_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
print(f"Model: {{type(model).__name__}}")
print("VLM ready!")
"""
        result = subprocess.run(
            [sys.executable, "-c", vlm_load],
            capture_output=True, text=True, timeout=900, cwd=str(PHYSX_ROOT)
        )

        log(f"VLM load stdout:\n{result.stdout}")
        if result.stderr:
            log(f"VLM load stderr:\n{result.stderr}", "WARNING")

        if result.returncode != 0:
            _warmup_error = f"VLM load failed (code {result.returncode})"
            _warmup_details["vlm_load"] = {
                "success": False,
                "stdout": result.stdout[-2000:],
                "stderr": result.stderr[-2000:],
            }
            log(_warmup_error, "ERROR")
            return

        _warmup_details["vlm_load"] = {"success": True}
        log("Model warmup completed!")

    except subprocess.TimeoutExpired:
        _warmup_error = "Warmup timed out"
        log(_warmup_error, "ERROR")
    except Exception as e:
        _warmup_error = f"Warmup exception: {e}"
        log(_warmup_error, "ERROR")
        log(traceback.format_exc(), "ERROR")
    finally:
        _models_ready.set()


# =============================================================================
# Mesh Rendering (GLB → Image)
# =============================================================================

def render_glb_to_image(glb_bytes: bytes, request_id: str) -> Optional[bytes]:
    """
    Render a GLB mesh to a single view image for VLM analysis.

    Args:
        glb_bytes: Raw GLB file bytes
        request_id: Request ID for logging

    Returns:
        PNG image bytes or None on failure
    """
    try:
        from mesh_to_views import render_mesh_views

        log(f"[{request_id}] Rendering GLB to image ({len(glb_bytes)} bytes)")

        images = render_mesh_views(
            glb_bytes,
            num_views=1,
            resolution=(512, 512),
            elevation_degrees=30.0,
            use_pyrender=False,  # Use trimesh for reliability
        )

        if images:
            log(f"[{request_id}] Rendered {len(images)} view(s)")
            return images[0]
        else:
            log(f"[{request_id}] Rendering failed, no images produced", "WARNING")
            return None

    except ImportError:
        log(f"[{request_id}] mesh_to_views not available", "WARNING")
        return None
    except Exception as e:
        log(f"[{request_id}] Rendering error: {e}", "ERROR")
        return None


# =============================================================================
# PhysX-Anything Pipeline
# =============================================================================

def run_physx_anything(image_bytes: bytes, request_id: str, glb_bytes: Optional[bytes] = None) -> Tuple[bytes, bytes]:
    """
    Run the PhysX-Anything pipeline.

    Args:
        image_bytes: Input image (PNG/JPEG)
        request_id: Unique request ID
        glb_bytes: Optional input GLB mesh (for mesh-conditioned generation)

    Returns:
        (mesh_glb_bytes, urdf_bytes)
    """
    if not PHYSX_ENTRY.is_file():
        raise PhysxError(f"Pipeline entry script not found: {PHYSX_ENTRY}")

    req_dir = TMP_ROOT / f"req_{request_id}"
    in_dir = req_dir / "input"
    out_dir = req_dir / "output"

    try:
        # Clean up
        if req_dir.exists():
            shutil.rmtree(req_dir, ignore_errors=True)

        in_dir.mkdir(parents=True)
        out_dir.mkdir(parents=True)

        # Save input image
        input_path = in_dir / "input.png"
        input_path.write_bytes(image_bytes)
        log(f"[{request_id}] Input image: {len(image_bytes)} bytes")

        # Optionally save input GLB
        if glb_bytes:
            glb_path = in_dir / "input.glb"
            glb_path.write_bytes(glb_bytes)
            log(f"[{request_id}] Input GLB: {len(glb_bytes)} bytes")

        # Build command
        cmd = [
            sys.executable,
            str(PHYSX_ENTRY),
            "--input_image", str(input_path),
            "--output_dir", str(out_dir),
            "--request_id", request_id,
        ]

        log(f"[{request_id}] Running: {' '.join(cmd)}")

        # Run pipeline
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=900,
            cwd=str(PHYSX_ROOT),
        )

        if result.stdout:
            log(f"[{request_id}] stdout:\n{result.stdout[-3000:]}")
        if result.stderr:
            log(f"[{request_id}] stderr:\n{result.stderr[-2000:]}", "WARNING")

        if result.returncode != 0:
            raise PhysxError(f"Pipeline failed (code {result.returncode})")

        # Find outputs
        mesh_path = out_dir / "part.glb"
        urdf_path = out_dir / "part.urdf"

        if not mesh_path.is_file():
            files = [str(f.name) for f in out_dir.rglob("*") if f.is_file()]
            raise PhysxError(f"Mesh not found. Files: {files}")

        if not urdf_path.is_file():
            files = [str(f.name) for f in out_dir.rglob("*") if f.is_file()]
            raise PhysxError(f"URDF not found. Files: {files}")

        mesh_bytes = mesh_path.read_bytes()
        urdf_bytes = urdf_path.read_bytes()

        log(f"[{request_id}] Output: mesh={len(mesh_bytes)} bytes, urdf={len(urdf_bytes)} bytes")

        return mesh_bytes, urdf_bytes

    except subprocess.TimeoutExpired:
        raise PhysxError("Pipeline timed out (900s)")
    finally:
        if req_dir.exists():
            shutil.rmtree(req_dir, ignore_errors=True)


def encode_response(mesh_bytes: bytes, urdf_bytes: bytes) -> dict:
    """Encode pipeline output to JSON response."""
    return {
        "mesh_base64": base64.b64encode(mesh_bytes).decode("ascii"),
        "urdf_base64": base64.b64encode(urdf_bytes).decode("ascii"),
        "placeholder": False,
        "generator": "physx-anything",
    }


# =============================================================================
# HTTP Endpoints
# =============================================================================

@app.route("/", methods=["GET"])
def healthcheck():
    """Health check endpoint."""
    if not _models_ready.is_set():
        return jsonify({
            "status": "warming_up",
            "message": "Models are loading...",
            "ready": False,
        }), 503

    if _warmup_error:
        return jsonify({
            "status": "error",
            "message": f"Warmup failed: {_warmup_error}",
            "ready": False,
            "details": _warmup_details if DEBUG_MODE else None,
        }), 503

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
    return f"not ready: {_warmup_error or 'loading'}", 503


@app.route("/debug", methods=["GET"])
def debug_info():
    """Debug endpoint with service state."""
    is_valid, msg, details = validate_vlm_model()

    return jsonify({
        "models_ready": _models_ready.is_set(),
        "warmup_error": _warmup_error,
        "warmup_details": _warmup_details,
        "model_validation": {
            "valid": is_valid,
            "message": msg,
            "details": details,
        },
        "paths": {
            "physx_root": str(PHYSX_ROOT),
            "physx_entry": str(PHYSX_ENTRY),
            "vlm_checkpoint": str(VLM_CHECKPOINT_PATH),
        },
    }), 200


@app.route("/", methods=["POST"])
def handle_request():
    """
    Main processing endpoint.

    Request JSON:
        image_base64: Base64 PNG/JPEG image (required unless glb_base64 provided)
        glb_base64: Base64 GLB mesh (optional, new for ZeroScene integration)

    Response JSON:
        mesh_base64: Base64 GLB mesh with articulation
        urdf_base64: Base64 URDF with joint definitions
        placeholder: False if successful
        generator: "physx-anything"
    """
    request_id = str(uuid.uuid4())[:8]
    log(f"[{request_id}] Request received")

    # Check service state
    if not _models_ready.is_set():
        return jsonify({"error": "Service warming up"}), 503

    if _warmup_error:
        return jsonify({"error": f"Warmup failed: {_warmup_error}"}), 503

    # Parse request
    data = request.get_json(force=True, silent=True) or {}

    img_b64 = data.get("image_base64")
    glb_b64 = data.get("glb_base64")

    # Validate input
    image_bytes: Optional[bytes] = None
    glb_bytes: Optional[bytes] = None

    # Decode GLB if provided
    if glb_b64:
        try:
            glb_bytes = base64.b64decode(glb_b64)
            log(f"[{request_id}] GLB input: {len(glb_bytes)} bytes")
        except Exception:
            return jsonify({"error": "Invalid glb_base64"}), 400

    # Decode or generate image
    if img_b64:
        try:
            image_bytes = base64.b64decode(img_b64)
            log(f"[{request_id}] Image input: {len(image_bytes)} bytes")
        except Exception:
            return jsonify({"error": "Invalid image_base64"}), 400
    elif glb_bytes:
        # Render image from GLB
        log(f"[{request_id}] No image provided, rendering from GLB...")
        image_bytes = render_glb_to_image(glb_bytes, request_id)
        if not image_bytes:
            return jsonify({"error": "Failed to render image from GLB"}), 400
    else:
        return jsonify({"error": "Either image_base64 or glb_base64 is required"}), 400

    # Acquire GPU lock
    log(f"[{request_id}] Waiting for GPU lock...")
    acquired = _gpu_lock.acquire(timeout=120)
    if not acquired:
        return jsonify({"error": "Service busy, retry later"}), 503

    try:
        log(f"[{request_id}] GPU lock acquired, running pipeline")
        mesh_bytes, urdf_bytes = run_physx_anything(image_bytes, request_id, glb_bytes)
        log(f"[{request_id}] Pipeline completed")
    except PhysxError as e:
        log(f"[{request_id}] Pipeline error: {e}", "ERROR")
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        log(f"[{request_id}] Unexpected error: {e}", "ERROR")
        log(traceback.format_exc(), "ERROR")
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500
    finally:
        _gpu_lock.release()
        log(f"[{request_id}] GPU lock released")

    return jsonify(encode_response(mesh_bytes, urdf_bytes)), 200


# =============================================================================
# Startup
# =============================================================================

def start_warmup():
    """Start warmup in background thread."""
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    log(f"PHYSX_ROOT={PHYSX_ROOT}")
    log(f"VLM_CHECKPOINT={VLM_CHECKPOINT_PATH}")
    log(f"DEBUG_MODE={DEBUG_MODE}")

    warmup_thread = threading.Thread(target=warmup_models, daemon=True)
    warmup_thread.start()


# Initialize on import
start_warmup()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, threaded=True)
