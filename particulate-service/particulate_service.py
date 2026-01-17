#!/usr/bin/env python3
"""
Particulate Service for Cloud Run.

This service wraps the Particulate model (Li et al., arXiv:2512.11798) to provide
mesh → articulation inference for 3D assets.

Particulate features:
- Takes a 3D mesh as input (GLB/OBJ)
- Outputs articulated structure (parts, joints, kinematic tree)
- Fast inference: ~10 seconds per object
- Excellent for internal parts detection

The service:
1. Accepts GLB mesh input (base64 encoded)
2. Runs Particulate inference (point cloud transformer)
3. Exports segmented mesh + URDF with joint definitions

Environment Variables:
    PARTICULATE_ROOT: Path to Particulate installation (default: /opt/particulate)
    PARTICULATE_DEBUG: Enable verbose logging (default: 1)
    PARTICULATE_DEBUG_TOKEN: Shared secret for /debug access (default: unset)
    PORT: Service port (default: 8080)

Cloud Run Settings (REQUIRED):
    --memory 16Gi
    --cpu 4
    --gpu 1 --gpu-type nvidia-l4
    --timeout 300
    --concurrency 1
    --min-instances 0
"""

import base64
import hmac
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import traceback
import uuid
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET
from xml.dom import minidom

from flask import Flask, jsonify, request

app = Flask(__name__)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

PARTICULATE_ROOT = Path(os.environ.get("PARTICULATE_ROOT", "/opt/particulate"))
PARTICULATE_INFER = PARTICULATE_ROOT / "infer.py"
TMP_ROOT = Path("/tmp/particulate")

DEBUG_MODE = os.environ.get("PARTICULATE_DEBUG", "1") == "1"
DEBUG_TOKEN = os.environ.get("PARTICULATE_DEBUG_TOKEN")

# GPU lock for serialization
_gpu_lock = threading.Semaphore(1)

# Warmup state
_models_ready = threading.Event()
_warmup_error: Optional[str] = None
_warmup_details: dict = {}


# =============================================================================
# Exceptions
# =============================================================================

class ParticulateError(RuntimeError):
    """Raised when the Particulate pipeline fails."""


# =============================================================================
# Logging
# =============================================================================

def log(msg: str, level: str = "INFO") -> None:
    """Log with prefix for easy filtering."""
    level_name = level.upper()
    log_level = getattr(logging, level_name, logging.INFO)
    logger.log(log_level, "[PARTICULATE-SERVICE] [%s] %s", level_name, msg)


# =============================================================================
# Dependency Probes
# =============================================================================

def _http_probe(url: str, timeout_s: float) -> Dict[str, Any]:
    result: Dict[str, Any] = {"ok": False, "url": url}
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout_s) as response:
            status_code = response.getcode()
            result["status_code"] = status_code
            result["ok"] = 200 <= status_code < 300
            if not result["ok"]:
                result["error"] = f"non_2xx_status:{status_code}"
    except urllib.error.HTTPError as exc:
        result["status_code"] = exc.code
        result["error"] = f"http_error:{exc.code}"
    except Exception as exc:
        result["error"] = f"request_failed:{exc}"
    return result


def _process_probe(pattern: str, timeout_s: float) -> Dict[str, Any]:
    result: Dict[str, Any] = {"ok": False, "pattern": pattern}
    try:
        completed = subprocess.run(
            ["pgrep", "-f", pattern],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        result["ok"] = completed.returncode == 0
        if not result["ok"]:
            result["error"] = "process_not_found"
    except FileNotFoundError:
        result["error"] = "pgrep_not_available"
    except Exception as exc:
        result["error"] = f"process_probe_failed:{exc}"
    return result


def _gpu_probe(timeout_s: float) -> Dict[str, Any]:
    result: Dict[str, Any] = {"ok": False}
    try:
        completed = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        result["stdout"] = completed.stdout.strip()
        result["stderr"] = completed.stderr.strip()
        result["ok"] = completed.returncode == 0
        if not result["ok"]:
            result["error"] = "nvidia_smi_failed"
    except FileNotFoundError:
        result["error"] = "nvidia_smi_not_found"
    except Exception as exc:
        result["error"] = f"gpu_probe_failed:{exc}"
    return result


def _llm_probe(timeout_s: float) -> Dict[str, Any]:
    result: Dict[str, Any] = {"ok": False}
    health_url = os.getenv("LLM_HEALTH_URL")
    if health_url:
        probe = _http_probe(health_url, timeout_s)
        result.update(probe)
        return result

    token_keys = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
    ]
    for key in token_keys:
        if os.getenv(key):
            result["ok"] = True
            result["status"] = "token_present"
            result["token_source"] = key
            return result

    result["error"] = "missing_credentials"
    return result


def _isaac_sim_probe(timeout_s: float) -> Dict[str, Any]:
    result: Dict[str, Any] = {"ok": False}
    health_url = os.getenv("ISAAC_SIM_HEALTH_URL")
    if health_url:
        probe = _http_probe(health_url, timeout_s)
        result.update(probe)
        return result

    process_pattern = os.getenv("ISAAC_SIM_PROCESS_PATTERN")
    if process_pattern:
        probe = _process_probe(process_pattern, timeout_s)
        result.update(probe)
        return result

    result["ok"] = True
    result["status"] = "skipped"
    result["reason"] = "not_configured"
    return result


def _dependency_health() -> Tuple[bool, Dict[str, Any]]:
    timeout_s = float(os.getenv("HEALTH_PROBE_TIMEOUT_S", "2.0"))
    gpu_required = os.getenv("GPU_HEALTH_REQUIRED", "true").lower() == "true"
    isaac_required = os.getenv("ISAAC_SIM_HEALTH_REQUIRED", "false").lower() == "true"
    llm_required = os.getenv("LLM_HEALTH_REQUIRED", "false").lower() == "true"

    dependencies: Dict[str, Any] = {
        "gpu": _gpu_probe(timeout_s),
        "isaac_sim": _isaac_sim_probe(timeout_s),
        "llm": _llm_probe(timeout_s),
    }
    errors: List[Dict[str, Any]] = []

    for name, required in [
        ("gpu", gpu_required),
        ("isaac_sim", isaac_required),
        ("llm", llm_required),
    ]:
        details = dependencies[name]
        details["required"] = required
        if required and not details.get("ok", False):
            errors.append({
                "dependency": name,
                "error": details.get("error", "dependency_unavailable"),
                "details": details,
            })

    return not errors, {"dependencies": dependencies, "errors": errors}


# =============================================================================
# Model Validation
# =============================================================================

def validate_particulate_model() -> Tuple[bool, str, dict]:
    """
    Validate that Particulate model files are complete.

    Returns:
        (is_valid, message, details)
    """
    details = {
        "path": str(PARTICULATE_ROOT),
        "exists": PARTICULATE_ROOT.is_dir(),
        "files": [],
    }

    if not PARTICULATE_ROOT.is_dir():
        return False, f"Particulate directory not found at {PARTICULATE_ROOT}", details

    if not PARTICULATE_INFER.is_file():
        return False, f"infer.py not found at {PARTICULATE_INFER}", details

    # Check for key directories
    required_dirs = ["particulate", "configs"]
    missing_dirs = []
    for d in required_dirs:
        if not (PARTICULATE_ROOT / d).is_dir():
            missing_dirs.append(d)

    if missing_dirs:
        return False, f"Missing directories: {missing_dirs}", details

    details["required_dirs"] = required_dirs
    details["missing_dirs"] = missing_dirs

    return True, "Particulate installation validated", details


# =============================================================================
# Model Warmup
# =============================================================================

def warmup_models() -> None:
    """
    Pre-load Particulate model at startup.

    This loads the model checkpoint to catch errors early and reduce
    first-request latency.
    """
    global _warmup_error, _warmup_details

    log("Starting model warmup...")

    try:
        # Step 1: Validate installation
        log("Step 1/3: Validating Particulate installation...")
        is_valid, msg, details = validate_particulate_model()
        _warmup_details["installation_validation"] = details

        if not is_valid:
            _warmup_error = f"Installation validation failed: {msg}"
            log(_warmup_error, "ERROR")
            return

        log(f"Installation validation passed")

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

        # Step 3: Load Particulate model (download from HuggingFace if needed)
        log("Step 3/3: Loading Particulate model...")
        model_load = f"""
import sys
sys.path.insert(0, "{PARTICULATE_ROOT}")

import torch
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf

# Download checkpoint if not cached
checkpoint = hf_hub_download(repo_id="rayli/Particulate", filename="model.pt")
print(f"Checkpoint: {{checkpoint}}")

# Load config and model
cfg_path = "{PARTICULATE_ROOT}/configs/pat_B.yaml"
cfg = OmegaConf.load(cfg_path)

from particulate.models import PAT_B
model = PAT_B(**cfg)
model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
model.to("cuda")
model.eval()

print("Model loaded successfully!")
print(f"Parameters: {{sum(p.numel() for p in model.parameters()):,}}")
"""
        result = subprocess.run(
            [sys.executable, "-c", model_load],
            capture_output=True, text=True, timeout=300, cwd=str(PARTICULATE_ROOT)
        )

        log(f"Model load stdout:\n{result.stdout}")
        if result.stderr:
            log(f"Model load stderr:\n{result.stderr}", "WARNING")

        if result.returncode != 0:
            _warmup_error = f"Model load failed (code {result.returncode})"
            _warmup_details["model_load"] = {
                "success": False,
                "stdout": result.stdout[-2000:],
                "stderr": result.stderr[-2000:],
            }
            log(_warmup_error, "ERROR")
            return

        _warmup_details["model_load"] = {"success": True}
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
# URDF Generation
# =============================================================================

def create_urdf_from_articulation(
    mesh_parts: Dict[int, bytes],
    part_ids: List[int],
    motion_hierarchy: Dict[int, int],
    joint_types: Dict[int, str],
    joint_axes: Dict[int, List[float]],
    joint_limits: Dict[int, Tuple[float, float]],
    output_dir: Path,
    robot_name: str = "robot",
) -> Path:
    """
    Generate URDF from Particulate articulation output.

    Args:
        mesh_parts: Dict mapping part_id -> OBJ mesh bytes
        part_ids: List of unique part IDs
        motion_hierarchy: Dict mapping child_id -> parent_id
        joint_types: Dict mapping part_id -> "revolute" | "prismatic" | "fixed"
        joint_axes: Dict mapping part_id -> [x, y, z] normalized axis
        joint_limits: Dict mapping part_id -> (lower, upper) in radians/meters
        output_dir: Directory to write URDF and mesh files
        robot_name: Name for the robot in URDF

    Returns:
        Path to generated URDF file
    """
    meshes_dir = output_dir / "meshes"
    meshes_dir.mkdir(parents=True, exist_ok=True)

    # Write mesh files
    for part_id, mesh_bytes in mesh_parts.items():
        mesh_path = meshes_dir / f"part_{part_id}.obj"
        mesh_path.write_bytes(mesh_bytes)

    # Build URDF
    robot = ET.Element("robot", name=robot_name)

    # Find root parts (no parent)
    child_ids = set(motion_hierarchy.keys())
    parent_ids = set(motion_hierarchy.values())
    root_ids = [pid for pid in part_ids if pid not in child_ids]

    if not root_ids:
        root_ids = [min(part_ids)] if part_ids else [0]

    # Create links
    for part_id in part_ids:
        link = ET.SubElement(robot, "link", name=f"link_{part_id}")

        # Visual
        visual = ET.SubElement(link, "visual")
        visual_geom = ET.SubElement(visual, "geometry")
        ET.SubElement(visual_geom, "mesh", filename=f"meshes/part_{part_id}.obj")

        # Collision
        collision = ET.SubElement(link, "collision")
        collision_geom = ET.SubElement(collision, "geometry")
        ET.SubElement(collision_geom, "mesh", filename=f"meshes/part_{part_id}.obj")

        # Inertial (placeholder values)
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "mass", value="1.0")
        ET.SubElement(inertial, "inertia",
                      ixx="0.1", ixy="0", ixz="0",
                      iyy="0.1", iyz="0", izz="0.1")

    # Create joints
    for child_id, parent_id in motion_hierarchy.items():
        if parent_id not in part_ids:
            continue

        joint_type = joint_types.get(child_id, "fixed")
        joint = ET.SubElement(robot, "joint",
                              name=f"joint_{parent_id}_{child_id}",
                              type=joint_type)

        ET.SubElement(joint, "parent", link=f"link_{parent_id}")
        ET.SubElement(joint, "child", link=f"link_{child_id}")
        ET.SubElement(joint, "origin", xyz="0 0 0", rpy="0 0 0")

        if joint_type != "fixed":
            axis = joint_axes.get(child_id, [0, 0, 1])
            # Normalize axis
            import math
            mag = math.sqrt(sum(a*a for a in axis)) + 1e-6
            axis = [a/mag for a in axis]
            ET.SubElement(joint, "axis", xyz=f"{axis[0]:.6f} {axis[1]:.6f} {axis[2]:.6f}")

            limits = joint_limits.get(child_id, (-3.14159, 3.14159))
            ET.SubElement(joint, "limit",
                          lower=str(limits[0]),
                          upper=str(limits[1]),
                          effort="1000",
                          velocity="100")

    # Format and write URDF
    urdf_str = ET.tostring(robot, encoding="unicode")
    urdf_pretty = minidom.parseString(urdf_str).toprettyxml(indent="  ")
    # Remove extra blank lines
    urdf_pretty = "\n".join(line for line in urdf_pretty.split("\n") if line.strip())

    urdf_path = output_dir / f"{robot_name}.urdf"
    urdf_path.write_text(urdf_pretty, encoding="utf-8")

    return urdf_path


def generate_static_urdf(output_dir: Path, mesh_path: Path, robot_name: str = "robot") -> Path:
    """Generate a static (non-articulated) URDF for fallback."""
    meshes_dir = output_dir / "meshes"
    meshes_dir.mkdir(parents=True, exist_ok=True)

    # Copy mesh to meshes directory
    dest_mesh = meshes_dir / mesh_path.name
    shutil.copy(mesh_path, dest_mesh)

    urdf_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<robot name="{robot_name}">
  <link name="base_link">
    <visual>
      <geometry>
        <mesh filename="meshes/{mesh_path.name}"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <mesh filename="meshes/{mesh_path.name}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>
</robot>
"""
    urdf_path = output_dir / f"{robot_name}.urdf"
    urdf_path.write_text(urdf_content, encoding="utf-8")
    return urdf_path


# =============================================================================
# Particulate Pipeline
# =============================================================================

def run_particulate(glb_bytes: bytes, request_id: str) -> Tuple[bytes, bytes, Dict[str, Any]]:
    """
    Run the Particulate inference pipeline.

    Args:
        glb_bytes: Input GLB mesh bytes
        request_id: Unique request ID

    Returns:
        (segmented_glb_bytes, urdf_bytes, articulation_metadata)
    """
    if not PARTICULATE_INFER.is_file():
        raise ParticulateError(f"infer.py not found: {PARTICULATE_INFER}")

    req_dir = TMP_ROOT / f"req_{request_id}"
    in_dir = req_dir / "input"
    out_dir = req_dir / "output"

    try:
        # Clean up
        if req_dir.exists():
            shutil.rmtree(req_dir, ignore_errors=True)

        in_dir.mkdir(parents=True)
        out_dir.mkdir(parents=True)

        # Save input mesh
        input_path = in_dir / "input.glb"
        input_path.write_bytes(glb_bytes)
        log(f"[{request_id}] Input mesh: {len(glb_bytes)} bytes")

        # Build inference command
        # Particulate infer.py --input_mesh <path> [options]
        cmd = [
            sys.executable,
            str(PARTICULATE_INFER),
            "--input_mesh", str(input_path),
            "--output_dir", str(out_dir),
            "--up_dir", "Y",  # Our meshes typically use Y-up
            "--export_urdf",  # Enable URDF export
            "--export_glb",   # Export segmented GLB
        ]

        log(f"[{request_id}] Running: {' '.join(cmd)}")

        # Run pipeline
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,  # 2 min timeout (should complete in ~10s)
            cwd=str(PARTICULATE_ROOT),
        )

        if result.stdout:
            log(f"[{request_id}] stdout:\n{result.stdout[-3000:]}")
        if result.stderr:
            log(f"[{request_id}] stderr:\n{result.stderr[-2000:]}", "WARNING")

        if result.returncode != 0:
            raise ParticulateError(f"Pipeline failed (code {result.returncode})")

        # Find outputs
        # Particulate outputs:
        # - segmented.glb (or input_segmented.glb)
        # - robot.urdf (or input.urdf)
        # - articulation.json (metadata)

        glb_candidates = list(out_dir.glob("*.glb"))
        urdf_candidates = list(out_dir.glob("*.urdf"))
        json_candidates = list(out_dir.glob("*.json"))

        if not glb_candidates:
            # Fall back to original mesh with static URDF
            log(f"[{request_id}] No segmented GLB found, using original", "WARNING")
            shutil.copy(input_path, out_dir / "segmented.glb")
            glb_candidates = [out_dir / "segmented.glb"]

            # Generate static URDF
            urdf_path = generate_static_urdf(out_dir, input_path, "robot")
            urdf_candidates = [urdf_path]

        mesh_path = glb_candidates[0]
        urdf_path = urdf_candidates[0] if urdf_candidates else None

        # Read outputs
        mesh_bytes = mesh_path.read_bytes()

        if urdf_path and urdf_path.is_file():
            urdf_bytes = urdf_path.read_bytes()
        else:
            # Generate static URDF if not produced
            urdf_path = generate_static_urdf(out_dir, mesh_path, "robot")
            urdf_bytes = urdf_path.read_bytes()

        # Read articulation metadata if available
        metadata = {}
        if json_candidates:
            try:
                metadata = json.loads(json_candidates[0].read_text())
            except Exception as e:
                log(f"[{request_id}] Failed to parse metadata: {e}", "WARNING")

        log(f"[{request_id}] Output: mesh={len(mesh_bytes)} bytes, urdf={len(urdf_bytes)} bytes")

        return mesh_bytes, urdf_bytes, metadata

    except subprocess.TimeoutExpired:
        raise ParticulateError("Pipeline timed out (120s)")
    finally:
        if req_dir.exists():
            shutil.rmtree(req_dir, ignore_errors=True)


def encode_response(mesh_bytes: bytes, urdf_bytes: bytes, metadata: Dict[str, Any]) -> dict:
    """Encode pipeline output to JSON response."""
    response = {
        "mesh_base64": base64.b64encode(mesh_bytes).decode("ascii"),
        "urdf_base64": base64.b64encode(urdf_bytes).decode("ascii"),
        "placeholder": False,
        "generator": "particulate",
    }

    # Add articulation summary from metadata
    if metadata:
        response["articulation"] = {
            "joint_count": metadata.get("joint_count", 0),
            "part_count": metadata.get("part_count", 0),
            "is_articulated": metadata.get("is_articulated", False),
        }

    return response


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

    deps_ok, dep_details = _dependency_health()
    if not deps_ok:
        return jsonify({
            "status": "error",
            "message": "Dependency check failed",
            "ready": False,
            "dependencies": dep_details.get("dependencies"),
            "errors": dep_details.get("errors"),
        }), 503

    return jsonify({
        "status": "ok",
        "ready": True,
        "service": "particulate",
        "details": _warmup_details if DEBUG_MODE else None,
        "dependencies": dep_details.get("dependencies"),
    }), 200


@app.route("/healthz", methods=["GET"])
def healthz():
    """Health check endpoint with dependency probes."""
    return healthcheck()


@app.route("/ready", methods=["GET"])
def readiness():
    """Readiness probe for Kubernetes/Cloud Run."""
    if _models_ready.is_set() and not _warmup_error:
        return "ready", 200
    return f"not ready: {_warmup_error or 'loading'}", 503


def _debug_authorized() -> bool:
    if not DEBUG_MODE or not DEBUG_TOKEN:
        return False

    auth_header = request.headers.get("Authorization", "")
    if not auth_header:
        return False

    token = auth_header
    if auth_header.lower().startswith("bearer "):
        token = auth_header[7:].strip()

    return hmac.compare_digest(token, DEBUG_TOKEN)


@app.route("/debug", methods=["GET"])
def debug_info():
    """Debug endpoint with service state."""
    if not _debug_authorized():
        return jsonify({"error": "debug access forbidden"}), 403

    is_valid, msg, details = validate_particulate_model()

    return jsonify({
        "models_ready": _models_ready.is_set(),
        "warmup_error": _warmup_error,
        "warmup_details": _warmup_details,
        "installation_validation": {
            "valid": is_valid,
            "message": msg,
            "details": details,
        },
        "paths": {
            "particulate_root": str(PARTICULATE_ROOT),
            "particulate_infer": str(PARTICULATE_INFER),
        },
    }), 200


@app.route("/", methods=["POST"])
def handle_request():
    """
    Main processing endpoint.

    Request JSON:
        glb_base64: Base64 GLB mesh (required)

    Response JSON:
        mesh_base64: Base64 segmented GLB mesh
        urdf_base64: Base64 URDF with joint definitions
        placeholder: False if successful
        generator: "particulate"
        articulation: { joint_count, part_count, is_articulated }
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

    glb_b64 = data.get("glb_base64")
    if not glb_b64:
        return jsonify({"error": "glb_base64 is required"}), 400

    # Decode GLB
    try:
        glb_bytes = base64.b64decode(glb_b64)
        log(f"[{request_id}] GLB input: {len(glb_bytes)} bytes")
    except Exception:
        return jsonify({"error": "Invalid glb_base64"}), 400

    # Acquire GPU lock
    log(f"[{request_id}] Waiting for GPU lock...")
    acquired = _gpu_lock.acquire(timeout=60)
    if not acquired:
        return jsonify({"error": "Service busy, retry later"}), 503

    try:
        log(f"[{request_id}] GPU lock acquired, running pipeline")
        mesh_bytes, urdf_bytes, metadata = run_particulate(glb_bytes, request_id)
        log(f"[{request_id}] Pipeline completed")
    except ParticulateError as e:
        log(f"[{request_id}] Pipeline error: {e}", "ERROR")
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        log(f"[{request_id}] Unexpected error: {e}", "ERROR")
        log(traceback.format_exc(), "ERROR")
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500
    finally:
        _gpu_lock.release()
        log(f"[{request_id}] GPU lock released")

    return jsonify(encode_response(mesh_bytes, urdf_bytes, metadata)), 200


# =============================================================================
# Startup
# =============================================================================

def start_warmup():
    """Start warmup in background thread."""
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    log(f"PARTICULATE_ROOT={PARTICULATE_ROOT}")
    log(f"DEBUG_MODE={DEBUG_MODE}")

    warmup_thread = threading.Thread(target=warmup_models, daemon=True)
    warmup_thread.start()


# Initialize on import
start_warmup()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, threaded=True)
