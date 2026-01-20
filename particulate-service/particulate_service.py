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
    ENV: Service environment (set to "production" to disable /debug)
    PARTICULATE_ROOT: Path to Particulate installation (default: /opt/particulate)
    ENABLE_DEBUG_ENDPOINT: Enable /debug in non-production (default: false)
    DEBUG_MODE: Legacy flag for /debug in non-production (default: 0)
    DEBUG_TOKEN: Shared secret required for /debug access (default: unset)
    PARTICULATE_DEBUG: Legacy flag for /debug (default: 0)
    PARTICULATE_DEBUG_TOKEN: Legacy shared secret for /debug access (default: unset)
    PARTICULATE_MAX_GLB_BYTES: Max decoded GLB size (default: 52428800)
    PORT: Service port (default: 8080)

Cloud Run Settings (REQUIRED):
    --memory 16Gi
    --cpu 4
    --gpu 1 --gpu-type nvidia-l4
    --timeout 300
    --concurrency 1
    --min-instances 0
"""

import atexit
import base64
import binascii
import hmac
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import stat
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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.config import load_pipeline_config
from tools.config.env import parse_bool_env
from tools.config.production_mode import resolve_production_mode

try:
    from tools.secrets import get_secret_or_env
    HAVE_SECRET_MANAGER = True
except ImportError:  # pragma: no cover
    HAVE_SECRET_MANAGER = False
    get_secret_or_env = None

from monitoring.alerting import send_alert

app = Flask(__name__)
logger = logging.getLogger(__name__)

_SECURITY_HEADERS = {
    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'; base-uri 'none'",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}


def _parse_allowed_origins() -> set[str]:
    raw_origins = os.getenv("CORS_ALLOWED_ORIGINS", "")
    if not raw_origins:
        return set()
    return {origin.strip() for origin in raw_origins.split(",") if origin.strip()}


def _allowed_origin(origin: str, allowed: set[str]) -> str | None:
    if not origin or not allowed:
        return None
    if "*" in allowed:
        return "*"
    if origin in allowed:
        return origin
    return None


@app.after_request
def _apply_security_headers(response):  # type: ignore[override]
    """Apply API security headers.

    These endpoints are API-only. If they are ever exposed to browsers for
    state-changing actions, enforce CSRF tokens on those routes.
    """
    for header, value in _SECURITY_HEADERS.items():
        response.headers.setdefault(header, value)

    allowed_origins = _parse_allowed_origins()
    origin = request.headers.get("Origin", "")
    allow_origin = _allowed_origin(origin, allowed_origins)
    if allow_origin:
        response.headers["Access-Control-Allow-Origin"] = allow_origin
        response.headers.setdefault("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        request_headers = request.headers.get("Access-Control-Request-Headers")
        if request_headers:
            response.headers.setdefault("Access-Control-Allow-Headers", request_headers)
        if allow_origin != "*":
            response.headers.setdefault("Vary", "Origin")
    return response

# =============================================================================
# Configuration
# =============================================================================

PARTICULATE_ROOT = Path(os.environ.get("PARTICULATE_ROOT", "/opt/particulate"))
PARTICULATE_INFER = PARTICULATE_ROOT / "infer.py"
_TMP_ENV_VAR = "PARTICULATE_TMP_DIR"

ENVIRONMENT = os.environ.get("ENV", "").lower()
PRODUCTION_MODE = resolve_production_mode()
ENABLE_DEBUG_ENDPOINT = os.environ.get("ENABLE_DEBUG_ENDPOINT", "false").lower() in ("1", "true", "yes")
DEBUG_MODE = os.environ.get("DEBUG_MODE", os.environ.get("PARTICULATE_DEBUG", "0")) == "1"
DEBUG_ENDPOINT_ENABLED = ENABLE_DEBUG_ENDPOINT or DEBUG_MODE


def _is_production_env() -> bool:
    return ENVIRONMENT == "production" or PRODUCTION_MODE


def _ensure_secure_directory(path: Path) -> None:
    try:
        os.chmod(path, 0o700)
    except OSError as exc:
        log(f"Failed to chmod temp directory {path}: {exc}", "ERROR")

    try:
        mode = stat.S_IMODE(path.stat().st_mode)
    except OSError as exc:
        log(f"Failed to stat temp directory {path}: {exc}", "ERROR")
        if _is_production_env():
            raise RuntimeError(f"Temp directory {path} is not accessible") from exc
        return

    if mode != 0o700:
        message = f"Temp directory {path} permissions are {oct(mode)}, expected 0o700"
        log(message, "ERROR")
        if _is_production_env():
            raise RuntimeError(message)


def _init_tmp_root() -> Path:
    base_dir = os.getenv(_TMP_ENV_VAR)
    if base_dir:
        base_path = Path(base_dir).expanduser()
        if not base_path.exists():
            base_path.mkdir(parents=True, exist_ok=True, mode=0o700)
        elif not base_path.is_dir():
            raise RuntimeError(f"{_TMP_ENV_VAR} must point to a directory: {base_path}")
        tmp_root = Path(tempfile.mkdtemp(prefix="particulate-", dir=str(base_path)))
    else:
        tmp_root = Path(tempfile.mkdtemp(prefix="particulate-"))

    _ensure_secure_directory(tmp_root)
    log(f"Using temp root: {tmp_root}")
    return tmp_root


TMP_ROOT = _init_tmp_root()


def _load_debug_token() -> Optional[str]:
    env_token = os.environ.get("DEBUG_TOKEN", os.environ.get("PARTICULATE_DEBUG_TOKEN", ""))
    if env_token is not None:
        env_token = env_token.strip() or None
    production_mode = _is_production_env()

    if HAVE_SECRET_MANAGER and get_secret_or_env is not None:
        try:
            token = get_secret_or_env(
                "particulate-debug-token",
                env_var="DEBUG_TOKEN",
                fallback_to_env=not production_mode,
            )
        except Exception as exc:
            if production_mode:
                logger.error(
                    "Debug token must be stored in Secret Manager in production; "
                    "env var DEBUG_TOKEN is not allowed. (%s)",
                    exc,
                )
                return None
            logger.warning(
                "Failed to fetch Secret Manager debug token; falling back to env vars: %s",
                exc,
            )
            token = env_token
        if token and not env_token:
            logger.info("Using Secret Manager for debug token.")
        if token is not None:
            token = token.strip() or None
        if token:
            return token
        if not production_mode:
            return env_token
        return None

    if production_mode:
        if env_token:
            logger.error(
                "Debug token must be stored in Secret Manager in production; "
                "env var DEBUG_TOKEN is not allowed."
            )
        return None

    return env_token


DEBUG_TOKEN = _load_debug_token()
PARTICULATE_MAX_GLB_BYTES = int(os.environ.get("PARTICULATE_MAX_GLB_BYTES", "52428800"))
PARTICULATE_MAX_GLB_B64_CHARS = ((PARTICULATE_MAX_GLB_BYTES + 2) // 3) * 4

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


def _estimate_base64_decoded_size(encoded: str) -> int:
    """Estimate decoded byte size from base64 string length."""
    padding = 0
    if encoded.endswith("=="):
        padding = 2
    elif encoded.endswith("="):
        padding = 1
    return (len(encoded) * 3) // 4 - padding


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
    timeout_s = _get_health_probe_timeout_s()
    gpu_required = parse_bool_env(os.getenv("GPU_HEALTH_REQUIRED"), default=True)
    isaac_required = parse_bool_env(os.getenv("ISAAC_SIM_HEALTH_REQUIRED"), default=False)
    llm_required = parse_bool_env(os.getenv("LLM_HEALTH_REQUIRED"), default=False)

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


def _get_health_probe_timeout_s() -> float:
    env_value = os.getenv("HEALTH_PROBE_TIMEOUT_S")
    if env_value is not None:
        return float(env_value)
    pipeline_config = load_pipeline_config()
    return float(pipeline_config.health_checks.probe_timeout_s)


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
        _alert_healthcheck_failure(
            "warmup_failed",
            {
                "error": _warmup_error,
                "details": _warmup_details,
            },
        )
        return jsonify({
            "status": "error",
            "message": f"Warmup failed: {_warmup_error}",
            "ready": False,
            "details": _warmup_details if _debug_authorized() else None,
        }), 503

    deps_ok, dep_details = _dependency_health()
    if not deps_ok:
        _alert_healthcheck_failure("dependency_check_failed", dep_details)
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
        "details": _warmup_details if _debug_authorized() else None,
        "dependencies": dep_details.get("dependencies"),
    }), 200


@app.route("/healthz", methods=["GET"])
def healthz():
    """Health check endpoint with dependency probes."""
    return healthcheck()


def _alert_healthcheck_failure(event_type: str, details: Dict[str, Any]) -> None:
    if os.getenv("ALERT_HEALTHCHECK_ENABLED", "true").lower() != "true":
        return

    threshold = int(os.getenv("ALERT_HEALTHCHECK_FAILURE_THRESHOLD", "1"))
    errors = details.get("errors") if isinstance(details, dict) else None
    failure_count = len(errors) if isinstance(errors, list) else 1

    if failure_count < threshold:
        return

    send_alert(
        event_type=f"particulate_healthcheck_{event_type}",
        summary="Particulate health check failed",
        details={
            "service": "particulate",
            "failure_count": failure_count,
            "details": details,
        },
        severity=os.getenv("ALERT_HEALTHCHECK_SEVERITY", "critical"),
    )


@app.route("/ready", methods=["GET"])
def readiness():
    """Readiness probe for Kubernetes/Cloud Run."""
    if _models_ready.is_set() and not _warmup_error:
        return "ready", 200
    return f"not ready: {_warmup_error or 'loading'}", 503


def _debug_block_reason() -> Optional[str]:
    if _is_production_env():
        return "production_environment"
    if not DEBUG_ENDPOINT_ENABLED:
        return "debug_endpoint_disabled"
    if not DEBUG_TOKEN:
        return "debug_token_missing"

    auth_header = request.headers.get("Authorization", "")
    if not auth_header:
        return "authorization_missing"

    token = auth_header
    if auth_header.lower().startswith("bearer "):
        token = auth_header[7:].strip()

    if not hmac.compare_digest(token, DEBUG_TOKEN):
        return "authorization_invalid"

    return None


def _debug_authorized() -> bool:
    return _debug_block_reason() is None


@app.route("/debug", methods=["GET"])
def debug_info():
    """Debug endpoint with service state."""
    request_id = request.headers.get("X-Request-Id") or request.headers.get("X-Request-ID")
    if not request_id:
        request_id = str(uuid.uuid4())[:8]
    remote_addr = request.remote_addr or "unknown"

    if not _debug_authorized():
        block_reason = _debug_block_reason() or "unknown"
        log(
            (
                "Denied debug request: "
                f"reason={block_reason} "
                f"request_id={request_id} "
                f"remote={remote_addr} "
                f"auth_header_present={bool(request.headers.get('Authorization'))}"
            ),
            level="WARNING",
        )
        return jsonify({"error": "debug access forbidden"}), 403

    log(
        (
            "Debug request authorized: "
            f"request_id={request_id} "
            f"remote={remote_addr}"
        )
    )

    warmup_status = "ready" if _models_ready.is_set() and not _warmup_error else "error"
    if not _models_ready.is_set():
        warmup_status = "warming_up"

    return jsonify({
        "models_ready": _models_ready.is_set(),
        "warmup_status": warmup_status,
        "warmup_ok": _warmup_error is None,
        "environment": ENVIRONMENT or "unknown",
        "debug_enabled": DEBUG_ENDPOINT_ENABLED,
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
    if not isinstance(glb_b64, str):
        return jsonify({"error": "glb_base64 must be a base64 string"}), 400

    b64_len = len(glb_b64)
    estimated_bytes = _estimate_base64_decoded_size(glb_b64)
    if b64_len > PARTICULATE_MAX_GLB_B64_CHARS or estimated_bytes > PARTICULATE_MAX_GLB_BYTES:
        log(
            (
                f"[{request_id}] Rejected payload: b64_len={b64_len} "
                f"estimated_bytes={estimated_bytes} limit_bytes={PARTICULATE_MAX_GLB_BYTES}"
            ),
            "WARNING",
        )
        return jsonify({"error": "Payload too large"}), 413

    # Decode GLB
    try:
        glb_bytes = base64.b64decode(glb_b64, validate=True)
        if len(glb_bytes) > PARTICULATE_MAX_GLB_BYTES:
            log(
                (
                    f"[{request_id}] Rejected payload: decoded_bytes={len(glb_bytes)} "
                    f"limit_bytes={PARTICULATE_MAX_GLB_BYTES}"
                ),
                "WARNING",
            )
            return jsonify({"error": "Payload too large"}), 413
        log(f"[{request_id}] GLB input: {len(glb_bytes)} bytes")
    except binascii.Error:
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
        return jsonify({"error": "internal_error", "request_id": request_id}), 500
    finally:
        _gpu_lock.release()
        log(f"[{request_id}] GPU lock released")

    return jsonify(encode_response(mesh_bytes, urdf_bytes, metadata)), 200


# =============================================================================
# Startup
# =============================================================================

def start_warmup():
    """Start warmup in background thread."""
    if os.getenv("PARTICULATE_SKIP_WARMUP", "").lower() in ("1", "true", "yes"):
        log("Skipping warmup (PARTICULATE_SKIP_WARMUP enabled)")
        return
    log(f"PARTICULATE_ROOT={PARTICULATE_ROOT}")
    log(f"DEBUG_MODE={DEBUG_MODE}")

    warmup_thread = threading.Thread(target=warmup_models, daemon=True)
    warmup_thread.start()


# Initialize on import
start_warmup()


def _cleanup_tmp_root() -> None:
    try:
        if TMP_ROOT.exists():
            shutil.rmtree(TMP_ROOT, ignore_errors=True)
            log(f"Cleaned up temp root: {TMP_ROOT}")
    except Exception as exc:
        log(f"Failed to cleanup temp root {TMP_ROOT}: {exc}", "WARNING")


atexit.register(_cleanup_tmp_root)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, threaded=True)
