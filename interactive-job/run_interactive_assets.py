#!/usr/bin/env python3
"""
Interactive Asset Pipeline for 3D-RE-GEN Integration.

This job processes 3D assets (GLB meshes) from 3D-RE-GEN to add articulation
data using Particulate - a fast feed-forward mesh articulation model (~10s per object).

The pipeline:
1. Receives GLB meshes from 3D-RE-GEN (or crop images as fallback)
2. Sends GLB directly to Particulate for articulation
3. Outputs URDF with articulation data + segmented meshes

Designed for the 3D-RE-GEN pipeline:
    3D-RE-GEN → interactive-job → simready-job → usd-assembly-job

Environment Variables:
    BUCKET: GCS bucket name
    SCENE_ID: Scene identifier
    ASSETS_PREFIX: Path to assets (contains scene_assets.json)
    PARTICULATE_ENDPOINT: Particulate Cloud Run service URL
    PARTICULATE_MODE: "remote" (default), "local", "mock", or "skip" for inference control
    PARTICULATE_LOCAL_ENDPOINT: Optional local Particulate URL for PARTICULATE_MODE=local
    PARTICULATE_LOCAL_MODEL: Identifier for the locally hosted Particulate model (required in labs/prod)
    APPROVED_PARTICULATE_MODELS: Comma-separated allowlist for local Particulate models (default: pat_b)
    ARTICULATION_BACKEND: "particulate", "heuristic", or "auto" (default)
    REGEN3D_PREFIX: Optional path to 3D-RE-GEN outputs (default: same as ASSETS_PREFIX)
    INTERACTIVE_MODE: "glb" (default) or "image" for legacy crop-based processing
    PIPELINE_ENV: Pipeline environment (e.g., "production") for production guardrails
    DISALLOW_PLACEHOLDER_URDF: "true" to fail if placeholder URDFs are generated
    LABS_MODE: "true" to enforce labs guardrails (Particulate required, no heuristics)
    ARTICULATION_MULTIVIEW_ENABLED: "true" to generate experimental synthetic multiview references
    ARTICULATION_MULTIVIEW_COUNT: Number of synthetic scaffold views (default: 4)
    ARTICULATION_MULTIVIEW_MODEL: Gemini image model for scaffold generation
"""
import base64
import importlib.util
import json
import math
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from tools.config.env_flags import env_flag
from tools.config.production_mode import resolve_production_mode_detail, resolve_pipeline_environment
from tools.scene_manifest.loader import load_manifest_or_scene_assets
from tools.validation.entrypoint_checks import validate_required_env_vars
from tools.workflow import FailureMarkerWriter
from tools.metrics.pipeline_metrics import get_metrics


# =============================================================================
# Configuration
# =============================================================================

# Service timeouts
PARTICULATE_WARMUP_TIMEOUT = int(os.getenv("PARTICULATE_WARMUP_TIMEOUT", "120"))  # 2 min
PARTICULATE_REQUEST_TIMEOUT = 120  # 2 min (inference takes ~10s)

MAX_RETRIES = 3

# Processing modes
MODE_GLB = "glb"      # 3D-RE-GEN GLB mesh input (default)
MODE_IMAGE = "image"  # Legacy crop image input

PARTICULATE_MODE_REMOTE = "remote"
PARTICULATE_MODE_LOCAL = "local"
PARTICULATE_MODE_MOCK = "mock"
PARTICULATE_MODE_SKIP = "skip"

ARTICULATION_BACKEND_PARTICULATE = "particulate"
ARTICULATION_BACKEND_HEURISTIC = "heuristic"
ARTICULATION_BACKEND_AUTO = "auto"


# =============================================================================
# Logging
# =============================================================================

def log(msg: str, level: str = "INFO", obj_id: str = "") -> None:
    """Log with prefix and optional object ID."""
    prefix = f"[{obj_id}] " if obj_id else ""
    stream = sys.stderr if level in ("ERROR", "WARNING") else sys.stdout
    print(f"[INTERACTIVE] [{level}] {prefix}{msg}", file=stream, flush=True)


def raise_placeholder_error(obj_id: str, reason: str) -> None:
    """Raise a clear error when placeholder outputs are generated."""
    raise RuntimeError(
        "Placeholder URDF generation blocked "
        f"for {obj_id} ({reason}). "
        "Set DISALLOW_PLACEHOLDER_URDF=false and disable production mode to allow placeholders."
    )


def normalize_particulate_mode(value: str) -> str:
    """Normalize particulate mode env var to a supported value."""
    mode = value.strip().lower()
    if mode not in {PARTICULATE_MODE_REMOTE, PARTICULATE_MODE_LOCAL, PARTICULATE_MODE_MOCK, PARTICULATE_MODE_SKIP}:
        log(
            f"Unknown PARTICULATE_MODE '{value}', defaulting to '{PARTICULATE_MODE_REMOTE}'",
            "WARNING",
        )
        return PARTICULATE_MODE_REMOTE
    return mode


def normalize_articulation_backend(value: str) -> str:
    """Normalize articulation backend env var to a supported value."""
    backend = value.strip().lower()
    if backend not in {
        ARTICULATION_BACKEND_PARTICULATE,
        ARTICULATION_BACKEND_HEURISTIC,
        ARTICULATION_BACKEND_AUTO,
    }:
        log(
            f"Unknown ARTICULATION_BACKEND '{value}', defaulting to '{ARTICULATION_BACKEND_AUTO}'",
            "WARNING",
        )
        return ARTICULATION_BACKEND_AUTO
    return backend


def resolve_articulation_backend(
    requested_backend: str,
    particulate_mode: str,
    particulate_endpoint: str,
) -> str:
    """Resolve backend selection (explicit or auto)."""
    if requested_backend == ARTICULATION_BACKEND_PARTICULATE:
        return ARTICULATION_BACKEND_PARTICULATE
    if requested_backend == ARTICULATION_BACKEND_HEURISTIC:
        return ARTICULATION_BACKEND_HEURISTIC

    if particulate_mode in {PARTICULATE_MODE_LOCAL, PARTICULATE_MODE_MOCK}:
        return ARTICULATION_BACKEND_PARTICULATE
    if particulate_mode == PARTICULATE_MODE_REMOTE and particulate_endpoint:
        return ARTICULATION_BACKEND_PARTICULATE
    return ARTICULATION_BACKEND_HEURISTIC


def parse_csv_env(value: str) -> List[str]:
    """Parse a comma-separated environment value into a normalized list."""
    return [item.strip() for item in value.split(",") if item.strip()]


def resolve_interactive_production_mode() -> Tuple[bool, str, Optional[str]]:
    """Resolve production mode from explicit override or pipeline environment."""
    production_mode, source, value = resolve_production_mode_detail()
    if production_mode:
        return production_mode, source or "PIPELINE_ENV", value

    pipeline_env = resolve_pipeline_environment()
    return False, "PIPELINE_ENV", pipeline_env


def resolve_particulate_endpoint(
    particulate_mode: str,
    remote_endpoint: str,
    local_endpoint: str,
) -> Tuple[str, Optional[str]]:
    """Resolve the particulate endpoint based on mode."""
    if particulate_mode == PARTICULATE_MODE_LOCAL:
        endpoint = local_endpoint or "http://localhost:8080"
        if not local_endpoint:
            log(
                f"PARTICULATE_LOCAL_ENDPOINT not set; using default {endpoint}",
                "WARNING",
            )
        return endpoint, "local"
    if particulate_mode == PARTICULATE_MODE_MOCK:
        return "", "mock"
    if particulate_mode == PARTICULATE_MODE_REMOTE:
        return remote_endpoint, "remote"
    return "", None


def write_failure_marker(
    assets_root: Path,
    failure_writer: Optional[FailureMarkerWriter],
    scene_id: str,
    reason: str,
    details: Optional[Dict[str, Any]] = None,
    errors: Optional[List[Dict[str, Any]]] = None,
    warnings: Optional[List[Dict[str, Any]]] = None,
    config_context: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write a failure marker to prevent downstream jobs from using low-fidelity assets."""
    marker_path = assets_root / ".interactive_failed"
    error_payload = {
        "code": reason,
        "message": reason.replace("_", " ").strip(),
        "details": details or {},
        "objects": errors or [],
        "warnings": warnings or [],
    }
    payload = {
        "scene_id": scene_id,
        "status": "failed",
        "success": False,
        "reason": reason,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "details": details or {},
        "error": error_payload,
    }
    save_json(payload, marker_path)
    if failure_writer is not None:
        failure_writer.write_failure(
            exception=RuntimeError(reason),
            failed_step="interactive-assets",
            input_params=config_context or {},
            partial_results={
                "details": details or {},
                "errors": errors or [],
                "warnings": warnings or [],
            },
            error_code=reason,
            config_context=config_context or {},
        )
    log(f"Wrote failure marker: {marker_path}", "ERROR")
    return marker_path


def write_status_marker(
    assets_root: Path,
    scene_id: str,
    status: str,
    summary: Dict[str, Any],
    errors: Optional[List[Dict[str, Any]]] = None,
    warnings: Optional[List[Dict[str, Any]]] = None,
) -> Path:
    """Write a status marker for workflows with summary + structured payloads."""
    marker_path = assets_root / ".interactive_complete"
    payload = {
        "scene_id": scene_id,
        "status": status,
        "success": status == "success",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": summary,
        "errors": errors or [],
        "warnings": warnings or [],
    }
    save_json(payload, marker_path)
    log(f"Wrote status marker: {marker_path} ({status})")
    return marker_path


def write_interactive_summary(
    assets_root: Path,
    scene_id: str,
    summary: Dict[str, Any],
) -> Path:
    """Write a concise interactive summary artifact for downstream diagnostics."""
    summary_path = assets_root / ".interactive_summary.json"
    payload = {
        "scene_id": scene_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **summary,
    }
    save_json(payload, summary_path)
    log(f"Wrote interactive summary: {summary_path}")
    return summary_path


def write_placeholder_warning_marker(
    assets_root: Path,
    scene_id: str,
    placeholders: List[Dict[str, Any]],
) -> Path:
    """Write a warning marker when placeholder URDFs are generated."""
    marker_path = assets_root / ".interactive_placeholder_warning"
    payload = {
        "scene_id": scene_id,
        "status": "warning",
        "reason": "placeholder_urdf_generated",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "placeholder_count": len(placeholders),
        "objects": placeholders,
    }
    save_json(payload, marker_path)
    log(f"Wrote placeholder warning marker: {marker_path}", "WARNING")
    return marker_path


def collect_result_payloads(results: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Collect structured error/warning payloads for workflows."""
    errors = []
    warnings = []
    for result in results:
        status = result.get("status")
        payload = {
            "id": result.get("id"),
            "status": status,
            "error": result.get("error"),
        }
        if status == "error":
            errors.append(payload)
        elif status in {"fallback", "static"}:
            warnings.append(payload)
    return errors, warnings


# =============================================================================
# File/Path Utilities
# =============================================================================

def ensure_dir(path: Path) -> None:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    """Load JSON file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, path: Path) -> None:
    """Save JSON file."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def find_glb_file(obj_dir: Path, obj_id: str) -> Optional[Path]:
    """
    Find GLB mesh file for an object.

    3D-RE-GEN outputs meshes as:
    - obj_{id}.glb (direct output)
    - mesh.glb (alternative naming)
    - part.glb (from Particulate)
    """
    candidates = [
        obj_dir / f"obj_{obj_id}.glb",
        obj_dir / "mesh.glb",
        obj_dir / "part.glb",
        obj_dir / f"{obj_id}.glb",
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    # Fallback: find any GLB in directory
    glb_files = list(obj_dir.glob("*.glb"))
    if glb_files:
        return glb_files[0]

    return None


def find_crop_image(obj_dir: Path, multiview_root: Path, obj_id: str) -> Optional[Path]:
    """
    Find crop/view image for an object.

    Looks in:
    1. Object directory (crop.png, view_0.png)
    2. Multiview directory structure
    """
    obj_name = f"obj_{obj_id}"

    # Check object directory
    for name in ["crop.png", "view_0.png", "input.png", "reference.png"]:
        candidate = obj_dir / name
        if candidate.is_file():
            return candidate

    # Check multiview directory
    multiview_obj = multiview_root / obj_name
    if multiview_obj.is_dir():
        for name in ["crop.png", "view_0.png"]:
            candidate = multiview_obj / name
            if candidate.is_file():
                return candidate

    # Find any PNG in object directory
    png_files = list(obj_dir.glob("*.png"))
    if png_files:
        return png_files[0]

    return None


def _coerce_optional_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on", "y"}:
            return True
        if normalized in {"0", "false", "no", "off", "n"}:
            return False
    return None


def is_required_articulation_object(obj: Dict[str, Any]) -> bool:
    """Return True when object is explicitly marked articulation-required."""
    explicit_required = _coerce_optional_bool(obj.get("articulation_required"))
    if explicit_required is True:
        return True
    articulation = obj.get("articulation")
    if isinstance(articulation, dict):
        explicit_from_articulation = _coerce_optional_bool(articulation.get("required"))
        if explicit_from_articulation is True:
            return True

    use_sim_role_defaults = env_flag(os.getenv("INTERACTIVE_REQUIRE_SIM_ROLE_DEFAULTS"), default=False)
    if use_sim_role_defaults:
        sim_role = str(obj.get("sim_role", "") or "").strip().lower()
        if sim_role in {"articulated_furniture", "articulated_appliance"}:
            return True
    return False


def is_articulation_candidate_object(obj: Dict[str, Any]) -> bool:
    """Return True when object should be considered for articulation processing."""
    if is_required_articulation_object(obj):
        return True

    explicit_candidate = _coerce_optional_bool(obj.get("articulation_candidate"))
    if explicit_candidate is not None:
        return explicit_candidate

    articulation = obj.get("articulation")
    if isinstance(articulation, dict):
        explicit_candidate = _coerce_optional_bool(articulation.get("candidate"))
        if explicit_candidate is not None:
            return explicit_candidate

    if obj.get("articulation_hint"):
        return True
    if isinstance(articulation, dict):
        if articulation.get("type"):
            return True
        hints = articulation.get("hints")
        if isinstance(hints, list) and len(hints) > 0:
            return True

    use_sim_role_candidates = env_flag(os.getenv("INTERACTIVE_CANDIDATE_FROM_SIM_ROLE"), default=True)
    if use_sim_role_candidates:
        sim_role = str(obj.get("sim_role", "") or "").strip().lower()
        if sim_role in {"articulated_furniture", "articulated_appliance"}:
            return True
    return False


# =============================================================================
# Mesh Rendering (GLB → Multi-View Images)
# =============================================================================

def render_mesh_views(glb_path: Path, output_dir: Path, num_views: int = 4) -> List[Path]:
    """
    Render multiple views of a GLB mesh for VLM analysis.

    Uses trimesh for cross-platform rendering (no GPU required).

    Args:
        glb_path: Path to GLB mesh file
        output_dir: Directory to save rendered views
        num_views: Number of views to render (default: 4)

    Returns:
        List of paths to rendered PNG images
    """
    try:
        import trimesh
        import numpy as np
    except ImportError:
        log("trimesh not available, skipping mesh rendering", "WARNING")
        return []

    ensure_dir(output_dir)
    view_paths = []

    try:
        # Load mesh
        mesh = trimesh.load(str(glb_path), force='mesh')

        if isinstance(mesh, trimesh.Scene):
            # Extract geometry from scene
            geometries = list(mesh.geometry.values())
            if geometries:
                mesh = trimesh.util.concatenate(geometries)
            else:
                log(f"No geometry found in {glb_path}", "WARNING")
                return []

        # Get bounding box for camera positioning
        bounds = mesh.bounds
        center = mesh.centroid
        size = np.max(bounds[1] - bounds[0])

        # Camera distance (ensure full object is visible)
        distance = size * 2.5

        # Render from multiple angles
        angles = np.linspace(0, 360, num_views, endpoint=False)

        for i, angle in enumerate(angles):
            try:
                # Create scene with camera at angle
                scene = trimesh.Scene(mesh)

                # Position camera
                rad = np.radians(angle)
                camera_pos = center + np.array([
                    distance * np.cos(rad),
                    distance * np.sin(rad),
                    distance * 0.5  # Slight elevation
                ])

                # Set camera transform (look at center)
                scene.set_camera(
                    angles=(0, 0, 0),
                    distance=distance,
                    center=center,
                )

                # Render to PNG
                view_path = output_dir / f"view_{i}.png"

                # Use offscreen rendering
                png_data = scene.save_image(resolution=(512, 512))
                if png_data:
                    view_path.write_bytes(png_data)
                    view_paths.append(view_path)
                    log(f"Rendered view {i} ({angle}°): {view_path}")

            except Exception as e:
                log(f"Failed to render view {i}: {e}", "WARNING")
                continue

    except Exception as e:
        log(f"Failed to load/render mesh {glb_path}: {e}", "WARNING")

    return view_paths


def render_mesh_to_single_view(glb_path: Path, output_path: Path) -> Optional[Path]:
    """
    Render a single front view of a GLB mesh.

    This is a simpler fallback when multi-view rendering fails.
    """
    try:
        import trimesh
    except ImportError:
        return None

    ensure_dir(output_path.parent)

    try:
        mesh = trimesh.load(str(glb_path), force='mesh')

        if isinstance(mesh, trimesh.Scene):
            geometries = list(mesh.geometry.values())
            if geometries:
                mesh = trimesh.util.concatenate(geometries)
            else:
                return None

        scene = trimesh.Scene(mesh)
        png_data = scene.save_image(resolution=(512, 512))

        if png_data:
            output_path.write_bytes(png_data)
            return output_path

    except Exception as e:
        log(f"Single view render failed: {e}", "WARNING")

    return None


def generate_multiview_scaffold(
    obj_name: str,
    obj_class: str,
    glb_path: Path,
    output_dir: Path,
    view_count: int,
    model: str,
) -> Dict[str, Any]:
    """Generate synthetic reference views for future articulation segmentation."""
    scaffold_dir = output_dir / "multiview_synth"
    ensure_dir(scaffold_dir)

    metadata: Dict[str, Any] = {
        "status": "pending",
        "object_id": obj_name,
        "class_name": obj_class,
        "model": model,
        "requested_count": int(view_count),
        "generated_count": 0,
        "generated_files": [],
        "errors": [],
    }

    seed_view_path = render_mesh_to_single_view(glb_path, scaffold_dir / "seed_mesh_view.png")
    if seed_view_path is not None:
        metadata["seed_view"] = seed_view_path.name

    try:
        from tools.llm_client.client import create_llm_client, LLMProvider
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["errors"].append(f"llm_client_unavailable: {exc}")
        save_json(metadata, output_dir / "multiview_scaffold.json")
        return metadata

    try:
        client = create_llm_client(
            provider=LLMProvider.GEMINI,
            model=model,
            fallback_enabled=False,
        )
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["errors"].append(f"llm_client_init_failed: {exc}")
        save_json(metadata, output_dir / "multiview_scaffold.json")
        return metadata

    count = max(1, int(view_count))
    for view_idx in range(count):
        prompt = (
            f"Generate a clean product image of a {obj_class} on a plain light background. "
            "Preserve realistic structure and proportions. "
            "This image will be used for articulated-part segmentation research. "
            f"Create view {view_idx + 1} of {count}. "
            "Prefer slight viewpoint changes and include plausible open/closed articulation states "
            "when drawers, doors, or lids are present."
        )
        try:
            response = client.generate_image(prompt=prompt, size="1024x1024")
            if not response.images:
                raise RuntimeError("no images returned")
            output_path = scaffold_dir / f"view_{view_idx}.png"
            output_path.write_bytes(response.images[0])
            metadata["generated_files"].append(output_path.name)
        except Exception as exc:
            metadata["errors"].append(f"view_{view_idx}: {exc}")

    metadata["generated_count"] = len(metadata["generated_files"])
    metadata["status"] = "success" if metadata["generated_files"] else "failed"
    save_json(metadata, output_dir / "multiview_scaffold.json")
    return metadata


# =============================================================================
# Particulate Service Communication
# =============================================================================

def check_service_health(endpoint: str, timeout: int = 30) -> Tuple[bool, str, Optional[dict]]:
    """
    Check service health status.

    Returns:
        (is_ready, status_message, response_data)
    """
    try:
        req = urllib.request.Request(endpoint, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            data = json.loads(body)
            is_ready = data.get("ready", False) and data.get("status") == "ok"
            return is_ready, data.get("status", "unknown"), data
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8", errors="replace")
            data = json.loads(body)
            return False, data.get("message", f"HTTP {e.code}"), data
        except (json.JSONDecodeError, UnicodeDecodeError, IOError) as parse_err:
            # Replace bare except with specific exceptions
            log(f"Failed to parse error response: {parse_err}", "WARNING")
            return False, f"HTTP {e.code}", None
    except urllib.error.URLError as e:
        return False, f"Network error: {e.reason}", None
    except Exception as e:
        return False, f"Error: {e}", None

def wait_for_particulate_ready(endpoint: str, max_wait: int = PARTICULATE_WARMUP_TIMEOUT) -> bool:
    """
    Wait for Particulate service to be ready.

    Particulate has fast cold starts (~1-2 min).
    """
    log(f"Waiting for Particulate service to be ready (max {max_wait}s)...")
    start_time = time.time()
    last_status = ""

    while time.time() - start_time < max_wait:
        elapsed = int(time.time() - start_time)
        is_ready, status, _ = check_service_health(endpoint)

        if is_ready:
            log(f"Particulate service ready after {elapsed}s")
            return True

        if status != last_status:
            log(f"Particulate service status: {status} ({elapsed}s elapsed)", "WARNING")
            last_status = status

        # Faster polling for Particulate (it starts faster)
        time.sleep(5)

    log(f"Particulate service not ready after {max_wait}s", "ERROR")
    return False


def call_particulate_service(
    endpoint: str,
    glb_path: Path,
    obj_id: str,
) -> Optional[dict]:
    """
    Call Particulate service with GLB mesh.

    The service expects:
    - glb_base64: Base64-encoded GLB mesh

    Returns:
        Service response dict or None on failure:
        {
            "mesh_base64": "<segmented GLB>",
            "urdf_base64": "<URDF>",
            "placeholder": false,
            "generator": "particulate",
            "articulation": { "joint_count": N, "part_count": N, "is_articulated": bool }
        }
    """
    if not glb_path or not glb_path.is_file():
        log(f"No GLB file provided", "ERROR", obj_id=obj_id)
        return None

    # Build request payload
    with glb_path.open("rb") as f:
        glb_bytes = f.read()

    payload = {
        "glb_base64": base64.b64encode(glb_bytes).decode("utf-8")
    }

    body = json.dumps(payload).encode("utf-8")
    log(f"Request payload: glb={len(glb_bytes)} bytes", obj_id=obj_id)

    # Retry loop
    for attempt in range(MAX_RETRIES + 1):
        try:
            timeout = PARTICULATE_REQUEST_TIMEOUT

            if attempt > 0:
                log(f"Retry {attempt}/{MAX_RETRIES} (timeout={timeout}s)", obj_id=obj_id)
            else:
                log(f"POST {endpoint} (timeout={timeout}s)", obj_id=obj_id)

            req = urllib.request.Request(
                endpoint,
                data=body,
                headers={"Content-Type": "application/json"},
            )

            metrics = get_metrics()
            scene_id = os.environ.get("SCENE_ID", "unknown")
            start = time.time()
            resp_status = None
            with metrics.track_api_call("particulate", "articulation", scene_id):
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    elapsed = int(time.time() - start)
                    resp_status = resp.status
                    log(f"Response: {resp.status} ({elapsed}s)", obj_id=obj_id)

                    text = resp.read().decode("utf-8", errors="replace")
                    try:
                        data = json.loads(text)
                    except json.JSONDecodeError as decode_err:
                        preview = text[:200]
                        log(
                            "Failed to decode Particulate response JSON "
                            f"({decode_err}); preview: {preview}",
                            level="ERROR",
                            obj_id=obj_id,
                        )
                        if resp_status is not None and 400 <= resp_status < 500 and resp_status != 429:
                            return None
                        if attempt < MAX_RETRIES:
                            time.sleep(10 * (attempt + 1))
                            continue
                        return None

                if "placeholder" not in data or "articulation" not in data:
                    log(
                        "Invalid Particulate response schema; "
                        "missing required keys 'placeholder' or 'articulation'.",
                        level="ERROR",
                        obj_id=obj_id,
                    )
                    return None

                is_placeholder = data.get("placeholder", True)
                articulation = data.get("articulation", {})
                joint_count = articulation.get("joint_count", 0)

                log(f"Success: placeholder={is_placeholder}, joints={joint_count}", obj_id=obj_id)

                return data

        except urllib.error.HTTPError as e:
            error_body = ""
            try:
                error_body = e.read().decode("utf-8", errors="replace")[:1000]
            except (UnicodeDecodeError, IOError, OSError) as read_err:
                # Replace bare except with specific exceptions
                error_body = f"[Failed to read error body: {read_err}]"

            log(f"HTTP {e.code}: {error_body}", level="ERROR", obj_id=obj_id)

            # Don't retry client errors (except 429)
            if 400 <= e.code < 500 and e.code != 429:
                return None

            if attempt < MAX_RETRIES:
                wait_time = 10 * (attempt + 1)  # Shorter backoff for Particulate
                log(f"Waiting {wait_time}s before retry...", "WARNING", obj_id)
                time.sleep(wait_time)

        except (urllib.error.URLError, TimeoutError) as e:
            log(f"Network error: {e}", "ERROR", obj_id=obj_id)
            if attempt < MAX_RETRIES:
                time.sleep(10 * (attempt + 1))

        except Exception as e:
            log(f"Unexpected error: {type(e).__name__}: {e}", "ERROR", obj_id=obj_id)
            if attempt < MAX_RETRIES:
                time.sleep(10)

    return None


def build_mock_particulate_response(
    glb_path: Path,
    obj_id: str,
    placeholder: bool,
) -> Optional[dict]:
    """Build a mock Particulate response payload for tests."""
    if not glb_path or not glb_path.is_file():
        log("Mock Particulate missing GLB input", "ERROR", obj_id=obj_id)
        return None

    glb_bytes = glb_path.read_bytes()
    response = {
        "placeholder": placeholder,
        "generator": "particulate-mock",
        "articulation": {
            "joint_count": 0,
            "part_count": 1,
            "is_articulated": False,
        },
    }

    if placeholder:
        return response

    response["mesh_base64"] = base64.b64encode(glb_bytes).decode("utf-8")
    static_urdf = generate_static_urdf(obj_id, "part.glb").encode("utf-8")
    response["urdf_base64"] = base64.b64encode(static_urdf).decode("utf-8")
    return response


# =============================================================================
# URDF Generation and Parsing
# =============================================================================


class URDFValidationResult:
    """Result of URDF validation."""

    def __init__(self):
        self.is_valid = True
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.repairs_made: List[str] = []

    def add_error(self, msg: str):
        self.errors.append(msg)
        self.is_valid = False

    def add_warning(self, msg: str):
        self.warnings.append(msg)

    def add_repair(self, msg: str):
        self.repairs_made.append(msg)


class SimulationValidationResult:
    """Result of simulation-backed URDF validation."""

    def __init__(self):
        self.is_valid = True
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.skipped = False
        self.backend: Optional[str] = None

    def add_error(self, msg: str):
        self.errors.append(msg)
        self.is_valid = False

    def add_warning(self, msg: str):
        self.warnings.append(msg)

    def mark_skipped(self, reason: str):
        self.skipped = True
        self.add_warning(reason)


def validate_urdf_in_pybullet(
    urdf_path: Path,
    mesh_dir: Optional[Path] = None,
    obj_id: str = "",
) -> SimulationValidationResult:
    """
    Validate a URDF by loading it in PyBullet and checking controllability.

    Checks:
    - URDF loads successfully in PyBullet
    - Joint limits are valid for revolute/prismatic joints
    - Joint axis is non-zero and normalized
    - Controllable joints accept position control commands
    """
    result = SimulationValidationResult()
    result.backend = "pybullet"

    if not urdf_path.is_file():
        result.add_error(f"URDF file does not exist: {urdf_path}")
        return result

    if importlib.util.find_spec("pybullet") is None:
        result.mark_skipped("PyBullet not available; skipping simulation validation")
        return result

    import pybullet as p

    client_id = p.connect(p.DIRECT)
    try:
        if mesh_dir:
            p.setAdditionalSearchPath(str(mesh_dir), physicsClientId=client_id)

        try:
            body_id = p.loadURDF(
                str(urdf_path),
                basePosition=[0, 0, 0],
                useFixedBase=True,
                flags=p.URDF_USE_INERTIA_FROM_FILE,
                physicsClientId=client_id,
            )
        except Exception as exc:
            result.add_error(f"Failed to load URDF in PyBullet: {exc}")
            return result

        if body_id < 0:
            result.add_error("PyBullet returned invalid body id")
            return result

        joint_count = p.getNumJoints(body_id, physicsClientId=client_id)
        for joint_index in range(joint_count):
            joint_info = p.getJointInfo(body_id, joint_index, physicsClientId=client_id)
            joint_name = joint_info[1].decode("utf-8", errors="ignore") if joint_info[1] else f"joint_{joint_index}"
            joint_type = joint_info[2]
            lower_limit = joint_info[8]
            upper_limit = joint_info[9]
            axis = joint_info[13]

            axis_norm = math.sqrt(sum(v * v for v in axis))
            if axis_norm <= 1e-6 and joint_type != p.JOINT_FIXED:
                result.add_error(f"Joint '{joint_name}': axis is zero vector in PyBullet")
            elif axis_norm > 1e-6 and abs(axis_norm - 1.0) > 0.05:
                result.add_warning(f"Joint '{joint_name}': axis not normalized (norm={axis_norm:.3f})")

            if joint_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                if lower_limit >= upper_limit:
                    result.add_error(
                        f"Joint '{joint_name}': lower limit ({lower_limit}) >= upper limit ({upper_limit})"
                    )

                mid = (lower_limit + upper_limit) * 0.5
                try:
                    p.resetJointState(body_id, joint_index, mid, physicsClientId=client_id)
                    p.setJointMotorControl2(
                        bodyIndex=body_id,
                        jointIndex=joint_index,
                        controlMode=p.POSITION_CONTROL,
                        targetPosition=mid,
                        force=100,
                        physicsClientId=client_id,
                    )
                except Exception as exc:
                    result.add_error(f"Joint '{joint_name}': not controllable in PyBullet ({exc})")

    finally:
        p.disconnect(physicsClientId=client_id)

    return result


def validate_urdf_in_simulator(
    urdf_path: Path,
    mesh_dir: Optional[Path] = None,
    obj_id: str = "",
) -> SimulationValidationResult:
    """
    Validate URDF in a simulator backend (PyBullet preferred).

    Returns a SimulationValidationResult with validation status and warnings.
    """
    return validate_urdf_in_pybullet(urdf_path, mesh_dir, obj_id)


def validate_urdf(urdf_path: Path, mesh_dir: Optional[Path] = None) -> URDFValidationResult:
    """
    Validate a URDF file for common issues.

    Checks:
    - XML syntax validity
    - Required robot element and name
    - Link structure (at least one link, base_link present)
    - Joint references (parent/child links must exist)
    - Mesh file references (if mesh_dir provided)
    - Inertial properties (mass > 0, valid inertia matrix)
    - Joint limits (lower < upper for limited joints)

    Args:
        urdf_path: Path to the URDF file
        mesh_dir: Optional directory to check mesh file references

    Returns:
        URDFValidationResult with errors, warnings, and repair info
    """
    result = URDFValidationResult()

    if not urdf_path.is_file():
        result.add_error(f"URDF file does not exist: {urdf_path}")
        return result

    # Parse XML
    try:
        tree = ET.parse(urdf_path)
        root = tree.getroot()
    except ET.ParseError as e:
        result.add_error(f"XML parse error: {e}")
        return result

    # Check root element
    if root.tag != "robot":
        result.add_error(f"Root element must be 'robot', found '{root.tag}'")
        return result

    robot_name = root.attrib.get("name", "")
    if not robot_name:
        result.add_warning("Robot has no name attribute")

    # Collect link names
    link_names = set()
    for link in root.findall("link"):
        link_name = link.attrib.get("name")
        if not link_name:
            result.add_error("Found link without name attribute")
            continue
        if link_name in link_names:
            result.add_warning(f"Duplicate link name: {link_name}")
        link_names.add(link_name)

    if not link_names:
        result.add_error("URDF has no links defined")
        return result

    if "base_link" not in link_names:
        result.add_warning("No 'base_link' found (recommended for Isaac Sim)")

    # Validate links
    for link in root.findall("link"):
        link_name = link.attrib.get("name", "unknown")

        # Check visual geometry
        visual = link.find("visual")
        if visual is not None:
            geom = visual.find("geometry")
            if geom is not None:
                mesh = geom.find("mesh")
                if mesh is not None and mesh_dir:
                    mesh_file = mesh.attrib.get("filename", "")
                    if mesh_file:
                        mesh_path = mesh_dir / mesh_file
                        if not mesh_path.is_file():
                            result.add_warning(f"Link '{link_name}': mesh file not found: {mesh_file}")

        # Check inertial
        inertial = link.find("inertial")
        if inertial is not None:
            mass = inertial.find("mass")
            if mass is not None:
                try:
                    mass_val = float(mass.attrib.get("value", "0"))
                    if mass_val <= 0:
                        result.add_warning(f"Link '{link_name}': mass must be positive, got {mass_val}")
                except ValueError:
                    result.add_error(f"Link '{link_name}': invalid mass value")

            inertia = inertial.find("inertia")
            if inertia is not None:
                # Check for positive diagonal elements
                for attr in ["ixx", "iyy", "izz"]:
                    try:
                        val = float(inertia.attrib.get(attr, "0"))
                        if val < 0:
                            result.add_warning(f"Link '{link_name}': {attr} should be non-negative")
                    except ValueError:
                        result.add_error(f"Link '{link_name}': invalid {attr} value")

    # Validate joints
    for joint in root.findall("joint"):
        joint_name = joint.attrib.get("name", "unnamed")
        joint_type = joint.attrib.get("type", "")

        if not joint_type:
            result.add_error(f"Joint '{joint_name}': missing type attribute")
            continue

        valid_types = ["revolute", "continuous", "prismatic", "fixed", "floating", "planar"]
        if joint_type not in valid_types:
            result.add_warning(f"Joint '{joint_name}': unknown type '{joint_type}'")

        # Check parent/child references
        parent = joint.find("parent")
        child = joint.find("child")

        if parent is None:
            result.add_error(f"Joint '{joint_name}': missing parent element")
        else:
            parent_link = parent.attrib.get("link", "")
            if parent_link not in link_names:
                result.add_error(f"Joint '{joint_name}': parent link '{parent_link}' not defined")

        if child is None:
            result.add_error(f"Joint '{joint_name}': missing child element")
        else:
            child_link = child.attrib.get("link", "")
            if child_link not in link_names:
                result.add_error(f"Joint '{joint_name}': child link '{child_link}' not defined")

        # Check joint limits for limited types
        if joint_type in ["revolute", "prismatic"]:
            limit = joint.find("limit")
            if limit is None:
                result.add_warning(f"Joint '{joint_name}': {joint_type} joint should have limits")
            else:
                try:
                    lower = float(limit.attrib.get("lower", "0"))
                    upper = float(limit.attrib.get("upper", "0"))
                    if lower >= upper:
                        result.add_warning(f"Joint '{joint_name}': lower limit ({lower}) >= upper limit ({upper})")
                except ValueError:
                    result.add_error(f"Joint '{joint_name}': invalid limit values")

        # Check axis for non-fixed joints
        if joint_type not in ["fixed"]:
            axis = joint.find("axis")
            if axis is not None:
                xyz = axis.attrib.get("xyz", "")
                if xyz:
                    try:
                        vals = [float(v) for v in xyz.split()]
                        if len(vals) != 3:
                            result.add_error(f"Joint '{joint_name}': axis xyz must have 3 values")
                        elif all(v == 0 for v in vals):
                            result.add_warning(f"Joint '{joint_name}': axis is zero vector")
                    except ValueError:
                        result.add_error(f"Joint '{joint_name}': invalid axis xyz values")

    return result


def repair_urdf(urdf_path: Path, output_path: Optional[Path] = None) -> Tuple[bool, URDFValidationResult]:
    """
    Attempt to repair common URDF issues.

    Repairs:
    - Missing inertial elements (adds reasonable defaults)
    - Zero or negative mass (sets to 1.0)
    - Missing joint limits (adds defaults based on type)
    - Zero axis vectors (sets to 0 0 1)

    Args:
        urdf_path: Path to the URDF file to repair
        output_path: Path to write repaired URDF (defaults to overwrite)

    Returns:
        (success, validation_result) tuple
    """
    result = URDFValidationResult()
    output_path = output_path or urdf_path

    if not urdf_path.is_file():
        result.add_error(f"URDF file does not exist: {urdf_path}")
        return False, result

    try:
        tree = ET.parse(urdf_path)
        root = tree.getroot()
    except ET.ParseError as e:
        result.add_error(f"Cannot parse URDF for repair: {e}")
        return False, result

    if root.tag != "robot":
        result.add_error("Cannot repair: root is not 'robot'")
        return False, result

    modified = False

    # Repair links
    for link in root.findall("link"):
        link_name = link.attrib.get("name", "unknown")

        # Add missing inertial
        inertial = link.find("inertial")
        if inertial is None:
            inertial = ET.SubElement(link, "inertial")
            ET.SubElement(inertial, "mass", {"value": "1.0"})
            ET.SubElement(inertial, "inertia", {
                "ixx": "0.1", "ixy": "0", "ixz": "0",
                "iyy": "0.1", "iyz": "0", "izz": "0.1"
            })
            result.add_repair(f"Added default inertial to link '{link_name}'")
            modified = True
        else:
            # Fix zero/negative mass
            mass = inertial.find("mass")
            if mass is not None:
                try:
                    mass_val = float(mass.attrib.get("value", "0"))
                    if mass_val <= 0:
                        mass.set("value", "1.0")
                        result.add_repair(f"Fixed non-positive mass in link '{link_name}'")
                        modified = True
                except ValueError:
                    mass.set("value", "1.0")
                    result.add_repair(f"Fixed invalid mass in link '{link_name}'")
                    modified = True
            else:
                ET.SubElement(inertial, "mass", {"value": "1.0"})
                result.add_repair(f"Added missing mass to link '{link_name}'")
                modified = True

            # Add missing inertia
            inertia = inertial.find("inertia")
            if inertia is None:
                ET.SubElement(inertial, "inertia", {
                    "ixx": "0.1", "ixy": "0", "ixz": "0",
                    "iyy": "0.1", "iyz": "0", "izz": "0.1"
                })
                result.add_repair(f"Added default inertia to link '{link_name}'")
                modified = True

    # Repair joints
    for joint in root.findall("joint"):
        joint_name = joint.attrib.get("name", "unnamed")
        joint_type = joint.attrib.get("type", "fixed")

        # Fix missing axis for non-fixed joints
        if joint_type not in ["fixed"]:
            axis = joint.find("axis")
            if axis is None:
                ET.SubElement(joint, "axis", {"xyz": "0 0 1"})
                result.add_repair(f"Added default axis to joint '{joint_name}'")
                modified = True
            else:
                xyz = axis.attrib.get("xyz", "")
                if xyz:
                    try:
                        vals = [float(v) for v in xyz.split()]
                        if all(v == 0 for v in vals):
                            axis.set("xyz", "0 0 1")
                            result.add_repair(f"Fixed zero axis in joint '{joint_name}'")
                            modified = True
                    except ValueError:
                        axis.set("xyz", "0 0 1")
                        result.add_repair(f"Fixed invalid axis in joint '{joint_name}'")
                        modified = True

        # Add missing limits for limited joints
        if joint_type in ["revolute", "prismatic"]:
            limit = joint.find("limit")
            if limit is None:
                if joint_type == "revolute":
                    # Default to ±90 degrees
                    ET.SubElement(joint, "limit", {
                        "lower": "-1.57", "upper": "1.57",
                        "effort": "100", "velocity": "1.0"
                    })
                else:  # prismatic
                    # Default to 0-0.5m
                    ET.SubElement(joint, "limit", {
                        "lower": "0", "upper": "0.5",
                        "effort": "100", "velocity": "0.5"
                    })
                result.add_repair(f"Added default limits to joint '{joint_name}'")
                modified = True

    if modified:
        # Write repaired URDF
        tree.write(output_path, encoding="unicode", xml_declaration=True)
        result.is_valid = True
        log(f"Repaired URDF: {len(result.repairs_made)} fixes applied")
    else:
        result.is_valid = True  # No repairs needed

    return True, result


def validate_and_repair_urdf(
    urdf_path: Path,
    mesh_dir: Optional[Path] = None,
    auto_repair: bool = True,
    obj_id: str = ""
) -> URDFValidationResult:
    """
    Validate URDF and optionally repair common issues.

    This is the main entry point for URDF validation in the pipeline.

    Args:
        urdf_path: Path to the URDF file
        mesh_dir: Optional directory to check mesh references
        auto_repair: If True, attempt to repair issues
        obj_id: Object ID for logging

    Returns:
        Combined validation result
    """
    # First validation pass
    result = validate_urdf(urdf_path, mesh_dir)

    if result.errors:
        log(f"URDF validation errors: {result.errors}", "ERROR", obj_id)
    if result.warnings:
        log(f"URDF validation warnings: {result.warnings}", "WARNING", obj_id)

    # Attempt repair if needed and allowed
    if auto_repair and (result.errors or result.warnings):
        log("Attempting URDF repair...", "INFO", obj_id)
        success, repair_result = repair_urdf(urdf_path)

        if success and repair_result.repairs_made:
            # Re-validate after repair
            result = validate_urdf(urdf_path, mesh_dir)
            result.repairs_made = repair_result.repairs_made
            log(f"URDF repairs applied: {repair_result.repairs_made}", "INFO", obj_id)

    return result


def parse_urdf_summary(urdf_path: Path) -> Dict[str, List[Dict]]:
    """Extract joint/link summary from URDF file."""
    summary: Dict[str, List[Dict]] = {"joints": [], "links": []}

    if not urdf_path.is_file():
        return summary

    try:
        tree = ET.parse(urdf_path)
        root = tree.getroot()
    except Exception as e:
        log(f"Failed to parse URDF: {e}", "WARNING")
        return summary

    # Extract links
    for link in root.findall("link"):
        link_data = {"name": link.attrib.get("name")}

        inertial = link.find("inertial")
        if inertial is not None:
            mass_elem = inertial.find("mass")
            if mass_elem is not None and mass_elem.attrib.get("value"):
                try:
                    link_data["mass"] = float(mass_elem.attrib["value"])
                except ValueError:
                    pass

        summary["links"].append(link_data)

    # Extract joints
    for joint in root.findall("joint"):
        joint_data = {
            "name": joint.attrib.get("name"),
            "type": joint.attrib.get("type"),
        }

        parent = joint.find("parent")
        child = joint.find("child")
        axis = joint.find("axis")
        limit = joint.find("limit")

        if parent is not None:
            joint_data["parent"] = parent.attrib.get("link")
        if child is not None:
            joint_data["child"] = child.attrib.get("link")
        if axis is not None:
            joint_data["axis"] = axis.attrib.get("xyz")
        if limit is not None:
            try:
                joint_data["lower"] = float(limit.attrib.get("lower", ""))
            except (ValueError, TypeError):
                pass
            try:
                joint_data["upper"] = float(limit.attrib.get("upper", ""))
            except (ValueError, TypeError):
                pass

        summary["joints"].append(joint_data)

    return summary


def generate_static_urdf(obj_id: str, mesh_filename: str = "mesh.glb") -> str:
    """
    Generate a static URDF for objects without articulation.

    This is used as a fallback when Particulate doesn't detect any joints.
    """
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<robot name="{obj_id}">
  <link name="base_link">
    <visual>
      <geometry>
        <mesh filename="{mesh_filename}"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <mesh filename="{mesh_filename}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>
</robot>
"""


def generate_placeholder_urdf(
    obj_id: str,
    reason: str = "service_unavailable",
    disallow_placeholder: bool = False,
) -> str:
    """Generate placeholder URDF when processing fails."""
    if disallow_placeholder:
        raise_placeholder_error(obj_id, reason)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!-- Placeholder URDF: {reason} -->
<robot name="{obj_id}_placeholder">
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.1 0.1 0.1"/>
      </geometry>
    </visual>
  </link>
</robot>
"""


def _load_heuristic_articulation_module() -> Any:
    """Load heuristic articulation module lazily."""
    module_path = REPO_ROOT / "interactive-job" / "heuristic_articulation.py"
    spec = importlib.util.spec_from_file_location("heuristic_articulation", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load heuristic articulation module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dimensions_from_object_metadata(obj: Dict[str, Any]) -> Optional[Any]:
    """Extract [width, depth, height] from object metadata when available."""
    dims = obj.get("dimensions_est")
    if isinstance(dims, dict):
        width = dims.get("width")
        depth = dims.get("depth")
        height = dims.get("height")
        if all(v is not None for v in (width, depth, height)):
            try:
                import numpy as np

                return np.array([float(width), float(depth), float(height)], dtype=float)
            except (TypeError, ValueError):
                return None
            except ImportError:
                return None
    return None


def materialize_heuristic_articulation(
    obj: Dict[str, Any],
    obj_name: str,
    obj_class: str,
    glb_path: Path,
    output_dir: Path,
) -> Tuple[Path, Path, Dict[str, Any]]:
    """Generate heuristic articulation output (mesh + URDF)."""
    ensure_dir(output_dir)
    mesh_path = output_dir / "mesh.glb"
    shutil.copy(glb_path, mesh_path)

    urdf_path = output_dir / f"{obj_name}.urdf"
    metadata: Dict[str, Any] = {
        "placeholder": False,
        "generator": "heuristic",
    }

    module = _load_heuristic_articulation_module()
    detector = module.HeuristicArticulationDetector()
    dims = _dimensions_from_object_metadata(obj)

    spec = detector.detect(
        object_id=obj_name,
        object_category=obj_class,
        object_dimensions=dims,
    )
    if spec is None:
        urdf_path.write_text(generate_static_urdf(obj_name, mesh_path.name), encoding="utf-8")
        metadata["fallback_static"] = True
        metadata["reason"] = "no_heuristic_pattern_match"
        return mesh_path, urdf_path, metadata

    ok = module.generate_urdf_from_spec(
        spec=spec,
        mesh_path=Path(mesh_path.name),
        output_path=urdf_path,
    )
    if not ok:
        urdf_path.write_text(generate_static_urdf(obj_name, mesh_path.name), encoding="utf-8")
        metadata["fallback_static"] = True
        metadata["reason"] = "heuristic_urdf_generation_failed"
        return mesh_path, urdf_path, metadata

    metadata["joint_type"] = spec.joint_type.value
    metadata["confidence"] = float(spec.confidence)
    return mesh_path, urdf_path, metadata


# =============================================================================
# Asset Materialization
# =============================================================================

def materialize_articulation_response(
    response: dict,
    output_dir: Path,
    obj_id: str,
    disallow_placeholder: bool = False,
) -> Tuple[Optional[Path], Optional[Path], Dict[str, Any]]:
    """
    Materialize Particulate articulation response to disk.

    Returns:
        (mesh_path, urdf_path, metadata)
    """
    ensure_dir(output_dir)

    mesh_path = output_dir / "part.glb"
    urdf_path = output_dir / f"{obj_id}.urdf"

    meta: Dict[str, Any] = {
        "placeholder": response.get("placeholder", False),
        "generator": response.get("generator", "particulate"),
    }

    # Handle placeholder response
    if response.get("placeholder"):
        if disallow_placeholder:
            raise_placeholder_error(obj_id, "particulate_placeholder_response")
        return None, None, meta

    # Decode mesh
    if response.get("mesh_base64"):
        try:
            mesh_bytes = base64.b64decode(response["mesh_base64"])
            mesh_path.write_bytes(mesh_bytes)
            log(f"Wrote mesh: {len(mesh_bytes)} bytes", obj_id=obj_id)
        except Exception as e:
            log(f"Failed to decode mesh: {e}", "ERROR", obj_id)
            return None, None, meta
    else:
        return None, None, meta

    # Decode URDF
    if response.get("urdf_base64"):
        try:
            urdf_bytes = base64.b64decode(response["urdf_base64"])
            urdf_path.write_bytes(urdf_bytes)
            log(f"Wrote URDF: {len(urdf_bytes)} bytes", obj_id=obj_id)
        except Exception as e:
            log(f"Failed to decode URDF: {e}", "ERROR", obj_id)
            return mesh_path, None, meta
    else:
        # Generate static URDF if not provided
        urdf_content = generate_static_urdf(obj_id, mesh_path.name)
        urdf_path.write_text(urdf_content, encoding="utf-8")
        meta["urdf_generated"] = True

    return mesh_path, urdf_path, meta


# =============================================================================
# Object Processing
# =============================================================================

def process_object(
    obj: dict,
    assets_root: Path,
    regen3d_root: Path,
    multiview_root: Path,
    particulate_endpoint: Optional[str],
    particulate_mode: str,
    articulation_backend: str,
    mode: str,
    disallow_placeholder_urdf: bool,
    multiview_enabled: bool,
    multiview_count: int,
    multiview_model: str,
    index: int,
    total: int,
) -> dict:
    """
    Process a single interactive object using configured articulation backend.

    Pipeline:
    1. Find GLB mesh from 3D-RE-GEN
    2. Run Particulate or heuristic articulation
    3. Materialize outputs (mesh + URDF)
    4. Generate manifest with joint summary
    """
    obj_id = str(obj.get("id"))
    obj_name = f"obj_{obj_id}"
    obj_class = obj.get("class_name", "unknown")

    log(f"Processing {index + 1}/{total}: {obj_class}", obj_id=obj_name)

    # Output directory
    output_dir = assets_root / "interactive" / obj_name
    ensure_dir(output_dir)

    # Result structure
    required_articulation = is_required_articulation_object(obj)
    articulation_hint = obj.get("articulation_hint")
    result = {
        "id": obj_id,
        "name": obj_name,
        "class_name": obj_class,
        "status": "pending",
        "mode": mode,
        "backend": articulation_backend,
        "particulate_mode": particulate_mode,
        "output_dir": str(output_dir),
        "mesh_path": None,
        "urdf_path": None,
        "joint_count": 0,
        "is_articulated": False,
        "placeholder": False,
        "required_articulation": required_articulation,
        "articulation_hint": articulation_hint,
    }

    # Find GLB mesh from 3D-RE-GEN
    glb_path: Optional[Path] = None

    if mode == MODE_GLB:
        regen3d_obj_dir = regen3d_root / obj_name
        glb_path = find_glb_file(regen3d_obj_dir, obj_id)

        if glb_path:
            log(f"Found GLB: {glb_path}", obj_id=obj_name)
            result["input_glb"] = str(glb_path)
        else:
            log(f"No GLB found in {regen3d_obj_dir}", "WARNING", obj_name)

    # Particulate requires GLB - if not found, try alternative locations
    if not glb_path or not glb_path.is_file():
        # Try assets root
        alt_paths = [
            assets_root / obj_name / f"obj_{obj_id}.glb",
            assets_root / obj_name / "mesh.glb",
            assets_root / "static" / obj_name / "mesh.glb",
        ]
        for alt_path in alt_paths:
            if alt_path.is_file():
                glb_path = alt_path
                log(f"Found GLB at alternative location: {glb_path}", obj_id=obj_name)
                result["input_glb"] = str(glb_path)
                break

    # If still no GLB, generate static URDF
    if not glb_path or not glb_path.is_file():
        log(f"No GLB mesh found, generating static URDF", "WARNING", obj_name)
        result["status"] = "static"
        result["error"] = "no_glb_found"

        urdf_path = output_dir / f"{obj_name}.urdf"
        urdf_path.write_text(
            generate_placeholder_urdf(
                obj_name,
                "no_glb",
                disallow_placeholder=disallow_placeholder_urdf,
            ),
            encoding="utf-8",
        )
        result["urdf_path"] = str(urdf_path)
        result["placeholder"] = True
        result["placeholder_reason"] = "no_glb"
        return result

    if multiview_enabled and (required_articulation or articulation_hint):
        scaffold_metadata = generate_multiview_scaffold(
            obj_name=obj_name,
            obj_class=obj_class,
            glb_path=glb_path,
            output_dir=output_dir,
            view_count=multiview_count,
            model=multiview_model,
        )
        result["multiview_scaffold"] = scaffold_metadata

    resolved_backend = resolve_articulation_backend(
        requested_backend=articulation_backend,
        particulate_mode=particulate_mode,
        particulate_endpoint=particulate_endpoint or "",
    )
    result["backend"] = resolved_backend

    mesh_path: Optional[Path] = None
    urdf_path: Optional[Path] = None
    meta: Dict[str, Any] = {}

    if resolved_backend == ARTICULATION_BACKEND_HEURISTIC:
        log("Using heuristic articulation backend", obj_id=obj_name)
        try:
            mesh_path, urdf_path, meta = materialize_heuristic_articulation(
                obj=obj,
                obj_name=obj_name,
                obj_class=obj_class,
                glb_path=glb_path,
                output_dir=output_dir,
            )
            if meta.get("fallback_static"):
                result["status"] = "static"
        except Exception as e:
            log(f"Heuristic articulation failed: {e}", "ERROR", obj_name)
            result["status"] = "error"
            result["error"] = "heuristic_failed"

            mesh_out = output_dir / "mesh.glb"
            shutil.copy(glb_path, mesh_out)
            result["mesh_path"] = str(mesh_out)

            fallback_urdf_path = output_dir / f"{obj_name}.urdf"
            fallback_urdf_path.write_text(generate_static_urdf(obj_name), encoding="utf-8")
            result["urdf_path"] = str(fallback_urdf_path)
            return result
    else:
        if not particulate_endpoint and particulate_mode != PARTICULATE_MODE_MOCK:
            log("No Particulate endpoint configured, using static URDF", "WARNING", obj_name)
            result["status"] = "static"
            mesh_out = output_dir / "mesh.glb"
            shutil.copy(glb_path, mesh_out)
            result["mesh_path"] = str(mesh_out)

            urdf_path = output_dir / f"{obj_name}.urdf"
            urdf_path.write_text(generate_static_urdf(obj_name), encoding="utf-8")
            result["urdf_path"] = str(urdf_path)
            return result

        if particulate_mode == PARTICULATE_MODE_MOCK:
            mock_placeholder = env_flag(os.getenv("PARTICULATE_MOCK_PLACEHOLDER"), default=False)
            log(f"Building mock Particulate response (placeholder={mock_placeholder})", obj_id=obj_name)
            response = build_mock_particulate_response(glb_path, obj_name, mock_placeholder)
        else:
            log("Calling Particulate for articulation", obj_id=obj_name)
            response = call_particulate_service(particulate_endpoint, glb_path, obj_name)

        if not response:
            log("Particulate service call failed", "ERROR", obj_name)
            result["status"] = "error"
            result["error"] = "service_failed"

            mesh_out = output_dir / "mesh.glb"
            shutil.copy(glb_path, mesh_out)
            result["mesh_path"] = str(mesh_out)

            urdf_path = output_dir / f"{obj_name}.urdf"
            urdf_path.write_text(generate_static_urdf(obj_name), encoding="utf-8")
            result["urdf_path"] = str(urdf_path)
            return result

        mesh_path, urdf_path, meta = materialize_articulation_response(
            response,
            output_dir,
            obj_name,
            disallow_placeholder=disallow_placeholder_urdf,
        )

    if not mesh_path or not urdf_path:
        log("Materialization failed, using fallback", "WARNING", obj_name)
        result["status"] = "fallback"

        # Use original GLB if available
        if glb_path and glb_path.is_file():
            mesh_out = output_dir / "mesh.glb"
            shutil.copy(glb_path, mesh_out)
            result["mesh_path"] = str(mesh_out)
            mesh_path = mesh_out

        urdf_path = output_dir / f"{obj_name}.urdf"
        urdf_path.write_text(generate_static_urdf(obj_name, mesh_path.name if mesh_path else "mesh.glb"), encoding="utf-8")
        result["urdf_path"] = str(urdf_path)
        return result

    # Validate and repair URDF
    urdf_validation = validate_and_repair_urdf(
        urdf_path,
        mesh_dir=output_dir,
        auto_repair=True,
        obj_id=obj_name
    )

    # Simulation-backed validation (PyBullet/Isaac Sim)
    sim_validation = validate_urdf_in_simulator(
        urdf_path,
        mesh_dir=output_dir,
        obj_id=obj_name,
    )
    if sim_validation.errors:
        log(f"Simulation validation errors: {sim_validation.errors}", "ERROR", obj_id=obj_name)
    if sim_validation.warnings:
        log(f"Simulation validation warnings: {sim_validation.warnings}", "WARNING", obj_id=obj_name)

    downgraded_to_static = False
    if not sim_validation.skipped and not sim_validation.is_valid:
        log(
            "Invalid URDF in simulator; downgrading to static asset.",
            "WARNING",
            obj_id=obj_name,
        )
        urdf_path.write_text(
            generate_static_urdf(obj_name, mesh_path.name),
            encoding="utf-8",
        )
        result["status"] = "static"
        downgraded_to_static = True
        urdf_validation = validate_urdf(urdf_path, mesh_dir=output_dir)

    # Parse URDF for joint information
    joint_summary = parse_urdf_summary(urdf_path)
    joint_count = len(joint_summary.get("joints", []))

    # Build manifest
    manifest = {
        "object_id": obj_id,
        "object_name": obj_name,
        "class_name": obj_class,
        "backend": resolved_backend,
        "required_articulation": required_articulation,
        "articulation_hint": articulation_hint,
        "generator": meta.get("generator", "particulate"),
        "endpoint": particulate_endpoint,
        "particulate_mode": particulate_mode,
        "mesh_path": mesh_path.name,
        "urdf_path": urdf_path.name,
        "is_articulated": joint_count > 0,
        "joint_summary": joint_summary,
        "input_mode": mode,
        "urdf_validation": {
            "is_valid": urdf_validation.is_valid,
            "errors": urdf_validation.errors,
            "warnings": urdf_validation.warnings,
            "repairs_made": urdf_validation.repairs_made,
        },
        "simulation_validation": {
            "backend": sim_validation.backend,
            "is_valid": sim_validation.is_valid,
            "errors": sim_validation.errors,
            "warnings": sim_validation.warnings,
            "skipped": sim_validation.skipped,
        },
        "downgraded_to_static": downgraded_to_static,
    }
    if result.get("multiview_scaffold"):
        manifest["multiview_scaffold"] = result["multiview_scaffold"]

    if glb_path:
        manifest["input_glb"] = str(glb_path)

    # Save manifest
    manifest_path = output_dir / "interactive_manifest.json"
    save_json(manifest, manifest_path)

    # Update result
    if result["status"] == "pending":
        result["status"] = "ok"
    result["mesh_path"] = str(mesh_path)
    result["urdf_path"] = str(urdf_path)
    result["manifest_path"] = str(manifest_path)
    result["joint_count"] = joint_count
    result["is_articulated"] = joint_count > 0
    result["urdf_valid"] = urdf_validation.is_valid
    result["urdf_repairs"] = len(urdf_validation.repairs_made)
    result["downgraded_to_static"] = downgraded_to_static

    log(f"Completed: {joint_count} joints detected, articulated={joint_count > 0}, urdf_valid={urdf_validation.is_valid}", obj_id=obj_name)

    return result


# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> None:
    """Main entry point for interactive asset processing using Particulate."""

    # Configuration from environment
    validate_required_env_vars(
        {
            "BUCKET": "GCS bucket name",
            "SCENE_ID": "Scene identifier",
        },
        label="[INTERACTIVE]",
    )

    bucket = os.environ["BUCKET"]
    scene_id = os.environ["SCENE_ID"]
    assets_prefix = os.getenv("ASSETS_PREFIX", "")
    multiview_prefix = os.getenv("MULTIVIEW_PREFIX", "")
    regen3d_prefix = os.getenv("REGEN3D_PREFIX", "")  # 3D-RE-GEN output path

    # Particulate endpoint
    particulate_endpoint = os.getenv("PARTICULATE_ENDPOINT", "")
    particulate_mode = normalize_particulate_mode(os.getenv("PARTICULATE_MODE", PARTICULATE_MODE_REMOTE))
    particulate_local_endpoint = os.getenv("PARTICULATE_LOCAL_ENDPOINT", "")
    particulate_endpoint, endpoint_source = resolve_particulate_endpoint(
        particulate_mode,
        particulate_endpoint,
        particulate_local_endpoint,
    )

    mode = os.getenv("INTERACTIVE_MODE", MODE_GLB)  # Default to GLB mode
    production_mode, production_source, production_value = resolve_interactive_production_mode()
    labs_mode = env_flag(os.getenv("LABS_MODE"), default=False)
    disallow_placeholder_env = env_flag(os.getenv("DISALLOW_PLACEHOLDER_URDF"), default=False)
    disallow_placeholder_urdf = production_mode or disallow_placeholder_env
    articulation_backend = normalize_articulation_backend(
        os.getenv("ARTICULATION_BACKEND", ARTICULATION_BACKEND_AUTO)
    )
    multiview_enabled = env_flag(os.getenv("ARTICULATION_MULTIVIEW_ENABLED"), default=False)
    try:
        multiview_count = max(1, int(os.getenv("ARTICULATION_MULTIVIEW_COUNT", "4")))
    except ValueError:
        multiview_count = 4
    multiview_model = os.getenv("ARTICULATION_MULTIVIEW_MODEL", "gemini-3-pro-image-preview")
    local_model_id = os.getenv("PARTICULATE_LOCAL_MODEL", "").strip()
    approved_models = parse_csv_env(os.getenv("APPROVED_PARTICULATE_MODELS", "pat_b"))
    failure_writer = FailureMarkerWriter(
        bucket=bucket,
        scene_id=scene_id or "unknown",
        job_name="interactive-job",
        base_path=assets_prefix or None,
    )

    if not assets_prefix:
        log("ASSETS_PREFIX is required", "ERROR")
        sys.exit(1)

    # Setup paths
    root = Path("/mnt/gcs")
    assets_root = root / assets_prefix
    multiview_root = root / multiview_prefix if multiview_prefix else assets_root / "multiview"
    regen3d_root = root / regen3d_prefix if regen3d_prefix else assets_root / "regen3d"

    # Load scene assets manifest (prefer canonical scene_manifest.json)
    scene_assets = load_manifest_or_scene_assets(assets_root)
    if scene_assets is None:
        manifest_path = assets_root / "scene_manifest.json"
        legacy_path = assets_root / "scene_assets.json"
        log(
            f"scene manifest not found at {manifest_path} or {legacy_path}",
            "ERROR",
        )
        sys.exit(1)
    objects = scene_assets.get("objects", [])

    process_all_interactive = env_flag(os.getenv("INTERACTIVE_PROCESS_ALL"), default=False)
    all_interactive_objects = [o for o in objects if o.get("type") == "interactive"]
    if process_all_interactive:
        interactive_objects = list(all_interactive_objects)
    else:
        interactive_objects = [o for o in all_interactive_objects if is_articulation_candidate_object(o)]
    required_interactive_objects = [o for o in interactive_objects if is_required_articulation_object(o)]
    interactive_summary_base = {
        "total_objects_in_manifest": len(objects),
        "interactive_objects_in_manifest": len(all_interactive_objects),
        "candidate_interactive_count": len(interactive_objects),
        "required_interactive_count": len(required_interactive_objects),
        "process_all_interactive": process_all_interactive,
    }

    # Print configuration
    log("=" * 60)
    log("Interactive Asset Pipeline (Particulate)")
    log("=" * 60)
    log(f"Bucket: {bucket}")
    log(f"Scene ID: {scene_id}")
    log(f"Assets: {assets_root}")
    log(f"3D-RE-GEN: {regen3d_root}")
    log(f"Multiview: {multiview_root}")
    log(f"Particulate Mode: {particulate_mode}")
    log(f"Particulate Endpoint: {particulate_endpoint or '(none - static mode)'}")
    log(f"Mode: {mode}")
    log(f"Interactive objects in manifest: {len(all_interactive_objects)}")
    log(f"Interactive candidate objects: {len(interactive_objects)}")
    log(f"Required articulation objects: {len(required_interactive_objects)}")
    log(f"INTERACTIVE_PROCESS_ALL: {process_all_interactive}")
    log(f"Production mode: {production_mode}")
    if production_mode:
        source_value = production_value.strip() if isinstance(production_value, str) else production_value
        log(f"Production mode source: {production_source}={source_value}")
    log(f"Labs mode: {labs_mode}")
    log(f"Disallow placeholder URDFs: {disallow_placeholder_urdf}")
    log(f"Articulation backend: {articulation_backend}")
    log(f"Multiview scaffold enabled: {multiview_enabled}")
    if multiview_enabled:
        log(f"Multiview scaffold config: count={multiview_count}, model={multiview_model}")
    log("=" * 60)

    config_context = {
        "assets_prefix": assets_prefix,
        "multiview_prefix": multiview_prefix or None,
        "regen3d_prefix": regen3d_prefix or None,
        "mode": mode,
        "particulate_mode": particulate_mode,
        "production_mode": production_mode,
        "production_mode_source": production_source,
        "production_mode_value": production_value,
        "labs_mode": labs_mode,
        "disallow_placeholder_urdf": disallow_placeholder_urdf,
        "particulate_endpoint": particulate_endpoint or None,
        "particulate_endpoint_source": endpoint_source,
        "articulation_backend": articulation_backend,
        "resolved_articulation_backend": resolve_articulation_backend(
            articulation_backend,
            particulate_mode,
            particulate_endpoint,
        ),
        "articulation_multiview_enabled": multiview_enabled,
        "articulation_multiview_count": multiview_count,
        "articulation_multiview_model": multiview_model,
        "particulate_local_model": local_model_id or None,
    }

    guardrails_enabled = production_mode or labs_mode

    resolved_backend = resolve_articulation_backend(
        articulation_backend,
        particulate_mode,
        particulate_endpoint,
    )

    if guardrails_enabled and resolved_backend == ARTICULATION_BACKEND_HEURISTIC:
        log("Heuristic articulation outputs are not allowed in labs/production mode", "ERROR")
        write_failure_marker(
            assets_root,
            failure_writer,
            scene_id,
            reason="heuristic_articulation_blocked",
            details={
                "articulation_backend": articulation_backend,
                "resolved_articulation_backend": resolved_backend,
            },
            config_context=config_context,
        )
        sys.exit(1)

    if guardrails_enabled and particulate_mode == PARTICULATE_MODE_LOCAL:
        if not local_model_id:
            log("PARTICULATE_LOCAL_MODEL is required for local Particulate in labs/production mode", "ERROR")
            write_failure_marker(
                assets_root,
                failure_writer,
                scene_id,
                reason="missing_particulate_local_model",
                details={"particulate_mode": particulate_mode},
                config_context=config_context,
            )
            sys.exit(1)
        if local_model_id not in approved_models:
            log(
                f"Local Particulate model '{local_model_id}' is not in the approved list",
                "ERROR",
            )
            write_failure_marker(
                assets_root,
                failure_writer,
                scene_id,
                reason="unapproved_particulate_model",
                details={
                    "particulate_local_model": local_model_id,
                    "approved_models": approved_models,
                },
                config_context=config_context,
            )
            sys.exit(1)

    if guardrails_enabled and particulate_mode != PARTICULATE_MODE_LOCAL and not particulate_endpoint:
        log("PARTICULATE_ENDPOINT is required in labs/production mode", "ERROR")
        write_failure_marker(
            assets_root,
            failure_writer,
            scene_id,
            reason="missing_particulate_endpoint",
            details={
                "mode": mode,
                "particulate_mode": particulate_mode,
            },
            config_context=config_context,
        )
        sys.exit(1)

    if particulate_mode == PARTICULATE_MODE_SKIP:
        if required_interactive_objects:
            details = {
                "required_objects": [str(o.get("id")) for o in required_interactive_objects],
                "particulate_mode": particulate_mode,
            }
            write_failure_marker(
                assets_root,
                failure_writer,
                scene_id,
                reason="required_articulation_unmet",
                details=details,
                config_context=config_context,
            )
            sys.exit(1)

        log("PARTICULATE_MODE=skip set; skipping articulation", "WARNING")
        warnings = [{
            "code": "particulate_skipped",
            "message": "Particulate inference skipped by configuration (PARTICULATE_MODE=skip).",
            "details": {
                "particulate_mode": particulate_mode,
            },
        }]
        results = [{
            "id": str(obj.get("id")),
            "name": f"obj_{obj.get('id')}",
            "class_name": obj.get("class_name", "unknown"),
            "status": "skipped",
            "mode": mode,
            "backend": resolved_backend,
            "particulate_mode": particulate_mode,
            "error": "particulate_skipped",
        } for obj in interactive_objects]

        results_path = assets_root / "interactive" / "interactive_results.json"
        ensure_dir(results_path.parent)
        save_json({
            "scene_id": scene_id,
            "total_objects": len(interactive_objects),
            "ok_count": 0,
            "articulated_count": 0,
            "error_count": 0,
            "fallback_count": 0,
            "skipped_count": len(interactive_objects),
            "mode": mode,
            "backend": resolved_backend,
            "particulate_mode": particulate_mode,
            "particulate_endpoint": particulate_endpoint or None,
            "objects": results,
        }, results_path)

        summary = {
            "total_objects": len(interactive_objects),
            "ok_count": 0,
            "articulated_count": 0,
            "error_count": 0,
            "fallback_count": 0,
            "skipped_count": len(interactive_objects),
            "mode": mode,
            "backend": resolved_backend,
            "particulate_mode": particulate_mode,
            "particulate_endpoint": particulate_endpoint or None,
            "required_interactive_count": len(required_interactive_objects),
        }
        write_status_marker(
            assets_root,
            scene_id,
            status="success",
            summary=summary,
            warnings=warnings,
        )
        write_interactive_summary(
            assets_root,
            scene_id,
            {
                **interactive_summary_base,
                "status": "success",
                "processed_count": len(interactive_objects),
                "ok_count": 0,
                "error_count": 0,
                "required_failure_count": 0,
            },
        )
        return

    # Early exit if no interactive objects
    if not interactive_objects:
        log("No interactive objects to process")

        # Write empty results
        results_path = assets_root / "interactive" / "interactive_results.json"
        ensure_dir(results_path.parent)
        save_json({
            "scene_id": scene_id,
            "total_objects": 0,
            "objects": [],
        }, results_path)

        summary = {
            "total_objects": 0,
            "ok_count": 0,
            "articulated_count": 0,
            "error_count": 0,
            "fallback_count": 0,
            "skipped_count": 0,
            "mode": mode,
            "backend": resolved_backend,
            "particulate_mode": particulate_mode,
            "particulate_endpoint": particulate_endpoint or None,
            "required_interactive_count": len(required_interactive_objects),
        }
        write_status_marker(
            assets_root,
            scene_id,
            status="success",
            summary=summary,
        )
        write_interactive_summary(
            assets_root,
            scene_id,
            {
                **interactive_summary_base,
                "status": "success",
                "processed_count": 0,
                "ok_count": 0,
                "error_count": 0,
                "required_failure_count": 0,
            },
        )

        log("Done (no objects)")
        return

    # Wait for Particulate service to be ready
    if resolved_backend == ARTICULATION_BACKEND_PARTICULATE and particulate_endpoint:
        log("Checking Particulate service health...")
        if not wait_for_particulate_ready(particulate_endpoint):
            if production_mode:
                log("Particulate service not ready in production mode", "ERROR")
                write_failure_marker(
                    assets_root,
                    failure_writer,
                    scene_id,
                    reason="particulate_unhealthy",
                    details={"endpoint": particulate_endpoint},
                    config_context=config_context,
                )
                sys.exit(1)
            log("Particulate service not ready, will attempt processing anyway", "WARNING")

    # Process each object
    results = []
    for i, obj in enumerate(interactive_objects):
        try:
            result = process_object(
                obj=obj,
                assets_root=assets_root,
                regen3d_root=regen3d_root,
                multiview_root=multiview_root,
                particulate_endpoint=particulate_endpoint,
                particulate_mode=particulate_mode,
                articulation_backend=articulation_backend,
                mode=mode,
                disallow_placeholder_urdf=disallow_placeholder_urdf,
                multiview_enabled=multiview_enabled,
                multiview_count=multiview_count,
                multiview_model=multiview_model,
                index=i,
                total=len(interactive_objects),
            )
            results.append(result)
        except Exception as e:
            log(f"Error processing obj_{obj.get('id')}: {e}", "ERROR")
            results.append({
                "id": obj.get("id"),
                "status": "error",
                "error": str(e),
                "required_articulation": is_required_articulation_object(obj),
            })

    # Summary statistics
    ok_count = sum(1 for r in results if r.get("status") == "ok")
    articulated_count = sum(1 for r in results if r.get("is_articulated"))
    error_count = sum(1 for r in results if r.get("status") == "error")
    fallback_count = sum(1 for r in results if r.get("status") in ("fallback", "static"))
    skipped_count = sum(1 for r in results if r.get("status") == "skipped")
    placeholder_results = [
        {
            "id": r.get("id"),
            "name": r.get("name"),
            "class_name": r.get("class_name"),
            "reason": r.get("placeholder_reason"),
            "urdf_path": r.get("urdf_path"),
        }
        for r in results
        if r.get("placeholder")
    ]
    error_payloads, warning_payloads = collect_result_payloads(results)
    required_failures = [
        {
            "id": r.get("id"),
            "status": r.get("status"),
            "backend": r.get("backend"),
            "error": r.get("error"),
        }
        for r in results
        if r.get("required_articulation") and not r.get("is_articulated")
    ]
    if required_failures:
        error_payloads.append(
            {
                "code": "required_articulation_unmet",
                "message": "Required articulated objects were not articulated.",
                "details": {"objects": required_failures},
            }
        )

    # Write results
    results_data = {
        "scene_id": scene_id,
        "total_objects": len(interactive_objects),
        "ok_count": ok_count,
        "articulated_count": articulated_count,
        "error_count": error_count,
        "fallback_count": fallback_count,
        "skipped_count": skipped_count,
        "mode": mode,
        "backend": resolved_backend,
        "particulate_mode": particulate_mode,
        "particulate_endpoint": particulate_endpoint or None,
        "objects": results,
    }

    results_path = assets_root / "interactive" / "interactive_results.json"
    ensure_dir(results_path.parent)
    save_json(results_data, results_path)
    log(f"Results written to {results_path}")

    if placeholder_results and not production_mode:
        write_placeholder_warning_marker(
            assets_root=assets_root,
            scene_id=scene_id,
            placeholders=placeholder_results,
        )

    if production_mode and ok_count == 0 and len(interactive_objects) > 0:
        log("No successful articulations in production mode", "ERROR")
        write_failure_marker(
            assets_root,
            failure_writer,
            scene_id,
            reason="articulation_failed",
            details={
                "total_objects": len(interactive_objects),
                "error_count": error_count,
                "fallback_count": fallback_count,
            },
            errors=error_payloads,
            warnings=warning_payloads,
            config_context=config_context,
        )
        sys.exit(1)

    summary = {
        "total_objects": len(interactive_objects),
        "ok_count": ok_count,
        "articulated_count": articulated_count,
        "error_count": error_count,
        "fallback_count": fallback_count,
        "skipped_count": skipped_count,
        "mode": mode,
        "backend": resolved_backend,
        "particulate_mode": particulate_mode,
        "particulate_endpoint": particulate_endpoint or None,
        "required_failures": required_failures,
        "required_interactive_count": len(required_interactive_objects),
    }

    if required_failures:
        status = "failure"
    elif len(interactive_objects) == 0:
        status = "success"
    elif ok_count == 0:
        status = "failure"
    elif error_count > 0:
        status = "partial"
    else:
        status = "success"

    write_status_marker(
        assets_root,
        scene_id,
        status=status,
        summary=summary,
        errors=error_payloads,
        warnings=warning_payloads,
    )
    write_interactive_summary(
        assets_root,
        scene_id,
        {
            **interactive_summary_base,
            "status": status,
            "processed_count": len(interactive_objects),
            "ok_count": ok_count,
            "error_count": error_count,
            "required_failure_count": len(required_failures),
            "required_failures": required_failures,
        },
    )
    if status == "failure":
        failure_reason = "required_articulation_unmet" if required_failures else "articulation_failed"
        write_failure_marker(
            assets_root,
            failure_writer,
            scene_id,
            reason=failure_reason,
            details=summary,
            errors=error_payloads,
            warnings=warning_payloads,
            config_context=config_context,
        )

    # Final summary
    log("=" * 60)
    log("SUMMARY")
    log("=" * 60)
    log(f"Total processed: {len(interactive_objects)}")
    log(f"OK: {ok_count}")
    log(f"Articulated: {articulated_count}")
    log(f"Fallback/Static: {fallback_count}")
    log(f"Errors: {error_count}")
    log("=" * 60)

    # Exit with error if all failed
    if ok_count == 0 and len(interactive_objects) > 0:
        log("WARNING: All objects failed or fell back to static!", "WARNING")
    if required_failures:
        log("ERROR: Required articulated objects did not receive articulation outputs.", "ERROR")
        sys.exit(1)


if __name__ == "__main__":
    from tools.startup_validation import validate_and_fail_fast

    validate_and_fail_fast(job_name="INTERACTIVE", validate_gcs=True)
    main()
