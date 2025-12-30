#!/usr/bin/env python3
"""
Interactive Asset Pipeline for ZeroScene Integration.

This job processes 3D assets (GLB meshes) from ZeroScene to add articulation
data using Particulate - a fast feed-forward mesh articulation model (~10s per object).

The pipeline:
1. Receives GLB meshes from ZeroScene (or crop images as fallback)
2. Sends GLB directly to Particulate for articulation
3. Outputs URDF with articulation data + segmented meshes

Designed for the ZeroScene pipeline:
    ZeroScene → interactive-job → simready-job → usd-assembly-job

Environment Variables:
    BUCKET: GCS bucket name
    SCENE_ID: Scene identifier
    ASSETS_PREFIX: Path to assets (contains scene_assets.json)
    PARTICULATE_ENDPOINT: Particulate Cloud Run service URL
    ZEROSCENE_PREFIX: Optional path to ZeroScene outputs (default: same as ASSETS_PREFIX)
    INTERACTIVE_MODE: "glb" (default) or "image" for legacy crop-based processing
"""
import base64
import json
import os
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

from tools.scene_manifest.loader import load_manifest_or_scene_assets


# =============================================================================
# Configuration
# =============================================================================

# Service timeouts
PARTICULATE_WARMUP_TIMEOUT = int(os.getenv("PARTICULATE_WARMUP_TIMEOUT", "120"))  # 2 min
PARTICULATE_REQUEST_TIMEOUT = 120  # 2 min (inference takes ~10s)

MAX_RETRIES = 3

# Processing modes
MODE_GLB = "glb"      # ZeroScene GLB mesh input (default)
MODE_IMAGE = "image"  # Legacy crop image input


# =============================================================================
# Logging
# =============================================================================

def log(msg: str, level: str = "INFO", obj_id: str = "") -> None:
    """Log with prefix and optional object ID."""
    prefix = f"[{obj_id}] " if obj_id else ""
    stream = sys.stderr if level in ("ERROR", "WARNING") else sys.stdout
    print(f"[INTERACTIVE] [{level}] {prefix}{msg}", file=stream, flush=True)


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

    ZeroScene outputs meshes as:
    - obj_{id}.glb (direct output)
    - mesh.glb (alternative naming)
    - part.glb (from PhysX-Anything)
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
        except:
            return False, f"HTTP {e.code}", None
    except urllib.error.URLError as e:
        return False, f"Network error: {e.reason}", None
    except Exception as e:
        return False, f"Error: {e}", None

def wait_for_particulate_ready(endpoint: str, max_wait: int = PARTICULATE_WARMUP_TIMEOUT) -> bool:
    """
    Wait for Particulate service to be ready.

    Particulate has much faster cold starts than PhysX-Anything (~1-2 min vs 10+ min).
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

            start = time.time()
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                elapsed = int(time.time() - start)
                log(f"Response: {resp.status} ({elapsed}s)", obj_id=obj_id)

                text = resp.read().decode("utf-8", errors="replace")
                data = json.loads(text)

                is_placeholder = data.get("placeholder", True)
                articulation = data.get("articulation", {})
                joint_count = articulation.get("joint_count", 0)

                log(f"Success: placeholder={is_placeholder}, joints={joint_count}", obj_id=obj_id)

                return data

        except urllib.error.HTTPError as e:
            error_body = ""
            try:
                error_body = e.read().decode("utf-8", errors="replace")[:1000]
            except:
                pass

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


# =============================================================================
# URDF Generation and Parsing
# =============================================================================

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

    This is used as a fallback when PhysX-Anything doesn't detect any joints.
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


def generate_placeholder_urdf(obj_id: str, reason: str = "service_unavailable") -> str:
    """Generate placeholder URDF when processing fails."""
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


# =============================================================================
# Asset Materialization
# =============================================================================

def materialize_articulation_response(
    response: dict,
    output_dir: Path,
    obj_id: str,
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
    zeroscene_root: Path,
    multiview_root: Path,
    particulate_endpoint: Optional[str],
    mode: str,
    index: int,
    total: int,
) -> dict:
    """
    Process a single interactive object using Particulate.

    Pipeline:
    1. Find GLB mesh from ZeroScene
    2. Send GLB to Particulate for articulation
    3. Materialize outputs (mesh + URDF)
    4. Generate manifest with joint summary

    Args:
        particulate_endpoint: Particulate service URL
    """
    obj_id = str(obj.get("id"))
    obj_name = f"obj_{obj_id}"
    obj_class = obj.get("class_name", "unknown")

    log(f"Processing {index + 1}/{total}: {obj_class}", obj_id=obj_name)

    # Output directory
    output_dir = assets_root / "interactive" / obj_name
    ensure_dir(output_dir)

    # Result structure
    result = {
        "id": obj_id,
        "name": obj_name,
        "class_name": obj_class,
        "status": "pending",
        "mode": mode,
        "backend": "particulate",
        "output_dir": str(output_dir),
        "mesh_path": None,
        "urdf_path": None,
        "joint_count": 0,
        "is_articulated": False,
    }

    # Find GLB mesh from ZeroScene
    glb_path: Optional[Path] = None

    if mode == MODE_GLB:
        zeroscene_obj_dir = zeroscene_root / obj_name
        glb_path = find_glb_file(zeroscene_obj_dir, obj_id)

        if glb_path:
            log(f"Found GLB: {glb_path}", obj_id=obj_name)
            result["input_glb"] = str(glb_path)
        else:
            log(f"No GLB found in {zeroscene_obj_dir}", "WARNING", obj_name)

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
        urdf_path.write_text(generate_placeholder_urdf(obj_name, "no_glb"), encoding="utf-8")
        result["urdf_path"] = str(urdf_path)
        return result

    # Check if Particulate endpoint is configured
    if not particulate_endpoint:
        log("No Particulate endpoint configured, using static URDF", "WARNING", obj_name)
        result["status"] = "static"

        # Copy GLB and generate static URDF
        import shutil
        mesh_out = output_dir / "mesh.glb"
        shutil.copy(glb_path, mesh_out)
        result["mesh_path"] = str(mesh_out)

        urdf_path = output_dir / f"{obj_name}.urdf"
        urdf_path.write_text(generate_static_urdf(obj_name), encoding="utf-8")
        result["urdf_path"] = str(urdf_path)
        return result

    # Call Particulate service
    log(f"Calling Particulate for articulation", obj_id=obj_name)
    response = call_particulate_service(particulate_endpoint, glb_path, obj_name)

    if not response:
        log("Particulate service call failed", "ERROR", obj_name)
        result["status"] = "error"
        result["error"] = "service_failed"

        # Fallback to static URDF with original mesh
        import shutil
        mesh_out = output_dir / "mesh.glb"
        shutil.copy(glb_path, mesh_out)
        result["mesh_path"] = str(mesh_out)

        urdf_path = output_dir / f"{obj_name}.urdf"
        urdf_path.write_text(generate_static_urdf(obj_name), encoding="utf-8")
        result["urdf_path"] = str(urdf_path)
        return result

    # Materialize response
    mesh_path, urdf_path, meta = materialize_articulation_response(response, output_dir, obj_name)

    if not mesh_path or not urdf_path:
        log("Materialization failed, using fallback", "WARNING", obj_name)
        result["status"] = "fallback"

        # Use original GLB if available
        if glb_path and glb_path.is_file():
            import shutil
            mesh_out = output_dir / "mesh.glb"
            shutil.copy(glb_path, mesh_out)
            result["mesh_path"] = str(mesh_out)
            mesh_path = mesh_out

        urdf_path = output_dir / f"{obj_name}.urdf"
        urdf_path.write_text(generate_static_urdf(obj_name, mesh_path.name if mesh_path else "mesh.glb"), encoding="utf-8")
        result["urdf_path"] = str(urdf_path)
        return result

    # Parse URDF for joint information
    joint_summary = parse_urdf_summary(urdf_path)
    joint_count = len(joint_summary.get("joints", []))

    # Build manifest
    manifest = {
        "object_id": obj_id,
        "object_name": obj_name,
        "class_name": obj_class,
        "generator": meta.get("generator", "physx-anything"),
        "endpoint": endpoint,
        "mesh_path": mesh_path.name,
        "urdf_path": urdf_path.name,
        "is_articulated": joint_count > 0,
        "joint_summary": joint_summary,
        "input_mode": mode,
    }

    if glb_path:
        manifest["input_glb"] = str(glb_path)
    if image_path:
        manifest["input_image"] = str(image_path)

    # Save manifest
    manifest_path = output_dir / "interactive_manifest.json"
    save_json(manifest, manifest_path)

    # Update result
    result["status"] = "ok"
    result["mesh_path"] = str(mesh_path)
    result["urdf_path"] = str(urdf_path)
    result["manifest_path"] = str(manifest_path)
    result["joint_count"] = joint_count
    result["is_articulated"] = joint_count > 0

    log(f"Completed: {joint_count} joints detected, articulated={joint_count > 0}", obj_id=obj_name)

    return result


# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> None:
    """Main entry point for interactive asset processing using Particulate."""

    # Configuration from environment
    bucket = os.getenv("BUCKET", "")
    scene_id = os.getenv("SCENE_ID", "")
    assets_prefix = os.getenv("ASSETS_PREFIX", "")
    multiview_prefix = os.getenv("MULTIVIEW_PREFIX", "")
    zeroscene_prefix = os.getenv("ZEROSCENE_PREFIX", "")  # ZeroScene output path

    # Particulate endpoint
    particulate_endpoint = os.getenv("PARTICULATE_ENDPOINT", "")

    mode = os.getenv("INTERACTIVE_MODE", MODE_GLB)  # Default to GLB mode

    if not assets_prefix:
        log("ASSETS_PREFIX is required", "ERROR")
        sys.exit(1)

    # Setup paths
    root = Path("/mnt/gcs")
    assets_root = root / assets_prefix
    multiview_root = root / multiview_prefix if multiview_prefix else assets_root / "multiview"
    zeroscene_root = root / zeroscene_prefix if zeroscene_prefix else assets_root / "zeroscene"

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

    # Filter interactive objects
    interactive_objects = [o for o in objects if o.get("type") == "interactive"]

    # Print configuration
    log("=" * 60)
    log("Interactive Asset Pipeline (Particulate)")
    log("=" * 60)
    log(f"Bucket: {bucket}")
    log(f"Scene ID: {scene_id}")
    log(f"Assets: {assets_root}")
    log(f"ZeroScene: {zeroscene_root}")
    log(f"Multiview: {multiview_root}")
    log(f"Particulate Endpoint: {particulate_endpoint or '(none - static mode)'}")
    log(f"Mode: {mode}")
    log(f"Interactive objects: {len(interactive_objects)}")
    log("=" * 60)

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

        # Write completion marker
        marker_path = assets_root / ".interactive_complete"
        marker_path.write_text("completed (no interactive objects)\n")

        log("Done (no objects)")
        return

    # Wait for Particulate service to be ready
    if particulate_endpoint:
        log("Checking Particulate service health...")
        if not wait_for_particulate_ready(particulate_endpoint):
            log("Particulate service not ready, will attempt processing anyway", "WARNING")

    # Process each object
    results = []
    for i, obj in enumerate(interactive_objects):
        try:
            result = process_object(
                obj=obj,
                assets_root=assets_root,
                zeroscene_root=zeroscene_root,
                multiview_root=multiview_root,
                particulate_endpoint=particulate_endpoint,
                mode=mode,
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
            })

    # Summary statistics
    ok_count = sum(1 for r in results if r.get("status") == "ok")
    articulated_count = sum(1 for r in results if r.get("is_articulated"))
    error_count = sum(1 for r in results if r.get("status") == "error")
    fallback_count = sum(1 for r in results if r.get("status") in ("fallback", "static"))

    # Write results
    results_data = {
        "scene_id": scene_id,
        "total_objects": len(interactive_objects),
        "ok_count": ok_count,
        "articulated_count": articulated_count,
        "error_count": error_count,
        "fallback_count": fallback_count,
        "mode": mode,
        "backend": "particulate",
        "particulate_endpoint": particulate_endpoint or None,
        "objects": results,
    }

    results_path = assets_root / "interactive" / "interactive_results.json"
    ensure_dir(results_path.parent)
    save_json(results_data, results_path)
    log(f"Results written to {results_path}")

    # Write completion marker
    marker_path = assets_root / ".interactive_complete"
    marker_path.write_text(f"completed: {ok_count} ok, {articulated_count} articulated, {error_count} errors\n")

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


if __name__ == "__main__":
    main()
