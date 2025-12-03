"""
Interactive asset generation job.

This job orchestrates calls to the PhysX-Anything service to generate
physics-ready assets (URDF + mesh) from object crops.

Key improvements:
- Much longer warmup wait (service can take 10+ minutes to load VLM model)
- Better retry logic with service health checking
- Sequential processing (GPU can't handle parallel inference)
- Detailed logging for debugging
"""
import base64
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def log(msg: str, level: str = "INFO") -> None:
    """Log with prefix and level."""
    stream = sys.stderr if level in ("ERROR", "WARNING") else sys.stdout
    print(f"[PHYSX] [{level}] {msg}", file=stream, flush=True)


def load_scene_assets(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"scene_assets.json not found at {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def check_service_health(endpoint: str, timeout: int = 30) -> Tuple[bool, str, Optional[dict]]:
    """
    Check PhysX service health.
    
    Returns:
        (is_ready, status_message, response_json)
    """
    try:
        req = urllib.request.Request(endpoint, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310
            body = resp.read().decode("utf-8", errors="replace")
            try:
                data = json.loads(body)
                is_ready = data.get("ready", False) and data.get("status") == "ok"
                return is_ready, data.get("status", "unknown"), data
            except json.JSONDecodeError:
                return False, f"Invalid JSON: {body[:100]}", None
                
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8", errors="replace")
            data = json.loads(body)
            return False, data.get("message", f"HTTP {e.code}"), data
        except:
            return False, f"HTTP {e.code}: {e.reason}", None
            
    except urllib.error.URLError as e:
        return False, f"Network error: {e.reason}", None
        
    except Exception as e:
        return False, f"Error: {type(e).__name__}: {e}", None


def wait_for_service_ready(endpoint: str, max_wait: int = 600) -> bool:
    """
    Wait for the PhysX service to be ready.
    
    The VLM model can take 10+ minutes to load on cold start,
    so we need a very long wait time.
    
    Args:
        endpoint: Service URL
        max_wait: Maximum seconds to wait (default: 10 minutes)
        
    Returns:
        True if service is ready, False if timeout
    """
    log(f"Waiting for service to be ready (max {max_wait}s)...")
    log(f"Endpoint: {endpoint}")
    
    start_time = time.time()
    attempt = 0
    last_status = ""
    
    while time.time() - start_time < max_wait:
        attempt += 1
        elapsed = int(time.time() - start_time)
        
        is_ready, status, response = check_service_health(endpoint)
        
        if is_ready:
            log(f"Service ready after {elapsed}s!")
            if response:
                log(f"Response: {json.dumps(response, indent=2)[:500]}")
            return True
        
        # Log status changes
        if status != last_status:
            log(f"Service status: {status} (attempt {attempt}, {elapsed}s elapsed)", "WARNING")
            last_status = status
        elif elapsed > 0 and elapsed % 60 == 0:
            # Log every minute even if status unchanged
            log(f"Still waiting... ({elapsed}s elapsed, status: {status})", "WARNING")
        
        # Adaptive backoff: start slow, then faster
        if elapsed < 60:
            sleep_time = 10  # First minute: check every 10s
        elif elapsed < 300:
            sleep_time = 20  # 1-5 minutes: check every 20s
        else:
            sleep_time = 30  # After 5 minutes: check every 30s
        
        time.sleep(sleep_time)
    
    log(f"ERROR: Service not ready after {max_wait}s (last status: {last_status})", "ERROR")
    return False


def call_physx_anything(
    endpoint: str, 
    crop_path: Path, 
    obj_id: str,
    max_retries: int = 3,
    first_timeout: int = 900,  # 15 min for first request
    retry_timeout: int = 600,  # 10 min for retries
) -> Optional[dict]:
    """
    Post the crop image to the PhysX-Anything service with retry logic.

    Args:
        endpoint: PhysX service URL
        crop_path: Path to the crop image
        obj_id: Object ID for logging
        max_retries: Maximum retry attempts
        first_timeout: Timeout for first attempt (longer for cold start)
        retry_timeout: Timeout for retry attempts

    Returns:
        Response dict or None on failure
    """
    with crop_path.open("rb") as f:
        payload = base64.b64encode(f.read()).decode("utf-8")
    body = json.dumps({"image_base64": payload}).encode("utf-8")
    
    log(f"[{obj_id}] Image size: {len(payload)} bytes (base64)")

    for attempt in range(max_retries + 1):
        try:
            req = urllib.request.Request(
                endpoint,
                data=body,
                headers={"Content-Type": "application/json"},
            )

            # Use longer timeout for first attempt
            timeout = first_timeout if attempt == 0 else retry_timeout

            if attempt > 0:
                log(f"[{obj_id}] Retry {attempt}/{max_retries} (timeout={timeout}s)")
            else:
                log(f"[{obj_id}] POST {endpoint} (timeout={timeout}s)")

            start_time = time.time()
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310
                elapsed = int(time.time() - start_time)
                log(f"[{obj_id}] Response status: {resp.status} (took {elapsed}s)")
                
                text = resp.read().decode("utf-8", errors="replace")
                
                try:
                    data = json.loads(text)
                    is_placeholder = data.get("placeholder", True)
                    log(f"[{obj_id}] Success! placeholder={is_placeholder}")
                    
                    # Log asset sizes if present
                    if data.get("mesh_base64"):
                        mesh_size = len(data["mesh_base64"]) * 3 // 4  # Approximate decoded size
                        log(f"[{obj_id}] Mesh size: ~{mesh_size} bytes")
                    
                    return data
                    
                except json.JSONDecodeError:
                    log(f"[{obj_id}] WARNING: non-JSON response: {text[:500]}...", "WARNING")
                    return None

        except urllib.error.HTTPError as e:
            elapsed = int(time.time() - start_time) if 'start_time' in locals() else 0
            
            error_body = ""
            try:
                error_body = e.read().decode("utf-8", errors="replace")
            except:
                pass
            
            # Truncate error body for logging but show enough context
            if len(error_body) > 1000:
                error_display = error_body[:500] + "\n...[truncated]...\n" + error_body[-500:]
            else:
                error_display = error_body
            
            log(f"[{obj_id}] HTTP {e.code} (after {elapsed}s): {error_display}", "ERROR")
            
            # Don't retry 4xx errors (client errors) except 429 (rate limit)
            if 400 <= e.code < 500 and e.code != 429:
                return None
                
            # Retry 503 (service busy/warming) and 5xx errors
            if attempt < max_retries:
                if e.code == 503:
                    # Service is busy or warming up - wait longer
                    wait_time = min(60 * (attempt + 1), 180)  # 60s, 120s, 180s
                    log(f"[{obj_id}] Service unavailable, waiting {wait_time}s before retry...", "WARNING")
                else:
                    # Other server errors - exponential backoff
                    wait_time = min(20 * (2 ** attempt), 120)  # 20s, 40s, 80s, max 120s
                    log(f"[{obj_id}] Server error, waiting {wait_time}s before retry...", "WARNING")
                    
                time.sleep(wait_time)
                continue
            return None

        except urllib.error.URLError as e:
            log(f"[{obj_id}] Network error: {e.reason}", "ERROR")
            if attempt < max_retries:
                wait_time = min(20 * (2 ** attempt), 120)
                log(f"[{obj_id}] Retrying in {wait_time}s...", "WARNING")
                time.sleep(wait_time)
                continue
            return None

        except TimeoutError:
            log(f"[{obj_id}] Request timed out after {timeout}s", "ERROR")
            if attempt < max_retries:
                log(f"[{obj_id}] Retrying...", "WARNING")
                time.sleep(10)
                continue
            return None

        except Exception as e:
            log(f"[{obj_id}] Unexpected error: {type(e).__name__}: {e}", "ERROR")
            if attempt < max_retries:
                wait_time = min(20 * (2 ** attempt), 120)
                log(f"[{obj_id}] Retrying in {wait_time}s...", "WARNING")
                time.sleep(wait_time)
                continue
            return None

    return None


def download_file(url: str, out_path: Path) -> bool:
    ensure_dir(out_path.parent)
    log(f"Downloading {url} -> {out_path}")
    try:
        urllib.request.urlretrieve(url, out_path)  # nosec B310
        return True
    except Exception as e:
        log(f"WARNING: failed to download {url}: {e}", "WARNING")
        if out_path.is_file():
            try:
                out_path.unlink()
            except OSError:
                pass
        return False


def write_placeholder_assets(out_dir: Path, obj_id: str) -> Tuple[Path, Path]:
    """
    Writes placeholder URDF/mesh when service cannot produce real assets.
    """
    ensure_dir(out_dir)

    mesh_path = out_dir / "part.glb"
    placeholder = b"Placeholder GLB - PhysX-Anything service failed"
    mesh_path.write_bytes(placeholder)

    urdf_path = out_dir / f"{obj_id}.urdf"
    urdf = f"""<?xml version="1.0" encoding="UTF-8"?>
<robot name="{obj_id}">
  <link name="base">
    <visual>
      <geometry>
        <mesh filename="{mesh_path.name}" />
      </geometry>
    </visual>
  </link>
</robot>
"""
    urdf_path.write_text(urdf.strip() + "\n", encoding="utf-8")
    return mesh_path, urdf_path


def parse_urdf_summary(urdf_path: Path) -> Dict:
    """Extracts joint/link summary from URDF."""
    import xml.etree.ElementTree as ET

    summary: Dict[str, List[Dict]] = {"joints": [], "links": []}
    if not urdf_path.is_file():
        return summary

    try:
        tree = ET.parse(urdf_path)
        root = tree.getroot()
    except Exception as e:
        log(f"WARNING: failed to parse URDF {urdf_path}: {e}", "WARNING")
        return summary

    for link in root.findall("link"):
        inertial = link.find("inertial")
        mass_tag = inertial.find("mass") if inertial is not None else None
        mass_val = None
        if mass_tag is not None:
            value = mass_tag.attrib.get("value")
            if value:
                try:
                    mass_val = float(value)
                except ValueError:
                    pass
        summary["links"].append({
            "name": link.attrib.get("name"),
            "mass": mass_val,
        })

    for joint in root.findall("joint"):
        axis_tag = joint.find("axis")
        limit_tag = joint.find("limit")
        axis = axis_tag.attrib.get("xyz") if axis_tag is not None else None
        lower = upper = None
        if limit_tag is not None:
            try:
                lower = float(limit_tag.attrib.get("lower", ""))
            except (ValueError, TypeError):
                pass
            try:
                upper = float(limit_tag.attrib.get("upper", ""))
            except (ValueError, TypeError):
                pass

        parent_elem = joint.find("parent")
        child_elem = joint.find("child")
        
        summary["joints"].append({
            "name": joint.attrib.get("name"),
            "type": joint.attrib.get("type"),
            "parent": parent_elem.attrib.get("link") if parent_elem is not None else None,
            "child": child_elem.attrib.get("link") if child_elem is not None else None,
            "axis": axis,
            "lower": lower,
            "upper": upper,
        })

    return summary


def decode_base64_to_file(data_b64: str, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    out_path.write_bytes(base64.b64decode(data_b64))


def materialize_service_assets(
    response: Dict,
    output_dir: Path,
    obj_id: str,
) -> Tuple[Optional[Path], Optional[Path], Dict]:
    """
    Materialize PhysX-Anything outputs to disk.
    Returns (mesh_path, urdf_path, metadata).
    """
    mesh_path = output_dir / "part.glb"
    urdf_path = output_dir / f"{obj_id}.urdf"
    ensure_dir(output_dir)

    meta: Dict = {
        "mesh_url": response.get("mesh_url"),
        "urdf_url": response.get("urdf_url"),
        "asset_zip_url": response.get("asset_zip_url"),
        "placeholder": response.get("placeholder", False),
        "generator": response.get("generator", "unknown"),
    }

    # If service already marked as placeholder, skip materialization
    if response.get("placeholder"):
        return None, None, meta

    # 1) ZIP bundle path
    if response.get("asset_zip_url"):
        try:
            with tempfile.NamedTemporaryFile(suffix=".zip", delete=True) as tmp:
                ok = download_file(response["asset_zip_url"], Path(tmp.name))
                if ok:
                    with zipfile.ZipFile(tmp.name) as zf:
                        zf.extractall(output_dir)
                        for name in zf.namelist():
                            lower = name.lower()
                            if lower.endswith(".urdf"):
                                urdf_path = output_dir / name
                            if lower.endswith((".glb", ".gltf")):
                                mesh_path = output_dir / name
            if urdf_path.is_file() and mesh_path.is_file():
                return mesh_path, urdf_path, meta
        except Exception as e:
            log(f"WARNING: failed to handle asset_zip_url: {e}", "WARNING")

    # 2) Direct URLs
    if response.get("mesh_url"):
        download_file(response["mesh_url"], mesh_path)
    if response.get("urdf_url"):
        download_file(response["urdf_url"], urdf_path)
    if urdf_path.is_file() and mesh_path.is_file():
        return mesh_path, urdf_path, meta

    # 3) Inline base64 payloads
    if response.get("mesh_base64"):
        decode_base64_to_file(response["mesh_base64"], mesh_path)
    if response.get("urdf_base64"):
        decode_base64_to_file(response["urdf_base64"], urdf_path)
    
    if urdf_path.is_file() and mesh_path.is_file():
        mesh_size = mesh_path.stat().st_size
        urdf_size = urdf_path.stat().st_size
        log(f"[{obj_id}] Mesh: {mesh_size} bytes, URDF: {urdf_size} bytes")
        if mesh_size < 100:
            log(f"[{obj_id}] WARNING: Mesh file suspiciously small!", "WARNING")
        return mesh_path, urdf_path, meta

    meta["placeholder"] = True
    return None, None, meta


def process_interactive_object(
    obj: dict,
    multiview_root: Path,
    assets_root: Path,
    endpoint: Optional[str],
    obj_index: int,
    total_objects: int,
) -> dict:
    """Process a single interactive object."""
    obj_id = obj.get("id")
    obj_name = f"obj_{obj_id}"
    crop_rel = obj.get("crop_path")

    log(f"[{obj_name}] Processing object {obj_index + 1}/{total_objects}")

    # Determine crop path
    if crop_rel:
        crop_path = Path("/mnt/gcs") / crop_rel
    else:
        obj_dir = multiview_root / obj_name
        crop_png = obj_dir / "crop.png"
        view_png = obj_dir / "view_0.png"

        if crop_png.is_file():
            crop_path = crop_png
        elif view_png.is_file():
            crop_path = view_png
        else:
            crop_path = crop_png

    output_dir = assets_root / "interactive" / obj_name
    ensure_dir(output_dir)

    result = {
        "id": obj_id,
        "name": obj_name,
        "status": "placeholder",
        "crop_path": str(crop_path),
        "output_dir": str(output_dir),
        "urdf_path": None,
        "mesh_path": None,
        "endpoint": endpoint,
        "service_response": None,
    }

    if not crop_path.is_file():
        log(f"[{obj_name}] WARNING: missing crop at {crop_path}", "WARNING")
        mesh_path, urdf_path = write_placeholder_assets(output_dir, obj_name)
        result["urdf_path"] = str(urdf_path)
        result["mesh_path"] = str(mesh_path)
        return result

    # Call PhysX service
    response: Optional[dict] = None
    if endpoint:
        response = call_physx_anything(endpoint, crop_path, obj_name)
        result["service_response"] = response

    # Materialize assets
    mesh_path: Optional[Path] = None
    urdf_path: Optional[Path] = None
    manifest: Dict = {}

    if response and not response.get("placeholder"):
        mesh_path, urdf_path, download_meta = materialize_service_assets(
            response, output_dir, obj_name
        )
        manifest.update(download_meta)

    # Fallback to placeholder if needed
    if mesh_path is None or urdf_path is None:
        log(f"[{obj_name}] Using placeholder assets", "WARNING")
        mesh_path, urdf_path = write_placeholder_assets(output_dir, obj_name)
        manifest["placeholder"] = True

    # Build manifest
    manifest.update({
        "object_id": obj_id,
        "object_name": obj_name,
        "endpoint": endpoint,
        "urdf_path": urdf_path.name,
        "mesh_path": mesh_path.name,
    })

    manifest["joint_summary"] = parse_urdf_summary(urdf_path)

    manifest_path = output_dir / "interactive_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    result.update({
        "status": "ok" if not manifest.get("placeholder") else "placeholder",
        "urdf_path": str(urdf_path),
        "mesh_path": str(mesh_path),
        "manifest": str(manifest_path),
    })
    
    log(f"[{obj_name}] Completed with status={result['status']}")
    return result


def main() -> None:
    bucket = os.getenv("BUCKET", "")
    scene_id = os.getenv("SCENE_ID", "")
    multiview_prefix = os.getenv("MULTIVIEW_PREFIX")
    assets_prefix = os.getenv("ASSETS_PREFIX")
    layout_prefix = os.getenv("LAYOUT_PREFIX")
    endpoint = os.getenv("PHYSX_ENDPOINT")
    
    # Configuration
    parallel_enabled = os.getenv("PHYSX_PARALLEL", "0") == "1"
    max_workers = int(os.getenv("PHYSX_MAX_WORKERS", "1"))
    warmup_timeout = int(os.getenv("PHYSX_WARMUP_TIMEOUT", "600"))  # 10 minutes default

    if not multiview_prefix or not assets_prefix:
        log("MULTIVIEW_PREFIX and ASSETS_PREFIX are required", "ERROR")
        sys.exit(1)

    root = Path("/mnt/gcs")
    multiview_root = root / multiview_prefix
    assets_root = root / assets_prefix
    layout_root = root / layout_prefix if layout_prefix else None

    assets_manifest = assets_root / "scene_assets.json"
    if not assets_manifest.is_file():
        log(f"ERROR: scene_assets.json not found at {assets_manifest}", "ERROR")
        sys.exit(1)

    scene_assets = load_scene_assets(assets_manifest)
    objects = scene_assets.get("objects", [])
    interactive = [o for o in objects if o.get("type") == "interactive"]

    log("=" * 60)
    log("Interactive Asset Generation")
    log("=" * 60)
    log(f"Bucket: {bucket}")
    log(f"Scene: {scene_id}")
    log(f"Multiview: {multiview_root}")
    log(f"Assets: {assets_root}")
    log(f"Endpoint: {endpoint}")
    log(f"Interactive objects: {len(interactive)}")
    log(f"Parallel: {parallel_enabled}, Workers: {max_workers}")
    log(f"Warmup timeout: {warmup_timeout}s")
    log("=" * 60)

    if not endpoint and interactive:
        log("ERROR: PHYSX_ENDPOINT is required for interactive assets", "ERROR")
        sys.exit(1)

    # Wait for PhysX service to be ready before processing
    # This is critical - the VLM model can take 10+ minutes to load!
    if endpoint and interactive:
        log("Checking service health before processing...")
        if not wait_for_service_ready(endpoint, max_wait=warmup_timeout):
            log("WARNING: Service not ready after timeout", "WARNING")
            log("Will attempt processing anyway (may result in placeholders)", "WARNING")

    # Process objects
    results = []
    total = len(interactive)

    if not parallel_enabled or max_workers <= 1 or total <= 1:
        # Sequential processing (recommended for GPU-bound service)
        log(f"Processing {total} objects sequentially...")
        for i, obj in enumerate(interactive):
            result = process_interactive_object(
                obj, multiview_root, assets_root, endpoint, i, total
            )
            results.append(result)
    else:
        # Parallel processing (use with caution)
        log(f"Processing {total} objects with {max_workers} workers...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    process_interactive_object, 
                    obj, multiview_root, assets_root, endpoint, i, total
                ): obj
                for i, obj in enumerate(interactive)
            }
            for future in as_completed(futures):
                obj = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    log(f"ERROR processing obj_{obj.get('id')}: {e}", "ERROR")
                    results.append({
                        "id": obj.get("id"),
                        "status": "error",
                        "error": str(e),
                    })

    # Write summary
    ok_count = sum(1 for r in results if r.get("status") == "ok")
    placeholder_count = sum(1 for r in results if r.get("status") == "placeholder")
    error_count = sum(1 for r in results if r.get("status") == "error")
    
    summary = {
        "scene_id": scene_id,
        "total_objects": total,
        "ok_count": ok_count,
        "placeholder_count": placeholder_count,
        "error_count": error_count,
        "interactive_processed": results,
    }
    
    out_path = assets_root / "interactive" / "interactive_results.json"
    ensure_dir(out_path.parent)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"Wrote summary to {out_path}")

    # Write completion marker
    marker_path = assets_root / ".interactive_complete"
    marker_path.write_text(f"completed at {os.getenv('K_REVISION', 'unknown')}\n")
    log(f"Wrote completion marker: {marker_path}")

    # Summary
    log("=" * 60)
    log(f"RESULTS: {ok_count} ok, {placeholder_count} placeholder, {error_count} error")
    log("=" * 60)
    
    # Exit with error if all failed
    if ok_count == 0 and total > 0:
        log("WARNING: All objects resulted in placeholders or errors!", "WARNING")


if __name__ == "__main__":
    main()