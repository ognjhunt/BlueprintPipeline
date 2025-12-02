"""
Interactive asset generation job.

This job orchestrates calls to the PhysX-Anything service to generate
physics-ready assets (URDF + mesh) from object crops.

Key improvements:
- Sequential processing by default (GPU can't handle parallel inference)
- Longer timeouts for cold starts and heavy ML inference
- Better retry logic with exponential backoff
- Service warmup before processing
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


def load_scene_assets(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"scene_assets.json not found at {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def wait_for_service_ready(endpoint: str, max_wait: int = 300) -> bool:
    """
    Wait for the PhysX service to be ready.
    This handles Cloud Run cold starts which can take several minutes
    for large ML containers.
    
    Args:
        endpoint: Service URL
        max_wait: Maximum seconds to wait (default: 5 minutes)
        
    Returns:
        True if service is ready, False if timeout
    """
    print(f"[PHYSX] Waiting for service to be ready (max {max_wait}s)...", flush=True)
    
    start_time = time.time()
    attempt = 0
    
    while time.time() - start_time < max_wait:
        attempt += 1
        try:
            req = urllib.request.Request(endpoint, method="GET")
            with urllib.request.urlopen(req, timeout=30) as resp:  # nosec B310
                if resp.status == 200:
                    body = resp.read().decode("utf-8", errors="replace")
                    print(f"[PHYSX] Service ready after {int(time.time() - start_time)}s: {body[:100]}", flush=True)
                    return True
                    
        except urllib.error.HTTPError as e:
            if e.code == 503:
                # Service is warming up
                elapsed = int(time.time() - start_time)
                print(f"[PHYSX] Service warming up (attempt {attempt}, {elapsed}s elapsed)...", 
                      file=sys.stderr, flush=True)
            else:
                print(f"[PHYSX] Health check HTTP {e.code}: {e.reason}", file=sys.stderr, flush=True)
                
        except urllib.error.URLError as e:
            print(f"[PHYSX] Health check network error: {e.reason}", file=sys.stderr, flush=True)
            
        except Exception as e:
            print(f"[PHYSX] Health check error: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        
        # Exponential backoff: 5s, 10s, 15s... up to 30s
        sleep_time = min(5 + (attempt * 5), 30)
        time.sleep(sleep_time)
    
    print(f"[PHYSX] ERROR: Service not ready after {max_wait}s", file=sys.stderr, flush=True)
    return False


def call_physx_anything(
    endpoint: str, 
    crop_path: Path, 
    obj_id: str,
    max_retries: int = 3,
    first_timeout: int = 900,  # 15 min for first request (model loading)
    retry_timeout: int = 600,  # 10 min for retries
) -> Optional[dict]:
    """
    Post the crop image to the PhysX-Anything service with retry logic.

    Args:
        endpoint: PhysX service URL
        crop_path: Path to the crop image
        obj_id: Object ID for logging
        max_retries: Maximum retry attempts (default: 3)
        first_timeout: Timeout for first attempt in seconds (default: 15 min)
        retry_timeout: Timeout for retry attempts in seconds (default: 10 min)

    Returns:
        Response dict or None on failure
    """
    with crop_path.open("rb") as f:
        payload = base64.b64encode(f.read()).decode("utf-8")
    body = json.dumps({"image_base64": payload}).encode("utf-8")

    for attempt in range(max_retries + 1):
        try:
            req = urllib.request.Request(
                endpoint,
                data=body,
                headers={"Content-Type": "application/json"},
            )

            # Use longer timeout for first attempt (cold start + model loading)
            timeout = first_timeout if attempt == 0 else retry_timeout

            if attempt > 0:
                print(f"[PHYSX] [{obj_id}] Retry {attempt}/{max_retries} (timeout={timeout}s)", flush=True)
            else:
                print(f"[PHYSX] [{obj_id}] POST {endpoint} (timeout={timeout}s)", flush=True)

            with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310
                print(f"[PHYSX] [{obj_id}] Response status: {resp.status}", flush=True)
                text = resp.read().decode("utf-8", errors="replace")
                try:
                    data = json.loads(text)
                    is_placeholder = data.get("placeholder", True)
                    print(f"[PHYSX] [{obj_id}] Success! placeholder={is_placeholder}", flush=True)
                    return data
                except json.JSONDecodeError:
                    print(f"[PHYSX] [{obj_id}] WARNING: non-JSON response: {text[:200]}...", 
                          file=sys.stderr, flush=True)
                    return None

        except urllib.error.HTTPError as e:
            error_body = ""
            try:
                error_body = e.read().decode("utf-8", errors="replace")[:500]
            except:
                pass
            print(f"[PHYSX] [{obj_id}] HTTP {e.code}: {error_body}", file=sys.stderr, flush=True)
            
            # Don't retry 4xx errors (client errors) except 429 (rate limit)
            if 400 <= e.code < 500 and e.code != 429:
                return None
                
            # Retry 503 (service busy/warming) and 5xx errors
            if attempt < max_retries:
                if e.code == 503:
                    # Service is busy or warming up - wait longer
                    wait_time = min(30 * (attempt + 1), 120)  # 30s, 60s, 90s, max 120s
                else:
                    # Other server errors - standard exponential backoff
                    wait_time = min(10 * (2 ** attempt), 60)  # 10s, 20s, 40s, max 60s
                    
                print(f"[PHYSX] [{obj_id}] Retrying in {wait_time}s...", file=sys.stderr, flush=True)
                time.sleep(wait_time)
                continue
            return None

        except urllib.error.URLError as e:
            print(f"[PHYSX] [{obj_id}] Network error: {e.reason}", file=sys.stderr, flush=True)
            if attempt < max_retries:
                wait_time = min(10 * (2 ** attempt), 60)
                print(f"[PHYSX] [{obj_id}] Retrying in {wait_time}s...", file=sys.stderr, flush=True)
                time.sleep(wait_time)
                continue
            return None

        except Exception as e:
            print(f"[PHYSX] [{obj_id}] Error: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            if attempt < max_retries:
                wait_time = min(10 * (2 ** attempt), 60)
                print(f"[PHYSX] [{obj_id}] Retrying in {wait_time}s...", file=sys.stderr, flush=True)
                time.sleep(wait_time)
                continue
            return None

    return None


def download_file(url: str, out_path: Path) -> bool:
    ensure_dir(out_path.parent)
    print(f"[PHYSX] Downloading {url} -> {out_path}", flush=True)
    try:
        urllib.request.urlretrieve(url, out_path)  # nosec B310
        return True
    except Exception as e:
        print(f"[PHYSX] WARNING: failed to download {url}: {e}", file=sys.stderr, flush=True)
        if out_path.is_file():
            try:
                out_path.unlink()
            except OSError:
                pass
        return False


def write_placeholder_assets(out_dir: Path, obj_id: str) -> Tuple[Path, Path]:
    """
    Writes placeholder URDF/mesh when the PhysX-Anything service cannot be contacted
    or does not return valid assets.
    """
    ensure_dir(out_dir)

    mesh_path = out_dir / "part.glb"
    placeholder = b"Placeholder GLB generated by run_interactive_assets.py"
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
  <joint name="placeholder_joint" type="fixed">
    <parent link="base"/>
    <child link="base"/>
  </joint>
</robot>
"""
    urdf_path.write_text(urdf.strip() + "\n", encoding="utf-8")
    return mesh_path, urdf_path


def parse_urdf_summary(urdf_path: Path) -> Dict:
    """Extracts a lightweight joint/link summary for bookkeeping."""
    import xml.etree.ElementTree as ET

    summary: Dict[str, List[Dict]] = {"joints": [], "links": []}
    if not urdf_path.is_file():
        return summary

    try:
        tree = ET.parse(urdf_path)
        root = tree.getroot()
    except Exception as e:
        print(f"[PHYSX] WARNING: failed to parse URDF {urdf_path}: {e}", file=sys.stderr, flush=True)
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
                    mass_val = None
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
            low_str = limit_tag.attrib.get("lower")
            up_str = limit_tag.attrib.get("upper")
            try:
                lower = float(low_str) if low_str else None
            except ValueError:
                lower = None
            try:
                upper = float(up_str) if up_str else None
            except ValueError:
                upper = None

        parent_link = None
        child_link = None
        parent_elem = joint.find("parent")
        child_elem = joint.find("child")
        if parent_elem is not None:
            parent_link = parent_elem.attrib.get("link")
        if child_elem is not None:
            child_link = child_elem.attrib.get("link")

        summary["joints"].append({
            "name": joint.attrib.get("name"),
            "type": joint.attrib.get("type"),
            "parent": parent_link,
            "child": child_link,
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
    Attempts to download or decode PhysX-Anything outputs.
    Returns paths to (mesh, urdf) along with metadata.
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

    # If service already marked as placeholder, skip asset materialization
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
            print(f"[PHYSX] WARNING: failed to handle asset_zip_url: {e}", file=sys.stderr, flush=True)

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
        # Verify files are not trivially small
        mesh_size = mesh_path.stat().st_size
        urdf_size = urdf_path.stat().st_size
        print(f"[PHYSX] [{obj_id}] Mesh size: {mesh_size} bytes, URDF size: {urdf_size} bytes", flush=True)
        if mesh_size < 100:
            print(f"[PHYSX] [{obj_id}] WARNING: Mesh file suspiciously small!", file=sys.stderr, flush=True)
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

    print(f"[PHYSX] [{obj_name}] Processing object {obj_index + 1}/{total_objects}", flush=True)

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
        print(f"[PHYSX] [{obj_name}] WARNING: missing crop at {crop_path}", file=sys.stderr, flush=True)
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
        mesh_path, urdf_path, download_meta = materialize_service_assets(response, output_dir, obj_name)
        manifest.update(download_meta)

    # Fallback to placeholder if needed
    if mesh_path is None or urdf_path is None:
        print(f"[PHYSX] [{obj_name}] Using placeholder assets", file=sys.stderr, flush=True)
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
    
    print(f"[PHYSX] [{obj_name}] Completed with status={result['status']}", flush=True)
    return result


def main() -> None:
    bucket = os.getenv("BUCKET", "")
    scene_id = os.getenv("SCENE_ID", "")
    multiview_prefix = os.getenv("MULTIVIEW_PREFIX")
    assets_prefix = os.getenv("ASSETS_PREFIX")
    layout_prefix = os.getenv("LAYOUT_PREFIX")
    endpoint = os.getenv("PHYSX_ENDPOINT")
    
    # Default to sequential processing (set PHYSX_PARALLEL=1 to enable parallel)
    parallel_enabled = os.getenv("PHYSX_PARALLEL", "0") == "1"
    max_workers = int(os.getenv("PHYSX_MAX_WORKERS", "1"))

    if not multiview_prefix or not assets_prefix:
        print("[PHYSX] MULTIVIEW_PREFIX and ASSETS_PREFIX are required", file=sys.stderr, flush=True)
        sys.exit(1)

    root = Path("/mnt/gcs")
    multiview_root = root / multiview_prefix
    assets_root = root / assets_prefix
    layout_root = root / layout_prefix if layout_prefix else None

    assets_manifest = assets_root / "scene_assets.json"
    if not assets_manifest.is_file():
        print(f"[PHYSX] ERROR: scene_assets.json not found at {assets_manifest}", file=sys.stderr, flush=True)
        sys.exit(1)

    scene_assets = load_scene_assets(assets_manifest)
    objects = scene_assets.get("objects", [])
    interactive = [o for o in objects if o.get("type") == "interactive"]

    print(f"[PHYSX] ========================================", flush=True)
    print(f"[PHYSX] Interactive Asset Generation", flush=True)
    print(f"[PHYSX] ========================================", flush=True)
    print(f"[PHYSX] Bucket: {bucket}", flush=True)
    print(f"[PHYSX] Scene: {scene_id}", flush=True)
    print(f"[PHYSX] Multiview: {multiview_root}", flush=True)
    print(f"[PHYSX] Assets: {assets_root}", flush=True)
    print(f"[PHYSX] Endpoint: {endpoint}", flush=True)
    print(f"[PHYSX] Interactive objects: {len(interactive)}", flush=True)
    print(f"[PHYSX] Parallel: {parallel_enabled}, Workers: {max_workers}", flush=True)
    print(f"[PHYSX] ========================================", flush=True)

    if not endpoint and interactive:
        print("[PHYSX] ERROR: PHYSX_ENDPOINT is required for interactive assets", file=sys.stderr, flush=True)
        sys.exit(1)

    # Wait for PhysX service to be ready before processing
    if endpoint and interactive:
        if not wait_for_service_ready(endpoint, max_wait=300):
            print("[PHYSX] WARNING: Service not ready, will attempt processing anyway", 
                  file=sys.stderr, flush=True)

    # Process objects
    results = []
    total = len(interactive)

    if not parallel_enabled or max_workers <= 1 or total <= 1:
        # Sequential processing (recommended for GPU-bound service)
        print(f"[PHYSX] Processing {total} objects sequentially...", flush=True)
        for i, obj in enumerate(interactive):
            result = process_interactive_object(
                obj, multiview_root, assets_root, endpoint, i, total
            )
            results.append(result)
    else:
        # Parallel processing (use with caution)
        print(f"[PHYSX] Processing {total} objects with {max_workers} workers...", flush=True)
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
                    print(f"[PHYSX] ERROR processing obj_{obj.get('id')}: {e}", 
                          file=sys.stderr, flush=True)
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
    print(f"[PHYSX] Wrote summary to {out_path}", flush=True)

    # Write completion marker
    marker_path = assets_root / ".interactive_complete"
    marker_path.write_text(f"completed at {os.getenv('K_REVISION', 'unknown')}\n")
    print(f"[PHYSX] Wrote completion marker: {marker_path}", flush=True)

    # Summary
    print(f"[PHYSX] ========================================", flush=True)
    print(f"[PHYSX] RESULTS: {ok_count} ok, {placeholder_count} placeholder, {error_count} error", flush=True)
    print(f"[PHYSX] ========================================", flush=True)


if __name__ == "__main__":
    main()