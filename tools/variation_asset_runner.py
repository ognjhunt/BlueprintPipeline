"""Hybrid local+VM orchestration for Stage 3 variation asset generation.

Generates reference images locally via Gemini 3.0 Pro Image, then
runs Hunyuan3D-2/2.1 on a remote GPU VM for 3D mesh generation.
Physics metadata is estimated locally from mesh volume + category defaults.

Reuses VMExecutor from tools/vm_executor.py for SSH/SCP.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import shlex
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Physics defaults by category (from variation-asset-pipeline-job)
# ---------------------------------------------------------------------------

PHYSICS_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "dishes": {"bulk_density": 2400, "static_friction": 0.4, "dynamic_friction": 0.35, "restitution": 0.1, "collision_shape": "convex"},
    "tableware": {"bulk_density": 2400, "static_friction": 0.4, "dynamic_friction": 0.35, "restitution": 0.1, "collision_shape": "convex"},
    "utensils": {"bulk_density": 7800, "static_friction": 0.3, "dynamic_friction": 0.25, "restitution": 0.15, "collision_shape": "convex"},
    "stationery": {"bulk_density": 1200, "static_friction": 0.4, "dynamic_friction": 0.35, "restitution": 0.1, "collision_shape": "convex"},
    "electronics": {"bulk_density": 1800, "static_friction": 0.3, "dynamic_friction": 0.25, "restitution": 0.05, "collision_shape": "box"},
    "groceries": {"bulk_density": 800, "static_friction": 0.5, "dynamic_friction": 0.4, "restitution": 0.05, "collision_shape": "box"},
    "bottles": {"bulk_density": 1200, "static_friction": 0.35, "dynamic_friction": 0.3, "restitution": 0.1, "collision_shape": "convex"},
    "tools": {"bulk_density": 3500, "static_friction": 0.5, "dynamic_friction": 0.4, "restitution": 0.1, "collision_shape": "convex"},
    "default": {"bulk_density": 600, "static_friction": 0.5, "dynamic_friction": 0.4, "restitution": 0.1, "collision_shape": "convex"},
}

# Target real-world sizes (max bounding box dimension in meters) by asset name/category.
# Hunyuan3D outputs meshes in a ~2m unit cube; we rescale to these targets.
TARGET_SIZES_M: Dict[str, float] = {
    # By asset name
    "drinking_vessel": 0.10,    # mug/glass ~10cm tall
    "writing_utensils": 0.15,   # pen ~15cm long
    "electronic_clutter": 0.15, # phone/calculator ~15cm
    # By category fallbacks
    "tableware": 0.12,
    "stationery": 0.15,
    "electronics": 0.15,
    "dishes": 0.20,
    "utensils": 0.25,
    "bottles": 0.25,
    "groceries": 0.20,
    "tools": 0.25,
    "default": 0.15,
}

# Gemini model for reference image generation
GEMINI_IMAGE_MODEL = "gemini-3-pro-image-preview"

# Remote paths
VM_INPUT_ROOT = "/tmp/variation_inputs"
VM_OUTPUT_ROOT = "/tmp/variation_outputs"
VM_SCRIPT_PATH = "/tmp/variation_hunyuan.py"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AssetSpec:
    """A single variation asset from the replicator manifest."""
    name: str
    category: str
    description: str
    semantic_class: str
    priority: str
    source_hint: Optional[str] = None
    example_variants: List[str] = field(default_factory=list)
    physics_hints: Dict[str, Any] = field(default_factory=dict)
    material_hint: Optional[str] = None
    style_hint: Optional[str] = None
    generation_prompt_hint: Optional[str] = None


@dataclass
class AssetResult:
    """Result of processing a single variation asset."""
    name: str
    success: bool
    stage_completed: str  # "image", "3d", "physics", "complete", "failed"
    reference_image: Optional[str] = None
    glb_path: Optional[str] = None
    metadata_path: Optional[str] = None
    error: Optional[str] = None
    timings: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

class VariationAssetRunner:
    """Orchestrates variation asset generation across local + remote VM."""

    def __init__(
        self,
        scene_dir: Path,
        hunyuan_path: str = "",
        venv_python: str = "",
        vm_zone: str = "us-east1-c",
        vm_host: str = "isaac-sim-ubuntu",
    ):
        self.scene_dir = Path(scene_dir)
        self.manifest_path = self.scene_dir / "replicator" / "variation_assets" / "manifest.json"
        self.output_dir = self.scene_dir / "variation_assets"

        # VM config — these may be overridden by pre-flight probing
        self.hunyuan_path = hunyuan_path
        self.venv_python = venv_python
        self.vm_zone = vm_zone
        self.vm_host = vm_host

        # Lazy-loaded
        self._gemini_client = None
        self._vm = None

    @property
    def gemini_client(self):
        if self._gemini_client is None:
            from google import genai
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable is required")
            self._gemini_client = genai.Client(api_key=api_key)
        return self._gemini_client

    @property
    def vm(self):
        if self._vm is None:
            from tools.vm_executor import VMExecutor, VMConfig
            self._vm = VMExecutor(
                VMConfig(host=self.vm_host, zone=self.vm_zone),
                verbose=True,
            )
        return self._vm

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """Run the full variation asset pipeline. Returns a summary dict."""
        t0 = time.time()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Load manifest
        manifest = self._load_manifest()
        # Filter to only known AssetSpec fields (manifest may have extra keys)
        import dataclasses
        known_fields = {f.name for f in dataclasses.fields(AssetSpec)}
        assets = [
            AssetSpec(**{k: v for k, v in a.items() if k in known_fields})
            for a in manifest.get("assets", [])
        ]
        logger.info(f"[VARIATION] Loaded {len(assets)} assets from manifest")

        # 2. Pre-flight VM checks
        self._preflight_vm()

        # 3. Upload remote script
        self._upload_remote_script()

        # 4. Process each asset
        results: List[AssetResult] = []
        for asset in assets:
            logger.info(f"[VARIATION] === Processing: {asset.name} ({asset.priority}) ===")
            result = self._process_asset(asset)
            results.append(result)
            status = "OK" if result.success else f"FAILED ({result.error})"
            logger.info(f"[VARIATION] {asset.name}: {status}")

        # 5. Write output manifests
        self._write_variation_assets_json(results, assets)
        self._write_simready_assets_json(results, assets)
        self._write_pipeline_summary(results, time.time() - t0)

        # 6. Write completion marker
        marker = self.output_dir / ".variation_pipeline_complete"
        marker.write_text(json.dumps({
            "status": "completed",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total": len(results),
            "succeeded": sum(1 for r in results if r.success),
        }, indent=2))

        succeeded = sum(1 for r in results if r.success)
        logger.info(f"[VARIATION] Pipeline complete: {succeeded}/{len(results)} assets in {time.time()-t0:.1f}s")

        return {
            "total": len(results),
            "succeeded": succeeded,
            "results": [asdict(r) for r in results],
        }

    # ------------------------------------------------------------------
    # Manifest loading
    # ------------------------------------------------------------------

    def _load_manifest(self) -> Dict[str, Any]:
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Variation manifest not found: {self.manifest_path}")
        with open(self.manifest_path) as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # VM pre-flight
    # ------------------------------------------------------------------

    def _preflight_vm(self):
        """Verify VM is running, GPU is available, and probe Hunyuan3D paths."""
        logger.info("[VARIATION] Pre-flight VM checks...")

        if not self.vm.check_vm_running():
            raise RuntimeError(
                f"VM {self.vm_host} is not running. Start it with:\n"
                f"  gcloud compute instances start {self.vm_host} --zone={self.vm_zone}"
            )

        if not self.vm.check_gpu_available():
            raise RuntimeError("GPU not available on VM")

        # Probe Hunyuan3D path if not set
        if not self.hunyuan_path:
            self.hunyuan_path = self._probe_hunyuan_path()
        logger.info(f"[VARIATION] Hunyuan3D path: {self.hunyuan_path}")

        # Probe venv python if not set
        if not self.venv_python:
            self.venv_python = self._probe_venv_python()
        logger.info(f"[VARIATION] Venv Python: {self.venv_python}")

        # Create remote directories
        self.vm.ensure_directory(VM_INPUT_ROOT)
        self.vm.ensure_directory(VM_OUTPUT_ROOT)

    def _probe_hunyuan_path(self) -> str:
        """Find Hunyuan3D install on the VM."""
        candidates = [
            "/home/nijelhunt1/Hunyuan3D-2.1",
            "/home/nijelhunt_1/Hunyuan3D-2.1",
        ]
        for path in candidates:
            rc, _, _ = self.vm.ssh_exec(
                f"test -d {shlex.quote(path)} && echo found",
                stream_logs=False, check=False,
            )
            if rc == 0:
                return path
        raise RuntimeError(
            f"Hunyuan3D not found on VM. Checked: {candidates}"
        )

    def _probe_venv_python(self) -> str:
        """Find the Python venv on the VM."""
        # Check relative to hunyuan_path first, then known locations
        candidates = [
            f"{self.hunyuan_path}/venv_py310/bin/python",
            "/home/nijelhunt1/Hunyuan3D-2.1/venv_py310/bin/python",
        ]
        for path in candidates:
            rc, _, _ = self.vm.ssh_exec(
                f"test -x {shlex.quote(path)} && echo found",
                stream_logs=False, check=False,
            )
            if rc == 0:
                return path
        raise RuntimeError(
            f"Python venv not found on VM. Checked: {candidates}"
        )

    def _upload_remote_script(self):
        """Upload variation_hunyuan_remote.py to the VM."""
        local_script = Path(__file__).parent / "variation_hunyuan_remote.py"
        if not local_script.is_file():
            raise FileNotFoundError(f"Remote script not found: {local_script}")
        self.vm.scp_upload(local_script, VM_SCRIPT_PATH)
        logger.info(f"[VARIATION] Uploaded remote script to {VM_SCRIPT_PATH}")

    # ------------------------------------------------------------------
    # Per-asset processing
    # ------------------------------------------------------------------

    def _process_asset(self, asset: AssetSpec) -> AssetResult:
        """Process a single variation asset: image → 3D → rescale → physics → USDZ → simready."""
        timings: Dict[str, float] = {}
        asset_dir = self.output_dir / _safe_name(asset.name)
        asset_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Generate reference image (local, Gemini)
        t0 = time.time()
        try:
            image_path = self._generate_reference_image(asset, asset_dir)
        except Exception as e:
            return AssetResult(
                name=asset.name, success=False, stage_completed="failed",
                error=f"Image generation failed: {e}", timings=timings,
            )
        timings["image_gen_s"] = time.time() - t0

        # Step 2: 3D reconstruction (remote VM, Hunyuan3D)
        t0 = time.time()
        try:
            glb_path = self._reconstruct_3d(asset, image_path, asset_dir)
        except Exception as e:
            return AssetResult(
                name=asset.name, success=False, stage_completed="image",
                reference_image=str(image_path), error=f"3D reconstruction failed: {e}",
                timings=timings,
            )
        timings["reconstruction_s"] = time.time() - t0

        # Step 3: Rescale mesh to real-world size
        t0 = time.time()
        try:
            scale_factor = self._rescale_mesh(asset, glb_path)
        except Exception as e:
            return AssetResult(
                name=asset.name, success=False, stage_completed="3d",
                reference_image=str(image_path), glb_path=str(glb_path),
                error=f"Mesh rescaling failed: {e}", timings=timings,
            )
        timings["rescale_s"] = time.time() - t0

        # Step 4: Physics estimation (local, trimesh + category defaults)
        t0 = time.time()
        try:
            metadata_path = self._estimate_physics(asset, glb_path, asset_dir)
        except Exception as e:
            return AssetResult(
                name=asset.name, success=False, stage_completed="3d",
                reference_image=str(image_path), glb_path=str(glb_path),
                error=f"Physics estimation failed: {e}", timings=timings,
            )
        timings["physics_s"] = time.time() - t0

        # Step 5: Convert GLB → USDZ
        t0 = time.time()
        try:
            usdz_path = self._convert_to_usdz(asset, glb_path, asset_dir)
        except Exception as e:
            logger.warning(f"[VARIATION] USDZ conversion failed for {asset.name}: {e}")
            usdz_path = None
        timings["usdz_convert_s"] = time.time() - t0

        # Step 6: Generate simready.usda (physics wrapper)
        t0 = time.time()
        try:
            simready_path = self._generate_simready_usd(
                asset, usdz_path or glb_path, metadata_path, asset_dir
            )
        except Exception as e:
            logger.warning(f"[VARIATION] SimReady USD generation failed for {asset.name}: {e}")
            simready_path = None
        timings["simready_s"] = time.time() - t0

        return AssetResult(
            name=asset.name, success=True, stage_completed="complete",
            reference_image=str(image_path), glb_path=str(glb_path),
            metadata_path=str(metadata_path), timings=timings,
        )

    # ------------------------------------------------------------------
    # Step 1: Reference image generation (Gemini)
    # ------------------------------------------------------------------

    def _generate_reference_image(self, asset: AssetSpec, asset_dir: Path) -> Path:
        """Generate a reference image using Gemini 3.0 Pro Image."""
        from google.genai import types
        from PIL import Image

        image_path = asset_dir / "reference.png"

        # Skip if already exists
        if image_path.is_file() and image_path.stat().st_size > 1000:
            logger.info(f"[VARIATION] Reference image already exists: {image_path}")
            return image_path

        prompt = _build_image_prompt(asset)
        logger.info(f"[VARIATION] Generating reference image for: {asset.name}")

        last_error = None
        for attempt in range(3):
            try:
                response = self.gemini_client.models.generate_content(
                    model=GEMINI_IMAGE_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_modalities=["IMAGE", "TEXT"],
                        temperature=0.8,
                    ),
                )

                # Extract image from response
                if response.candidates and len(response.candidates) > 0:
                    candidate = response.candidates[0]
                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            if hasattr(part, "inline_data") and part.inline_data:
                                image_data = part.inline_data.data
                                # Gemini returns base64-encoded bytes — always try decode
                                if isinstance(image_data, (bytes, bytearray)):
                                    # Check if it's base64-encoded (starts with JPEG/PNG b64 header)
                                    try:
                                        image_bytes = base64.b64decode(image_data)
                                    except Exception:
                                        image_bytes = image_data
                                elif isinstance(image_data, str):
                                    image_bytes = base64.b64decode(image_data)
                                else:
                                    continue

                                # Try opening as-is first; if that fails, it might
                                # need the raw (non-decoded) bytes
                                for attempt_bytes in [image_bytes, image_data if isinstance(image_data, (bytes, bytearray)) else b""]:
                                    if not attempt_bytes:
                                        continue
                                    try:
                                        img = Image.open(io.BytesIO(attempt_bytes))
                                        img.save(str(image_path), format="PNG")
                                        logger.info(f"[VARIATION] Saved reference image: {image_path} ({img.size})")
                                        return image_path
                                    except Exception:
                                        continue

                raise ValueError("No image data in Gemini response")

            except Exception as e:
                last_error = e
                logger.warning(f"[VARIATION] Image gen attempt {attempt+1}/3 failed: {e}")
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))

        raise RuntimeError(f"Image generation failed after 3 attempts: {last_error}")

    # ------------------------------------------------------------------
    # Step 2: 3D reconstruction (Hunyuan3D on VM)
    # ------------------------------------------------------------------

    def _reconstruct_3d(self, asset: AssetSpec, image_path: Path, asset_dir: Path) -> Path:
        """Upload image to VM, run Hunyuan3D, download GLB."""
        glb_path = asset_dir / "model.glb"

        # Skip if already exists
        if glb_path.is_file() and glb_path.stat().st_size > 1000:
            logger.info(f"[VARIATION] GLB already exists: {glb_path}")
            return glb_path

        safe = _safe_name(asset.name)
        remote_input_dir = f"{VM_INPUT_ROOT}/{safe}"
        remote_image = f"{remote_input_dir}/reference.png"
        remote_output = f"{VM_OUTPUT_ROOT}/{safe}/model.glb"

        # Create remote dirs and upload image
        self.vm.ensure_directory(remote_input_dir)
        self.vm.ensure_directory(f"{VM_OUTPUT_ROOT}/{safe}")
        self.vm.scp_upload(image_path, remote_image)

        # Run Hunyuan3D
        cmd = (
            f"cd {shlex.quote(self.hunyuan_path)} && "
            f"PYTHONNOUSERSITE=1 "
            f"PYTORCH_SKIP_CUDA_MISMATCH_CHECK=1 "
            f"CUDA_HOME=/usr/local/cuda "
            f"{shlex.quote(self.venv_python)} {VM_SCRIPT_PATH} "
            f"--image {shlex.quote(remote_image)} "
            f"--output {shlex.quote(remote_output)} "
            f"--hunyuan-path {shlex.quote(self.hunyuan_path)} "
            f"--steps 50"
        )

        logger.info(f"[VARIATION] Running Hunyuan3D for {asset.name}...")
        self.vm.ssh_exec(cmd, timeout=300, stream_logs=True, check=True)

        # Download GLB
        self.vm.scp_download(remote_output, glb_path)

        if not glb_path.is_file() or glb_path.stat().st_size < 100:
            raise RuntimeError(f"GLB file missing or too small: {glb_path}")

        size_mb = glb_path.stat().st_size / (1024 * 1024)
        logger.info(f"[VARIATION] Downloaded GLB: {glb_path} ({size_mb:.1f} MB)")
        return glb_path

    # ------------------------------------------------------------------
    # Step 3: Physics estimation
    # ------------------------------------------------------------------

    def _estimate_physics(
        self, asset: AssetSpec, glb_path: Path, asset_dir: Path
    ) -> Path:
        """Estimate physics properties from mesh geometry + category defaults."""
        metadata_path = asset_dir / "metadata.json"

        # Get category defaults
        cat = asset.category.lower()
        defaults = PHYSICS_DEFAULTS.get(cat, PHYSICS_DEFAULTS["default"])

        # Try trimesh for volume/bounds
        volume_m3 = None
        bounds_m = None
        try:
            import trimesh
            mesh = trimesh.load(str(glb_path), force="mesh")
            if mesh.is_watertight:
                volume_m3 = float(mesh.volume)
            else:
                volume_m3 = float(mesh.convex_hull.volume)
            bounds_m = [float(x) for x in mesh.bounding_box.extents]
        except Exception as e:
            logger.warning(f"[VARIATION] Trimesh failed for {asset.name}: {e}")

        # Compute mass from volume and density
        mass_kg = None
        if volume_m3 and volume_m3 > 0:
            mass_kg = volume_m3 * defaults["bulk_density"]
        else:
            # Use physics_hints from manifest if available
            hints = asset.physics_hints
            if hints.get("mass_range_kg"):
                mass_range = hints["mass_range_kg"]
                mass_kg = (mass_range[0] + mass_range[1]) / 2
            else:
                mass_kg = 0.5  # generic fallback

        # Clamp to manifest range if available
        hints = asset.physics_hints
        if hints.get("mass_range_kg") and mass_kg is not None:
            lo, hi = hints["mass_range_kg"]
            mass_kg = max(lo, min(hi, mass_kg))

        metadata = {
            "name": asset.name,
            "category": asset.category,
            "semantic_class": asset.semantic_class,
            "description": asset.description,
            "physics": {
                "mass_kg": round(mass_kg, 4) if mass_kg else 0.5,
                "static_friction": hints.get("friction", defaults["static_friction"]),
                "dynamic_friction": defaults["dynamic_friction"],
                "restitution": defaults["restitution"],
                "collision_shape": hints.get("collision_shape", defaults["collision_shape"]),
                "center_of_mass": "geometric",
            },
            "geometry": {
                "volume_m3": round(volume_m3, 6) if volume_m3 else None,
                "bounding_box_m": bounds_m,
            },
            "files": {
                "glb": "model.glb",
                "usdz": "asset.usdz",
                "simready_usda": "simready.usda",
                "reference_image": "reference.png",
            },
            "generation": {
                "image_model": GEMINI_IMAGE_MODEL,
                "mesh_model": "Hunyuan3D-2",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(
            f"[VARIATION] Physics: {asset.name} -> "
            f"mass={metadata['physics']['mass_kg']}kg, "
            f"friction={metadata['physics']['static_friction']}"
        )
        return metadata_path

    # ------------------------------------------------------------------
    # Step 3: Mesh rescaling (Hunyuan3D unit cube → real-world meters)
    # ------------------------------------------------------------------

    def _rescale_mesh(self, asset: AssetSpec, glb_path: Path) -> float:
        """Rescale GLB mesh in-place to real-world dimensions. Returns scale factor."""
        import trimesh

        mesh = trimesh.load(str(glb_path), force="mesh")
        current_max = max(mesh.bounding_box.extents)

        if current_max < 1e-6:
            logger.warning(f"[VARIATION] Mesh has zero extent, skipping rescale: {asset.name}")
            return 1.0

        # Look up target size: by name first, then category, then default
        target = TARGET_SIZES_M.get(
            asset.name,
            TARGET_SIZES_M.get(asset.category.lower(), TARGET_SIZES_M["default"]),
        )

        scale_factor = target / current_max
        logger.info(
            f"[VARIATION] Rescaling {asset.name}: "
            f"current_max={current_max:.3f}m -> target={target:.3f}m "
            f"(scale={scale_factor:.4f})"
        )

        mesh.apply_scale(scale_factor)
        mesh.export(str(glb_path))

        # Verify
        new_max = max(mesh.bounding_box.extents)
        logger.info(f"[VARIATION] After rescale: max_extent={new_max:.4f}m")
        return scale_factor

    # ------------------------------------------------------------------
    # Step 5: GLB → USDZ conversion
    # ------------------------------------------------------------------

    def _convert_to_usdz(self, asset: AssetSpec, glb_path: Path, asset_dir: Path) -> Path:
        """Convert GLB to USDZ using pxr (OpenUSD Python bindings)."""
        usdz_path = asset_dir / "asset.usdz"

        # Skip if already exists
        if usdz_path.is_file() and usdz_path.stat().st_size > 1000:
            logger.info(f"[VARIATION] USDZ already exists: {usdz_path}")
            return usdz_path

        import tempfile
        import trimesh
        from pxr import Usd, UsdGeom, UsdShade, Gf, Sdf, UsdUtils

        # Load mesh with trimesh for geometry data
        mesh = trimesh.load(str(glb_path), force="mesh")

        with tempfile.TemporaryDirectory() as tmp_dir:
            usda_path = Path(tmp_dir) / "model.usda"

            # Create USD stage
            stage = Usd.Stage.CreateNew(str(usda_path))
            stage.SetMetadata("metersPerUnit", 1.0)
            stage.SetMetadata("upAxis", "Y")

            # Create mesh prim
            mesh_prim = UsdGeom.Mesh.Define(stage, "/Asset/Visual/Mesh")

            # Set vertices
            vertices = mesh.vertices.tolist()
            mesh_prim.GetPointsAttr().Set([Gf.Vec3f(*v) for v in vertices])

            # Set faces
            face_counts = [3] * len(mesh.faces)
            face_indices = mesh.faces.flatten().tolist()
            mesh_prim.GetFaceVertexCountsAttr().Set(face_counts)
            mesh_prim.GetFaceVertexIndicesAttr().Set(face_indices)

            # Set normals if available
            if mesh.vertex_normals is not None and len(mesh.vertex_normals) > 0:
                normals = [Gf.Vec3f(*n) for n in mesh.vertex_normals.tolist()]
                mesh_prim.GetNormalsAttr().Set(normals)
                mesh_prim.SetNormalsInterpolation("vertex")

            # Set extent
            bounds = mesh.bounds
            mesh_prim.GetExtentAttr().Set([
                Gf.Vec3f(*bounds[0].tolist()),
                Gf.Vec3f(*bounds[1].tolist()),
            ])

            # Set default prim
            stage.SetDefaultPrim(stage.GetPrimAtPath("/Asset"))

            stage.GetRootLayer().Save()

            # Package as USDZ
            usdz_path.parent.mkdir(parents=True, exist_ok=True)
            success = UsdUtils.CreateNewUsdzPackage(
                Sdf.AssetPath(str(usda_path)),
                str(usdz_path),
            )
            if not success:
                raise RuntimeError("UsdUtils.CreateNewUsdzPackage failed")

        size_mb = usdz_path.stat().st_size / (1024 * 1024)
        logger.info(f"[VARIATION] Created USDZ: {usdz_path} ({size_mb:.1f} MB)")
        return usdz_path

    # ------------------------------------------------------------------
    # Step 6: SimReady USD generation (physics wrapper)
    # ------------------------------------------------------------------

    def _generate_simready_usd(
        self,
        asset: AssetSpec,
        visual_path: Path,
        metadata_path: Path,
        asset_dir: Path,
    ) -> Path:
        """Generate simready.usda wrapping the visual asset with physics properties."""
        simready_path = asset_dir / "simready.usda"

        # Load metadata
        with open(metadata_path) as f:
            metadata = json.load(f)

        physics = metadata["physics"]
        geometry = metadata.get("geometry", {})

        mass = float(physics.get("mass_kg", 0.5))
        static_friction = float(physics.get("static_friction", 0.5))
        dynamic_friction = float(physics.get("dynamic_friction", 0.4))
        restitution = float(physics.get("restitution", 0.1))
        collision_shape = str(physics.get("collision_shape", "box")).lower()

        # Bounds from metadata (already in real-world meters after rescaling)
        bounds = geometry.get("bounding_box_m") or [0.1, 0.1, 0.1]

        # Compute center (assume centered at origin after rescaling)
        center = [0.0, 0.0, 0.0]

        # Collision padding (1%)
        pad = 0.01
        min_pad = 0.001
        size_pad = [max(b * (1.0 + pad), b + min_pad) for b in bounds]

        # Asset reference relative to simready.usda
        asset_rel = "./" + visual_path.name

        lines: List[str] = []
        lines.append("#usda 1.0")
        lines.append("(")
        lines.append("    metersPerUnit = 1")
        lines.append("    kilogramsPerUnit = 1")
        lines.append(")")
        lines.append("")

        # Asset root with physics APIs
        lines.append('def Xform "Asset" (')
        lines.append('    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]')
        lines.append(")")
        lines.append("{")
        lines.append("    bool physics:rigidBodyEnabled = 1")
        lines.append(f"    float physics:mass = {mass:.6f}")
        lines.append(f"    point3f physics:centerOfMass = ({center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f})")

        # Visual reference
        lines.append("")
        lines.append('    def Xform "Visual" (')
        lines.append(f"        prepend references = @{asset_rel}@")
        lines.append("    )")
        lines.append("    {")
        lines.append("    }")

        # Physics material
        lines.append("")
        lines.append('    def Scope "Looks"')
        lines.append("    {")
        lines.append('        def Material "PhysicsMaterial" (')
        lines.append('            prepend apiSchemas = ["PhysicsMaterialAPI"]')
        lines.append("        )")
        lines.append("        {")
        lines.append(f"            float physics:staticFriction = {static_friction:.4f}")
        lines.append(f"            float physics:dynamicFriction = {dynamic_friction:.4f}")
        lines.append(f"            float physics:restitution = {restitution:.4f}")
        lines.append("        }")
        lines.append("    }")

        # Collision proxy
        lines.append("")
        lines.append('    def Xform "Collision"')
        lines.append("    {")
        material_path = "</Asset/Looks/PhysicsMaterial>"

        if collision_shape == "convex":
            lines.append('        def Cube "Collider" (')
            lines.append('            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI"]')
            lines.append("        )")
            lines.append("        {")
            lines.append(f"            rel material:binding:physics = {material_path}")
            lines.append("            double size = 1")
            lines.append(f"            double3 xformOp:translate = ({center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f})")
            lines.append(f"            float3 xformOp:scale = ({size_pad[0]:.6f}, {size_pad[1]:.6f}, {size_pad[2]:.6f})")
            lines.append('            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]')
            lines.append("            float physxCollision:contactOffset = 0.005000")
            lines.append("            float physxCollision:restOffset = 0.001000")
            lines.append("            bool physxCollision:gpuCollision = true")
            lines.append("        }")
        else:
            # Default box proxy
            lines.append('        def Cube "Collider" (')
            lines.append('            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI"]')
            lines.append("        )")
            lines.append("        {")
            lines.append(f"            rel material:binding:physics = {material_path}")
            lines.append("            double size = 1")
            lines.append(f"            double3 xformOp:translate = ({center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f})")
            lines.append(f"            float3 xformOp:scale = ({size_pad[0]:.6f}, {size_pad[1]:.6f}, {size_pad[2]:.6f})")
            lines.append('            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]')
            lines.append("            float physxCollision:contactOffset = 0.005000")
            lines.append("            float physxCollision:restOffset = 0.001000")
            lines.append("            bool physxCollision:gpuCollision = true")
            lines.append("        }")

        lines.append("    }")

        # Robotics metadata
        lines.append("")
        lines.append("    # Robotics metadata for perception and manipulation")
        lines.append(f'    string semantic:class = "{asset.semantic_class}"')
        lines.append(f'    string simready:material = "{asset.material_hint or "generic"}"')
        lines.append(f"    bool simready:graspable = true")
        lines.append(f"    float simready:surfaceRoughness = {static_friction:.4f}")

        lines.append("}")
        lines.append("")

        simready_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"[VARIATION] Generated SimReady USD: {simready_path}")
        return simready_path

    # ------------------------------------------------------------------
    # Output manifest writing
    # ------------------------------------------------------------------

    def _write_variation_assets_json(
        self, results: List[AssetResult], assets: List[AssetSpec]
    ):
        """Write downstream-compatible variation_assets.json."""
        objects = []
        for result, asset in zip(results, assets):
            entry = {
                "id": asset.name,
                "name": asset.name,
                "category": asset.category,
                "short_description": asset.description,
                "semantic_class": asset.semantic_class,
                "priority": asset.priority,
                "status": "complete" if result.success else "failed",
            }
            if result.success and result.glb_path:
                entry["glb_path"] = result.glb_path
            if result.success and result.metadata_path:
                entry["metadata_path"] = result.metadata_path
            if result.reference_image:
                entry["reference_image"] = result.reference_image
            objects.append(entry)

        out = self.output_dir / "variation_assets.json"
        with open(out, "w") as f:
            json.dump({"objects": objects, "count": len(objects)}, f, indent=2)
        logger.info(f"[VARIATION] Wrote {out}")

    def _write_simready_assets_json(
        self, results: List[AssetResult], assets: List[AssetSpec]
    ):
        """Write simready_assets.json listing GLBs with physics."""
        entries = []
        for result, asset in zip(results, assets):
            if not result.success:
                continue
            safe = _safe_name(asset.name)
            entries.append({
                "id": asset.name,
                "path": f"{safe}/model.glb",
                "metadata": f"{safe}/metadata.json",
                "category": asset.category,
            })

        out = self.output_dir / "simready_assets.json"
        with open(out, "w") as f:
            json.dump({"assets": entries, "count": len(entries)}, f, indent=2)
        logger.info(f"[VARIATION] Wrote {out}")

    def _write_pipeline_summary(self, results: List[AssetResult], total_seconds: float):
        """Write pipeline timing summary."""
        summary = {
            "total_duration_s": round(total_seconds, 1),
            "total_assets": len(results),
            "succeeded": sum(1 for r in results if r.success),
            "failed": sum(1 for r in results if not r.success),
            "per_asset": [],
        }
        for r in results:
            summary["per_asset"].append({
                "name": r.name,
                "success": r.success,
                "stage_completed": r.stage_completed,
                "timings": r.timings,
                "error": r.error,
            })

        out = self.output_dir / "pipeline_summary.json"
        with open(out, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"[VARIATION] Wrote {out}")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _safe_name(name: str) -> str:
    """Convert name to filesystem-safe string."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


def _build_image_prompt(asset: AssetSpec) -> str:
    """Build Gemini prompt for product photography reference image."""
    if asset.generation_prompt_hint:
        base = asset.generation_prompt_hint
    else:
        base = f"a {asset.description}"

    material = asset.material_hint or "appropriate realistic materials"
    style = asset.style_hint or "photorealistic product photography"

    return f"""Generate a professional product photography image of {base}.

Style Requirements:
- {style}
- Materials: {material}
- Single isolated object, centered in frame
- Pure white or light gray studio background
- Soft, even studio lighting (3-point lighting setup)
- Front-facing view with slight elevation (15-20 degrees from front)
- Object fills approximately 70-80% of frame
- Sharp focus throughout, high detail
- No shadows on background (floating appearance)
- Photorealistic quality suitable for 3D reconstruction
- Square aspect ratio (1:1)

The image should look like a professional product catalog photo suitable for converting to a 3D model."""
