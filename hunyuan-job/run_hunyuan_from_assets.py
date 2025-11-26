#!/usr/bin/env python

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, List, Set

from PIL import Image
import trimesh  # for OBJ -> GLB without Blender

# Root where GCS is mounted in the container
GCS_ROOT = Path("/mnt/gcs")

# Where we cloned Hunyuan3D-2.1 in the Dockerfile
HUNYUAN_REPO_ROOT = Path(os.getenv("HUNYUAN_REPO_ROOT", "/app/Hunyuan3D-2.1"))

# Make the Hunyuan packages importable
sys.path.insert(0, str(HUNYUAN_REPO_ROOT))
sys.path.insert(0, str(HUNYUAN_REPO_ROOT / "hy3dshape"))
sys.path.insert(0, str(HUNYUAN_REPO_ROOT / "hy3dpaint"))

# -------------------------------------------------------------------
# Torchvision compatibility shim (fixes basicsr/RealESRGAN imports)
# This uses Tencent's official torchvision_fix.py from the repo.
# -------------------------------------------------------------------
try:
    from torchvision_fix import apply_fix as _torchvision_apply_fix  # type: ignore
except Exception as e:
    print(
        f"[HUNYUAN] WARNING: torchvision_fix not available ({e}); "
        f"RealESRGAN/basicsr may fail with functional_tensor import errors.",
        file=sys.stderr,
    )
else:
    try:
        if _torchvision_apply_fix():
            print("[HUNYUAN] Torchvision functional_tensor compatibility fix applied")
        else:
            print(
                "[HUNYUAN] WARNING: torchvision_fix.apply_fix() reported failure",
                file=sys.stderr,
            )
    except Exception as e:
        print(
            f"[HUNYUAN] WARNING: error while applying torchvision_fix: {e}",
            file=sys.stderr,
        )

from hy3dshape.rembg import BackgroundRemover
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig  # type: ignore


def getenv_bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes", "on"}


def convert_glb_to_usdz(glb_path: Path, usdz_path: Path) -> bool:
    """Convert a GLB into USDZ using the usd_from_gltf CLI if available."""
    usd_from_gltf = shutil.which("usd_from_gltf")
    if usd_from_gltf is None:
        print("[HUNYUAN] usd_from_gltf not found; skipping USDZ export", file=sys.stderr)
        return False

    usdz_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [usd_from_gltf, str(glb_path), "-o", str(usdz_path)]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        print(f"[HUNYUAN] Saved USDZ -> {usdz_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(
            f"[HUNYUAN] WARNING: usd_from_gltf failed for {glb_path}: {e.stderr or e}",
            file=sys.stderr,
        )
        return False


def pick_reference_image(root: Path, obj: dict) -> Optional[Path]:
    """
    Choose the best reference image for a given object, using the same
    fields as the SAM3D job: preferred_view -> multiview view_0.png -> crop.
    """
    crop_rel = obj.get("crop_path")
    preferred_rel = obj.get("preferred_view")
    mv_rel = obj.get("multiview_dir")

    candidates: List[Path] = []

    if preferred_rel:
        candidates.append(root / preferred_rel)
    if mv_rel:
        candidates.append(root / mv_rel / "view_0.png")
    if crop_rel:
        candidates.append(root / crop_rel)

    for p in candidates:
        if p.is_file():
            return p

    return None


def load_asset_plan(assets_root: Path) -> Optional[Path]:
    """scene_assets.json is produced by the assets-plan job."""
    plan_path = assets_root / "scene_assets.json"
    if not plan_path.is_file():
        print(f"[HUNYUAN] ERROR: assets plan not found at {plan_path}", file=sys.stderr)
        return None
    return plan_path


def main() -> None:
    bucket = os.getenv("BUCKET", "")
    scene_id = os.getenv("SCENE_ID", "")
    assets_prefix = os.getenv("ASSETS_PREFIX")  # scenes/<sceneId>/assets

    # Quality knobs (can be overridden with env vars if needed)
    num_steps = int(os.getenv("HUNYUAN_NUM_STEPS", "75"))        # shape steps (default in code is 50)
    max_num_view = int(os.getenv("HUNYUAN_MAX_NUM_VIEW", "9"))   # docs say 6–9
    resolution = int(os.getenv("HUNYUAN_RESOLUTION", "768"))     # docs say 512 or 768
    model_path = os.getenv("HUNYUAN_MODEL_PATH", "tencent/Hunyuan3D-2.1")

    if not assets_prefix:
        print("[HUNYUAN] ASSETS_PREFIX is required", file=sys.stderr)
        sys.exit(1)

    assets_root = GCS_ROOT / assets_prefix
    plan_path = load_asset_plan(assets_root)
    if plan_path is None:
        sys.exit(1)

    print(f"[HUNYUAN] Bucket={bucket}")
    print(f"[HUNYUAN] Scene={scene_id}")
    print(f"[HUNYUAN] Assets root={assets_root}")
    print(f"[HUNYUAN] Using model: {model_path}")
    print(f"[HUNYUAN] num_steps={num_steps}, max_num_view={max_num_view}, resolution={resolution}")

    plan = json.loads(plan_path.read_text())
    objects = plan.get("objects", [])
    print(f"[HUNYUAN] Loaded plan with {len(objects)} objects")

    # Focus only on static objects with IDs
    static_objects: List[dict] = [
        obj for obj in objects
        if obj.get("type") == "static" and obj.get("id") is not None
    ]
    print(f"[HUNYUAN] Found {len(static_objects)} static objects to process")

    if not static_objects:
        print("[HUNYUAN] No static objects to process; exiting.")
        return

    # ------------------------------------------------------------------
    # Stage 1: Shape generation (image -> mesh.glb for each static obj)
    # ------------------------------------------------------------------
    print("[HUNYUAN] Loading shape-generation pipeline...")
    pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)
    print("[HUNYUAN] Shape pipeline loaded")

    rembg = BackgroundRemover()
    successful_obj_ids: Set[str] = set()

    for obj in static_objects:
        oid = obj["id"]

        image_path = pick_reference_image(GCS_ROOT, obj)
        if image_path is None:
            print(
                f"[HUNYUAN] WARNING: no reference image found for obj {oid} (shape stage)",
                file=sys.stderr,
            )
            continue

        print(f"[HUNYUAN] [Shape] Generating mesh for obj {oid} from {image_path}")

        try:
            image = Image.open(image_path)
        except Exception as e:
            print(
                f"[HUNYUAN] ERROR: failed to open image for obj {oid}: {e}",
                file=sys.stderr,
            )
            continue

        # RGBA + optional background removal
        if image.mode == "RGB":
            image = rembg(image)
        if image.mode != "RGBA":
            image = image.convert("RGBA")

        # --- Shape generation ---------------------------------------
        try:
            meshes = pipeline_shapegen(image=image, num_inference_steps=num_steps)
        except Exception as e:
            print(
                f"[HUNYUAN] ERROR: shape pipeline failed for obj {oid}: {e}",
                file=sys.stderr,
            )
            continue

        # Handle both [mesh] and [[mesh]] styles robustly
        if isinstance(meshes, list):
            first = meshes[0] if meshes else None
            if isinstance(first, list):
                mesh = first[0] if first else None
            else:
                mesh = first
        else:
            mesh = meshes

        if mesh is None:
            print(
                f"[HUNYUAN] ERROR: shape pipeline returned no mesh for obj {oid}",
                file=sys.stderr,
            )
            continue

        out_dir = assets_root / f"obj_{oid}"
        out_dir.mkdir(parents=True, exist_ok=True)

        mesh_glb_path = out_dir / "mesh.glb"
        try:
            mesh.export(str(mesh_glb_path))
        except Exception as e:
            print(
                f"[HUNYUAN] ERROR: failed to export mesh.glb for obj {oid}: {e}",
                file=sys.stderr,
            )
            continue

        print(f"[HUNYUAN] Saved mesh.glb for obj {oid} -> {mesh_glb_path}")
        successful_obj_ids.add(oid)

    print(
        f"[HUNYUAN] Shape generation finished. "
        f"{len(successful_obj_ids)}/{len(static_objects)} static objects succeeded."
    )

    # Free shape pipeline GPU memory before loading the paint model
    try:
        import torch
    except ImportError:
        torch = None  # type: ignore[assignment]

    del pipeline_shapegen
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("[HUNYUAN] Cleared CUDA cache after shape generation")

    # ------------------------------------------------------------------
    # Stage 2: Texture generation (mesh.glb -> textured model.glb/usdz)
    # ------------------------------------------------------------------
    print("[HUNYUAN] Configuring texture (paint) pipeline...")
    conf = Hunyuan3DPaintConfig(max_num_view=max_num_view, resolution=resolution)
    # Make sure paths are correct inside the cloned repo
    conf.realesrgan_ckpt_path = str(
        HUNYUAN_REPO_ROOT / "hy3dpaint" / "ckpt" / "RealESRGAN_x4plus.pth"
    )
    conf.multiview_cfg_path = str(
        HUNYUAN_REPO_ROOT / "hy3dpaint" / "cfgs" / "hunyuan-paint-pbr.yaml"
    )
    # This matches examples which use a custom pipeline name under hy3dpaint
    conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"

    # --- SAFE INITIALIZATION WITH FALLBACK --------------------------
    paint_pipeline = None
    try:
        paint_pipeline = Hunyuan3DPaintPipeline(conf)
        print("[HUNYUAN] Paint pipeline loaded")
    except ModuleNotFoundError as e:
        # This is where the 'diffusers_modules.local.modules' error happens
        print(
            "[HUNYUAN] ERROR: texture pipeline requires custom diffusers dynamic "
            "modules ('diffusers_modules.local.*') that are not available in this "
            "Cloud Run environment. Skipping texture stage; meshes will remain "
            "untextured.",
            file=sys.stderr,
        )
    except Exception as e:
        print(
            f"[HUNYUAN] ERROR: failed to initialize paint pipeline: {e}. "
            "Skipping texture stage; meshes will remain untextured.",
            file=sys.stderr,
        )

    if paint_pipeline is None:
        # Fallback: at least ensure asset.glb exists for each successful object
        for obj in static_objects:
            oid = obj["id"]
            if oid not in successful_obj_ids:
                continue
            out_dir = assets_root / f"obj_{oid}"
            mesh_glb_path = out_dir / "mesh.glb"
            asset_glb_path = out_dir / "asset.glb"
            if mesh_glb_path.is_file() and not asset_glb_path.exists():
                shutil.copy(mesh_glb_path, asset_glb_path)
                print(f"[HUNYUAN] (fallback) Copied mesh.glb -> asset.glb for obj {oid}")

        print("[HUNYUAN] Done (shape-only; texture stage skipped).")
        return

    # --- FULL TEXTURE LOOP (only runs if paint_pipeline loaded) -----
    for obj in static_objects:
        oid = obj["id"]
        if oid not in successful_obj_ids:
            print(
                f"[HUNYUAN] Skipping texture for obj {oid} "
                f"(no mesh.glb from shape stage)",
                file=sys.stderr,
            )
            continue

        image_path = pick_reference_image(GCS_ROOT, obj)
        if image_path is None:
            print(
                f"[HUNYUAN] WARNING: no reference image found for obj {oid} "
                f"during texture stage",
                file=sys.stderr,
            )
            continue

        out_dir = assets_root / f"obj_{oid}"
        mesh_glb_path = out_dir / "mesh.glb"

        if not mesh_glb_path.is_file():
            print(
                f"[HUNYUAN] WARNING: mesh.glb missing for obj {oid} at {mesh_glb_path}",
                file=sys.stderr,
            )
            continue

        print(f"[HUNYUAN] [Texture] Texturing mesh for obj {oid}")

        model_obj_path = out_dir / "model.obj"
        model_glb_path = out_dir / "model.glb"

        # IMPORTANT: disable Hunyuan's internal GLB export (uses Blender/bpy),
        # we convert OBJ -> GLB ourselves using trimesh.
        try:
            textured_obj_path = paint_pipeline(
                mesh_path=str(mesh_glb_path),
                image_path=str(image_path),
                output_mesh_path=str(model_obj_path),
                save_glb=False,   # NO Blender-based GLB export
            )
        except Exception as e:
            print(
                f"[HUNYUAN] ERROR: paint pipeline failed for obj {oid}: {e}",
                file=sys.stderr,
            )
            continue

        # Hunyuan returns the OBJ path; ensure we have it
        if textured_obj_path is None:
            textured_obj_path = model_obj_path

        # --- Convert OBJ -> GLB without Blender (using trimesh) -----
        try:
            # Using scene mode to keep materials/textures if present
            scene = trimesh.load(str(textured_obj_path), force="scene")
            scene.export(str(model_glb_path))
            print(f"[HUNYUAN] Converted OBJ -> GLB for obj {oid} -> {model_glb_path}")
        except Exception as e:
            print(
                f"[HUNYUAN] WARNING: failed to convert OBJ to GLB for obj {oid}: {e}",
                file=sys.stderr,
            )

        if model_glb_path.is_file():
            print(
                f"[HUNYUAN] Saved textured model.glb for obj {oid} -> {model_glb_path}"
            )
        else:
            print(
                f"[HUNYUAN] WARNING: textured GLB not found for obj {oid}",
                file=sys.stderr,
            )

        # Keep SAM3D-style naming: asset.glb as an alias for model.glb
        asset_glb_path = out_dir / "asset.glb"
        if model_glb_path.is_file() and not asset_glb_path.exists():
            shutil.copy(model_glb_path, asset_glb_path)
            print(f"[HUNYUAN] Copied model.glb -> asset.glb for obj {oid}")

        # --- Optional USDZ export -----------------------------------
        if model_glb_path.is_file():
            usdz_path = out_dir / "model.usdz"
            convert_glb_to_usdz(model_glb_path, usdz_path)

    print("[HUNYUAN] Done.")


if __name__ == "__main__":
    main()
