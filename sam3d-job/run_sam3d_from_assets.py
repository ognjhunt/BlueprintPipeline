import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

SAM3D_REPO_ROOT = Path(os.getenv("SAM3D_REPO_ROOT", Path("/app/sam3d-objects")))
SAM3D_CHECKPOINT_ROOT = Path(
    os.getenv("SAM3D_CHECKPOINT_ROOT", SAM3D_REPO_ROOT / "checkpoints")
)
SAM3D_HF_REPO = os.getenv("SAM3D_HF_REPO", "facebook/sam-3d-objects")
SAM3D_HF_REVISION = os.getenv("SAM3D_HF_REVISION")

# SAM 3D Objects inference utilities live under notebook/
NOTEBOOK_ROOTS = [
    Path(__file__).parent / "notebook",
    SAM3D_REPO_ROOT / "notebook",
    Path("/workspace/sam3d-objects/notebook"),
]
for nb_root in NOTEBOOK_ROOTS:
    if nb_root.is_dir():
        sys.path.append(str(nb_root))

try:
    from inference import Inference
except Exception as e:  # pragma: no cover - defensive import
    print(f"[SAM3D] ERROR: failed to import SAM 3D inference utilities: {e}", file=sys.stderr)
    raise


def build_alpha_mask(crop: np.ndarray, background: int = 240) -> np.ndarray:
    gray = crop.mean(axis=-1)
    mask = (np.abs(gray - background) > 1).astype(np.uint8)
    return mask


def load_rgba_with_mask(image_path: Path, polygon: Optional[list] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build an RGBA view for SAM 3D. We primarily rely on the background-masked
    crop produced upstream; if a polygon is provided and appears normalized,
    we fall back to the image-derived alpha to avoid coordinate mismatches.
    """
    image = Image.open(image_path).convert("RGB")
    rgb = np.array(image)
    mask = build_alpha_mask(rgb)
    return rgb, mask


def save_basecolor_texture(texture: np.ndarray, out_path: Path) -> Optional[Path]:
    """Persist a basecolor texture if the inference output provides one."""

    if texture is None:
        return None

    try:
        tex = np.asarray(texture)
        if tex.dtype != np.uint8:
            tex = np.clip(tex * 255.0, 0, 255).astype(np.uint8)
        image = Image.fromarray(tex)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(out_path)
        print(f"[SAM3D] Saved basecolor texture -> {out_path}")
        return out_path
    except Exception as e:  # pragma: no cover - best-effort export
        print(f"[SAM3D] WARNING: failed to save texture {out_path.name}: {e}", file=sys.stderr)
        return None


def convert_glb_to_usdz(glb_path: Path, usdz_path: Path) -> bool:
    """Convert a GLB into USDZ using the usd_from_gltf CLI if available."""

    usd_from_gltf = shutil.which("usd_from_gltf")
    if usd_from_gltf is None:
        print("[SAM3D] WARNING: usd_from_gltf not available; skipping USDZ export", file=sys.stderr)
        return False

    usdz_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [usd_from_gltf, str(glb_path), "-o", str(usdz_path)]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        print(f"[SAM3D] Saved USDZ -> {usdz_path}")
        return True
    except subprocess.CalledProcessError as e:  # pragma: no cover - runtime dependency
        print(
            f"[SAM3D] WARNING: usd_from_gltf failed for {glb_path}: {e.stderr or e}",
            file=sys.stderr,
        )
        return False


def download_sam3d_checkpoints(target_root: Path) -> Optional[Path]:
    """Fetch checkpoints from Hugging Face if they are not already present."""

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # pragma: no cover - optional dependency
        print(
            f"[SAM3D] WARNING: huggingface_hub not available, cannot download checkpoints ({exc})",
            file=sys.stderr,
        )
        return None

    allow_patterns = ["checkpoints/hf/**"]
    target_root.mkdir(parents=True, exist_ok=True)

    try:
        local_path = snapshot_download(
            repo_id=SAM3D_HF_REPO,
            repo_type="model",
            revision=SAM3D_HF_REVISION,
            allow_patterns=allow_patterns,
            local_dir=target_root.parent,
            local_dir_use_symlinks=False,
        )
        config_path = Path(local_path) / "checkpoints/hf/pipeline.yaml"
        if config_path.is_file():
            print(
                f"[SAM3D] Downloaded SAM 3D checkpoints from {SAM3D_HF_REPO} -> {config_path.parent}"
            )
            return config_path

        print(
            f"[SAM3D] WARNING: Download finished but pipeline.yaml not found under {config_path.parent}",
            file=sys.stderr,
        )
    except Exception as exc:  # pragma: no cover - runtime dependency
        print(
            f"[SAM3D] WARNING: Failed to download checkpoints from {SAM3D_HF_REPO}: {exc}",
            file=sys.stderr,
        )

    return None


def validate_checkpoint_bundle(config_path: Path) -> bool:
    """Ensure the pipeline config and nearby weights are present."""

    if not config_path.is_file():
        print(
            f"[SAM3D] ERROR: pipeline config is missing: {config_path}", file=sys.stderr
        )
        return False

    weights_dir = config_path.parent
    weight_files = sorted(weights_dir.glob("*.ckpt")) + sorted(
        weights_dir.glob("*.safetensors")
    )
    if not weight_files:
        print(
            f"[SAM3D] ERROR: no weight files found next to {config_path}. "
            "Download checkpoints with HUGGINGFACE_TOKEN/HF_TOKEN set or provide a custom SAM3D_CONFIG_PATH.",
            file=sys.stderr,
        )
        return False

    print(
        f"[SAM3D] Found {len(weight_files)} checkpoint files under {weights_dir}"
    )
    return True


def main() -> None:
    bucket = os.getenv("BUCKET", "")
    scene_id = os.getenv("SCENE_ID", "")
    assets_prefix = os.getenv("ASSETS_PREFIX")  # scenes/<sceneId>/assets
    sam3d_config_path = os.getenv("SAM3D_CONFIG_PATH")

    if not assets_prefix:
        print("[SAM3D] ASSETS_PREFIX is required", file=sys.stderr)
        sys.exit(1)

    root = Path("/mnt/gcs")
    assets_root = root / assets_prefix
    plan_path = assets_root / "scene_assets.json"

    print(f"[SAM3D] Bucket={bucket}")
    print(f"[SAM3D] Scene={scene_id}")
    print(f"[SAM3D] Assets root={assets_root}")

    if not plan_path.is_file():
        print(f"[SAM3D] ERROR: assets plan not found: {plan_path}", file=sys.stderr)
        sys.exit(1)

    config_path = None
    if sam3d_config_path:
        candidate = Path(sam3d_config_path)
        if candidate.is_file():
            config_path = candidate
    if config_path is None:
        default_cfg = SAM3D_CHECKPOINT_ROOT / "hf/pipeline.yaml"
        legacy_cfg = Path("/mnt/gcs/sam3d/checkpoints/hf/pipeline.yaml")
        fallback_cfg = Path("/workspace/sam3d-objects/checkpoints/hf/pipeline.yaml")
        for candidate in (default_cfg, legacy_cfg, fallback_cfg):
            if candidate.is_file():
                config_path = candidate
                break

    if config_path is None or not config_path.is_file():
        print("[SAM3D] No local pipeline.yaml found; attempting download from Hugging Face")
        config_path = download_sam3d_checkpoints(SAM3D_CHECKPOINT_ROOT)

    if config_path is None or not validate_checkpoint_bundle(config_path):
        print(
            "[SAM3D] ERROR: SAM 3D checkpoints unavailable. Set SAM3D_CONFIG_PATH to a valid pipeline.yaml, "
            "or ensure HUGGINGFACE_TOKEN/HF_TOKEN is set so checkpoints can be downloaded.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[SAM3D] Using config: {config_path}")
    inference = Inference(str(config_path), compile=False)

    plan = json.loads(plan_path.read_text())
    objs = plan.get("objects", [])
    print(f"[SAM3D] Loaded plan with {len(objs)} objects")

    for obj in objs:
        if obj.get("pipeline") != "sam3d":
            continue

        oid = obj.get("id")
        crop_rel = obj.get("crop_path")
        if not crop_rel:
            print(f"[SAM3D] WARNING: object {oid} missing crop_path", file=sys.stderr)
            continue

        crop_path = root / crop_rel
        if not crop_path.is_file():
            print(f"[SAM3D] WARNING: crop not found for obj {oid}: {crop_path}", file=sys.stderr)
            continue

        print(f"[SAM3D] Reconstructing object {oid} from {crop_path}")
        rgb, mask = load_rgba_with_mask(crop_path)
        output = inference(rgb, mask, seed=42)

        out_dir = assets_root / "static" / f"obj_{oid}"
        out_dir.mkdir(parents=True, exist_ok=True)

        if "gs" in output:
            output["gs"].save_ply(str(out_dir / "splat.ply"))
            print(f"[SAM3D] Saved gaussian splat for obj {oid}")

        mesh = output.get("mesh")
        if mesh is not None and hasattr(mesh, "export"):
            mesh_glb_path = out_dir / "mesh.glb"
            mesh.export(str(mesh_glb_path))
            print(f"[SAM3D] Saved mesh for obj {oid}")

            model_glb_path = out_dir / "model.glb"
            if not model_glb_path.exists():
                shutil.copy(mesh_glb_path, model_glb_path)
                print(f"[SAM3D] Saved mesh copy -> {model_glb_path.name}")

            texture = output.get("texture") or output.get("texture_image")
            save_basecolor_texture(texture, out_dir / "texture_0_basecolor.png")

            usdz_path = out_dir / "model.usdz"
            convert_glb_to_usdz(model_glb_path, usdz_path)

    print("[SAM3D] Done.")


if __name__ == "__main__":
    main()
