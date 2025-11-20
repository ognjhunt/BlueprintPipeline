import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

# SAM 3D Objects inference utilities live under notebook/
NOTEBOOK_ROOTS = [
    Path(__file__).parent / "notebook",
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
        default_cfg = Path("/mnt/gcs/sam3d/checkpoints/hf/pipeline.yaml")
        fallback_cfg = Path("/workspace/sam3d-objects/checkpoints/hf/pipeline.yaml")
        if default_cfg.is_file():
            config_path = default_cfg
        elif fallback_cfg.is_file():
            config_path = fallback_cfg

    if config_path is None or not config_path.is_file():
        print(
            "[SAM3D] ERROR: SAM 3D config not found. Set SAM3D_CONFIG_PATH or "
            "sync checkpoints under /mnt/gcs/sam3d/checkpoints/hf/pipeline.yaml",
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
            mesh.export(str(out_dir / "mesh.glb"))
            print(f"[SAM3D] Saved mesh for obj {oid}")

    print("[SAM3D] Done.")


if __name__ == "__main__":
    main()
