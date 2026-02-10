#!/usr/bin/env python3
"""Standalone Hunyuan3D shape generation for variation assets.

Uploaded to VM and executed remotely. Generates a single GLB mesh from
a reference image using Hunyuan3D-2/2.1.

Usage:
    python variation_hunyuan_remote.py \
        --image /tmp/variation_inputs/drinking_vessel/reference.png \
        --output /tmp/variation_outputs/drinking_vessel/model.glb \
        --hunyuan-path /home/nijelhunt1/3D-RE-GEN
"""

import argparse
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Hunyuan3D shape generation for variation assets")
    parser.add_argument("--image", required=True, help="Path to reference image PNG")
    parser.add_argument("--output", required=True, help="Path to output GLB")
    parser.add_argument("--hunyuan-path", required=True, help="Path to Hunyuan3D repo")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps")
    args = parser.parse_args()

    hunyuan = Path(args.hunyuan_path)
    image_path = Path(args.image)
    output_path = Path(args.output)

    if not image_path.is_file():
        print(f"ERROR: Image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    # Add Hunyuan3D to path
    for subdir in [hunyuan, hunyuan / "hy3dshape", hunyuan / "hy3dpaint",
                   hunyuan / "hy3dgen", hunyuan / "hy3dgen" / "shapegen"]:
        if subdir.is_dir():
            sys.path.insert(0, str(subdir))
    sys.path.insert(0, str(hunyuan))

    # Probe import paths: hy3dgen (Hunyuan3D-2) vs hy3dshape (Hunyuan3D-2.1)
    Pipeline = None
    BackgroundRemover = None

    try:
        from hy3dgen.shapegen.pipelines import Hunyuan3DDiTFlowMatchingPipeline
        Pipeline = Hunyuan3DDiTFlowMatchingPipeline
        print("[VARIATION] Using hy3dgen (Hunyuan3D-2) imports")
    except ImportError:
        pass

    if Pipeline is None:
        try:
            from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
            Pipeline = Hunyuan3DDiTFlowMatchingPipeline
            print("[VARIATION] Using hy3dshape (Hunyuan3D-2.1) imports")
        except ImportError:
            pass

    if Pipeline is None:
        print("ERROR: Could not import Hunyuan3D pipeline from hy3dgen or hy3dshape",
              file=sys.stderr)
        sys.exit(1)

    try:
        from hy3dgen.rembg import BackgroundRemover as BR
        BackgroundRemover = BR
    except ImportError:
        try:
            from hy3dshape.rembg import BackgroundRemover as BR
            BackgroundRemover = BR
        except ImportError:
            print("[VARIATION] WARNING: No BackgroundRemover found, skipping background removal")

    # Load model
    print(f"[VARIATION] Loading Hunyuan3D model...")
    t0 = time.time()
    pipeline = Pipeline.from_pretrained("tencent/Hunyuan3D-2")
    print(f"[VARIATION] Model loaded in {time.time() - t0:.1f}s")

    # Load and prepare image
    from PIL import Image
    image = Image.open(str(image_path))
    print(f"[VARIATION] Image: {image.size}, mode={image.mode}")

    # Remove background if possible
    if BackgroundRemover is not None and image.mode == "RGB":
        print("[VARIATION] Removing background...")
        rembg = BackgroundRemover()
        image = rembg(image)

    if image.mode != "RGBA":
        image = image.convert("RGBA")

    # Generate mesh
    print(f"[VARIATION] Generating shape ({args.steps} steps)...")
    t0 = time.time()
    mesh = pipeline(image=image, num_inference_steps=args.steps)

    # Unwrap nested lists (pipeline sometimes returns [[mesh]])
    for _ in range(3):
        if isinstance(mesh, (list, tuple)) and len(mesh) > 0:
            mesh = mesh[0]
        else:
            break

    if mesh is None:
        print("ERROR: Pipeline returned no mesh", file=sys.stderr)
        sys.exit(1)

    duration = time.time() - t0
    print(f"[VARIATION] Shape generated in {duration:.1f}s")

    # Save GLB
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(output_path))
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"[VARIATION] Saved: {output_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
