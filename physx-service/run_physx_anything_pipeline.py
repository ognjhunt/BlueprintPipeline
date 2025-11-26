#!/usr/bin/env python
"""
Simple driver that runs the PhysX-Anything pipeline on a single input image
and copies the first produced GLB + URDF into a target output directory.

Usage (called from physx_service.py):

    python run_physx_anything_pipeline.py \
        --input_image /tmp/physx_anything/req_123/input.png \
        --output_dir /tmp/physx_anything/req_123/output
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def run_cmd(args, cwd: Path) -> None:
    print("[PHYSX-PIPELINE]", " ".join(str(a) for a in args), " (cwd=", cwd, ")")
    subprocess.run(args, check=True, cwd=str(cwd))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    input_image = Path(args.input_image).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_image.is_file():
        raise SystemExit(f"Input image not found: {input_image}")

    # ------------------------------------------------------------------
    # 1) Prepare a demo directory for this image
    # ------------------------------------------------------------------
    demo_dir = repo_root / "demo_from_service"
    test_demo_dir = repo_root / "test_demo"

    # Clean any previous runs to avoid mixing outputs.
    shutil.rmtree(demo_dir, ignore_errors=True)
    shutil.rmtree(test_demo_dir, ignore_errors=True)

    demo_dir.mkdir(parents=True, exist_ok=True)

    # Copy the image into demo_dir (the VLM script expects a directory)
    target_img = demo_dir / input_image.name
    shutil.copy2(input_image, target_img)
    print(f"[PHYSX-PIPELINE] Copied input image -> {target_img}")

    # ------------------------------------------------------------------
    # 2) Run the official pipeline scripts (per the project README)
    #    NOTE: if the repo changes these script names or arguments,
    #    update this section accordingly.
    # ------------------------------------------------------------------

    # Step 1: Vision-language model demo
    #   python 1_vlm_demo.py --demo_path ./demo --save_part_ply True \
    #          --remove_bg False --ckpt ./pretrain/vlm
    cmd1 = [
        sys.executable,
        "1_vlm_demo.py",
        "--demo_path",
        str(demo_dir),
        "--save_part_ply",
        "True",
        "--remove_bg",
        "False",
        "--ckpt",
        "./pretrain/vlm",
    ]
    run_cmd(cmd1, repo_root)

    # Step 2: Decoder
    cmd2 = [sys.executable, "2_decoder.py"]
    run_cmd(cmd2, repo_root)

    # Step 3: Split
    cmd3 = [sys.executable, "3_split.py"]
    run_cmd(cmd3, repo_root)

    # Step 4: Sim-ready generation
    #   python 4_simready_gen.py --voxel_define 32 --basepath ./test_demo \
    #          --process 0 --fixed_base 0 --deformable 0
    cmd4 = [
        sys.executable,
        "4_simready_gen.py",
        "--voxel_define",
        "32",
        "--basepath",
        str(test_demo_dir),
        "--process",
        "0",
        "--fixed_base",
        "0",
        "--deformable",
        "0",
    ]
    run_cmd(cmd4, repo_root)

    # ------------------------------------------------------------------
    # 3) Locate the first GLB and URDF produced under test_demo_dir,
    #    and copy them into output_dir as part.glb / part.urdf
    # ------------------------------------------------------------------
    if not test_demo_dir.is_dir():
        raise SystemExit(f"Expected test_demo directory at {test_demo_dir}, but it does not exist")

    mesh_path = None
    urdf_path = None

    for p in test_demo_dir.rglob("*.glb"):
        mesh_path = p
        break

    for p in test_demo_dir.rglob("*.urdf"):
        urdf_path = p
        break

    if mesh_path is None or urdf_path is None:
        raise SystemExit(
            f"Could not find both mesh (.glb) and URDF (.urdf) in {test_demo_dir} "
            f"(mesh={mesh_path}, urdf={urdf_path})"
        )

    out_mesh = output_dir / "part.glb"
    out_urdf = output_dir / "part.urdf"
    shutil.copy2(mesh_path, out_mesh)
    shutil.copy2(urdf_path, out_urdf)

    print(f"[PHYSX-PIPELINE] Exported mesh -> {out_mesh}")
    print(f"[PHYSX-PIPELINE] Exported URDF -> {out_urdf}")


if __name__ == "__main__":
    main()
