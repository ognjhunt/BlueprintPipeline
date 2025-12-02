import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from google.cloud import storage

from depth_anything_3.api import DepthAnything3


def main() -> None:
    dataset_prefix = os.environ.get("DATASET_PREFIX")
    out_prefix = os.environ.get("OUT_PREFIX")
    bucket = os.environ.get("BUCKET", "")
    scene_id = os.environ.get("SCENE_ID", "")

    if not dataset_prefix or not out_prefix:
        print(
            "[DA3] DATASET_PREFIX and OUT_PREFIX env vars are required",
            file=sys.stderr,
        )
        sys.exit(1)

    root = Path("/mnt/gcs")
    dataset_dir = root / dataset_prefix
    out_dir = root / out_prefix

    print(f"[DA3] Bucket: {bucket}")
    print(f"[DA3] Scene ID: {scene_id}")
    print(f"[DA3] Dataset dir: {dataset_dir}")
    print(f"[DA3] Output dir: {out_dir}")

    if not dataset_dir.is_dir():
        print(
            f"[DA3] ERROR: dataset directory does not exist: {dataset_dir}",
            file=sys.stderr,
        )
        try:
            for p in root.glob("**/*"):
                print("[DA3] FS:", p)
        except Exception as e:
            print("[DA3] Failed to list /mnt/gcs:", e, file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    # -------- Collect scene images from dataset --------
    # Prefer dataset/valid/images; fallback to dataset/images
    search_dirs = [
        dataset_dir / "valid" / "images",
        dataset_dir / "images",
    ]

    image_paths: list[str] = []
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")

    for d in search_dirs:
        if not d.is_dir():
            continue
        candidates: list[str] = []
        for pattern in exts:
            candidates.extend(str(p) for p in d.glob(pattern))
        if candidates:
            image_paths = sorted(set(candidates))
            print(f"[DA3] Using images from {d}")
            break

    if not image_paths:
        print(
            f"[DA3] ERROR: no images found in dataset under {dataset_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[DA3] Found {len(image_paths)} image(s) for DA3:")
    for p in image_paths:
        print(f"  - {p}")

    # -------- Load Depth Anything 3 --------
    model_id = os.environ.get("MODEL_ID", "depth-anything/da3-large")
    print(f"[DA3] Using model: {model_id}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DA3] Torch device: {device}")

    model = DepthAnything3.from_pretrained(model_id)
    model = model.to(device=device)

    # -------- Run inference --------
    print("[DA3] Running inference -> GLB")
    prediction = model.inference(
        image_paths,
        export_dir=str(out_dir),
        export_format="glb",  # per DA3 docs: glb, npz, ply, mini_npz, gs_ply, gs_video
    )

    depth = np.asarray(prediction.depth)        # [N, H, W]
    conf = np.asarray(prediction.conf)          # [N, H, W]
    extrinsics = np.asarray(prediction.extrinsics)  # [N, 3, 4]
    intrinsics = np.asarray(prediction.intrinsics)  # [N, 3, 3]

    # Save cameras on their own
    cams_path = out_dir / "cameras_da3.npz"
    try:
        np.savez_compressed(
            cams_path,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            image_paths=np.array(image_paths, dtype="U"),
        )
        print(
            f"[DA3] Saved cameras to {cams_path} "
            f"(extrinsics {extrinsics.shape}, intrinsics {intrinsics.shape})"
        )
    except Exception as e:
        print(f"[DA3] WARNING: failed to save cameras_da3.npz: {e}", file=sys.stderr)

    # Save depth + confidence + cameras in a single bundle for later steps
    # IMPORTANT: Write to temp file first, then upload atomically to avoid
    # multiple GCS events that would trigger duplicate layout-jobs
    geom_path = out_dir / "da3_geom.npz"
    try:
        # Write to local temp file first
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.npz') as tmp_file:
            tmp_path = tmp_file.name
            np.savez_compressed(
                tmp_path,
                depth=depth,
                conf=conf,
                extrinsics=extrinsics,
                intrinsics=intrinsics,
                image_paths=np.array(image_paths, dtype="U"),
            )

        # Upload atomically to GCS (single event)
        if bucket and out_prefix:
            try:
                storage_client = storage.Client()
                bucket_obj = storage_client.bucket(bucket)
                blob_path = f"{out_prefix}/da3_geom.npz"
                blob = bucket_obj.blob(blob_path)
                blob.upload_from_filename(tmp_path)
                print(
                    f"[DA3] Uploaded geometry bundle to gs://{bucket}/{blob_path} "
                    f"(depth {depth.shape}, conf {conf.shape})"
                )
            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        else:
            # Fallback: move to GCS FUSE mount if bucket info not available
            import shutil
            shutil.move(tmp_path, geom_path)
            print(
                f"[DA3] Saved geometry bundle to {geom_path} "
                f"(depth {depth.shape}, conf {conf.shape})"
            )
    except Exception as e:
        print(f"[DA3] WARNING: failed to save da3_geom.npz: {e}", file=sys.stderr)

    # List major outputs
    print("[DA3] Exported files:")
    for ext in ("*.glb", "*.ply", "cameras_da3.npz", "da3_geom.npz"):
        for p in sorted(out_dir.glob(ext)):
            print(f" - {p.name}")

    print("[DA3] Done.")


if __name__ == "__main__":
    main()
