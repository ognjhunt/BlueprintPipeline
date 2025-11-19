import os
import sys
import shutil
from pathlib import Path

from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology


def parse_classes_from_env() -> list[str]:
    """
    Read GSAM2_CLASSES from env, split on commas.
    If not set, fall back to a reasonable indoor-scene default.
    """
    raw = os.environ.get("GSAM2_CLASSES", "")
    if raw.strip():
        return [c.strip() for c in raw.split(",") if c.strip()]

    # Default: indoor-ish things that matter for robotics
    return [
        "chair",
        "table",
        "desk",
        "sofa",
        "couch",
        "bed",
        "cabinet",
        "drawer",
        "door",
        "window",
        "shelf",
        "lamp",
        "monitor",
        "television",
        "robot",
        "box",
        "plant",
    ]


def main() -> None:
    images_prefix = os.environ.get("IMAGES_PREFIX")
    seg_prefix = os.environ.get("SEG_PREFIX")
    bucket = os.environ.get("BUCKET", "")

    if not images_prefix or not seg_prefix:
        print(
            "[GSAM] IMAGES_PREFIX and SEG_PREFIX env vars are required",
            file=sys.stderr,
        )
        sys.exit(1)

    root = Path("/mnt/gcs")
    images_dir = root / images_prefix
    out_dir = root / seg_prefix
    work_images_dir = out_dir / "images"
    dataset_dir = out_dir / "dataset"

    print(f"[GSAM] Bucket: {bucket}")
    print(f"[GSAM] Images dir: {images_dir}")
    print(f"[GSAM] Output dir: {out_dir}")
    print(f"[GSAM] Work images dir: {work_images_dir}")
    print(f"[GSAM] Dataset dir (annotations): {dataset_dir}")

    if not images_dir.is_dir():
        print(
            f"[GSAM] ERROR: images directory does not exist: {images_dir}",
            file=sys.stderr,
        )
        # List filesystem for debugging
        try:
            for p in root.glob("**/*"):
                print("[GSAM] FS:", p)
        except Exception as e:
            print("[GSAM] Failed to list /mnt/gcs:", e, file=sys.stderr)
        sys.exit(1)

    work_images_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Collect PNG + JPG + JPEG images (case-insensitive)
    exts_to_scan = ["*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG"]
    image_paths: list[Path] = []
    for pattern in exts_to_scan:
        image_paths.extend(images_dir.glob(pattern))

    image_paths = sorted(set(image_paths))

    if not image_paths:
        print(
            f"[GSAM] ERROR: no .png/.jpg/.jpeg images found in {images_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    # OPTIONAL: only process a specific file name, e.g. room.png
    target_name = os.environ.get("GSAM_TARGET_IMAGE", "").strip()
    if target_name:
        filtered = [p for p in image_paths if p.name == target_name]
        if not filtered:
            print(
                f"[GSAM] WARNING: GSAM_TARGET_IMAGE={target_name} not found in {images_dir}",
                file=sys.stderr,
            )
        else:
            image_paths = filtered

    print(f"[GSAM] Found {len(image_paths)} images (png/jpg/jpeg)")

    # Copy images into work_images_dir so autodistill can treat this as a dataset folder
    for src in image_paths:
        dst = work_images_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
    print(f"[GSAM] Copied images into {work_images_dir}")

    classes = parse_classes_from_env()
    print(f"[GSAM] Using ontology classes: {classes}")

    ontology = CaptionOntology({name: name for name in classes})

    # Instantiate the Grounded SAM base model (GroundingDINO + SAM1)
    base_model = GroundedSAM(ontology=ontology)

    # Run labeling for each extension type that actually exists in work_images_dir
    present_exts = set(
        p.suffix.lower()
        for p in work_images_dir.iterdir()
        if p.is_file()
    )
    print(f"[GSAM] Present extensions in work_images_dir: {present_exts}")

    supported_exts = [".png", ".jpg", ".jpeg"]

    any_labeled = False
    for ext in supported_exts:
        if ext in present_exts:
            print(f"[GSAM] Running GroundedSAM.label() on *{ext} images")
            base_model.label(
                input_folder=str(work_images_dir),
                output_folder=str(dataset_dir),
                extension=ext,
            )
            any_labeled = True

    if not any_labeled:
        print(
            "[GSAM] WARNING: No supported extensions found after copy; "
            "nothing was labeled.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("[GSAM] Labeling done. Check the dataset/ directory for annotations.")
    print("[GSAM] Listing contents of output dir for sanity:")
    for p in sorted(out_dir.rglob("*")):
        rel = p.relative_to(out_dir)
        print(f"  - {rel}")

    print("[GSAM] Completed segmentation pipeline.")


if __name__ == "__main__":
    main()
