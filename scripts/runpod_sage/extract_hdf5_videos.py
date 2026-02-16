#!/usr/bin/env python3
"""
Extract per-demo MP4 videos and sample frames from a SAGE robomimic HDF5 dataset.

Usage:
    python extract_hdf5_videos.py --hdf5 path/to/dataset.hdf5 --output path/to/output/
    python extract_hdf5_videos.py --hdf5 path/to/dataset.hdf5 --output out/ --cameras agentview agentview_2
    python extract_hdf5_videos.py --hdf5 path/to/dataset.hdf5 --output out/ --frames-only --every-n 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np


def extract_videos(
    hdf5_path: Path,
    output_dir: Path,
    cameras: list[str],
    fps: int = 10,
    frames_only: bool = False,
    every_n: int = 1,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(hdf5_path), "r") as f:
        if "data" not in f:
            print(f"ERROR: No 'data' group in {hdf5_path}")
            sys.exit(1)

        data = f["data"]
        demo_keys = sorted(data.keys(), key=lambda k: int(k.split("_")[-1]))
        print(f"Found {len(demo_keys)} demos in {hdf5_path}")

        for demo_key in demo_keys:
            demo = data[demo_key]
            if "obs" not in demo:
                print(f"  {demo_key}: no obs group, skipping")
                continue

            obs = demo["obs"]
            for cam in cameras:
                rgb_key = f"{cam}_rgb"
                if rgb_key not in obs:
                    print(f"  {demo_key}/{cam}: no RGB data, skipping")
                    continue

                rgb = obs[rgb_key][:]  # (T, H, W, 3)
                T, H, W, _ = rgb.shape
                is_black = rgb.max() == 0
                rgb_std = rgb.astype(np.float32).std()

                status = "BLACK" if is_black else f"std={rgb_std:.1f}"
                print(f"  {demo_key}/{cam}: {T} frames, {W}x{H}, {status}")

                if is_black:
                    continue

                if frames_only or every_n > 1:
                    frames_dir = output_dir / f"{demo_key}_{cam}"
                    frames_dir.mkdir(parents=True, exist_ok=True)
                    from PIL import Image

                    for t in range(0, T, every_n):
                        img = Image.fromarray(rgb[t])
                        img.save(frames_dir / f"frame_{t:04d}.png")
                    print(f"    -> Saved {len(range(0, T, every_n))} frames to {frames_dir}/")

                if not frames_only:
                    mp4_path = output_dir / f"{demo_key}_{cam}.mp4"
                    try:
                        import imageio.v3 as iio
                        iio.imwrite(str(mp4_path), rgb, fps=fps, codec="libx264")
                    except (ImportError, TypeError):
                        import imageio
                        imageio.mimwrite(str(mp4_path), [frame for frame in rgb], fps=fps)
                    print(f"    -> {mp4_path}")

        # Also print depth statistics for diagnostics
        print("\nDepth diagnostics:")
        for demo_key in demo_keys:
            obs = data[demo_key].get("obs", {})
            for cam in cameras:
                depth_key = f"{cam}_depth"
                if depth_key not in obs:
                    continue
                depth = obs[depth_key][:]
                finite_count = np.isfinite(depth).sum()
                total = depth.size
                pct = 100 * finite_count / total if total > 0 else 0
                nonzero = (depth != 0).sum()
                print(f"  {demo_key}/{cam}_depth: {pct:.1f}% finite, {nonzero}/{total} nonzero")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract videos/frames from SAGE HDF5 dataset")
    parser.add_argument("--hdf5", required=True, type=Path, help="Path to dataset.hdf5")
    parser.add_argument("--output", required=True, type=Path, help="Output directory")
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=["agentview", "agentview_2"],
        help="Camera names to extract (default: agentview agentview_2)",
    )
    parser.add_argument("--fps", type=int, default=10, help="Video FPS (default: 10)")
    parser.add_argument(
        "--frames-only",
        action="store_true",
        help="Save individual PNG frames instead of MP4",
    )
    parser.add_argument(
        "--every-n",
        type=int,
        default=1,
        help="Save every Nth frame (default: 1, all frames). Also saves frames alongside MP4.",
    )
    args = parser.parse_args()

    if not args.hdf5.exists():
        print(f"ERROR: {args.hdf5} not found")
        sys.exit(1)

    extract_videos(
        hdf5_path=args.hdf5,
        output_dir=args.output,
        cameras=args.cameras,
        fps=args.fps,
        frames_only=args.frames_only,
        every_n=args.every_n,
    )


if __name__ == "__main__":
    main()
