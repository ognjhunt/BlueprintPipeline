#!/usr/bin/env python3
"""
Generate mock Genie Sim local outputs for contract tests.

Usage:
    python fixtures/generate_mock_geniesim_local.py --output-dir ./tmp --episodes 2
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List


def _build_frame(step: int) -> Dict[str, object]:
    return {
        "step": step,
        "observation": {
            "joint_positions": [0.1 * step, 0.2 * step],
            "sensor": {"temperature": 300.0 + step},
        },
        "action": [0.01 * step, 0.02 * step],
        "timestamp": step / 30.0,
    }


def generate_mock_geniesim_local(
    output_dir: Path,
    run_id: str = "mock_run",
    episodes: int = 2,
) -> Path:
    """Generate mock Genie Sim local outputs.

    Args:
        output_dir: Base output directory.
        run_id: Identifier for the run.
        episodes: Number of episodes to generate.

    Returns:
        Path to generated output directory.
    """
    base_dir = output_dir / "geniesim_local" / run_id
    recordings_dir = base_dir / "recordings"
    metadata_dir = base_dir / "metadata"
    recordings_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    episodes_index: List[Dict[str, object]] = []
    total_frames = 0
    quality_scores: List[float] = []
    for idx in range(episodes):
        episode_id = f"episode_{idx:06d}"
        frames = [_build_frame(step) for step in range(12)]
        quality_score = 0.9 if idx % 2 == 0 else 0.6
        validation_passed = quality_score >= 0.7
        payload = {
            "episode_id": episode_id,
            "task_name": "pick",
            "frames": frames,
            "frame_count": len(frames),
            "quality_score": quality_score,
            "validation_passed": validation_passed,
        }
        (recordings_dir / f"{episode_id}.json").write_text(json.dumps(payload, indent=2))

        episodes_index.append({
            "episode_id": episode_id,
            "episode_index": idx,
            "num_frames": len(frames),
            "duration_seconds": frames[-1]["timestamp"] if frames else 0.0,
            "quality_score": quality_score,
            "validation_passed": validation_passed,
            "file": f"{episode_id}.json",
        })
        total_frames += len(frames)
        quality_scores.append(quality_score)

    dataset_info = {
        "dataset_type": "lerobot",
        "format_version": "1.0",
        "episodes": episodes_index,
        "total_frames": total_frames,
        "average_quality_score": sum(quality_scores) / max(len(quality_scores), 1),
        "source": "genie_sim_mock",
        "converted_at": datetime.utcnow().isoformat() + "Z",
        "total_episodes": episodes,
        "skipped_episodes": 0,
    }
    (metadata_dir / "dataset_info.json").write_text(json.dumps(dataset_info, indent=2))

    episodes_index_path = metadata_dir / "episodes.jsonl"
    with episodes_index_path.open("w") as handle:
        for entry in episodes_index:
            handle.write(json.dumps(entry) + "\n")

    print(f"[MOCK-GENIESIM-LOCAL] Generated {episodes} episodes at {base_dir}")
    return base_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate mock Genie Sim local outputs")
    parser.add_argument("--output-dir", default="./tmp", help="Output directory")
    parser.add_argument("--run-id", default="mock_run", help="Run identifier")
    parser.add_argument("--episodes", type=int, default=2, help="Number of episodes")
    args = parser.parse_args()

    generate_mock_geniesim_local(
        output_dir=Path(args.output_dir),
        run_id=args.run_id,
        episodes=args.episodes,
    )


if __name__ == "__main__":
    main()
