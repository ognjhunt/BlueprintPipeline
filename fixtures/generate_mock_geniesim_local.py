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
import math
import random
from typing import Dict, List, Optional, Sequence, Tuple


def _generate_timestamps(
    steps: int,
    rng: random.Random,
    base_dt: float = 1 / 30.0,
    jitter: float = 0.004,
) -> List[float]:
    timestamps: List[float] = []
    current = 0.0
    for _ in range(steps):
        delta = base_dt + rng.uniform(-jitter, jitter)
        current += max(0.01, delta)
        timestamps.append(round(current, 4))
    return timestamps


def _smooth_wave(
    t: float,
    amplitude: float,
    frequency: float,
    phase: float,
    offset: float,
) -> float:
    return offset + amplitude * math.sin(frequency * t + phase)


def _generate_joint_series(
    timestamps: Sequence[float],
    joint_count: int,
    rng: random.Random,
) -> Tuple[List[List[float]], List[List[float]]]:
    profiles = []
    for _ in range(joint_count):
        amplitude = rng.uniform(0.2, 0.6)
        frequency = rng.uniform(0.5, 1.8)
        phase = rng.uniform(0.0, math.tau)
        offset = rng.uniform(-0.4, 0.4)
        profiles.append((amplitude, frequency, phase, offset))

    positions: List[List[float]] = []
    velocities: List[List[float]] = []
    previous = None
    for idx, t in enumerate(timestamps):
        joint_positions = [
            _smooth_wave(t, amplitude, frequency, phase, offset)
            for amplitude, frequency, phase, offset in profiles
        ]
        positions.append(joint_positions)
        if previous is None:
            velocities.append([0.0] * joint_count)
        else:
            dt = max(1e-3, t - timestamps[idx - 1])
            velocities.append(
                [
                    (joint_positions[j] - previous[j]) / dt
                    for j in range(joint_count)
                ]
            )
        previous = joint_positions
    return positions, velocities


def _build_rgb_frame(
    rng: random.Random,
    step: int,
    size: Tuple[int, int] = (8, 8),
) -> List[List[List[int]]]:
    height, width = size
    rgb_frame: List[List[List[int]]] = []
    for y in range(height):
        row: List[List[int]] = []
        for x in range(width):
            base = 90 + int(30 * math.sin(0.2 * step + (x + y) * 0.1))
            noise = rng.randint(-10, 10)
            r = max(0, min(255, base + noise))
            g = max(0, min(255, base + 20 + noise))
            b = max(0, min(255, base + 40 + noise))
            row.append([r, g, b])
        rgb_frame.append(row)
    return rgb_frame


def _build_physics_signals(step: int, rng: random.Random) -> Dict[str, object]:
    contacts = []
    collisions = []
    if step % 4 == 0:
        contacts.append(
            {
                "body_a": "gripper",
                "body_b": "object",
                "impulse": round(0.8 + rng.random() * 0.4, 3),
                "position": [0.45, 0.12, 0.2],
                "normal": [0.0, 0.0, 1.0],
            }
        )
    if step % 7 == 0:
        collisions.append(
            {
                "body_a": "object",
                "body_b": "table",
                "depth": round(0.002 + rng.random() * 0.001, 4),
                "position": [0.4, 0.0, 0.15],
            }
        )
    return {
        "contacts": contacts,
        "collisions": collisions,
    }


def _build_frame(
    step: int,
    timestamp: float,
    joint_positions: Sequence[float],
    joint_velocities: Sequence[float],
    rng: random.Random,
    include_camera: bool = True,
    include_physics: bool = True,
) -> Dict[str, object]:
    rgb_frame = _build_rgb_frame(rng, step) if include_camera else None
    physics_signals = _build_physics_signals(step, rng) if include_physics else None
    robot_state = list(joint_positions) + list(joint_velocities)
    action = [max(-2.0, min(2.0, v * 0.1)) for v in joint_velocities]

    frame = {
        "step": step,
        "observation": {
            "joint_positions": list(joint_positions),
            "joint_velocities": list(joint_velocities),
            "robot_state": robot_state,
            "sensor": {"temperature": round(295.0 + 0.3 * step, 2)},
        },
        "action": action,
        "timestamp": timestamp,
        "robot_state": robot_state,
    }
    if include_camera and rgb_frame is not None:
        frame["rgb_image"] = rgb_frame
        frame["observation"]["rgb_image"] = rgb_frame
    if include_physics and physics_signals is not None:
        frame["physics"] = physics_signals
        frame["observation"]["physics"] = physics_signals

    return frame


def generate_mock_geniesim_local(
    output_dir: Path,
    run_id: str = "mock_run",
    episodes: int = 2,
    seed: Optional[int] = None,
    include_camera: bool = True,
    include_physics: bool = True,
) -> Path:
    """Generate mock Genie Sim local outputs.

    Args:
        output_dir: Base output directory.
        run_id: Identifier for the run.
        episodes: Number of episodes to generate.
        seed: Optional random seed for deterministic output.
        include_camera: Whether to include RGB camera frames.
        include_physics: Whether to include physics contact/collision signals.

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
    rng = random.Random(seed)
    for idx in range(episodes):
        episode_id = f"episode_{idx:06d}"
        step_count = rng.randint(12, 18)
        timestamps = _generate_timestamps(step_count, rng)
        joint_count = rng.randint(6, 7)
        positions, velocities = _generate_joint_series(timestamps, joint_count, rng)
        frames = [
            _build_frame(
                step=step,
                timestamp=timestamps[step],
                joint_positions=positions[step],
                joint_velocities=velocities[step],
                rng=rng,
                include_camera=include_camera,
                include_physics=include_physics,
            )
            for step in range(step_count)
        ]
        quality_score = 0.92 if idx % 2 == 0 else 0.78
        validation_passed = quality_score >= 0.7
        payload = {
            "episode_id": episode_id,
            "task_name": "pick" if idx % 2 == 0 else "place",
            "frames": frames,
            "frame_count": len(frames),
            "quality_score": quality_score,
            "quality_components": {
                "collision_free": 0.95,
                "task_success": 0.9 if validation_passed else 0.6,
                "trajectory_smoothness": 0.92,
            },
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
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    generate_mock_geniesim_local(
        output_dir=Path(args.output_dir),
        run_id=args.run_id,
        episodes=args.episodes,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
