from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from tools.benchmarks.co_training_uplift import compute_uplift_report


def _write_lerobot_like_dataset(
    root: Path,
    *,
    episodes: List[Tuple[np.ndarray, np.ndarray, str]],
) -> None:
    """Write a minimal LeRobot-like dataset:

    - meta/episodes.jsonl with parquet_path + parquet_row_group
    - meta/splits.json
    - data/chunk-000/file-0000.parquet with one row group per episode
    """
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    parquet_rel = "data/chunk-000/file-0000.parquet"
    parquet_path = root / parquet_rel

    # Write row groups.
    writer: pq.ParquetWriter | None = None
    metas = []
    splits = {"train": [], "test": []}
    try:
        for ep_idx, (states, actions, split) in enumerate(episodes):
            table = pa.table(
                {
                    "observation.state": pa.array(states.tolist(), type=pa.list_(pa.float32())),
                    "action": pa.array(actions.tolist(), type=pa.list_(pa.float32())),
                }
            )
            if writer is None:
                writer = pq.ParquetWriter(parquet_path, schema=table.schema, compression="zstd")
            writer.write_table(table)

            metas.append(
                {
                    "episode_index": ep_idx,
                    "length": int(states.shape[0]),
                    "split": split,
                    "parquet_path": parquet_rel,
                    "parquet_row_group": ep_idx,
                }
            )
            splits.setdefault(split, []).append(ep_idx)
    finally:
        if writer is not None:
            writer.close()

    (root / "meta" / "episodes.jsonl").write_text("".join(json.dumps(m) + "\n" for m in metas))
    (root / "meta" / "splits.json").write_text(json.dumps(splits, indent=2))


def test_co_training_uplift_reports_positive_delta(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    state_dim = 4
    action_dim = 3

    # Real: very small/noisy train set, clean test set.
    real_root = tmp_path / "real"
    real_episodes = []
    for _ in range(2):
        states = rng.normal(size=(5, state_dim)).astype(np.float32)
        actions = (states[:, :action_dim] * 2.0 + rng.normal(scale=0.25, size=(5, action_dim))).astype(np.float32)
        real_episodes.append((states, actions, "train"))
    for _ in range(2):
        states = rng.normal(size=(50, state_dim)).astype(np.float32)
        actions = (states[:, :action_dim] * 2.0).astype(np.float32)
        real_episodes.append((states, actions, "test"))
    _write_lerobot_like_dataset(real_root, episodes=real_episodes)

    # Synthetic: more diverse + clean mapping.
    synth_root = tmp_path / "synth"
    synth_episodes = []
    for _ in range(4):
        states = rng.normal(size=(80, state_dim)).astype(np.float32)
        actions = (states[:, :action_dim] * 2.0).astype(np.float32)
        synth_episodes.append((states, actions, "train"))
    _write_lerobot_like_dataset(synth_root, episodes=synth_episodes)

    report = compute_uplift_report(
        real_dataset=real_root,
        synthetic_dataset=synth_root,
        seed=123,
        real_train_episodes=None,
        real_test_episodes=None,
        synthetic_train_episodes=None,
        ridge_lambda=1e-6,
    )

    assert report["benchmark"] == "co_training_uplift"
    uplift = report.get("uplift") or {}
    assert uplift.get("relative_uplift") is not None
    assert float(uplift["relative_uplift"]) > 0.0

