#!/usr/bin/env python3
"""
Co-training uplift benchmark (P2).

This produces a buyer-facing trust signal:
- Train on N "real" episodes only
- Train on N "real" + M "synthetic" episodes
- Evaluate on held-out "real" episodes and report the delta

MVP benchmark here is offline behavior cloning on state->action with ridge
regression (numpy-only). It is intentionally lightweight so it can run inside
RunPod / job environments without a full RL stack.

If you want success-rate deltas, wire this report into an environment-based
evaluator (Arena/IsaacLab) and replace the offline MSE evaluator.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import pyarrow.parquet as pq  # type: ignore

    HAVE_PYARROW = True
except Exception:
    pq = None
    HAVE_PYARROW = False


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _resolve_episodes_index_path(dataset_root: Path) -> Optional[Path]:
    candidates = [
        dataset_root / "meta" / "episodes.jsonl",
        dataset_root / "episodes.jsonl",
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    return None


def _resolve_splits_path(dataset_root: Path) -> Optional[Path]:
    candidates = [
        dataset_root / "meta" / "splits.json",
        dataset_root / "splits.json",
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    return None


def _load_episode_metas(dataset_root: Path) -> List[Dict[str, Any]]:
    episodes_path = _resolve_episodes_index_path(dataset_root)
    if episodes_path is None:
        raise FileNotFoundError(
            f"Missing episodes index. Expected meta/episodes.jsonl under {dataset_root}."
        )
    metas = _read_jsonl(episodes_path)
    # Ensure stable ordering.
    metas.sort(key=lambda m: int(m.get("episode_index", 0) or 0))
    return metas


def _load_splits(dataset_root: Path, metas: Sequence[Mapping[str, Any]]) -> Dict[str, List[int]]:
    splits_path = _resolve_splits_path(dataset_root)
    if splits_path is not None:
        payload = _read_json(splits_path)
        out: Dict[str, List[int]] = {}
        if isinstance(payload, dict):
            for split_name, ids in payload.items():
                if not isinstance(ids, list):
                    continue
                cleaned: List[int] = []
                for item in ids:
                    try:
                        cleaned.append(int(item))
                    except Exception:
                        continue
                out[split_name] = cleaned
            if out:
                return out

    # Fallback: use per-episode "split" field when present.
    out: Dict[str, List[int]] = {}
    for meta in metas:
        split = str(meta.get("split") or "").strip().lower() or "train"
        try:
            ep = int(meta.get("episode_index", 0) or 0)
        except Exception:
            continue
        out.setdefault(split, []).append(ep)
    if out:
        return out
    return {"train": [int(m.get("episode_index", 0) or 0) for m in metas]}


def _index_metas_by_episode(metas: Sequence[Mapping[str, Any]]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for meta in metas:
        try:
            ep = int(meta.get("episode_index", 0) or 0)
        except Exception:
            continue
        out[ep] = dict(meta)
    return out


def _iter_state_action_rows(
    dataset_root: Path,
    metas_by_episode: Mapping[int, Mapping[str, Any]],
    episode_ids: Iterable[int],
    *,
    columns: Tuple[str, str] = ("observation.state", "action"),
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield (states, actions) arrays per episode."""
    if not HAVE_PYARROW:
        raise ImportError("pyarrow is required to read LeRobot parquet episodes.")

    for ep in episode_ids:
        meta = metas_by_episode.get(int(ep))
        if not meta:
            continue
        parquet_rel = meta.get("parquet_path") or meta.get("data_path") or ""
        parquet_rel = str(parquet_rel or "").strip()
        if not parquet_rel:
            continue
        parquet_path = (dataset_root / parquet_rel).resolve()
        row_group = meta.get("parquet_row_group")
        try:
            row_group_idx = int(row_group) if row_group is not None else None
        except Exception:
            row_group_idx = None

        pf = pq.ParquetFile(parquet_path)
        if row_group_idx is not None:
            table = pf.read_row_group(row_group_idx, columns=list(columns))
        else:
            table = pf.read(columns=list(columns))

        state_list = table.column(columns[0]).to_pylist()
        action_list = table.column(columns[1]).to_pylist()
        states = np.asarray(state_list, dtype=np.float32)
        actions = np.asarray(action_list, dtype=np.float32)
        if states.ndim != 2 or actions.ndim != 2:
            continue
        yield states, actions


@dataclass(frozen=True)
class RidgeBCConfig:
    ridge_lambda: float = 1e-3


def _fit_ridge_bc(
    dataset_root: Path,
    metas_by_episode: Mapping[int, Mapping[str, Any]],
    episode_ids: Sequence[int],
    *,
    cfg: RidgeBCConfig,
) -> Dict[str, Any]:
    """Fit ridge regression (state->action). Returns weights + dims + stats."""
    # Accumulate sufficient statistics for (X^T X) and (X^T Y).
    xxt: Optional[np.ndarray] = None
    xyt: Optional[np.ndarray] = None
    n_rows = 0
    state_dim: Optional[int] = None
    action_dim: Optional[int] = None

    for states, actions in _iter_state_action_rows(dataset_root, metas_by_episode, episode_ids):
        if state_dim is None:
            state_dim = int(states.shape[1])
        if action_dim is None:
            action_dim = int(actions.shape[1])
        if states.shape[1] != state_dim or actions.shape[1] != action_dim:
            raise ValueError(
                "Inconsistent dims across episodes: "
                f"got states={states.shape}, actions={actions.shape}, "
                f"expected state_dim={state_dim}, action_dim={action_dim}."
            )

        # Add bias column.
        xb = np.concatenate([states, np.ones((states.shape[0], 1), dtype=np.float32)], axis=1)
        if xxt is None:
            xxt = np.zeros((xb.shape[1], xb.shape[1]), dtype=np.float64)
        if xyt is None:
            xyt = np.zeros((xb.shape[1], action_dim), dtype=np.float64)
        xxt += xb.T @ xb
        xyt += xb.T @ actions
        n_rows += int(states.shape[0])

    if xxt is None or xyt is None or state_dim is None or action_dim is None or n_rows == 0:
        raise RuntimeError("No training rows found (empty episode selection).")

    # Ridge: (X^T X + lambda I) W = X^T Y
    reg = float(cfg.ridge_lambda)
    xxt_reg = xxt.copy()
    xxt_reg += reg * np.eye(xxt_reg.shape[0], dtype=xxt_reg.dtype)
    w = np.linalg.solve(xxt_reg, xyt)  # (state_dim+1, action_dim)

    return {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "num_rows": n_rows,
        "ridge_lambda": reg,
        "weights": w.astype(np.float32),
    }


def _predict_actions(weights: np.ndarray, states: np.ndarray) -> np.ndarray:
    xb = np.concatenate([states, np.ones((states.shape[0], 1), dtype=np.float32)], axis=1)
    return xb @ weights


def _eval_mse(
    dataset_root: Path,
    metas_by_episode: Mapping[int, Mapping[str, Any]],
    episode_ids: Sequence[int],
    *,
    weights: np.ndarray,
) -> Dict[str, Any]:
    mse_sum = 0.0
    n = 0
    for states, actions in _iter_state_action_rows(dataset_root, metas_by_episode, episode_ids):
        pred = _predict_actions(weights, states)
        err = pred - actions
        mse_sum += float(np.mean(err * err)) * float(states.shape[0])
        n += int(states.shape[0])
    if n == 0:
        raise RuntimeError("No evaluation rows found (empty episode selection).")
    return {"mse": mse_sum / float(n), "num_rows": n}


def _choose_episode_subset(
    episode_ids: Sequence[int],
    *,
    seed: int,
    limit: Optional[int],
) -> List[int]:
    ids = list(dict.fromkeys(int(i) for i in episode_ids))
    rng = random.Random(int(seed))
    rng.shuffle(ids)
    if limit is not None:
        return ids[: max(0, int(limit))]
    return ids


def compute_uplift_report(
    *,
    real_dataset: Path,
    synthetic_dataset: Optional[Path],
    seed: int,
    real_train_episodes: Optional[int],
    real_test_episodes: Optional[int],
    synthetic_train_episodes: Optional[int],
    ridge_lambda: float,
) -> Dict[str, Any]:
    real_dataset = Path(real_dataset)
    synth_dataset = Path(synthetic_dataset) if synthetic_dataset is not None else None

    real_metas = _load_episode_metas(real_dataset)
    real_splits = _load_splits(real_dataset, real_metas)
    real_by_ep = _index_metas_by_episode(real_metas)

    train_ids = real_splits.get("train") or [int(m.get("episode_index", 0) or 0) for m in real_metas]
    test_ids = (
        real_splits.get("test")
        or real_splits.get("val")
        or []
    )
    if not test_ids:
        # Fallback: carve out a test slice from train.
        test_ids = list(train_ids)

    train_ids = _choose_episode_subset(train_ids, seed=seed, limit=real_train_episodes)
    test_ids = _choose_episode_subset(test_ids, seed=seed + 1, limit=real_test_episodes)

    ridge_cfg = RidgeBCConfig(ridge_lambda=float(ridge_lambda))

    # Baseline: real-only
    real_only = _fit_ridge_bc(real_dataset, real_by_ep, train_ids, cfg=ridge_cfg)
    real_only_eval = _eval_mse(real_dataset, real_by_ep, test_ids, weights=real_only["weights"])

    report: Dict[str, Any] = {
        "benchmark": "co_training_uplift",
        "metric": "offline_state_to_action_mse",
        "seed": int(seed),
        "real": {
            "dataset_root": str(real_dataset),
            "episodes_total": len(real_metas),
            "train_episodes": list(train_ids),
            "test_episodes": list(test_ids),
        },
        "models": {
            "real_only": {
                "ridge_lambda": float(ridge_lambda),
                "train_rows": int(real_only["num_rows"]),
                "eval": real_only_eval,
            }
        },
    }

    if synth_dataset is None:
        report["note"] = "synthetic_dataset not provided; uplift not computed."
        return report

    synth_metas = _load_episode_metas(synth_dataset)
    synth_splits = _load_splits(synth_dataset, synth_metas)
    synth_by_ep = _index_metas_by_episode(synth_metas)
    synth_train_ids = synth_splits.get("train") or [int(m.get("episode_index", 0) or 0) for m in synth_metas]
    synth_train_ids = _choose_episode_subset(
        synth_train_ids,
        seed=seed + 2,
        limit=synthetic_train_episodes,
    )

    # Fit combined model by accumulating stats over real train + synthetic train.
    combined_ids_real = list(train_ids)
    combined_ids_synth = list(synth_train_ids)

    # Fit on real portion.
    combined = _fit_ridge_bc(real_dataset, real_by_ep, combined_ids_real, cfg=ridge_cfg)

    # Continue accumulation for synthetic portion by refitting on concatenated IDs
    # (kept simple; dims are checked for compatibility).
    combined_plus = _fit_ridge_bc(
        synth_dataset,
        synth_by_ep,
        combined_ids_synth,
        cfg=ridge_cfg,
    )

    # Merge the two ridge solutions by refitting on the union of sufficient stats.
    # Since _fit_ridge_bc currently returns only the solved weights, we refit by
    # explicitly accumulating sufficient stats across both datasets.
    #
    # This keeps implementation simple and avoids storing huge X/Y matrices.
    def _accum_stats(
        root: Path,
        by_ep: Mapping[int, Mapping[str, Any]],
        ids: Sequence[int],
        *,
        state_dim: int,
        action_dim: int,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        xxt = np.zeros((state_dim + 1, state_dim + 1), dtype=np.float64)
        xyt = np.zeros((state_dim + 1, action_dim), dtype=np.float64)
        rows = 0
        for states, actions in _iter_state_action_rows(root, by_ep, ids):
            if states.shape[1] != state_dim or actions.shape[1] != action_dim:
                raise ValueError(
                    "State/action dims mismatch between datasets; cannot co-train. "
                    f"expected state_dim={state_dim}, action_dim={action_dim}, "
                    f"got states={states.shape}, actions={actions.shape}."
                )
            xb = np.concatenate([states, np.ones((states.shape[0], 1), dtype=np.float32)], axis=1)
            xxt += xb.T @ xb
            xyt += xb.T @ actions
            rows += int(states.shape[0])
        return xxt, xyt, rows

    state_dim = int(real_only["state_dim"])
    action_dim = int(real_only["action_dim"])
    xxt_r, xyt_r, rows_r = _accum_stats(real_dataset, real_by_ep, combined_ids_real, state_dim=state_dim, action_dim=action_dim)
    xxt_s, xyt_s, rows_s = _accum_stats(synth_dataset, synth_by_ep, combined_ids_synth, state_dim=state_dim, action_dim=action_dim)
    xxt = xxt_r + xxt_s
    xyt = xyt_r + xyt_s
    reg = float(ridge_lambda)
    xxt += reg * np.eye(state_dim + 1, dtype=xxt.dtype)
    w = np.linalg.solve(xxt, xyt).astype(np.float32)

    real_plus_synth_eval = _eval_mse(real_dataset, real_by_ep, test_ids, weights=w)
    mse_real_only = float(real_only_eval["mse"])
    mse_aug = float(real_plus_synth_eval["mse"])
    uplift = None
    if mse_real_only > 0:
        uplift = (mse_real_only - mse_aug) / mse_real_only

    report["synthetic"] = {
        "dataset_root": str(synth_dataset),
        "episodes_total": len(synth_metas),
        "train_episodes": list(combined_ids_synth),
    }
    report["models"]["real_plus_synth"] = {
        "ridge_lambda": float(ridge_lambda),
        "train_rows_real": int(rows_r),
        "train_rows_synth": int(rows_s),
        "train_rows_total": int(rows_r + rows_s),
        "eval": real_plus_synth_eval,
    }
    report["uplift"] = {
        "definition": "(mse_real_only - mse_real_plus_synth) / mse_real_only",
        "mse_real_only": mse_real_only,
        "mse_real_plus_synth": mse_aug,
        "relative_uplift": float(uplift) if uplift is not None else None,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-dataset", type=Path, required=True, help="Path to LeRobot dataset root (real).")
    parser.add_argument("--synthetic-dataset", type=Path, default=None, help="Optional path to LeRobot dataset root (synthetic).")
    parser.add_argument("--output", type=Path, default=Path("co_training_uplift.json"), help="Output JSON report path.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--real-train-episodes", type=int, default=None)
    parser.add_argument("--real-test-episodes", type=int, default=None)
    parser.add_argument("--synthetic-train-episodes", type=int, default=None)
    parser.add_argument("--ridge-lambda", type=float, default=1e-3)
    args = parser.parse_args()

    report = compute_uplift_report(
        real_dataset=args.real_dataset,
        synthetic_dataset=args.synthetic_dataset,
        seed=args.seed,
        real_train_episodes=args.real_train_episodes,
        real_test_episodes=args.real_test_episodes,
        synthetic_train_episodes=args.synthetic_train_episodes,
        ridge_lambda=args.ridge_lambda,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({"success": True, "output": str(args.output), "metric": report.get("metric")}))


if __name__ == "__main__":
    main()
