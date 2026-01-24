"""Action and state normalization for Cosmos Policy.

Cosmos Policy expects all actions and proprioception normalized to [-1, +1].
This module computes per-dimension statistics from episode data and provides
normalization/denormalization utilities compatible with the Cosmos Policy
training loop.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class NormalizationStats:
    """Per-dimension normalization statistics."""

    mean: np.ndarray
    std: np.ndarray
    min_val: np.ndarray
    max_val: np.ndarray
    dim: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "min": self.min_val.tolist(),
            "max": self.max_val.tolist(),
            "dim": self.dim,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NormalizationStats":
        return cls(
            mean=np.array(data["mean"], dtype=np.float32),
            std=np.array(data["std"], dtype=np.float32),
            min_val=np.array(data["min"], dtype=np.float32),
            max_val=np.array(data["max"], dtype=np.float32),
            dim=data["dim"],
        )


class ActionNormalizer:
    """Normalizes actions and proprioception to [-1, +1] for Cosmos Policy.

    Cosmos Policy uses min-max normalization to [-1, +1] by default,
    matching their paper's approach. This ensures the latent frame
    injection works correctly with the diffusion model's noise schedule.
    """

    def __init__(
        self,
        target_range: Tuple[float, float] = (-1.0, 1.0),
        clip: bool = True,
        eps: float = 1e-8,
    ):
        self.target_min, self.target_max = target_range
        self.clip = clip
        self.eps = eps

        self.action_stats: Optional[NormalizationStats] = None
        self.state_stats: Optional[NormalizationStats] = None
        self.proprio_stats: Optional[NormalizationStats] = None

    def fit(
        self,
        actions: List[np.ndarray],
        states: Optional[List[np.ndarray]] = None,
        proprio: Optional[List[np.ndarray]] = None,
    ) -> None:
        """Compute normalization statistics from episode data.

        Args:
            actions: List of action arrays, each shape (T, action_dim)
            states: Optional list of state arrays, each shape (T, state_dim)
            proprio: Optional list of proprioception arrays
        """
        if actions:
            all_actions = np.concatenate(actions, axis=0)
            self.action_stats = self._compute_stats(all_actions, "actions")

        if states:
            all_states = np.concatenate(states, axis=0)
            self.state_stats = self._compute_stats(all_states, "states")

        if proprio:
            all_proprio = np.concatenate(proprio, axis=0)
            self.proprio_stats = self._compute_stats(all_proprio, "proprio")

    def _compute_stats(self, data: np.ndarray, name: str) -> NormalizationStats:
        """Compute min/max/mean/std per dimension."""
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        stats = NormalizationStats(
            mean=np.mean(data, axis=0).astype(np.float32),
            std=np.std(data, axis=0).astype(np.float32),
            min_val=np.min(data, axis=0).astype(np.float32),
            max_val=np.max(data, axis=0).astype(np.float32),
            dim=data.shape[1],
        )

        logger.info(
            "[COSMOS-NORM] %s stats: dim=%d, range=[%.3f, %.3f]",
            name, stats.dim,
            float(stats.min_val.min()), float(stats.max_val.max()),
        )

        return stats

    def normalize_actions(self, actions: np.ndarray) -> np.ndarray:
        """Normalize actions to target range [-1, +1].

        Uses min-max normalization: x_norm = (x - min) / (max - min) * 2 - 1
        """
        if self.action_stats is None:
            raise ValueError("Must call fit() before normalize_actions()")
        return self._normalize(actions, self.action_stats)

    def normalize_states(self, states: np.ndarray) -> np.ndarray:
        """Normalize robot states to target range."""
        if self.state_stats is None:
            raise ValueError("Must call fit() before normalize_states()")
        return self._normalize(states, self.state_stats)

    def normalize_proprio(self, proprio: np.ndarray) -> np.ndarray:
        """Normalize proprioception to target range."""
        if self.proprio_stats is None:
            raise ValueError("Must call fit() before normalize_proprio()")
        return self._normalize(proprio, self.proprio_stats)

    def _normalize(self, data: np.ndarray, stats: NormalizationStats) -> np.ndarray:
        """Apply min-max normalization to target range."""
        data_range = stats.max_val - stats.min_val
        # Avoid division by zero for constant dimensions
        data_range = np.where(data_range < self.eps, 1.0, data_range)

        # Normalize to [0, 1]
        normalized = (data - stats.min_val) / data_range

        # Scale to target range
        normalized = normalized * (self.target_max - self.target_min) + self.target_min

        if self.clip:
            normalized = np.clip(normalized, self.target_min, self.target_max)

        return normalized.astype(np.float32)

    def denormalize_actions(self, normalized: np.ndarray) -> np.ndarray:
        """Denormalize actions back to original range."""
        if self.action_stats is None:
            raise ValueError("Must call fit() before denormalize_actions()")
        return self._denormalize(normalized, self.action_stats)

    def _denormalize(self, data: np.ndarray, stats: NormalizationStats) -> np.ndarray:
        """Reverse min-max normalization."""
        data_range = stats.max_val - stats.min_val
        data_range = np.where(data_range < self.eps, 1.0, data_range)

        # Reverse: [target_min, target_max] -> [0, 1] -> original
        unit = (data - self.target_min) / (self.target_max - self.target_min)
        return (unit * data_range + stats.min_val).astype(np.float32)

    def save(self, path: Path) -> None:
        """Save normalization statistics to JSON."""
        payload = {
            "target_range": [self.target_min, self.target_max],
            "clip": self.clip,
        }
        if self.action_stats:
            payload["action_stats"] = self.action_stats.to_dict()
        if self.state_stats:
            payload["state_stats"] = self.state_stats.to_dict()
        if self.proprio_stats:
            payload["proprio_stats"] = self.proprio_stats.to_dict()

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

        logger.info("[COSMOS-NORM] Saved normalization stats to %s", path)

    @classmethod
    def load(cls, path: Path) -> "ActionNormalizer":
        """Load normalization statistics from JSON."""
        with open(path) as f:
            payload = json.load(f)

        normalizer = cls(
            target_range=tuple(payload["target_range"]),
            clip=payload.get("clip", True),
        )

        if "action_stats" in payload:
            normalizer.action_stats = NormalizationStats.from_dict(
                payload["action_stats"]
            )
        if "state_stats" in payload:
            normalizer.state_stats = NormalizationStats.from_dict(
                payload["state_stats"]
            )
        if "proprio_stats" in payload:
            normalizer.proprio_stats = NormalizationStats.from_dict(
                payload["proprio_stats"]
            )

        return normalizer
