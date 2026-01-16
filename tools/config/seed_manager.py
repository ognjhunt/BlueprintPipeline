"""Seed management for deterministic pipeline runs."""

from __future__ import annotations

import importlib
import importlib.util
import os
import random
from typing import Optional

import numpy as np


DEFAULT_SEED_ENV_VAR = "PIPELINE_SEED"


def get_pipeline_seed(env_var: str = DEFAULT_SEED_ENV_VAR) -> Optional[int]:
    """Return the configured pipeline seed, if available."""
    value = os.getenv(env_var)
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Invalid {env_var}: {value!r} is not an integer") from exc


def set_global_seed(seed: int) -> None:
    """Set Python, NumPy, and optional framework seeds."""
    if seed is None:
        return

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)

    if importlib.util.find_spec("torch") is not None:
        torch = importlib.import_module("torch")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def configure_pipeline_seed(env_var: str = DEFAULT_SEED_ENV_VAR) -> Optional[int]:
    """Resolve and apply the pipeline seed from environment variables."""
    seed = get_pipeline_seed(env_var=env_var)
    if seed is None:
        return None
    set_global_seed(seed)
    return seed
