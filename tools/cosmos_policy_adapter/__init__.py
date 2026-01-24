"""Cosmos Policy export adapter for BlueprintPipeline.

Converts episode data from the pipeline's internal format to
NVIDIA Cosmos Policy's expected training format. Cosmos Policy
fine-tunes a pretrained video diffusion model (Cosmos-Predict2-2B)
to serve as a unified policy, world model, and value function.

Expected output structure:
    cosmos_policy/
    ├── meta/
    │   ├── info.json              # Dataset metadata + normalization stats
    │   ├── tasks.jsonl            # Task descriptions for T5 encoding
    │   └── episodes.jsonl         # Episode index
    ├── data/
    │   ├── episode_000000.parquet # Observation + action + proprio per frame
    │   └── ...
    ├── videos/
    │   ├── {camera_id}/
    │   │   ├── episode_000000.mp4
    │   │   └── ...
    │   └── ...
    └── config/
        ├── training_config.yaml   # Turnkey fine-tuning config
        └── normalization_stats.json

Reference: https://arxiv.org/abs/2601.16163
GitHub:    https://github.com/nvlabs/cosmos-policy
"""

from tools.cosmos_policy_adapter.exporter import CosmosPolicyExporter
from tools.cosmos_policy_adapter.config import (
    CosmosPolicyConfig,
    CosmosPolicyTrainingConfig,
    COSMOS_POLICY_DEFAULTS,
)
from tools.cosmos_policy_adapter.normalizer import ActionNormalizer

__all__ = [
    "CosmosPolicyExporter",
    "CosmosPolicyConfig",
    "CosmosPolicyTrainingConfig",
    "ActionNormalizer",
    "COSMOS_POLICY_DEFAULTS",
]
