"""Sim-to-Real Validation Framework.

Tools for measuring and validating sim-to-real transfer of trained policies.
"""

from .validation import (
    Sim2RealValidator,
    Sim2RealExperiment,
    Sim2RealResult,
    TransferMetrics,
)
from .metrics import (
    compute_transfer_gap,
    compute_success_rate,
    compute_policy_divergence,
)
from .experiments import (
    ExperimentTracker,
    register_experiment,
    log_real_world_result,
)

__all__ = [
    "Sim2RealValidator",
    "Sim2RealExperiment",
    "Sim2RealResult",
    "TransferMetrics",
    "compute_transfer_gap",
    "compute_success_rate",
    "compute_policy_divergence",
    "ExperimentTracker",
    "register_experiment",
    "log_real_world_result",
]
