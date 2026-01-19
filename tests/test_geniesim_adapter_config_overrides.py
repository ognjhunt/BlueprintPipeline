from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_geniesim_circuit_breaker_threshold_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GENIESIM_CIRCUIT_BREAKER_FAILURE_THRESHOLD", "7")
    monkeypatch.setenv("GENIESIM_CIRCUIT_BREAKER_SUCCESS_THRESHOLD", "4")
    monkeypatch.setenv("GENIESIM_CIRCUIT_BREAKER_RECOVERY_TIMEOUT_S", "12.5")

    from tools.geniesim_adapter.config import (
        get_geniesim_circuit_breaker_failure_threshold,
        get_geniesim_circuit_breaker_recovery_timeout_s,
        get_geniesim_circuit_breaker_success_threshold,
    )

    assert get_geniesim_circuit_breaker_failure_threshold() == 7
    assert get_geniesim_circuit_breaker_success_threshold() == 4
    assert get_geniesim_circuit_breaker_recovery_timeout_s() == 12.5


def test_task_confidence_threshold_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GENIESIM_TASK_CONFIDENCE_THRESHOLD", "0.95")

    from tools.geniesim_adapter.task_config import TaskConfigGenerator

    generator = TaskConfigGenerator(verbose=False)
    priority = generator._calculate_priority(
        "pick_place",
        {
            "semantics": {
                "affordances": [{"type": "Graspable", "confidence": 0.9}],
            }
        },
    )

    assert priority == 4
