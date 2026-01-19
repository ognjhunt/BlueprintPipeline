from tools.quality_gates.quality_gate import QualityGateRegistry


def _run_pre_episode_gate(steps_completed: int) -> object:
    registry = QualityGateRegistry(verbose=False)
    gate = registry.gates["qg-5-pre-episode"]
    context = {
        "simulation_check": {
            "scene_loads": True,
            "physics_stable": True,
            "steps_completed": steps_completed,
        }
    }
    return gate.check(context)


def test_pre_episode_gate_defaults_to_50_steps(add_repo_to_path, monkeypatch):
    monkeypatch.delenv("BP_QUALITY_SIMULATION_MIN_STABLE_STEPS", raising=False)

    result = _run_pre_episode_gate(steps_completed=49)

    assert result.passed is False
    assert result.details["min_stable_steps_required"] == 50


def test_pre_episode_gate_env_override_changes_threshold(add_repo_to_path, monkeypatch):
    monkeypatch.setenv("BP_QUALITY_SIMULATION_MIN_STABLE_STEPS", "5")

    result = _run_pre_episode_gate(steps_completed=6)

    assert result.passed is True
    assert result.details["min_stable_steps_required"] == 5
