import pytest


@pytest.mark.unit
def test_isaac_sim_availability_helpers(load_job_module, monkeypatch) -> None:
    isaac_sim = load_job_module("episode_generation", "isaac_sim_integration.py")

    monkeypatch.setattr(isaac_sim, "_ISAAC_SIM_AVAILABLE", True)
    monkeypatch.setattr(isaac_sim, "_PHYSX_AVAILABLE", True)
    monkeypatch.setattr(isaac_sim, "_REPLICATOR_AVAILABLE", False)

    assert isaac_sim.is_isaac_sim_available() is True
    assert isaac_sim.is_physx_available() is True
    assert isaac_sim.is_replicator_available() is False

    status = isaac_sim.get_availability_status()
    assert status is not isaac_sim._AVAILABILITY_STATUS


@pytest.mark.unit
def test_isaac_sim_session_singleton(load_job_module) -> None:
    isaac_sim = load_job_module("episode_generation", "isaac_sim_integration.py")

    session_a = isaac_sim.get_isaac_sim_session()
    session_b = isaac_sim.get_isaac_sim_session()

    assert session_a is session_b


@pytest.mark.unit
def test_physics_simulator_mock_step_records_tracked_objects(load_job_module, monkeypatch) -> None:
    isaac_sim = load_job_module("episode_generation", "isaac_sim_integration.py")

    monkeypatch.setattr(isaac_sim, "_PHYSX_AVAILABLE", False)
    monkeypatch.setattr(isaac_sim, "_ISAAC_SIM_AVAILABLE", False)

    simulator = isaac_sim.PhysicsSimulator(verbose=False)
    simulator.add_tracked_object("crate", "/World/Crate")

    result = simulator.step()

    assert result.success is True
    assert "crate" in result.object_states
