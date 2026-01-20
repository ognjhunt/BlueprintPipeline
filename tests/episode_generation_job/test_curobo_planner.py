import pytest


@pytest.mark.unit
def test_curobo_availability_helpers(load_job_module, monkeypatch) -> None:
    curobo_planner = load_job_module("episode_generation", "curobo_planner.py")

    monkeypatch.setattr(curobo_planner, "CUROBO_AVAILABLE", True)
    assert curobo_planner.is_curobo_available() is True

    monkeypatch.setattr(curobo_planner, "CUROBO_AVAILABLE", False)
    assert curobo_planner.is_curobo_available() is False


@pytest.mark.unit
def test_requires_curobo_env_flag(load_job_module, monkeypatch) -> None:
    curobo_planner = load_job_module("episode_generation", "curobo_planner.py")

    monkeypatch.setenv("DATA_QUALITY_LEVEL", "production")
    monkeypatch.delenv("LABS_STAGING", raising=False)
    assert curobo_planner._requires_curobo() is True

    monkeypatch.setenv("DATA_QUALITY_LEVEL", "")
    monkeypatch.setenv("LABS_STAGING", "true")
    assert curobo_planner._requires_curobo() is True

    monkeypatch.setenv("DATA_QUALITY_LEVEL", "")
    monkeypatch.setenv("LABS_STAGING", "0")
    assert curobo_planner._requires_curobo() is False


@pytest.mark.unit
def test_create_curobo_planner_missing_dependency(load_job_module, monkeypatch) -> None:
    curobo_planner = load_job_module("episode_generation", "curobo_planner.py")

    monkeypatch.setattr(curobo_planner, "CUROBO_AVAILABLE", False)
    monkeypatch.setenv("DATA_QUALITY_LEVEL", "")
    monkeypatch.setenv("LABS_STAGING", "0")

    assert curobo_planner.create_curobo_planner() is None


@pytest.mark.unit
def test_create_curobo_planner_missing_dependency_required(load_job_module, monkeypatch) -> None:
    curobo_planner = load_job_module("episode_generation", "curobo_planner.py")

    monkeypatch.setattr(curobo_planner, "CUROBO_AVAILABLE", False)
    monkeypatch.setenv("DATA_QUALITY_LEVEL", "production")

    with pytest.raises(RuntimeError, match="cuRobo is required for production"):
        curobo_planner.create_curobo_planner()
