from tools.geniesim_adapter.mock_mode import resolve_geniesim_mock_mode


def test_mock_mode_disabled_when_flag_false():
    decision = resolve_geniesim_mock_mode({"GENIESIM_MOCK_MODE": "false"})

    assert decision.enabled is False
    assert decision.requested is False


def test_mock_mode_blocked_in_production():
    env = {
        "GENIESIM_MOCK_MODE": "true",
        "ALLOW_GENIESIM_MOCK": "1",
        "PIPELINE_ENV": "production",
    }

    decision = resolve_geniesim_mock_mode(env)

    assert decision.enabled is False
    assert decision.production_mode is True


def test_mock_mode_enabled_with_override_in_non_prod():
    env = {
        "GENIESIM_MOCK_MODE": "true",
        "ALLOW_GENIESIM_MOCK": "1",
        "PIPELINE_ENV": "staging",
    }

    decision = resolve_geniesim_mock_mode(env)

    assert decision.enabled is True
    assert decision.allow_override is True
