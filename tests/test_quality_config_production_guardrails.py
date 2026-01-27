from tools.config import ConfigLoader


def test_quality_config_production_floor_enforced(monkeypatch):
    config = {
        "thresholds": {
            "episodes": {
                "collision_free_rate_min": 0.5,
                "quality_pass_rate_min": 0.5,
                "quality_score_min": 0.5,
                "min_episodes_required": 1,
            }
        }
    }

    monkeypatch.setenv("PIPELINE_ENV", "production")
    errors = ConfigLoader.validate_config(config, config_type="quality")
    assert errors is not None
    assert "thresholds.episodes.collision_free_rate_min" in errors
    assert "thresholds.episodes.quality_pass_rate_min" in errors
    assert "thresholds.episodes.quality_score_min" in errors
    assert "thresholds.episodes.min_episodes_required" in errors

    monkeypatch.setenv("PIPELINE_ENV", "development")
    errors = ConfigLoader.validate_config(config, config_type="quality")
    assert errors is None
