import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_quality_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "genie-sim-import-job" / "quality_config.py"
    module_dir = module_path.parent
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))
    spec = importlib.util.spec_from_file_location("quality_config", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load quality_config module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_quality_config_defaults_match_json():
    quality_module = _load_quality_module()
    repo_root = Path(__file__).resolve().parents[1]
    payload = json.loads((repo_root / "genie-sim-import-job" / "quality_config.json").read_text())

    config = quality_module.load_quality_config()

    assert config.default_min_quality_score == float(payload["default_min_quality_score"])
    assert config.min_allowed == float(payload["min_allowed"])
    assert config.max_allowed == float(payload["max_allowed"])
    assert config.description == payload["description"]


def test_resolve_min_quality_score_env_override_within_bounds():
    quality_module = _load_quality_module()
    config = quality_module.load_quality_config()

    resolved = quality_module.resolve_min_quality_score("0.7", config)

    assert resolved == 0.7


def test_resolve_min_quality_score_defaults_when_env_missing():
    quality_module = _load_quality_module()
    config = quality_module.load_quality_config()

    resolved = quality_module.resolve_min_quality_score(None, config)

    assert resolved == config.default_min_quality_score


def test_resolve_min_quality_score_rejects_invalid_value():
    quality_module = _load_quality_module()
    config = quality_module.load_quality_config()

    with pytest.raises(ValueError, match="MIN_QUALITY_SCORE must be a number"):
        quality_module.resolve_min_quality_score("not-a-number", config)


def test_resolve_min_quality_score_enforces_bounds():
    quality_module = _load_quality_module()
    config = quality_module.load_quality_config()

    with pytest.raises(ValueError, match="MIN_QUALITY_SCORE must be between"):
        quality_module.resolve_min_quality_score(str(config.min_allowed - 0.1), config)

    with pytest.raises(ValueError, match="MIN_QUALITY_SCORE must be between"):
        quality_module.resolve_min_quality_score(str(config.max_allowed + 0.1), config)
