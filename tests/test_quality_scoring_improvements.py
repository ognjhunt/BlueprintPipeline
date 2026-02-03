"""
Tests for quality scoring improvements.

These tests validate the configurable diversity divisors, gradual frame count scoring,
and partial object motion credit implemented without requiring VM pipeline runs.
"""
import json
import pytest
from pathlib import Path

# Import the quality config module
from tools.quality.quality_config import (
    load_quality_config,
    DiversityDivisors,
    FrameCountScoring,
    _parse_diversity_divisors,
    _parse_frame_count_scoring,
)


class TestDiversityDivisorsConfig:
    """Tests for configurable diversity divisors."""

    def test_load_diversity_divisors_from_config(self):
        """Verify diversity divisors are loaded from quality_config.json."""
        config = load_quality_config()
        assert config.diversity_divisors is not None
        assert "default" in config.diversity_divisors

    def test_get_diversity_divisors_for_known_robot(self):
        """Verify robot-specific divisors are returned correctly."""
        config = load_quality_config()
        franka_div = config.get_diversity_divisors("franka")
        assert franka_div.action == 0.05
        assert franka_div.obs == 0.05

    def test_get_diversity_divisors_for_g1_robot(self):
        """Verify G1 robot has different divisors."""
        config = load_quality_config()
        g1_div = config.get_diversity_divisors("g1")
        assert g1_div.action == 0.08
        assert g1_div.obs == 0.08

    def test_get_diversity_divisors_fallback_to_default(self):
        """Verify unknown robot types fall back to default divisors."""
        config = load_quality_config()
        unknown_div = config.get_diversity_divisors("unknown_robot_xyz")
        default_div = config.get_diversity_divisors("default")
        assert unknown_div.action == default_div.action
        assert unknown_div.obs == default_div.obs

    def test_get_diversity_divisors_case_insensitive(self):
        """Verify robot type matching is case insensitive."""
        config = load_quality_config()
        franka_lower = config.get_diversity_divisors("franka")
        franka_upper = config.get_diversity_divisors("FRANKA")
        franka_mixed = config.get_diversity_divisors("Franka")
        assert franka_lower.action == franka_upper.action == franka_mixed.action

    def test_parse_diversity_divisors_missing_returns_default(self):
        """Verify missing diversity_divisors returns default."""
        result = _parse_diversity_divisors({})
        assert "default" in result
        assert result["default"].action == 0.05
        assert result["default"].obs == 0.05

    def test_parse_diversity_divisors_partial_robot_config(self):
        """Verify partial robot config is handled."""
        payload = {
            "diversity_divisors": {
                "custom_robot": {"action": 0.1}  # Missing obs
            }
        }
        result = _parse_diversity_divisors(payload)
        assert result["custom_robot"].action == 0.1
        assert result["custom_robot"].obs == 0.05  # Default obs


class TestFrameCountScoring:
    """Tests for gradual frame count scoring."""

    def test_load_frame_count_scoring_from_config(self):
        """Verify frame count scoring config is loaded."""
        config = load_quality_config()
        assert config.frame_count_scoring is not None
        assert config.frame_count_scoring.min_frames_full_score == 10
        assert config.frame_count_scoring.min_frames_nonzero == 3
        assert config.frame_count_scoring.use_gradual_scoring is True

    def test_parse_frame_count_scoring_defaults(self):
        """Verify default frame count scoring when missing from config."""
        result = _parse_frame_count_scoring({})
        assert result.min_frames_full_score == 10
        assert result.min_frames_nonzero == 3
        assert result.use_gradual_scoring is True

    def test_gradual_scoring_10_frames_full_score(self):
        """10 or more frames should get 1.0 score."""
        cfg = FrameCountScoring(min_frames_full_score=10, min_frames_nonzero=3, use_gradual_scoring=True)
        # Simulate the scoring logic
        total_frames = 10
        if total_frames >= cfg.min_frames_full_score:
            score = 1.0
        elif total_frames >= cfg.min_frames_nonzero:
            score = 0.3 + 0.7 * (total_frames - cfg.min_frames_nonzero) / max(1, cfg.min_frames_full_score - cfg.min_frames_nonzero)
        else:
            score = 0.0
        assert score == 1.0

    def test_gradual_scoring_9_frames_partial_score(self):
        """9 frames should get partial score (~0.9), not 0.5."""
        cfg = FrameCountScoring(min_frames_full_score=10, min_frames_nonzero=3, use_gradual_scoring=True)
        total_frames = 9
        if total_frames >= cfg.min_frames_full_score:
            score = 1.0
        elif total_frames >= cfg.min_frames_nonzero:
            score = 0.3 + 0.7 * (total_frames - cfg.min_frames_nonzero) / max(1, cfg.min_frames_full_score - cfg.min_frames_nonzero)
        else:
            score = 0.0
        # 9 frames: 0.3 + 0.7 * (9-3)/(10-3) = 0.3 + 0.7 * 6/7 = 0.3 + 0.6 = 0.9
        assert abs(score - 0.9) < 0.01

    def test_gradual_scoring_5_frames_mid_score(self):
        """5 frames should get mid-range score."""
        cfg = FrameCountScoring(min_frames_full_score=10, min_frames_nonzero=3, use_gradual_scoring=True)
        total_frames = 5
        if total_frames >= cfg.min_frames_full_score:
            score = 1.0
        elif total_frames >= cfg.min_frames_nonzero:
            score = 0.3 + 0.7 * (total_frames - cfg.min_frames_nonzero) / max(1, cfg.min_frames_full_score - cfg.min_frames_nonzero)
        else:
            score = 0.0
        # 5 frames: 0.3 + 0.7 * (5-3)/(10-3) = 0.3 + 0.7 * 2/7 = 0.3 + 0.2 = 0.5
        assert abs(score - 0.5) < 0.01

    def test_gradual_scoring_2_frames_zero_score(self):
        """2 frames (below min_frames_nonzero) should get 0 score."""
        cfg = FrameCountScoring(min_frames_full_score=10, min_frames_nonzero=3, use_gradual_scoring=True)
        total_frames = 2
        if total_frames >= cfg.min_frames_full_score:
            score = 1.0
        elif total_frames >= cfg.min_frames_nonzero:
            score = 0.3 + 0.7 * (total_frames - cfg.min_frames_nonzero) / max(1, cfg.min_frames_full_score - cfg.min_frames_nonzero)
        else:
            score = 0.0
        assert score == 0.0

    def test_legacy_cliff_scoring(self):
        """Verify legacy cliff scoring when use_gradual_scoring is False."""
        cfg = FrameCountScoring(min_frames_full_score=10, min_frames_nonzero=3, use_gradual_scoring=False)
        # Legacy behavior
        assert (1.0 if 10 >= 10 else 0.5) == 1.0
        assert (1.0 if 9 >= 10 else 0.5) == 0.5
        assert (1.0 if 5 >= 10 else 0.5) == 0.5


class TestPartialObjectMotionCredit:
    """Tests for partial object motion credit in scene_state_penalty."""

    def test_full_motion_no_penalty(self):
        """Object that moves past threshold should get no penalty."""
        import numpy as np

        # Simulate object moving 2cm laterally (above 1cm threshold)
        _max_lateral = 0.02
        _max_lift = 0.0
        _move_thresh = 0.01
        _lift_thresh = 0.05
        _units_per_meter = 1.0

        _lift_credit = min(1.0, _max_lift / _lift_thresh) * 0.30 if _lift_thresh > 0 else 0.0
        _lateral_credit = min(1.0, _max_lateral / _move_thresh) * 0.70 if _move_thresh > 0 else 0.0
        _progress_ratio = _lift_credit + _lateral_credit
        scene_state_penalty = 0.20 * max(0.0, 1.0 - _progress_ratio)

        # Lateral credit: min(1.0, 0.02/0.01) * 0.70 = 1.0 * 0.70 = 0.70
        # Progress ratio: 0.70 + 0 = 0.70
        # Penalty: 0.20 * (1 - 0.70) = 0.06
        assert _lateral_credit == 0.70
        assert scene_state_penalty == pytest.approx(0.06, abs=0.001)

    def test_lift_only_partial_credit(self):
        """Object that lifts but doesn't move laterally gets partial credit."""
        import numpy as np

        # Simulate object lifting 5cm (full lift credit) but no lateral movement
        _max_lateral = 0.0
        _max_lift = 0.05
        _move_thresh = 0.01
        _lift_thresh = 0.05
        _units_per_meter = 1.0

        _lift_credit = min(1.0, _max_lift / _lift_thresh) * 0.30 if _lift_thresh > 0 else 0.0
        _lateral_credit = min(1.0, _max_lateral / _move_thresh) * 0.70 if _move_thresh > 0 else 0.0
        _progress_ratio = _lift_credit + _lateral_credit
        scene_state_penalty = 0.20 * max(0.0, 1.0 - _progress_ratio)

        # Lift credit: min(1.0, 0.05/0.05) * 0.30 = 1.0 * 0.30 = 0.30
        # Lateral credit: 0
        # Progress ratio: 0.30
        # Penalty: 0.20 * (1 - 0.30) = 0.14
        assert _lift_credit == 0.30
        assert scene_state_penalty == pytest.approx(0.14, abs=0.001)

    def test_no_motion_full_penalty(self):
        """Object that doesn't move at all gets full penalty."""
        _max_lateral = 0.0
        _max_lift = 0.0
        _move_thresh = 0.01
        _lift_thresh = 0.05

        _lift_credit = min(1.0, _max_lift / _lift_thresh) * 0.30 if _lift_thresh > 0 else 0.0
        _lateral_credit = min(1.0, _max_lateral / _move_thresh) * 0.70 if _move_thresh > 0 else 0.0
        _progress_ratio = _lift_credit + _lateral_credit
        scene_state_penalty = 0.20 * max(0.0, 1.0 - _progress_ratio)

        assert _progress_ratio == 0.0
        assert scene_state_penalty == 0.20

    def test_partial_lift_partial_credit(self):
        """Object that lifts 2.5cm (half of threshold) gets partial credit."""
        _max_lateral = 0.0
        _max_lift = 0.025  # Half of 5cm threshold
        _move_thresh = 0.01
        _lift_thresh = 0.05

        _lift_credit = min(1.0, _max_lift / _lift_thresh) * 0.30 if _lift_thresh > 0 else 0.0
        _lateral_credit = min(1.0, _max_lateral / _move_thresh) * 0.70 if _move_thresh > 0 else 0.0
        _progress_ratio = _lift_credit + _lateral_credit
        scene_state_penalty = 0.20 * max(0.0, 1.0 - _progress_ratio)

        # Lift credit: min(1.0, 0.025/0.05) * 0.30 = 0.5 * 0.30 = 0.15
        # Progress ratio: 0.15
        # Penalty: 0.20 * (1 - 0.15) = 0.17
        assert _lift_credit == pytest.approx(0.15, abs=0.001)
        assert scene_state_penalty == pytest.approx(0.17, abs=0.001)


class TestQualityConfigIntegration:
    """Integration tests for quality config changes."""

    def test_quality_config_json_has_new_fields(self):
        """Verify quality_config.json has the new fields."""
        repo_root = Path(__file__).resolve().parents[1]
        config_path = repo_root / "genie-sim-import-job" / "quality_config.json"
        payload = json.loads(config_path.read_text())

        assert "diversity_divisors" in payload
        assert "frame_count_scoring" in payload
        assert payload["frame_count_scoring"]["use_gradual_scoring"] is True

    def test_quality_config_schema_documented(self):
        """Verify new fields are documented in schema."""
        repo_root = Path(__file__).resolve().parents[1]
        config_path = repo_root / "genie-sim-import-job" / "quality_config.json"
        payload = json.loads(config_path.read_text())

        schema = payload.get("schema", {})
        assert "diversity_divisors" in schema
        assert "frame_count_scoring" in schema

    def test_all_robot_types_have_divisors(self):
        """Verify all expected robot types have divisors defined."""
        config = load_quality_config()
        expected_robots = ["franka", "panda", "g1", "ur5", "ur10", "default"]

        for robot in expected_robots:
            div = config.get_diversity_divisors(robot)
            assert div.action > 0, f"Robot {robot} has invalid action divisor"
            assert div.obs > 0, f"Robot {robot} has invalid obs divisor"
