#!/usr/bin/env python3
"""Tests for Genie Sim premium exporter factories."""

from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Dict

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPORT_ROOT = REPO_ROOT / "genie-sim-export-job"
sys.path.insert(0, str(EXPORT_ROOT))


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _assert_audio_defaults(config: Any) -> None:
    assert config.enabled is True
    assert config.voice_presets
    assert config.default_voice_preset == "narrator"
    # google provider requires GOOGLE_APPLICATION_CREDENTIALS env; only local+mock are always available
    assert "local" in config.tts_providers
    assert "mock" in config.tts_providers
    assert config.audio_output.sample_rate == 22050
    assert config.narration_templates
    assert config.narrate_all_episodes is True
    assert config.max_episodes_per_scene == -1
    assert config.include_skill_segments is True


def _assert_premium_defaults(config: Any) -> None:
    assert config.enabled is True
    assert config.output_format == "parquet"
    assert config.telemetry is not None
    assert config.failure_analysis is not None
    assert config.grasp_analytics is not None
    assert config.parallel_eval is not None


def _assert_sim2real_validation_defaults(config: Any) -> None:
    assert config.enabled is True
    assert "basic" in config.tiers
    assert config.guarantee_config is not None
    assert "sim_success_rate" in config.tracked_metrics
    assert "grasp_failure" in config.failure_modes
    assert "markdown" in config.report_formats


def _assert_generalization_defaults(output_dir: Path, manifests: Dict[str, Path]) -> None:
    config = _load_json(Path(manifests["generalization_analysis_config"]))
    assert config["enabled"] is True
    assert config["scene_id"] == "scene_001"
    assert config["analysis_config"]["learning_curve_window"] == 50
    assert "variation_types" in config["analysis_config"]


def _assert_language_defaults(output_dir: Path, manifests: Dict[str, Path]) -> None:
    config = _load_json(Path(manifests["language_annotations_config"]))
    assert config["enabled"] is True
    assert config["scene_id"] == "scene_001"
    assert config["annotation_config"]["num_variations_per_task"] == 10
    assert config["annotation_config"]["llm_provider"] == "gemini"
    assert "imperative" in config["annotation_config"]["styles"]


def _assert_policy_leaderboard_defaults(output_dir: Path, manifests: Dict[str, Path]) -> None:
    config = _load_json(Path(manifests["policy_leaderboard_config"]))
    assert config["enabled"] is True
    assert config["scene_id"] == "scene_001"
    assert config["leaderboard_config"]["confidence_level"] == 0.95
    assert config["leaderboard_config"]["bootstrap_samples"] == 10000
    assert Path(manifests["policy_leaderboard_utils"]).exists()


def _assert_sim2real_fidelity_defaults(output_dir: Path, manifests: Dict[str, Path]) -> None:
    config = _load_json(Path(manifests["sim2real_fidelity_matrix"]))
    assert config["scene_id"] == "scene_001"
    assert config["robot_type"] == "franka"
    assert "overall_scores" in config
    assert "component_fidelity" in config


def _assert_sim2real_validation_manifest(output_dir: Path, manifests: Dict[str, Path]) -> None:
    config = _load_json(Path(manifests["config"]))
    assert config["scene_id"] == "scene_001"
    assert config["robot_type"] == "franka"
    assert config["config"]["enabled"] is True
    assert "tiers" in config["config"]


def _assert_tactile_defaults(output_dir: Path, manifests: Dict[str, Path]) -> None:
    config = _load_json(Path(manifests["tactile_sensor_config"]))
    assert config["enabled"] is True
    assert config["scene_id"] == "scene_001"
    assert config["sensor_config"]["sensor_type"] == "gelslim"


def _assert_trajectory_defaults(output_dir: Path, manifests: Dict[str, Path]) -> None:
    config = _load_json(Path(manifests["trajectory_optimality_config"]))
    assert config["enabled"] is True
    assert config["scene_id"] == "scene_001"
    assert config["analysis_config"]["compute_path_efficiency"] is True
    assert Path(manifests["trajectory_analysis_utils"]).exists()


def _assert_embodiment_defaults(output_dir: Path, manifests: Dict[str, Path]) -> None:
    config = _load_json(Path(manifests["embodiment_transfer_matrix"]))
    assert config["scene_id"] == "scene_001"
    assert config["source_robot"] == "franka"
    assert "compatibility_matrix" in config


def _assert_premium_analytics_export(manifests: Dict[str, Path]) -> None:
    assert Path(manifests["config"]).exists()


EXPORTER_CASES = [
    {
        "name": "audio_narration",
        "module": "default_audio_narration",
        "factory": "create_default_audio_narration_exporter",
        "exporter_class": "DefaultAudioNarrationExporter",
        "factory_kwargs": lambda output_dir: {"scene_id": "scene_001", "output_dir": output_dir},
        "config_assert": _assert_audio_defaults,
        "marker_files": [".audio_narration_enabled"],
    },
    {
        "name": "embodiment_transfer",
        "module": "default_embodiment_transfer",
        "factory": "create_default_embodiment_transfer_exporter",
        "exporter_class": None,
        "factory_kwargs": lambda output_dir: {
            "scene_id": "scene_001",
            "source_robot": "franka",
            "output_dir": output_dir,
        },
        "manifest_assert": _assert_embodiment_defaults,
    },
    {
        "name": "generalization_analyzer",
        "module": "default_generalization_analyzer",
        "factory": "create_default_generalization_analyzer_exporter",
        "exporter_class": None,
        "factory_kwargs": lambda output_dir: {"scene_id": "scene_001", "output_dir": output_dir},
        "manifest_assert": _assert_generalization_defaults,
    },
    {
        "name": "language_annotations",
        "module": "default_language_annotations",
        "factory": "create_default_language_annotations_exporter",
        "exporter_class": None,
        "factory_kwargs": lambda output_dir: {"scene_id": "scene_001", "output_dir": output_dir},
        "manifest_assert": _assert_language_defaults,
    },
    {
        "name": "policy_leaderboard",
        "module": "default_policy_leaderboard",
        "factory": "create_default_policy_leaderboard_exporter",
        "exporter_class": None,
        "factory_kwargs": lambda output_dir: {"scene_id": "scene_001", "output_dir": output_dir},
        "manifest_assert": _assert_policy_leaderboard_defaults,
    },
    {
        "name": "premium_analytics",
        "module": "default_premium_analytics",
        "factory": "create_default_premium_analytics_exporter",
        "exporter_class": "DefaultPremiumAnalyticsExporter",
        "factory_kwargs": lambda output_dir: {"scene_id": "scene_001", "output_dir": output_dir},
        "config_assert": _assert_premium_defaults,
        "marker_files": [".premium_analytics_enabled"],
        "manifest_assert": lambda output_dir, manifests: _assert_premium_analytics_export(manifests),
    },
    {
        "name": "sim2real_fidelity",
        "module": "default_sim2real_fidelity",
        "factory": "create_default_sim2real_fidelity_exporter",
        "exporter_class": None,
        "factory_kwargs": lambda output_dir: {
            "scene_id": "scene_001",
            "robot_type": "franka",
            "output_dir": output_dir,
        },
        "manifest_assert": _assert_sim2real_fidelity_defaults,
    },
    {
        "name": "sim2real_validation",
        "module": "default_sim2real_validation",
        "factory": "create_default_sim2real_validation_exporter",
        "exporter_class": "DefaultSim2RealValidationExporter",
        "factory_kwargs": lambda output_dir: {
            "scene_id": "scene_001",
            "robot_type": "franka",
            "output_dir": output_dir,
        },
        "config_assert": _assert_sim2real_validation_defaults,
        "manifest_assert": _assert_sim2real_validation_manifest,
        "marker_files": [".sim2real_validation_enabled"],
    },
    {
        "name": "tactile_sensor",
        "module": "default_tactile_sensor_sim",
        "factory": "create_default_tactile_sensor_exporter",
        "exporter_class": None,
        "factory_kwargs": lambda output_dir: {"scene_id": "scene_001", "output_dir": output_dir},
        "manifest_assert": _assert_tactile_defaults,
    },
    {
        "name": "trajectory_optimality",
        "module": "default_trajectory_optimality",
        "factory": "create_default_trajectory_optimality_exporter",
        "exporter_class": None,
        "factory_kwargs": lambda output_dir: {"scene_id": "scene_001", "output_dir": output_dir},
        "manifest_assert": _assert_trajectory_defaults,
    },
]


@pytest.mark.parametrize("case", EXPORTER_CASES, ids=lambda case: case["name"])
def test_geniesim_premium_exporters(case: Dict[str, Any], tmp_path: Path) -> None:
    # Remove cached module and ensure genie-sim-export-job is first in sys.path
    # to avoid pollution from arena-export-job (which has identically named modules)
    sys.modules.pop(case["module"], None)
    if str(EXPORT_ROOT) in sys.path:
        sys.path.remove(str(EXPORT_ROOT))
    sys.path.insert(0, str(EXPORT_ROOT))
    module = import_module(case["module"])
    factory: Callable[..., Any] = getattr(module, case["factory"])
    output_dir = tmp_path / case["name"]

    result = factory(**case["factory_kwargs"](output_dir))

    exporter_class = case.get("exporter_class")
    if exporter_class:
        exporter_type = getattr(module, exporter_class)
        assert isinstance(result, exporter_type)
        assert result.output_dir == output_dir
        config_assert = case.get("config_assert")
        if config_assert:
            config_assert(result.config)
        manifests = result.export_all_manifests()
        assert manifests
        for path in manifests.values():
            path = Path(path)
            assert path.exists()
            assert output_dir == path or output_dir in path.parents
        for marker in case.get("marker_files", []):
            assert (output_dir / marker).exists()
        manifest_assert = case.get("manifest_assert")
        if manifest_assert:
            manifest_assert(output_dir, manifests)
        return

    assert isinstance(result, dict)
    assert result
    for path in result.values():
        path = Path(path)
        assert path.exists()
        assert output_dir == path or output_dir in path.parents
    manifest_assert = case.get("manifest_assert")
    if manifest_assert:
        manifest_assert(output_dir, result)
