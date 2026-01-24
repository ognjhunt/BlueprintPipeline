import json
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.unit
def test_sensor_data_validation_and_episode_errors(load_job_module) -> None:
    sensor_module = load_job_module("episode_generation", "sensor_data_capture.py")

    config = sensor_module.SensorDataConfig.from_data_pack(
        tier=sensor_module.DataPackTier.PLUS,
        num_cameras=1,
        resolution=(64, 48),
    )
    assert config.include_semantic_labels is True
    assert config.include_instance_ids is True

    frame = sensor_module.FrameSensorData(
        frame_index=0,
        timestamp=0.0,
        rgb_images={"wrist": np.zeros((48, 64, 3), dtype=np.uint8)},
        depth_maps={"wrist": np.ones((48, 64), dtype=np.float32)},
        semantic_masks={"wrist": np.zeros((48, 64), dtype=np.uint8)},
        instance_masks={"wrist": np.zeros((48, 64), dtype=np.uint16)},
    )
    assert frame.validate_rgb_frames() == []
    assert frame.validate_depth_frames() == []
    assert frame.validate_segmentation_frames() == []

    bad_frame = sensor_module.FrameSensorData(
        frame_index=1,
        timestamp=0.1,
        rgb_images={"wrist": np.zeros((48, 64), dtype=np.float32)},
    )
    errors = bad_frame.validate_rgb_frames()
    assert any("Expected shape" in error for error in errors)
    assert any("Expected dtype" in error for error in errors)

    episode = sensor_module.EpisodeSensorData(
        episode_id="episode_1",
        config=config,
        frames=[frame, bad_frame],
    )
    episode_errors = episode.validate_all_frames()
    assert any("Frame 1" in error for error in episode_errors)


@pytest.mark.unit
def test_quality_certificate_generation(load_job_module, tmp_path: Path) -> None:
    quality_module = load_job_module("episode_generation", "quality_certificate.py")
    enforcement_module = load_job_module("episode_generation", "isaac_sim_enforcement.py")

    capabilities = enforcement_module.EnvironmentCapabilities(
        isaac_sim_available=True,
        physx_available=True,
        replicator_available=True,
        gpu_available=True,
        production_mode=True,
        allow_mock_capture=False,
    )

    generator = quality_module.QualityCertificateGenerator(capabilities)
    cert = generator.generate_certificate(
        episode_id="ep_001",
        scene_id="scene_001",
        task_id="task_001",
        trajectory_metrics=quality_module.TrajectoryQualityMetrics(smoothness_score=1.0),
        visual_metrics=quality_module.VisualQualityMetrics(
            target_visibility_ratio=1.0,
            mean_sharpness=100.0,
            viewpoint_diversity=1.0,
        ),
        task_metrics=quality_module.TaskQualityMetrics(
            goal_achievement_score=1.0,
            skill_correctness_ratio=1.0,
            constraint_satisfaction_score=1.0,
        ),
        diversity_metrics=quality_module.DiversityMetrics(
            trajectory_novelty=1.0,
            state_space_coverage=1.0,
        ),
        sim2real_metrics=quality_module.Sim2RealMetrics(
            physics_plausibility_score=1.0,
            timing_realism_score=1.0,
        ),
        validation_passed=True,
        frame_count=2,
        camera_count=1,
        episode_data_hash="abc123",
    )

    assert cert.overall_quality_score >= 0.9
    assert cert.recommended_use in {"production_training", "fine_tuning"}

    output_path = tmp_path / "certificate.json"
    cert.save(output_path)
    saved = json.loads(output_path.read_text())
    assert saved["episode_id"] == "ep_001"
    assert saved["frame_count"] == 2


@pytest.mark.unit
@pytest.mark.parametrize(
    ("env_var", "env_value"),
    [
        ("PIPELINE_ENV", "production"),
        ("DATA_QUALITY_LEVEL", "production"),
    ],
)
def test_mock_capture_blocked_in_production_envs(
    monkeypatch,
    load_job_module,
    env_var: str,
    env_value: str,
) -> None:
    sensor_module = load_job_module("episode_generation", "sensor_data_capture.py")
    monkeypatch.setenv(env_var, env_value)
    monkeypatch.setenv("ALLOW_MOCK_CAPTURE", "true")

    with pytest.raises(RuntimeError, match="Mock sensor capture is not permitted"):
        sensor_module.create_sensor_capture(
            capture_mode=sensor_module.SensorDataCaptureMode.MOCK_DEV,
            allow_mock_capture=True,
            use_mock=True,
        )
