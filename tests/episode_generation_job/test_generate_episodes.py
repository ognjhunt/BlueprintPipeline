import numpy as np
import pytest


def _build_joint_trajectory(trajectory_solver, trajectory_id="traj_test"):
    robot_config = trajectory_solver.ROBOT_CONFIGS["franka"]
    states = [
        trajectory_solver.JointState(
            frame_idx=0,
            timestamp=0.0,
            joint_positions=robot_config.default_joint_positions.copy(),
            gripper_position=0.04,
        ),
        trajectory_solver.JointState(
            frame_idx=1,
            timestamp=0.5,
            joint_positions=robot_config.default_joint_positions.copy() + 0.05,
            gripper_position=0.02,
        ),
    ]
    return trajectory_solver.JointTrajectory(
        trajectory_id=trajectory_id,
        robot_type="franka",
        robot_config=robot_config,
        states=states,
        fps=30.0,
        total_duration=0.5,
    )


@pytest.mark.unit
def test_convert_episode_for_multiformat_reward(load_job_module) -> None:
    generate_episodes = load_job_module("episode_generation", "generate_episodes.py")
    trajectory_solver = load_job_module("episode_generation", "trajectory_solver.py")

    trajectory = _build_joint_trajectory(trajectory_solver)
    episode = generate_episodes.GeneratedEpisode(
        episode_id="episode_1",
        task_name="pick",
        task_description="Pick object",
        trajectory=trajectory,
        motion_plan=None,
        is_valid=True,
        quality_score=0.9,
    )

    result = generate_episodes._convert_episode_for_multiformat(
        episode,
        resolution=(640, 480),
        default_fps=30.0,
    )

    rewards = [frame["reward"] for frame in result["frames"]]
    assert rewards == [0.0, 1.0]


@pytest.mark.unit
def test_solve_trajectory_with_replan_returns_replanned_plan(monkeypatch, tmp_path, load_job_module) -> None:
    generate_episodes = load_job_module("episode_generation", "generate_episodes.py")
    motion_planner = load_job_module("episode_generation", "motion_planner.py")
    trajectory_solver = load_job_module("episode_generation", "trajectory_solver.py")

    config = generate_episodes.EpisodeGenerationConfig(
        scene_id="scene",
        manifest_path=tmp_path / "scene_manifest.json",
        output_dir=tmp_path,
        use_cpgen=False,
        use_validation=False,
        capture_sensor_data=False,
        max_retries=1,
    )
    generator = generate_episodes.EpisodeGenerator(config, verbose=False)

    waypoints = [
        motion_planner.Waypoint(
            position=np.array([0.0, 0.0, 0.5]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            duration_to_next=0.4,
        ),
        motion_planner.Waypoint(
            position=np.array([0.1, 0.0, 0.6]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            duration_to_next=0.4,
        ),
    ]
    motion_plan = motion_planner.MotionPlan(
        plan_id="plan-original",
        task_name="pick",
        task_description="Pick object",
        waypoints=waypoints,
    )

    replanned_motion_plan = motion_planner.MotionPlan(
        plan_id="plan-replanned",
        task_name="pick",
        task_description="Pick object",
        waypoints=waypoints,
    )
    solver_calls = {"count": 0}

    def fake_solve(_self, _plan):
        solver_calls["count"] += 1
        if solver_calls["count"] == 1:
            raise trajectory_solver.TrajectoryIKError("IK failure")
        return _build_joint_trajectory(trajectory_solver, trajectory_id="traj_replanned")

    def fake_plan_motion(**kwargs):
        assert kwargs["task_name"] == "pick"
        assert kwargs["task_description"] == "Pick object"
        return replanned_motion_plan

    monkeypatch.setattr(trajectory_solver.TrajectorySolver, "solve", fake_solve)
    monkeypatch.setattr(generator.motion_planner, "plan_motion", fake_plan_motion)

    returned_plan, returned_trajectory = generator._solve_trajectory_with_replan(
        motion_plan=motion_plan,
        task_name="pick",
        task_description="Pick object",
        target_object={"id": "mug"},
        place_position=[0.2, 0.0, 0.5],
        articulation_info=None,
        context_label="seed",
    )

    assert solver_calls["count"] == 2
    assert returned_plan is replanned_motion_plan
    assert returned_plan.task_name == "pick"
    assert returned_plan.task_description == "Pick object"
    assert returned_trajectory.trajectory_id == "traj_replanned"


@pytest.mark.unit
def test_build_quality_certificate_passes_metrics(monkeypatch, tmp_path, load_job_module) -> None:
    generate_episodes = load_job_module("episode_generation", "generate_episodes.py")
    trajectory_solver = load_job_module("episode_generation", "trajectory_solver.py")
    sim_validator = load_job_module("episode_generation", "sim_validator.py")

    class FakeCertificate:
        def __init__(self):
            self.validation_warnings = []
            self.validation_errors = []
            self.sensor_source = None
            self.physics_backend = None
            self.overall_quality_score = None
            self.recommended_use = None
            self.confidence_score = None
            self.data_quality_level = None

        def add_error(self, message: str) -> None:
            self.validation_errors.append(message)

        def add_warning(self, message: str) -> None:
            self.validation_warnings.append(message)

        def assess_training_suitability(self) -> str:
            return "production_training"

    class FakeQualityCertificateGenerator:
        def __init__(self, *_args, **_kwargs):
            self.last_kwargs = None

        def generate_certificate(self, **kwargs):
            self.last_kwargs = kwargs
            return FakeCertificate()

        def _compute_confidence_score(self, _cert):
            return 0.42

    monkeypatch.setattr(generate_episodes, "QualityCertificateGenerator", FakeQualityCertificateGenerator)
    monkeypatch.setattr(generate_episodes, "HAVE_QUALITY_SYSTEM", True)

    config = generate_episodes.EpisodeGenerationConfig(
        scene_id="scene",
        manifest_path=tmp_path / "scene_manifest.json",
        output_dir=tmp_path,
        use_cpgen=False,
        use_validation=False,
        capture_sensor_data=False,
    )
    generator = generate_episodes.EpisodeGenerator(config, verbose=False)

    trajectory = _build_joint_trajectory(trajectory_solver)
    metrics = sim_validator.QualityMetrics(
        task_success=True,
        grasp_success=True,
        placement_success=False,
        total_collisions=2,
        joint_limit_violations=1,
        torque_limit_violations=1,
        path_length=1.2,
        jerk_integral=0.3,
        velocity_smoothness=0.75,
        total_duration=2.0,
    )
    validation_result = sim_validator.ValidationResult(
        episode_id="episode_2",
        status=sim_validator.ValidationStatus.PASSED,
        metrics=metrics,
        physics_backend="isaac_sim",
    )

    episode = generate_episodes.GeneratedEpisode(
        episode_id="episode_2",
        task_name="pick",
        task_description="Pick object",
        trajectory=trajectory,
        motion_plan=None,
        is_valid=True,
        quality_score=0.88,
        validation_result=validation_result,
        validation_errors=["warning"],
    )

    fake_generator = generate_episodes.QualityCertificateGenerator("env")
    cert = generator._build_quality_certificate(episode, fake_generator)

    assert cert.sensor_source == "disabled"
    assert cert.physics_backend == generate_episodes.PhysicsValidationBackend.PHYSX.value
    assert cert.overall_quality_score == 0.88

    passed_metrics = fake_generator.last_kwargs["trajectory_metrics"]
    assert passed_metrics.collision_count == 2
    assert passed_metrics.joint_limit_violations == 1
    assert passed_metrics.torque_limit_violations == 1
    assert passed_metrics.trajectory_length_meters == 1.2
    assert passed_metrics.mean_jerk == 0.3
    assert passed_metrics.smoothness_score == 0.75
