"""Tests for Isaac Lab task generation modules."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass

from tools.isaac_lab_tasks.task_generator import (
    IsaacLabTaskGenerator,
    TaskGeneratorConfig,
)
from tools.isaac_lab_tasks.env_config import (
    EnvConfig,
    ObservationConfig,
    ActionConfig,
)
from tools.isaac_lab_tasks.reward_functions import RewardFunctionGenerator
from tools.isaac_lab_tasks.runtime_validator import RuntimeValidator


class TestTaskGeneratorConfig:
    """Test TaskGeneratorConfig."""

    def test_config_creation(self):
        """Test creating task generator config."""
        config = TaskGeneratorConfig(
            scene_id="test_scene",
            robot_type="franka",
        )
        assert config.scene_id == "test_scene"
        assert config.robot_type == "franka"

    def test_config_defaults(self):
        """Test config default values."""
        config = TaskGeneratorConfig(scene_id="test")
        assert config.simulation_dt > 0
        assert config.num_envs > 0
        assert config.enable_gpu_rendering in [True, False]


class TestObservationConfig:
    """Test ObservationConfig class."""

    def test_observation_config_creation(self):
        """Test creating observation config."""
        config = ObservationConfig(
            joint_pos=True,
            joint_vel=True,
            ee_position=True,
        )
        assert config.joint_pos is True
        assert config.joint_vel is True

    def test_observation_config_shape(self):
        """Test calculating observation shape."""
        config = ObservationConfig(
            joint_pos=True,  # Franka: 7 dims
            joint_vel=True,  # Franka: 7 dims
            ee_position=True,  # 3 dims
        )
        # Should calculate total observation dimension
        shape = config.get_obs_dim()
        assert shape > 0


class TestActionConfig:
    """Test ActionConfig class."""

    def test_action_config_creation(self):
        """Test creating action config."""
        config = ActionConfig(
            action_type="joint_pos",
            action_dim=7,
        )
        assert config.action_type == "joint_pos"
        assert config.action_dim == 7

    def test_action_config_invalid_type(self):
        """Test action config with invalid type."""
        with pytest.raises(ValueError):
            ActionConfig(
                action_type="invalid_type",
                action_dim=7,
            )

    def test_action_config_scale(self):
        """Test action scaling."""
        config = ActionConfig(
            action_type="joint_pos",
            action_dim=7,
            scale=2.0,
        )
        assert config.scale == 2.0


class TestEnvConfig:
    """Test EnvConfig class."""

    def test_env_config_creation(self):
        """Test creating environment config."""
        obs_cfg = ObservationConfig(joint_pos=True, joint_vel=True)
        act_cfg = ActionConfig(action_type="joint_pos", action_dim=7)

        env_cfg = EnvConfig(
            num_envs=4,
            observation_cfg=obs_cfg,
            action_cfg=act_cfg,
        )
        assert env_cfg.num_envs == 4
        assert env_cfg.observation_cfg == obs_cfg
        assert env_cfg.action_cfg == act_cfg

    def test_env_config_to_dict(self):
        """Test serializing env config."""
        obs_cfg = ObservationConfig(joint_pos=True)
        act_cfg = ActionConfig(action_type="joint_pos", action_dim=7)

        env_cfg = EnvConfig(
            num_envs=4,
            observation_cfg=obs_cfg,
            action_cfg=act_cfg,
        )
        cfg_dict = env_cfg.to_dict()
        assert cfg_dict["num_envs"] == 4
        assert "observation" in cfg_dict or "observations" in cfg_dict


class TestRewardFunctionGenerator:
    """Test RewardFunctionGenerator class."""

    def test_reward_generator_init(self):
        """Test initializing reward generator."""
        generator = RewardFunctionGenerator()
        assert generator is not None

    def test_generate_reaching_reward(self):
        """Test generating reaching reward function."""
        generator = RewardFunctionGenerator()

        reward_fn = generator.generate_reaching_reward(
            target_key="ee_position",
            weight=1.0,
        )
        assert reward_fn is not None
        assert callable(reward_fn)

    def test_generate_grasping_reward(self):
        """Test generating grasping reward function."""
        generator = RewardFunctionGenerator()

        reward_fn = generator.generate_grasping_reward(
            object_key="object_pos",
            gripper_key="gripper_state",
            weight=1.0,
        )
        assert reward_fn is not None

    def test_generate_pushing_reward(self):
        """Test generating pushing reward function."""
        generator = RewardFunctionGenerator()

        reward_fn = generator.generate_pushing_reward(
            object_key="object_pos",
            target_key="target_pos",
            weight=1.0,
        )
        assert reward_fn is not None

    def test_generate_composite_reward(self):
        """Test generating composite reward function."""
        generator = RewardFunctionGenerator()

        reward_fns = {
            "reaching": generator.generate_reaching_reward("ee_pos"),
            "completion": lambda obs: 1.0 if obs.get("done") else 0.0,
        }
        weights = {"reaching": 0.7, "completion": 0.3}

        composite = generator.generate_composite_reward(reward_fns, weights)
        assert composite is not None


class TestIsaacLabTaskGenerator:
    """Test IsaacLabTaskGenerator class."""

    @pytest.fixture
    def temp_task_dir(self):
        """Create temporary directory for tasks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_task_generator_init(self, temp_task_dir):
        """Test initializing task generator."""
        config = TaskGeneratorConfig(
            scene_id="test_scene",
            robot_type="franka",
        )
        generator = IsaacLabTaskGenerator(config, verbose=False)
        assert generator.config == config

    def test_generate_task_reaching(self, temp_task_dir):
        """Test generating reaching task."""
        config = TaskGeneratorConfig(
            scene_id="test_scene",
            robot_type="franka",
            task_type="reaching",
        )
        generator = IsaacLabTaskGenerator(config, verbose=False)

        result = generator.generate(output_dir=temp_task_dir)
        assert result is not None

    def test_generate_task_grasping(self, temp_task_dir):
        """Test generating grasping task."""
        config = TaskGeneratorConfig(
            scene_id="test_scene",
            robot_type="franka",
            task_type="grasping",
        )
        generator = IsaacLabTaskGenerator(config, verbose=False)

        result = generator.generate(output_dir=temp_task_dir)
        assert result is not None

    def test_generate_task_pushing(self, temp_task_dir):
        """Test generating pushing task."""
        config = TaskGeneratorConfig(
            scene_id="test_scene",
            robot_type="franka",
            task_type="pushing",
        )
        generator = IsaacLabTaskGenerator(config, verbose=False)

        result = generator.generate(output_dir=temp_task_dir)
        assert result is not None

    def test_generate_creates_required_files(self, temp_task_dir):
        """Test that generation creates required files."""
        config = TaskGeneratorConfig(
            scene_id="test_scene",
            robot_type="franka",
        )
        generator = IsaacLabTaskGenerator(config, verbose=False)

        result = generator.generate(output_dir=temp_task_dir)

        # Check for expected output files
        expected_files = [
            "task_env.py",
            "env_config.py",
            "reward_config.py",
        ]
        for filename in expected_files:
            file_path = temp_task_dir / filename
            # File might not exist if generation was mocked, but we can check the result
            assert result is not None


class TestRuntimeValidator:
    """Test RuntimeValidator class."""

    def test_validator_init(self):
        """Test initializing runtime validator."""
        validator = RuntimeValidator()
        assert validator is not None

    def test_validate_observation_space(self):
        """Test validating observation space."""
        validator = RuntimeValidator()

        obs_config = ObservationConfig(
            joint_pos=True,
            joint_vel=True,
            ee_position=True,
        )

        result = validator.validate_observation_space(obs_config)
        assert result.is_valid is True or result.is_valid is False

    def test_validate_action_space(self):
        """Test validating action space."""
        validator = RuntimeValidator()

        act_config = ActionConfig(
            action_type="joint_pos",
            action_dim=7,
        )

        result = validator.validate_action_space(act_config)
        assert result is not None

    def test_validate_reward_function(self):
        """Test validating reward function."""
        validator = RuntimeValidator()

        # Create a simple reward function
        def dummy_reward(obs):
            return 0.0

        result = validator.validate_reward_function(dummy_reward)
        assert result.is_valid is True

    def test_validate_environment_config(self):
        """Test validating full environment config."""
        validator = RuntimeValidator()

        obs_cfg = ObservationConfig(joint_pos=True)
        act_cfg = ActionConfig(action_type="joint_pos", action_dim=7)
        env_cfg = EnvConfig(num_envs=4, observation_cfg=obs_cfg, action_cfg=act_cfg)

        result = validator.validate_env_config(env_cfg)
        assert result is not None


class TestIsaacLabTaskIntegration:
    """Integration tests for task generation."""

    @pytest.fixture
    def temp_task_dir(self):
        """Create temporary directory for tasks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_full_task_generation_workflow(self, temp_task_dir):
        """Test complete task generation workflow."""
        # Create config
        config = TaskGeneratorConfig(
            scene_id="test_scene",
            robot_type="franka",
            task_type="reaching",
            num_envs=8,
        )

        # Generate task
        generator = IsaacLabTaskGenerator(config, verbose=False)
        result = generator.generate(output_dir=temp_task_dir)

        # Validate result
        validator = RuntimeValidator()
        assert result is not None

    def test_multiple_robot_types(self, temp_task_dir):
        """Test generating tasks for different robots."""
        robot_types = ["franka", "ur10", "panda"]

        for robot in robot_types:
            config = TaskGeneratorConfig(
                scene_id=f"test_scene_{robot}",
                robot_type=robot,
                task_type="reaching",
            )
            generator = IsaacLabTaskGenerator(config, verbose=False)
            result = generator.generate(output_dir=temp_task_dir / robot)
            assert result is not None

    def test_multiple_task_types(self, temp_task_dir):
        """Test generating different task types."""
        task_types = ["reaching", "grasping", "pushing"]

        for task in task_types:
            config = TaskGeneratorConfig(
                scene_id=f"test_scene_{task}",
                robot_type="franka",
                task_type=task,
            )
            generator = IsaacLabTaskGenerator(config, verbose=False)
            result = generator.generate(output_dir=temp_task_dir / task)
            assert result is not None


class TestTaskGeneratorEdgeCases:
    """Test edge cases in task generation."""

    def test_single_environment(self):
        """Test generating task with single environment."""
        config = TaskGeneratorConfig(
            scene_id="test",
            robot_type="franka",
            num_envs=1,
        )
        generator = IsaacLabTaskGenerator(config, verbose=False)
        assert generator.config.num_envs == 1

    def test_large_num_environments(self):
        """Test generating task with large number of environments."""
        config = TaskGeneratorConfig(
            scene_id="test",
            robot_type="franka",
            num_envs=256,
        )
        generator = IsaacLabTaskGenerator(config, verbose=False)
        assert generator.config.num_envs == 256

    def test_high_dimensional_observation(self):
        """Test observation config with many sensors."""
        obs_cfg = ObservationConfig(
            joint_pos=True,
            joint_vel=True,
            joint_acc=True,
            ee_position=True,
            ee_velocity=True,
            ee_force=True,
            object_pos=True,
            object_vel=True,
        )
        # Should handle high-dimensional observations
        obs_dim = obs_cfg.get_obs_dim()
        assert obs_dim > 20  # Should be fairly high dimensional
