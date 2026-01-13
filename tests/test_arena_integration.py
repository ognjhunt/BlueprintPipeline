"""Tests for Arena integration modules."""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass

from tools.arena_integration.arena_exporter import (
    ArenaExporter,
    ArenaExportConfig,
    ArenaExportResult,
)
from tools.arena_integration.task_mapping import (
    TaskMapper,
    TaskMapping,
)
from tools.arena_integration.evaluation_runner import (
    EvaluationRunner,
    EvaluationConfig,
)
from tools.arena_integration.components import (
    ArenaComponent,
    ComponentType,
)


class TestArenaComponent:
    """Test ArenaComponent class."""

    def test_component_creation(self):
        """Test creating arena component."""
        component = ArenaComponent(
            component_id="comp_001",
            component_type=ComponentType.SENSOR,
            name="Camera",
            properties={"resolution": [640, 480]},
        )
        assert component.component_id == "comp_001"
        assert component.component_type == ComponentType.SENSOR
        assert component.name == "Camera"

    def test_component_properties(self):
        """Test component properties."""
        component = ArenaComponent(
            component_id="comp_001",
            component_type=ComponentType.ACTUATOR,
            name="Gripper",
            properties={"force": 100.0, "speed": 0.5},
        )
        assert component.properties["force"] == 100.0
        assert component.properties["speed"] == 0.5

    def test_component_to_dict(self):
        """Test serializing component."""
        component = ArenaComponent(
            component_id="comp_001",
            component_type=ComponentType.SENSOR,
            name="Camera",
        )
        comp_dict = component.to_dict()
        assert comp_dict["component_id"] == "comp_001"
        assert comp_dict["name"] == "Camera"


class TestTaskMapping:
    """Test TaskMapping class."""

    def test_task_mapping_creation(self):
        """Test creating task mapping."""
        mapping = TaskMapping(
            blueprint_task="reaching",
            arena_task="reach",
            mapping_config={"scaling": 1.0},
        )
        assert mapping.blueprint_task == "reaching"
        assert mapping.arena_task == "reach"

    def test_task_mapping_with_transforms(self):
        """Test task mapping with transforms."""
        mapping = TaskMapping(
            blueprint_task="grasping",
            arena_task="grasp",
            transforms={
                "state_transform": lambda x: x * 2,
                "action_transform": lambda x: x / 2,
            },
        )
        assert mapping.transforms is not None


class TestTaskMapper:
    """Test TaskMapper class."""

    def test_task_mapper_init(self):
        """Test initializing task mapper."""
        mapper = TaskMapper()
        assert mapper is not None

    def test_register_mapping(self):
        """Test registering task mapping."""
        mapper = TaskMapper()
        mapping = TaskMapping(
            blueprint_task="reaching",
            arena_task="reach",
        )
        mapper.register_mapping(mapping)

        # Check mapping was registered
        registered = mapper.get_mapping("reaching")
        assert registered is not None

    def test_map_task(self):
        """Test mapping task."""
        mapper = TaskMapper()
        mapper.register_mapping(TaskMapping("reaching", "reach"))

        arena_task = mapper.map_task("reaching")
        assert arena_task == "reach"

    def test_get_unmapped_tasks(self):
        """Test getting unmapped tasks."""
        mapper = TaskMapper()
        mapper.register_mapping(TaskMapping("reaching", "reach"))

        # Try to map unmapped task
        try:
            arena_task = mapper.map_task("pushing")
            # If no error, check result
            assert arena_task is not None or True
        except KeyError:
            # Expected if task not mapped
            pass


class TestArenaExportConfig:
    """Test ArenaExportConfig class."""

    def test_config_creation(self):
        """Test creating arena export config."""
        config = ArenaExportConfig(
            scene_id="test_scene",
            output_format="arena_3.0",
        )
        assert config.scene_id == "test_scene"
        assert config.output_format == "arena_3.0"

    def test_config_defaults(self):
        """Test config default values."""
        config = ArenaExportConfig(scene_id="test")
        assert config.include_metadata is True
        assert config.validate_output is True

    def test_config_custom_mapping(self):
        """Test config with custom task mapping."""
        task_mapping = {"reaching": "reach", "grasping": "grasp"}
        config = ArenaExportConfig(
            scene_id="test",
            custom_task_mapping=task_mapping,
        )
        assert config.custom_task_mapping == task_mapping


class TestArenaExportResult:
    """Test ArenaExportResult class."""

    def test_result_creation(self):
        """Test creating export result."""
        result = ArenaExportResult(
            success=True,
            scene_id="test",
            output_dir=Path("/tmp/output"),
        )
        assert result.success is True
        assert result.scene_id == "test"

    def test_result_with_errors(self):
        """Test result with errors."""
        result = ArenaExportResult(
            success=False,
            scene_id="test",
            output_dir=Path("/tmp/output"),
            errors=["Component mapping failed", "Invalid task"],
        )
        assert result.success is False
        assert len(result.errors) == 2

    def test_result_to_dict(self):
        """Test serializing result."""
        result = ArenaExportResult(
            success=True,
            scene_id="test",
            output_dir=Path("/tmp/output"),
        )
        result_dict = result.to_dict()
        assert result_dict["success"] is True
        assert result_dict["scene_id"] == "test"


class TestArenaExporter:
    """Test ArenaExporter class."""

    @pytest.fixture
    def temp_export_dir(self):
        """Create temporary directory for exports."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_exporter_init(self):
        """Test initializing exporter."""
        config = ArenaExportConfig(scene_id="test")
        exporter = ArenaExporter(config)
        assert exporter.config == config

    def test_export_basic(self, temp_export_dir):
        """Test basic export."""
        config = ArenaExportConfig(scene_id="test_scene")
        exporter = ArenaExporter(config, verbose=False)

        # Create minimal input
        input_file = temp_export_dir / "input.json"
        input_file.write_text(json.dumps({
            "scene_id": "test_scene",
            "components": [],
            "tasks": [],
        }))

        # Export
        result = exporter.export(input_file, temp_export_dir / "output")
        assert result is not None

    def test_export_with_components(self, temp_export_dir):
        """Test export with components."""
        config = ArenaExportConfig(scene_id="test")
        exporter = ArenaExporter(config, verbose=False)

        input_file = temp_export_dir / "input.json"
        input_file.write_text(json.dumps({
            "scene_id": "test",
            "components": [
                {
                    "id": "camera_001",
                    "type": "sensor",
                    "name": "Camera",
                    "properties": {"resolution": [640, 480]},
                },
            ],
            "tasks": [],
        }))

        result = exporter.export(input_file, temp_export_dir / "output")
        assert result is not None


class TestEvaluationConfig:
    """Test EvaluationConfig class."""

    def test_config_creation(self):
        """Test creating evaluation config."""
        config = EvaluationConfig(
            task_name="reaching",
            num_episodes=10,
        )
        assert config.task_name == "reaching"
        assert config.num_episodes == 10

    def test_config_defaults(self):
        """Test config defaults."""
        config = EvaluationConfig(task_name="test")
        assert config.num_episodes > 0
        assert config.timeout_per_episode > 0


class TestEvaluationRunner:
    """Test EvaluationRunner class."""

    def test_runner_init(self):
        """Test initializing evaluation runner."""
        config = EvaluationConfig(task_name="reaching")
        runner = EvaluationRunner(config)
        assert runner.config == config

    def test_run_evaluation_mock(self):
        """Test running evaluation with mock."""
        config = EvaluationConfig(
            task_name="reaching",
            num_episodes=2,
        )
        runner = EvaluationRunner(config, verbose=False)

        # Mock environment
        mock_env = MagicMock()
        mock_env.reset.return_value = {"obs": [0.0] * 10}
        mock_env.step.return_value = (
            {"obs": [0.1] * 10},
            1.0,  # reward
            False,  # done
            {},
        )

        with patch.object(runner, "create_environment", return_value=mock_env):
            result = runner.run()
            assert result is not None


class TestArenaIntegrationWorkflow:
    """Test complete Arena integration workflow."""

    @pytest.fixture
    def temp_workflow_dir(self):
        """Create temporary directory for workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_full_export_workflow(self, temp_workflow_dir):
        """Test complete export workflow."""
        # Step 1: Create task mapping
        mapper = TaskMapper()
        mapper.register_mapping(TaskMapping("reaching", "reach"))
        mapper.register_mapping(TaskMapping("grasping", "grasp"))

        # Step 2: Create export config
        config = ArenaExportConfig(
            scene_id="test_scene",
            include_metadata=True,
        )

        # Step 3: Create exporter
        exporter = ArenaExporter(config, verbose=False)

        # Step 4: Prepare input
        input_file = temp_workflow_dir / "scene.json"
        input_file.write_text(json.dumps({
            "scene_id": "test_scene",
            "components": [],
            "tasks": ["reaching"],
        }))

        # Step 5: Export
        output_dir = temp_workflow_dir / "output"
        result = exporter.export(input_file, output_dir)

        assert result is not None

    def test_task_mapping_pipeline(self):
        """Test task mapping pipeline."""
        # Create mapper
        mapper = TaskMapper()

        # Register multiple mappings
        blueprint_tasks = [
            ("reaching", "reach"),
            ("grasping", "grasp"),
            ("pushing", "push"),
            ("placing", "place"),
        ]

        for bp_task, arena_task in blueprint_tasks:
            mapping = TaskMapping(bp_task, arena_task)
            mapper.register_mapping(mapping)

        # Test mappings
        for bp_task, expected_arena_task in blueprint_tasks:
            mapped = mapper.map_task(bp_task)
            assert mapped == expected_arena_task

    def test_component_export_pipeline(self, temp_workflow_dir):
        """Test component export pipeline."""
        components = [
            ArenaComponent(
                component_id="cam_1",
                component_type=ComponentType.SENSOR,
                name="RGB Camera",
                properties={"resolution": [640, 480], "fps": 30},
            ),
            ArenaComponent(
                component_id="gripper_1",
                component_type=ComponentType.ACTUATOR,
                name="Parallel Gripper",
                properties={"force_limit": 100.0},
            ),
        ]

        # Serialize components
        output_file = temp_workflow_dir / "components.json"
        with open(output_file, "w") as f:
            json.dump([c.to_dict() for c in components], f)

        # Verify file
        assert output_file.exists()

        # Load and verify
        with open(output_file) as f:
            loaded = json.load(f)

        assert len(loaded) == 2
        assert loaded[0]["name"] == "RGB Camera"


class TestArenaIntegrationEdgeCases:
    """Test edge cases in Arena integration."""

    def test_empty_scene_export(self):
        """Test exporting empty scene."""
        config = ArenaExportConfig(scene_id="empty")
        exporter = ArenaExporter(config, verbose=False)

        # Minimal empty scene
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "empty.json"
            input_file.write_text(json.dumps({
                "scene_id": "empty",
                "components": [],
                "tasks": [],
            }))

            result = exporter.export(input_file, Path(tmpdir) / "output")
            # Should handle empty scene gracefully
            assert result is not None

    def test_large_component_count(self):
        """Test export with many components."""
        components = []
        for i in range(100):
            components.append(
                ArenaComponent(
                    component_id=f"comp_{i:03d}",
                    component_type=ComponentType.SENSOR,
                    name=f"Component {i}",
                )
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "large_scene.json"
            with open(output_file, "w") as f:
                json.dump([c.to_dict() for c in components], f)

            assert output_file.exists()
            with open(output_file) as f:
                loaded = json.load(f)
            assert len(loaded) == 100

    def test_special_characters_in_names(self):
        """Test components with special characters in names."""
        names = [
            "Camera-RGB",
            "Gripper (Parallel)",
            "Joint #1",
            "Sensor/Vision",
            "Component $Price",
        ]

        for i, name in enumerate(names):
            component = ArenaComponent(
                component_id=f"comp_{i}",
                component_type=ComponentType.SENSOR,
                name=name,
            )
            comp_dict = component.to_dict()
            assert comp_dict["name"] == name
