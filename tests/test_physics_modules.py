from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.physics import dynamic_scene, multi_agent, soft_body


@pytest.fixture()
def coordinator():
    coordinator = multi_agent.MultiAgentCoordinator(enable_logging=False)
    coordinator.add_agent(
        multi_agent.AgentConfiguration(
            agent_id="a1",
            robot_type="franka_panda",
            spawn_position=(-1.0, 0.0, 0.0),
            workspace_bounds=(1.0, 1.0, 1.0),
        )
    )
    coordinator.add_agent(
        multi_agent.AgentConfiguration(
            agent_id="a2",
            robot_type="franka_panda",
            spawn_position=(1.0, 0.0, 0.0),
            workspace_bounds=(1.0, 1.0, 1.0),
        )
    )
    return coordinator


def test_multi_agent_nearest_neighbor_allocation_and_conflict(coordinator):
    coordinator.set_object_positions(
        {
            "obj_left": (-0.8, 0.0, 0.0),
            "obj_right": (0.9, 0.0, 0.0),
        }
    )
    allocation = coordinator.allocate_tasks(
        ["obj_left", "obj_right"],
        strategy=multi_agent.TaskAllocationStrategy.NEAREST_NEIGHBOR,
    )

    assert allocation["a1"] == ["obj_left"]
    assert allocation["a2"] == ["obj_right"]

    conflicts = coordinator.detect_workspace_conflicts()
    assert conflicts == []


def test_multi_agent_invalid_strategy_raises(coordinator):
    with pytest.raises(ValueError, match="Unknown allocation strategy"):
        coordinator.allocate_tasks(["obj"], strategy="not-a-strategy")


def test_dynamic_scene_conveyor_belt_updates_and_export():
    manager = dynamic_scene.DynamicSceneManager(enable_logging=False)
    obstacle = manager.create_conveyor_belt(
        obstacle_id="belt_1",
        position=(0.0, 0.0, 0.0),
        length=2.0,
        speed=1.5,
        direction=(2.0, 0.0, 0.0),
    )

    assert obstacle.velocity == (1.5, 0.0, 0.0)

    workspace = ((-0.1, -0.1, -0.1), (0.2, 0.2, 0.2))
    collisions = manager.detect_potential_collisions(workspace, time_horizon=1.0)
    assert collisions
    assert collisions[0]["obstacle_id"] == "belt_1"

    exported = manager.export_isaac_sim_config()
    assert exported["dynamic_obstacles"][0]["name"] == "belt_1"
    assert exported["dynamic_obstacles"][0]["linear_velocity"] == [1.5, 0.0, 0.0]


def test_dynamic_scene_empty_scene_no_collisions():
    manager = dynamic_scene.DynamicSceneManager(enable_logging=False)
    collisions = manager.detect_potential_collisions(
        ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)),
        time_horizon=1.0,
    )
    assert collisions == []


def test_soft_body_properties_and_mass_for_cloth():
    physics = soft_body.SoftBodyPhysics(enable_logging=False)
    obj_data = {"category": "towel", "material_name": "cotton"}
    bounds = {"size_m": [1.0, 0.5, 0.1], "volume_m3": 0.05}

    props = physics.generate_soft_body_properties(obj_data, bounds=bounds)

    assert props.soft_body_type == soft_body.SoftBodyType.CLOTH
    assert props.material == soft_body.DeformableMaterial.COTTON
    assert props.mass_per_area is not None
    assert props.total_mass == pytest.approx(props.mass_per_area * 1.0 * 0.5 * 2)


def test_soft_body_missing_definition_returns_none_and_no_attachments():
    physics = soft_body.SoftBodyPhysics(enable_logging=False)
    obj_data = {"category": "unknown", "material_name": "metal"}

    assert physics.detect_soft_body_type(obj_data) is None
    assert physics.is_soft_body({}) is False

    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    attachments = physics.generate_attachment_constraints(obj_data, vertices)
    assert attachments == []
