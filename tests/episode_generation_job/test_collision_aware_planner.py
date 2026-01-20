import numpy as np
import pytest


@pytest.mark.unit
def test_collision_primitive_aabb_shapes(load_job_module) -> None:
    planner_module = load_job_module("episode_generation", "collision_aware_planner.py")

    sphere = planner_module.CollisionPrimitive(
        prim_type="sphere",
        position=np.array([1.0, 2.0, 3.0]),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        radius=0.5,
    )
    min_pt, max_pt = sphere.get_aabb()
    assert np.allclose(min_pt, [0.5, 1.5, 2.5])
    assert np.allclose(max_pt, [1.5, 2.5, 3.5])

    box = planner_module.CollisionPrimitive(
        prim_type="box",
        position=np.array([0.0, 0.0, 0.0]),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        dimensions=np.array([2.0, 4.0, 6.0]),
    )
    min_pt, max_pt = box.get_aabb()
    assert np.allclose(min_pt, [-1.0, -2.0, -3.0])
    assert np.allclose(max_pt, [1.0, 2.0, 3.0])

    capsule = planner_module.CollisionPrimitive(
        prim_type="capsule",
        position=np.array([0.0, 0.0, 1.0]),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        radius=0.25,
        height=1.0,
    )
    min_pt, max_pt = capsule.get_aabb()
    assert np.allclose(min_pt, [-0.25, -0.25, 0.25])
    assert np.allclose(max_pt, [0.25, 0.25, 1.75])

    mesh = planner_module.CollisionPrimitive(
        prim_type="mesh",
        position=np.array([1.0, 0.0, 0.0]),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        vertices=np.array([[-1.0, -1.0, -1.0], [2.0, 1.0, 3.0]]),
    )
    min_pt, max_pt = mesh.get_aabb()
    assert np.allclose(min_pt, [0.0, -1.0, -1.0])
    assert np.allclose(max_pt, [3.0, 1.0, 3.0])


@pytest.mark.unit
def test_scene_collision_checker_point_and_segment(load_job_module) -> None:
    planner_module = load_job_module("episode_generation", "collision_aware_planner.py")

    checker = planner_module.SceneCollisionChecker(verbose=False)
    checker.add_obstacle(position=np.array([0.0, 0.0, 0.0]), dimensions=np.array([1.0, 1.0, 1.0]))

    assert checker.check_collision_point(np.array([0.0, 0.0, 0.0])) is True
    assert checker.check_collision_point(np.array([2.0, 0.0, 0.0])) is False

    assert checker.check_collision_segment(
        start=np.array([-1.0, 0.0, 0.0]),
        end=np.array([1.0, 0.0, 0.0]),
        radius=0.01,
    ) is True
    assert checker.check_collision_segment(
        start=np.array([2.0, 2.0, 0.0]),
        end=np.array([3.0, 2.0, 0.0]),
        radius=0.01,
    ) is False


@pytest.mark.unit
def test_collision_aware_planner_rrt_fallback(load_job_module, monkeypatch) -> None:
    planner_module = load_job_module("episode_generation", "collision_aware_planner.py")
    monkeypatch.setattr(planner_module, "_CUROBO_AVAILABLE", False)

    planner = planner_module.CollisionAwarePlanner(verbose=False, use_curobo=True)
    expected_path = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])]

    calls = {"rrt": 0}

    def fake_segment(*_args, **_kwargs):
        return True

    def fake_rrt_plan(*_args, **_kwargs):
        calls["rrt"] += 1
        return expected_path

    monkeypatch.setattr(planner.collision_checker, "check_collision_segment", fake_segment)
    monkeypatch.setattr(planner.rrt_planner, "plan", fake_rrt_plan)

    result = planner.plan_cartesian_path(
        start_pos=np.array([0.0, 0.0, 0.0]),
        goal_pos=np.array([1.0, 0.0, 0.0]),
    )

    assert result == expected_path
    assert calls["rrt"] == 1
    assert planner.is_using_curobo() is False
