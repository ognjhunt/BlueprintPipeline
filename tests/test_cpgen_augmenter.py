"""
Regression tests for CP-Gen style augmentation.

Covers pick-and-place style constraints with replanning validation.
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "episode-generation-job"))

from cpgen_augmenter import (
    ConstraintPreservingAugmenter,
    ConstraintSolver,
    KeypointConstraintModule,
    ObjectTransform,
)
from motion_planner import MotionPlan, MotionPhase, Waypoint
from task_specifier import (
    ConstraintType,
    Keypoint,
    KeypointConstraint,
    SegmentType,
    SkillSegment,
    TaskSpecification,
)


class TrackingConstraintModule(KeypointConstraintModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.solve_called = False
        self.validate_called = False

    def solve_waypoint(self, *args, **kwargs):
        self.solve_called = True
        return super().solve_waypoint(*args, **kwargs)

    def validate_trajectory(self, *args, **kwargs):
        self.validate_called = True
        return super().validate_trajectory(*args, **kwargs)


def _build_pick_place_spec() -> TaskSpecification:
    keypoint = Keypoint(
        keypoint_id="ee",
        frame="ee",
        local_position=np.zeros(3),
        description="End-effector keypoint",
    )
    pick_constraint = KeypointConstraint(
        constraint_id="pick_at_block",
        keypoint_id="ee",
        constraint_type=ConstraintType.RELATIVE_POSITION,
        reference_object_id="block",
        reference_offset=np.zeros(3),
        start_time=0.0,
        end_time=0.7,
    )
    place_constraint = KeypointConstraint(
        constraint_id="place_at_block",
        keypoint_id="ee",
        constraint_type=ConstraintType.POSITION,
        reference_offset=np.array([0.2, 0.1, 0.5]),
        start_time=1.1,
        end_time=1.6,
    )
    return TaskSpecification(
        spec_id="spec_pick_place",
        task_name="pick_place",
        task_description="Pick and place a block.",
        goal_object_id="block",
        goal_position=np.array([0.2, 0.1, 0.5]),
        segments=[
            SkillSegment(
                segment_id="pick",
                segment_type=SegmentType.SKILL,
                skill_name="pick",
                start_time=0.0,
                end_time=0.7,
                manipulated_object_id="block",
                keypoints=[keypoint],
                constraints=[pick_constraint],
            ),
            SkillSegment(
                segment_id="move",
                segment_type=SegmentType.FREE_SPACE,
                skill_name="move",
                start_time=0.7,
                end_time=1.1,
            ),
            SkillSegment(
                segment_id="place",
                segment_type=SegmentType.SKILL,
                skill_name="place",
                start_time=1.1,
                end_time=1.6,
                manipulated_object_id="block",
                keypoints=[keypoint],
                constraints=[place_constraint],
            ),
        ],
    )


def _build_pick_place_plan() -> MotionPlan:
    waypoints = [
        Waypoint(
            position=np.array([0.5, 0.0, 0.5]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            gripper_aperture=1.0,
            duration_to_next=0.4,
            phase=MotionPhase.APPROACH,
        ),
        Waypoint(
            position=np.array([0.5, 0.0, 0.5]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            gripper_aperture=0.0,
            duration_to_next=0.4,
            phase=MotionPhase.GRASP,
        ),
        Waypoint(
            position=np.array([0.3, 0.0, 0.6]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            gripper_aperture=0.0,
            duration_to_next=0.4,
            phase=MotionPhase.TRANSPORT,
        ),
        Waypoint(
            position=np.array([0.2, 0.1, 0.5]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            gripper_aperture=0.0,
            duration_to_next=0.0,
            phase=MotionPhase.PLACE,
        ),
    ]
    return MotionPlan(
        plan_id="plan_pick_place",
        task_name="pick_place",
        task_description="Pick and place a block.",
        waypoints=waypoints,
        target_object_id="block",
        target_object_position=np.array([0.5, 0.0, 0.5]),
        place_position=np.array([0.2, 0.1, 0.5]),
        robot_type="franka",
    )


def test_cpgen_pick_place_constraints_preserved():
    scene_objects = [
        {
            "id": "block",
            "position": [0.5, 0.0, 0.5],
            "dimensions": [0.05, 0.05, 0.05],
        }
    ]
    task_spec = _build_pick_place_spec()
    motion_plan = _build_pick_place_plan()
    augmenter = ConstraintPreservingAugmenter(
        robot_type="franka",
        use_collision_planner=False,
        use_physics_validation=False,
        verbose=False,
    )

    seed = augmenter.create_seed_episode(task_spec, motion_plan, scene_objects)
    transform = ObjectTransform(
        object_id="block",
        position_offset=np.array([0.1, 0.0, 0.0]),
    )
    augmented = augmenter.augment(
        seed=seed,
        object_transforms={"block": transform},
        obstacles=[
            {
                "id": "block",
                "position": [0.6, 0.0, 0.5],
                "dimensions": [0.05, 0.05, 0.05],
            }
        ],
        variation_index=0,
    )

    assert augmented.metrics.num_constraints_checked > 0
    assert augmented.metrics.constraint_satisfaction_ratio == 1.0
    assert augmented.metrics.violations == []
    assert augmented.planning_success is True
    assert np.allclose(augmented.motion_plan.waypoints[0].position, [0.6, 0.0, 0.5])
    assert np.allclose(augmented.motion_plan.waypoints[-1].position, [0.2, 0.1, 0.5])


def test_cpgen_allows_pluggable_constraint_module():
    scene_objects = [
        {
            "id": "block",
            "position": [0.5, 0.0, 0.5],
            "dimensions": [0.05, 0.05, 0.05],
        }
    ]
    task_spec = _build_pick_place_spec()
    motion_plan = _build_pick_place_plan()
    tracking_module = TrackingConstraintModule(solver=ConstraintSolver(verbose=False))
    augmenter = ConstraintPreservingAugmenter(
        robot_type="franka",
        use_collision_planner=False,
        use_physics_validation=False,
        constraint_module=tracking_module,
        verbose=False,
    )
    seed = augmenter.create_seed_episode(task_spec, motion_plan, scene_objects)
    transform = ObjectTransform(
        object_id="block",
        position_offset=np.array([0.1, 0.0, 0.0]),
    )
    augmenter.augment(
        seed=seed,
        object_transforms={"block": transform},
        obstacles=[
            {
                "id": "block",
                "position": [0.6, 0.0, 0.5],
                "dimensions": [0.05, 0.05, 0.05],
            }
        ],
        variation_index=0,
    )

    assert tracking_module.solve_called is True
    assert tracking_module.validate_called is True
