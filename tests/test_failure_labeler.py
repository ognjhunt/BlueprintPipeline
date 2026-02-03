from tools.failure_labeler import FailureLabeler, FailureType


def test_failure_labeler_collision_includes_contact_positions():
    labeler = FailureLabeler(collision_threshold=10.0)
    contact_events = [
        {
            "frame": 5,
            "timestamp": 0.5,
            "body_a": "gripper_link",
            "body_b": "table",
            "force_magnitude": 20.0,
            "position": {"x": 1.0, "y": 2.0, "z": 3.0},
        }
    ]

    info = labeler.label_episode(
        task_success=False,
        num_frames=10,
        contact_events=contact_events,
    )

    assert info.failure_type == FailureType.COLLISION
    assert info.contact_positions == [[1.0, 2.0, 3.0]]


def test_failure_labeler_slip_includes_contact_positions():
    labeler = FailureLabeler()
    contact_events = [
        {
            "frame": 1,
            "body_a": "gripper",
            "body_b": "object",
            "force_magnitude": 1.0,
            "position": [0.0, 0.0, 0.0],
        },
        {
            "frame": 2,
            "body_a": "gripper",
            "body_b": "object",
            "force_magnitude": 0.0,
            "position": [0.1, 0.2, 0.3],
        },
    ]

    info = labeler.label_episode(
        task_success=False,
        num_frames=5,
        contact_events=contact_events,
    )

    assert info.failure_type == FailureType.SLIP
    assert info.contact_positions == [[0.1, 0.2, 0.3]]
