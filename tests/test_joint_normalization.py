from tools.geniesim_adapter import local_framework as lf


def test_normalize_joint_arrays_truncates() -> None:
    joint_positions = [0.0] * 7
    joint_velocities = list(range(34))
    joint_efforts = list(range(34))
    joint_accelerations = list(range(10))
    joint_names = [f"joint_{i}" for i in range(34)]

    jp, jv, je, ja, jn = lf._normalize_joint_arrays(
        joint_positions,
        joint_velocities,
        joint_efforts,
        joint_accelerations,
        joint_names,
    )

    assert len(jp) == 7
    assert len(jv) == 7
    assert len(je) == 7
    assert len(ja) == 7
    assert len(jn) == 7
