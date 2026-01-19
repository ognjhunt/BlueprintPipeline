import ast
from pathlib import Path


EXPECTED_COMMAND_IDS = {
    "GET_CAMERA_DATA": 1,
    "GET_SEMANTIC_DATA": 10,
    "LINEAR_MOVE": 2,
    "SET_JOINT_POSITION": 3,
    "GET_JOINT_POSITION": 8,
    "GET_EE_POSE": 18,
    "GET_IK_STATUS": 19,
    "SET_TRAJECTORY_LIST": 25,
    "GET_GRIPPER_STATE": 4,
    "SET_GRIPPER_STATE": 9,
    "GET_OBJECT_POSE": 5,
    "ADD_OBJECT": 6,
    "GET_ROBOT_LINK_POSE": 7,
    "GET_OBJECT_JOINT": 26,
    "GET_PART_DOF_JOINT": 32,
    "SET_OBJECT_POSE": 24,
    "SET_TARGET_POINT": 27,
    "SET_LINEAR_VELOCITY": 33,
    "ATTACH_OBJ": 13,
    "DETACH_OBJ": 14,
    "ATTACH_OBJ_TO_PARENT": 50,
    "DETACH_OBJ_FROM_PARENT": 51,
    "REMOVE_OBJS_FROM_OBSTACLE": 52,
    "GET_OBSERVATION": 11,
    "START_RECORDING": 15,
    "STOP_RECORDING": 20,
    "RESET": 12,
    "EXIT": 17,
    "INIT_ROBOT": 21,
    "TASK_STATUS": 16,
    "ADD_CAMERA": 22,
    "SET_FRAME_STATE": 28,
    "SET_LIGHT": 30,
    "SET_CODE_FACE_ORIENTATION": 34,
    "SET_TASK_METRIC": 53,
    "STORE_CURRENT_STATE": 54,
    "PLAYBACK": 55,
    "GET_CHECKER_STATUS": 56,
}


def _load_command_enum_values() -> dict[str, int]:
    repo_root = Path(__file__).resolve().parents[1]
    source_path = repo_root / "tools" / "geniesim_adapter" / "local_framework.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    command_values: dict[str, int] = {}

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "CommandType":
            for statement in node.body:
                if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                    target = statement.targets[0]
                    if isinstance(target, ast.Name) and isinstance(statement.value, ast.Constant):
                        if isinstance(statement.value.value, int):
                            command_values[target.id] = statement.value.value
            break

    return command_values


def test_command_type_values_are_unique() -> None:
    values = list(_load_command_enum_values().values())
    assert len(values) == len(set(values))


def test_command_type_values_match_expected_ids() -> None:
    actual = _load_command_enum_values()
    assert actual == EXPECTED_COMMAND_IDS
