import json

from tools.config import ConfigLoader, PIPELINE_CONFIG_PATH


def test_pipeline_config_rejects_unknown_geniesim_robot():
    with open(PIPELINE_CONFIG_PATH, "r") as handle:
        config = json.load(handle)

    robot_config = config.setdefault("robot_config", {})
    supported = list(robot_config.get("supported_robots", []))
    supported.append("not_a_robot")
    robot_config["supported_robots"] = supported

    errors = ConfigLoader.validate_config(config, "pipeline")
    assert errors is not None
    assert "robot_config.supported_robots" in errors
