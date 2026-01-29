import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DWM_ROOT = REPO_ROOT / "dwm-preparation-job"


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_module(
    "dwm_physics_policy_runner",
    DWM_ROOT / "trajectory_generator" / "physics_policy_runner.py",
)
models = _load_module(
    "dwm_models",
    DWM_ROOT / "models.py",
)


@pytest.mark.unit
def test_robot_action_index_maps_frame_indices():
    action = models.RobotAction(
        frame_idx=2,
        wrist_pose=np.eye(4),
        joint_positions=[0.0, 0.1],
    )
    action_index = runner._RobotActionIndex.from_actions([action])

    assert action_index.get(2) == action
    assert action_index.get(99) is None


@pytest.mark.unit
def test_require_isaac_raises_when_unavailable(monkeypatch):
    monkeypatch.setattr(runner, "_isaac_available", lambda: False)

    with pytest.raises(RuntimeError, match="Isaac Sim/Lab modules are not available"):
        runner._require_isaac()
