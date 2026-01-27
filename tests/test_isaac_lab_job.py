from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest


@dataclass
class _FakeRuntimeResult:
    is_valid: bool = True
    errors: list[str] = None
    warnings: list[str] = None
    rollout_fps: float = 0.0
    observation_shapes: dict | None = None
    action_shape: tuple | None = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []

    def to_dict(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
        }


def _valid_env_cfg() -> str:
    return "\n".join(
        [
            "from omni.isaac.lab import foo",
            "class KitchenEnvCfg(ManagerBasedEnvCfg):",
            "    pass",
        ]
    )


def _valid_task_code() -> str:
    return "\n".join(
        [
            "class MyTaskEnv:",
            "    pass",
        ]
    )


def test_validate_code_helpers(load_job_module):
    module = load_job_module("isaac_lab", "generate_isaac_lab_task.py")

    valid_env = module.validate_isaac_lab_env_config(_valid_env_cfg())
    assert valid_env.is_valid

    invalid_env = module.validate_isaac_lab_env_config("def broken(")
    assert not invalid_env.is_valid

    valid_task = module.validate_isaac_lab_task(_valid_task_code())
    assert valid_task.is_valid

    invalid_task = module.validate_isaac_lab_task("def broken(")
    assert not invalid_task.is_valid

    valid_syntax = module.validate_python_syntax("x = 1")
    assert valid_syntax.is_valid

    invalid_syntax = module.validate_python_syntax("def broken(")
    assert not invalid_syntax.is_valid


def test_run_isaac_lab_job_main_flow(load_job_module, tmp_path: Path, monkeypatch):
    module = load_job_module("isaac_lab", "generate_isaac_lab_task.py")

    assets_prefix = "scenes/test_scene/assets"
    usd_prefix = "scenes/test_scene/usd"
    replicator_prefix = "scenes/test_scene/replicator"
    isaac_lab_prefix = "scenes/test_scene/isaac_lab"

    assets_dir = tmp_path / assets_prefix
    assets_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / usd_prefix).mkdir(parents=True, exist_ok=True)
    (tmp_path / replicator_prefix).mkdir(parents=True, exist_ok=True)

    manifest = {"objects": []}
    (assets_dir / "scene_manifest.json").write_text(json.dumps(manifest))

    task_files = {
        "env_cfg.py": _valid_env_cfg(),
        "task_policy.py": _valid_task_code(),
        "train_cfg.yaml": "runner:\n  num_envs: 4\nexperiment: {}\n",
    }

    class _FakeGenerator:
        def __init__(self, policy_config):
            self.policy_config = policy_config

        def generate(self, recipe, policy_id, robot_type, num_envs):
            return SimpleNamespace(task_name="test_task", files=task_files)

        def save(self, task, output_dir: str):
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            for filename, content in task.files.items():
                (output_path / filename).write_text(content)
            return {name: str(output_path / name) for name in task.files.keys()}

    class _FakeRuntimeValidator:
        def __init__(self, verbose: bool = False):
            self.verbose = verbose

        def validate(self, **kwargs):
            return _FakeRuntimeResult(
                is_valid=True,
                rollout_fps=12.0,
                observation_shapes={"obs": (4,)},
                action_shape=(2,),
            )

    monkeypatch.setattr(module, "IsaacLabTaskGenerator", _FakeGenerator)
    monkeypatch.setattr(module, "IsaacLabRuntimeValidator", _FakeRuntimeValidator)

    exit_code = module.run_isaac_lab_job(
        root=tmp_path,
        scene_id="test_scene",
        assets_prefix=assets_prefix,
        usd_prefix=usd_prefix,
        replicator_prefix=replicator_prefix,
        isaac_lab_prefix=isaac_lab_prefix,
        environment_type="generic",
        policy_id="manipulation",
        robot_type="franka",
        num_envs=4,
        run_runtime_validation=True,
    )

    isaac_lab_dir = tmp_path / isaac_lab_prefix
    assert exit_code == 0
    assert (isaac_lab_dir / ".isaac_lab_complete").is_file()
    assert (isaac_lab_dir / "generation_metadata.json").is_file()
