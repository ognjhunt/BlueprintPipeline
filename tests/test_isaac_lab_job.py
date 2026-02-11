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
            "from isaaclab import foo",
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
    replicator_dir = tmp_path / replicator_prefix
    assets_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / usd_prefix).mkdir(parents=True, exist_ok=True)
    replicator_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"objects": []}
    (assets_dir / "scene_manifest.json").write_text(json.dumps(manifest))
    (replicator_dir / "affordance_graph.json").write_text(
        json.dumps(
            {
                "scene_id": "test_scene",
                "environment_type": "generic",
                "regions": [
                    {
                        "id": "default_region",
                        "surface_type": "horizontal",
                        "affordances": ["support_surface", "placeable", "containable"],
                    }
                ],
                "policy_region_map": {"manipulation": ["default_region"]},
                "policy_asset_map": {},
                "asset_to_region_candidates": {},
            }
        )
    )

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
    assert (isaac_lab_dir / "task_catalog.json").is_file()
    assert (isaac_lab_dir / "task_spawn_plan_baseline.json").is_file()
    assert (isaac_lab_dir / "task_spawn_plan.json").is_file()

    metadata = json.loads((isaac_lab_dir / "generation_metadata.json").read_text())
    assert "task_catalog_summary" in metadata
    assert set(metadata["task_catalog_summary"].keys()) == {"humanoid", "manipulator"}
    assert metadata["task_spawn_plan_summary"]["feasible_tasks"] >= 1


def test_run_isaac_lab_refresh_mode_uses_variation_assets(
    load_job_module,
    tmp_path: Path,
    monkeypatch,
):
    module = load_job_module("isaac_lab", "generate_isaac_lab_task.py")

    assets_prefix = "scenes/test_scene/assets"
    usd_prefix = "scenes/test_scene/usd"
    replicator_prefix = "scenes/test_scene/replicator"
    variation_prefix = "scenes/test_scene/variation_assets"
    isaac_lab_prefix = "scenes/test_scene/isaac_lab"

    assets_dir = tmp_path / assets_prefix
    replicator_dir = tmp_path / replicator_prefix
    variation_dir = tmp_path / variation_prefix
    assets_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / usd_prefix).mkdir(parents=True, exist_ok=True)
    replicator_dir.mkdir(parents=True, exist_ok=True)
    variation_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "objects": [
            {"id": "plate_01", "sim_role": "manipulable_object"},
        ]
    }
    (assets_dir / "scene_manifest.json").write_text(json.dumps(manifest))
    (replicator_dir / "affordance_graph.json").write_text(
        json.dumps(
            {
                "scene_id": "test_scene",
                "environment_type": "kitchen",
                "regions": [
                    {
                        "id": "counter_region",
                        "surface_type": "horizontal",
                        "affordances": ["support_surface", "placeable", "containable"],
                    }
                ],
                "policy_region_map": {"dish_loading": ["counter_region"]},
                "policy_asset_map": {"dish_loading": ["dirty_plate", "dirty_mug"]},
                "asset_to_region_candidates": {
                    "dirty_plate": ["counter_region"],
                    "dirty_mug": ["counter_region"],
                },
                "articulation_targets": [],
            }
        )
    )

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

    monkeypatch.setattr(module, "IsaacLabTaskGenerator", _FakeGenerator)

    baseline_exit = module.run_isaac_lab_job(
        root=tmp_path,
        scene_id="test_scene",
        assets_prefix=assets_prefix,
        usd_prefix=usd_prefix,
        replicator_prefix=replicator_prefix,
        isaac_lab_prefix=isaac_lab_prefix,
        environment_type="kitchen",
        policy_id="dish_loading",
        robot_type="franka",
        num_envs=4,
        run_runtime_validation=False,
    )
    assert baseline_exit == 0

    (variation_dir / "variation_assets.json").write_text(
        json.dumps(
            {
                "scene_id": "test_scene",
                "objects": [
                    {
                        "id": "dirty_plate",
                        "generated_3d": {"status": "success", "usdz_path": "dirty_plate.usdz"},
                    },
                    {
                        "id": "dirty_mug",
                        "generated_3d": {"status": "failed"},
                    },
                ],
            }
        )
    )
    (variation_dir / ".simready_complete").write_text(
        json.dumps({"simready_assets": {"dirty_plate": "variation_assets/obj_dirty_plate/simready.usda"}})
    )

    refresh_exit = module.run_isaac_lab_job(
        root=tmp_path,
        scene_id="test_scene",
        assets_prefix=assets_prefix,
        usd_prefix=usd_prefix,
        replicator_prefix=replicator_prefix,
        isaac_lab_prefix=isaac_lab_prefix,
        environment_type="kitchen",
        policy_id="dish_loading",
        robot_type="franka",
        num_envs=4,
        run_runtime_validation=False,
        isaac_refresh_only=True,
        variation_assets_prefix=variation_prefix,
    )
    assert refresh_exit == 0

    isaac_lab_dir = tmp_path / isaac_lab_prefix
    refresh_plan = json.loads((isaac_lab_dir / "task_spawn_plan_refresh.json").read_text())
    assert (isaac_lab_dir / ".isaac_lab_refresh_complete").is_file()
    assert refresh_plan["mode"] == "refresh_only"
    assert refresh_plan["variation_asset_filter"]["available_asset_ids"] == ["dirty_plate"]


def test_build_scene_task_catalog_has_robot_tracks(load_job_module):
    module = load_job_module("isaac_lab", "generate_isaac_lab_task.py")

    manifest = {
        "objects": [
            {"id": "obj_washer", "category": "washer"},
            {"id": "obj_dryer", "category": "dryer"},
            {"id": "obj_hamper", "category": "laundry_hamper"},
        ]
    }
    policy_config = {
        "environments": {
            "laundry": {
                "default_policies": [
                    "laundry_sorting",
                    "door_manipulation",
                ]
            }
        },
        "policies": {
            "laundry_sorting": {
                "display_name": "Laundry Sorting",
                "description": "Sort garments by type and destination.",
                "reward_components": ["sorting_accuracy", "collision_penalty"],
            },
            "door_manipulation": {
                "display_name": "Door Manipulation",
                "description": "Open and close articulated doors.",
                "reward_components": ["joint_progress", "handle_grasp"],
            },
        },
    }

    catalog = module.build_scene_task_catalog(
        scene_id="scene_1",
        environment_type="laundry",
        manifest=manifest,
        policy_config=policy_config,
    )

    assert catalog["scene_id"] == "scene_1"
    tracks = catalog["tracks"]
    assert set(tracks.keys()) == {"humanoid", "manipulator"}
    assert tracks["humanoid"]["tasks"]
    assert tracks["manipulator"]["tasks"]


def test_build_region_constrained_task_plan_infeasible(load_job_module):
    module = load_job_module("isaac_lab", "generate_isaac_lab_task.py")

    task_catalog = {
        "scene_id": "scene_2",
        "environment_type": "kitchen",
        "tracks": {
            "manipulator": {
                "robot_types": ["franka"],
                "tasks": [
                    {
                        "task_id": "manipulator_dish_loading_v1",
                        "policy_id": "dish_loading",
                        "robot_track": "manipulator",
                        "scene_anchors": ["counter", "sink"],
                    }
                ],
            }
        },
    }
    affordance_graph = {
        "regions": [
            {
                "id": "counter_region",
                "surface_type": "horizontal",
                "affordances": ["support_surface", "placeable"],
            }
        ],
        "policy_region_map": {"dish_loading": ["counter_region"]},
        "policy_asset_map": {"dish_loading": []},
        "asset_to_region_candidates": {},
    }

    task_plan = module.build_region_constrained_task_plan(
        task_catalog=task_catalog,
        affordance_graph=affordance_graph,
    )
    feasibility = module.evaluate_task_plan_feasibility(task_plan)
    assert not feasibility["is_feasible"]
    assert "no_feasible_tasks_for_track:manipulator" in feasibility["reasons"]


def test_run_isaac_lab_job_fails_without_affordance_graph_in_strict_mode(
    load_job_module,
    tmp_path: Path,
    monkeypatch,
):
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

    monkeypatch.setattr(module, "IsaacLabTaskGenerator", _FakeGenerator)
    monkeypatch.setenv("AFFORDANCE_FEASIBILITY_MODE", "strict")

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
        run_runtime_validation=False,
    )

    assert exit_code == 1
