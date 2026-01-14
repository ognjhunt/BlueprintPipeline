#!/usr/bin/env python3
"""
Minimal end-to-end integration tests for job entrypoints.

These tests use mock inputs and assert output schema integrity for:
* episode-generation-job
* genie-sim-export-job
"""

from __future__ import annotations

import json
from pathlib import Path
import importlib.util
import sys

import pytest


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


REPO_ROOT = Path(__file__).resolve().parents[1]
EPISODE_JOB_DIR = REPO_ROOT / "episode-generation-job"
GENIESIM_EXPORT_DIR = REPO_ROOT / "genie-sim-export-job"

sys.path.insert(0, str(EPISODE_JOB_DIR))
sys.path.insert(0, str(GENIESIM_EXPORT_DIR))

episode_module = _load_module(
    "episode_generation_job.generate_episodes",
    EPISODE_JOB_DIR / "generate_episodes.py",
)
geniesim_export_module = _load_module(
    "genie_sim_export_job.export_to_geniesim",
    GENIESIM_EXPORT_DIR / "export_to_geniesim.py",
)


def _write_scene_manifest(path: Path, scene_id: str) -> None:
    manifest = {
        "version": "1.0.0",
        "scene_id": scene_id,
        "scene": {
            "environment_type": "kitchen",
            "coordinate_frame": "y_up",
            "meters_per_unit": 1.0,
            "room": {"bounds": {"width": 2.0, "depth": 2.0, "height": 2.0}},
        },
        "objects": [
            {
                "id": "mug_0",
                "name": "mug_0",
                "category": "mug",
                "description": "coffee mug",
                "sim_role": "manipulable_object",
                "dimensions_est": {"width": 0.08, "depth": 0.08, "height": 0.1},
                "transform": {
                    "position": {"x": 0.1, "y": 0.2, "z": 0.8},
                    "rotation_quaternion": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
                    "scale": {"x": 1.0, "y": 1.0, "z": 1.0},
                },
                "asset": {"path": "mug_0.usd"},
                "physics": {"mass": 0.2},
                "physics_hints": {"material_type": "ceramic"},
                "semantics": {"affordances": ["Graspable", "Containable"]},
                "relationships": [],
            }
        ],
    }
    path.write_text(json.dumps(manifest, indent=2))


def _write_episode_outputs(output_dir: Path, scene_id: str) -> None:
    lerobot_dir = output_dir / "lerobot"
    meta_dir = lerobot_dir / "meta"
    data_dir = lerobot_dir / "data" / "chunk-000"
    meta_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    info = {
        "dataset_name": scene_id,
        "robot_type": "franka",
        "episodes": 1,
    }
    stats = {"num_episodes": 1, "num_frames": 10}
    meta_dir.joinpath("info.json").write_text(json.dumps(info, indent=2))
    meta_dir.joinpath("stats.json").write_text(json.dumps(stats, indent=2))
    meta_dir.joinpath("tasks.jsonl").write_text(json.dumps({"task": "pick_mug"}) + "\n")
    meta_dir.joinpath("episodes.jsonl").write_text(
        json.dumps({"episode_id": "episode_000000", "task": "pick_mug"}) + "\n"
    )
    data_dir.joinpath("episode_000000.parquet").write_bytes(b"mock")

    manifests_dir = output_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    generation_manifest = {
        "scene_id": scene_id,
        "episodes": [
            {
                "id": "episode_000000",
                "task_name": "pick_mug",
                "quality_score": 1.0,
            }
        ],
        "generated_at": "2025-01-01T00:00:00Z",
        "generator_version": "test",
    }
    manifests_dir.joinpath("generation_manifest.json").write_text(
        json.dumps(generation_manifest, indent=2)
    )
    manifests_dir.joinpath("task_coverage.json").write_text(
        json.dumps(
            {
                "tasks": {"pick_mug": 1},
                "total_unique_tasks": 1,
                "average_episodes_per_task": 1.0,
            },
            indent=2,
        )
    )

    quality_dir = output_dir / "quality"
    quality_dir.mkdir(parents=True, exist_ok=True)
    quality_dir.joinpath("validation_report.json").write_text(
        json.dumps({"scene_id": scene_id, "summary": {"status": "mock"}}, indent=2)
    )


def test_episode_generation_job_minimal_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scene_id = "mock_scene"
    assets_dir = tmp_path / "scenes" / scene_id / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    _write_scene_manifest(assets_dir / "scene_manifest.json", scene_id)

    def _mock_init(self, config, verbose: bool = True) -> None:
        self.config = config
        self.verbose = verbose

    def _mock_generate(self, manifest):
        _write_episode_outputs(self.config.output_dir, scene_id)
        output = episode_module.EpisodeGenerationOutput(
            scene_id=scene_id,
            robot_type=self.config.robot_type,
        )
        output.total_episodes = 1
        output.valid_episodes = 1
        output.tasks_generated = {"pick_mug": 1}
        output.output_dir = self.config.output_dir
        output.lerobot_dataset_path = self.config.output_dir / "lerobot"
        output.manifest_path = self.config.output_dir / "manifests" / "generation_manifest.json"
        return output

    monkeypatch.setattr(episode_module.EpisodeGenerator, "__init__", _mock_init)
    monkeypatch.setattr(episode_module.EpisodeGenerator, "generate", _mock_generate)

    exit_code = episode_module.run_episode_generation_job(
        root=tmp_path,
        bucket="",
        scene_id=scene_id,
        assets_prefix=f"scenes/{scene_id}/assets",
        episodes_prefix=f"scenes/{scene_id}/episodes",
        robot_type="franka",
        episodes_per_variation=1,
        max_variations=1,
        fps=10.0,
        use_llm=False,
        use_cpgen=False,
        capture_sensor_data=False,
    )

    assert exit_code == 0

    output_dir = tmp_path / "scenes" / scene_id / "episodes"
    info = json.loads((output_dir / "lerobot" / "meta" / "info.json").read_text())
    assert {"dataset_name", "robot_type", "episodes"}.issubset(info)

    generation_manifest = json.loads(
        (output_dir / "manifests" / "generation_manifest.json").read_text()
    )
    assert {"scene_id", "episodes", "generated_at", "generator_version"}.issubset(
        generation_manifest
    )
    assert (output_dir / "lerobot" / "data" / "chunk-000" / "episode_000000.parquet").is_file()
    assert (output_dir / "quality" / "validation_report.json").is_file()


def test_geniesim_export_job_minimal_end_to_end(tmp_path: Path) -> None:
    scene_id = "mock_scene"
    assets_dir = tmp_path / "scenes" / scene_id / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    _write_scene_manifest(assets_dir / "scene_manifest.json", scene_id)
    (assets_dir / ".usd_assembly_complete").write_text("ok")

    exit_code = geniesim_export_module.run_geniesim_export_job(
        root=tmp_path,
        scene_id=scene_id,
        assets_prefix=f"scenes/{scene_id}/assets",
        geniesim_prefix=f"scenes/{scene_id}/geniesim",
        robot_type="franka",
        max_tasks=5,
        generate_embeddings=False,
        filter_commercial=True,
        copy_usd=False,
        enable_multi_robot=False,
        enable_bimanual=False,
        enable_vla_packages=False,
        enable_rich_annotations=False,
        enable_premium_analytics=False,
    )

    assert exit_code == 0

    output_dir = tmp_path / "scenes" / scene_id / "geniesim"
    scene_graph = json.loads((output_dir / "scene_graph.json").read_text())
    assert {"scene_id", "coordinate_system", "meters_per_unit", "nodes", "edges", "metadata"}.issubset(
        scene_graph
    )

    asset_index = json.loads((output_dir / "asset_index.json").read_text())
    assert {"assets", "metadata"}.issubset(asset_index)
    assert asset_index["assets"], "Expected at least one asset"
    assert {"asset_id", "usd_path", "semantic_description", "categories"}.issubset(
        asset_index["assets"][0]
    )

    task_config = json.loads((output_dir / "task_config.json").read_text())
    assert {"scene_id", "environment_type", "suggested_tasks", "robot_config"}.issubset(task_config)
    assert (output_dir / "scene_config.yaml").is_file()
    assert (output_dir / "export_manifest.json").is_file()
    assert (output_dir / "enhanced_features.json").is_file()
    assert (output_dir / "_GENIESIM_EXPORT_COMPLETE").is_file()


def test_geniesim_export_job_blocks_on_error_gate(tmp_path: Path) -> None:
    scene_id = "blocked_scene"
    assets_dir = tmp_path / "scenes" / scene_id / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": "1.0.0",
        "scene_id": scene_id,
        "objects": [],
    }
    (assets_dir / "scene_manifest.json").write_text(json.dumps(manifest, indent=2))
    (assets_dir / ".usd_assembly_complete").write_text("ok")

    exit_code = geniesim_export_module.run_geniesim_export_job(
        root=tmp_path,
        scene_id=scene_id,
        assets_prefix=f"scenes/{scene_id}/assets",
        geniesim_prefix=f"scenes/{scene_id}/geniesim",
        robot_type="franka",
        max_tasks=1,
        generate_embeddings=False,
        filter_commercial=True,
        copy_usd=False,
        enable_multi_robot=False,
        enable_bimanual=False,
        enable_vla_packages=False,
        enable_rich_annotations=False,
        enable_premium_analytics=False,
        require_quality_gates=True,
    )

    assert exit_code == 1
