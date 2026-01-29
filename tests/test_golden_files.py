#!/usr/bin/env python3
"""Golden fixture validation for minimal pipeline outputs."""

from __future__ import annotations

import json
import shutil
import sys
from importlib import util
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "golden" / "mock_scene"
USD_PLACEHOLDER = "<USD_SOURCE_DIR>"
UNSTABLE_KEYS = {"timestamp", "generated_at", "generated_at_utc"}


def _load_module(module_name: str, file_path: Path):
    spec = util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {module_name} from {file_path}")
    module = util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


sys.path.insert(0, str(REPO_ROOT / "episode-generation-job"))
sys.path.insert(0, str(REPO_ROOT / "genie-sim-export-job"))

episode_module = _load_module(
    "generate_episodes",
    REPO_ROOT / "episode-generation-job" / "generate_episodes.py",
)
geniesim_export_module = _load_module(
    "genie_sim_export_job.export_to_geniesim",
    REPO_ROOT / "genie-sim-export-job" / "export_to_geniesim.py",
)


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


def _normalize_json(data: Any, usd_root: Path | None = None) -> Any:
    if isinstance(data, dict):
        normalized = {}
        for key, value in data.items():
            if key in UNSTABLE_KEYS:
                continue
            normalized[key] = _normalize_json(value, usd_root)
        return normalized
    if isinstance(data, list):
        normalized_list = [_normalize_json(item, usd_root) for item in data]
        if not normalized_list:
            return normalized_list
        if all(isinstance(item, dict) for item in normalized_list):
            if all("asset_id" in item for item in normalized_list):
                return sorted(normalized_list, key=lambda item: item["asset_id"])
            if all("id" in item for item in normalized_list):
                return sorted(normalized_list, key=lambda item: item["id"])
            if all("task_type" in item for item in normalized_list):
                return sorted(normalized_list, key=lambda item: item["task_type"])
            if all(
                all(key in item for key in ("source", "target", "relation"))
                for item in normalized_list
            ):
                return sorted(
                    normalized_list,
                    key=lambda item: (item["source"], item["target"], item["relation"]),
                )
        if all(isinstance(item, (str, int, float, bool)) for item in normalized_list):
            return sorted(normalized_list)
        return normalized_list
    if isinstance(data, str) and usd_root is not None:
        usd_root_str = str(usd_root)
        if usd_root_str in data:
            return data.replace(usd_root_str, USD_PLACEHOLDER)
    return data


def _assert_json_matches(actual_path: Path, expected_path: Path, usd_root: Path | None = None) -> None:
    actual = json.loads(actual_path.read_text())
    expected = json.loads(expected_path.read_text())
    assert _normalize_json(actual, usd_root) == _normalize_json(expected, usd_root)


@pytest.mark.integration
def test_golden_files_minimal_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scene_id = "mock_scene"
    inputs_dir = FIXTURE_ROOT / "inputs"
    expected_dir = FIXTURE_ROOT / "expected"

    assets_dir = tmp_path / "scenes" / scene_id / "assets"
    usd_dir = tmp_path / "scenes" / scene_id / "usd"
    variation_dir = tmp_path / "scenes" / scene_id / "variation_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    usd_dir.mkdir(parents=True, exist_ok=True)
    variation_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(inputs_dir / "scene_manifest.json", assets_dir / "scene_manifest.json")
    shutil.copy2(inputs_dir / "usd" / "scene.usda", usd_dir / "scene.usda")
    shutil.copy2(inputs_dir / "variation_assets.json", variation_dir / "variation_assets.json")
    (assets_dir / ".usd_assembly_complete").write_text("ok")

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
    monkeypatch.setenv("BYPASS_QUALITY_GATES", "1")

    episode_exit = episode_module.run_episode_generation_job(
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
    assert episode_exit == 0

    geniesim_exit = geniesim_export_module.run_geniesim_export_job(
        root=tmp_path,
        scene_id=scene_id,
        assets_prefix=f"scenes/{scene_id}/assets",
        geniesim_prefix=f"scenes/{scene_id}/geniesim",
        robot_type="franka",
        max_tasks=5,
        generate_embeddings=False,
        filter_commercial=True,
        copy_usd=True,
        enable_multi_robot=False,
        enable_bimanual=False,
        enable_vla_packages=False,
        enable_rich_annotations=False,
        enable_premium_analytics=False,
        variation_assets_prefix=f"scenes/{scene_id}/variation_assets",
    )
    assert geniesim_exit == 0

    episode_output_dir = tmp_path / "scenes" / scene_id / "episodes"
    expected_episode_dir = expected_dir / "episode_generation"
    for rel_path in [
        Path("lerobot/meta/info.json"),
        Path("lerobot/meta/stats.json"),
        Path("manifests/generation_manifest.json"),
        Path("manifests/task_coverage.json"),
        Path("quality/validation_report.json"),
    ]:
        _assert_json_matches(
            episode_output_dir / rel_path,
            expected_episode_dir / rel_path,
        )

    geniesim_output_dir = tmp_path / "scenes" / scene_id / "geniesim"
    expected_geniesim_dir = expected_dir / "geniesim"
    for rel_path in [
        Path("scene_graph.json"),
        Path("asset_index.json"),
        Path("task_config.json"),
        Path("enhanced_features.json"),
    ]:
        _assert_json_matches(
            geniesim_output_dir / rel_path,
            expected_geniesim_dir / rel_path,
            usd_root=usd_dir,
        )

    assert (
        geniesim_output_dir / "usd" / "scene.usda"
    ).read_text() == (expected_geniesim_dir / "usd" / "scene.usda").read_text()
