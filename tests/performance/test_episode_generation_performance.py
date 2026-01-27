import time
from dataclasses import replace

import pytest


_DEF_TASK_COUNT = 3


def _assert_time_scales(
    small_elapsed: float,
    large_elapsed: float,
    small_size: int,
    large_size: int,
    *,
    tolerance: float = 6.0,
) -> None:
    per_small = max(small_elapsed / small_size, 1e-6)
    per_large = large_elapsed / large_size
    assert per_large <= per_small * tolerance


def _build_tasks(generate_episodes, task_specifier):
    tasks_with_specs = []
    for idx in range(_DEF_TASK_COUNT):
        task_name = f"task_{idx}"
        task = {
            "task_id": f"task-{idx}",
            "task_name": task_name,
            "description": f"Task {idx}",
            "task_steps": [],
        }
        spec = task_specifier.TaskSpecification(
            spec_id=f"spec-{idx}",
            task_name=task_name,
            task_description=task["description"],
        )
        tasks_with_specs.append((task, spec))
    return tasks_with_specs


def _build_episodes(generate_episodes, count: int):
    episodes = []
    for idx in range(count):
        task_name = f"task_{idx % _DEF_TASK_COUNT}"
        episodes.append(
            generate_episodes.GeneratedEpisode(
                episode_id=f"episode-{idx}",
                task_name=task_name,
                task_description=f"{task_name} description",
                trajectory=None,
                motion_plan=None,
                scene_id="scene-1",
                variation_index=idx,
                is_seed=idx == 0,
                is_valid=True,
            )
        )
    return episodes


@pytest.mark.slow
def test_generation_manifest_write_scales_with_episode_count(tmp_path, load_job_module):
    generate_episodes = load_job_module("episode_generation", "generate_episodes.py")
    task_specifier = load_job_module("episode_generation", "task_specifier.py")

    config = generate_episodes.EpisodeGenerationConfig(
        scene_id="scene-1",
        manifest_path=tmp_path / "scene_manifest.json",
        output_dir=tmp_path / "output",
    )
    generator = generate_episodes.EpisodeGenerator.__new__(generate_episodes.EpisodeGenerator)
    generator.config = config

    tasks_with_specs = _build_tasks(generate_episodes, task_specifier)

    def run_write(count: int, output_dir):
        episodes = _build_episodes(generate_episodes, count)
        tasks_generated = {task["task_name"]: 0 for task, _ in tasks_with_specs}
        for episode in episodes:
            tasks_generated[episode.task_name] += 1

        output = generate_episodes.EpisodeGenerationOutput(
            scene_id=config.scene_id,
            robot_type=config.robot_type,
            tasks_generated=tasks_generated,
        )
        generator.config = replace(config, output_dir=output_dir)

        start = time.perf_counter()
        generator._write_manifest(episodes, tasks_with_specs, output)
        return time.perf_counter() - start

    small_count = 25
    large_count = 100
    small_elapsed = run_write(small_count, tmp_path / "small")
    large_elapsed = run_write(large_count, tmp_path / "large")

    _assert_time_scales(
        small_elapsed,
        large_elapsed,
        small_count,
        large_count,
    )
