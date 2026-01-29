from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from fixtures.generate_mock_geniesim_local import generate_mock_geniesim_local


def _write_stub_lerobot_metadata(
    module,
    output_dir: Path,
    episode_metadata_list: List[Any],
    job_id: str,
    scene_id: str,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_info = module._build_dataset_info(
        job_id=job_id,
        scene_id=scene_id,
        source="genie_sim",
        converted_at="2025-01-01T00:00:00Z",
    )
    total_frames = 0
    quality_scores = []
    for idx, episode in enumerate(episode_metadata_list):
        entry = {
            "episode_id": episode.episode_id,
            "episode_index": idx,
            "num_frames": episode.frame_count,
            "duration_seconds": episode.duration_seconds,
            "quality_score": episode.quality_score,
            "quality_components": episode.quality_components,
            "validation_passed": episode.validation_passed,
            "file": f"episode_{idx:06d}.parquet",
        }
        dataset_info["episodes"].append(entry)
        total_frames += episode.frame_count
        quality_scores.append(episode.quality_score)

    dataset_info["total_frames"] = total_frames
    dataset_info["total_episodes"] = len(dataset_info["episodes"])
    dataset_info["skipped_episodes"] = 0
    dataset_info["skip_rate_percent"] = 0.0
    dataset_info["average_quality_score"] = (
        sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
    )

    dataset_info_path = output_dir / "dataset_info.json"
    dataset_info_path.write_text(json.dumps(dataset_info, indent=2))

    episodes_path = output_dir / "episodes.jsonl"
    episodes_payload = "\n".join(json.dumps(entry) for entry in dataset_info["episodes"])
    if episodes_payload:
        episodes_payload += "\n"
    episodes_path.write_text(episodes_payload)

    return {
        "success": True,
        "converted_count": len(dataset_info["episodes"]),
        "skipped_count": 0,
        "skip_rate_percent": 0.0,
        "conversion_failures": [],
        "output_dir": output_dir,
        "metadata_file": dataset_info_path,
    }


def test_missing_lerobot_dir_triggers_conversion(
    tmp_path: Path,
    load_job_module,
    monkeypatch,
) -> None:
    module = load_job_module("geniesim_import", "import_from_geniesim.py")
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    recordings_root = generate_mock_geniesim_local(
        output_dir=tmp_path,
        run_id="conversion_missing_dir",
        episodes=1,
        seed=11,
    )
    recordings_dir = recordings_root / "recordings"

    monkeypatch.setattr(module, "_resolve_recordings_dir", lambda *args, **kwargs: recordings_dir)
    called: Dict[str, Any] = {}

    def _fake_convert_to_lerobot(
        *,
        episodes_dir: Path,
        output_dir: Path,
        episode_metadata_list: List[Any],
        min_quality_score: float,
        quality_component_thresholds: Dict[str, float],
        job_id: str,
        scene_id: str,
    ) -> Dict[str, Any]:
        called["episodes_dir"] = episodes_dir
        called["output_dir"] = output_dir
        called["episode_ids"] = [ep.episode_id for ep in episode_metadata_list]
        called["min_quality_score"] = min_quality_score
        called["quality_component_thresholds"] = quality_component_thresholds
        called["job_id"] = job_id
        called["scene_id"] = scene_id
        return _write_stub_lerobot_metadata(
            module,
            output_dir,
            episode_metadata_list,
            job_id,
            scene_id,
        )

    monkeypatch.setattr(module, "convert_to_lerobot", _fake_convert_to_lerobot)

    config = module._create_import_config(
        {
            "job_id": "job123",
            "output_dir": output_dir,
            "min_quality_score": 0.5,
            "quality_component_thresholds": {},
            "enable_validation": False,
        }
    )

    result = module.run_local_import_job(config)

    assert called["episodes_dir"] == recordings_dir
    assert called["output_dir"] == output_dir / "lerobot"
    assert called["episode_ids"]
    assert result.lerobot_conversion_success is True
    assert (output_dir / "lerobot" / "dataset_info.json").exists()
    assert (output_dir / "lerobot" / "episodes.jsonl").exists()


def test_conversion_failure_reports_error_when_required(
    tmp_path: Path,
    load_job_module,
    monkeypatch,
) -> None:
    module = load_job_module("geniesim_import", "import_from_geniesim.py")
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    recordings_root = generate_mock_geniesim_local(
        output_dir=tmp_path,
        run_id="conversion_failure",
        episodes=1,
        seed=19,
    )
    recordings_dir = recordings_root / "recordings"

    monkeypatch.setattr(module, "_resolve_recordings_dir", lambda *args, **kwargs: recordings_dir)

    def _failing_convert_to_lerobot(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(module, "convert_to_lerobot", _failing_convert_to_lerobot)

    config = module._create_import_config(
        {
            "job_id": "job123",
            "output_dir": output_dir,
            "min_quality_score": 0.5,
            "quality_component_thresholds": {},
            "enable_validation": False,
            "require_lerobot": True,
            "require_lerobot_raw_value": "true",
            "require_lerobot_source": "env",
        }
    )

    result = module.run_local_import_job(config)

    assert any("LeRobot conversion failed" in err for err in result.errors)
