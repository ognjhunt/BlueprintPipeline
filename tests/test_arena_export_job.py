from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.contract_utils import validate_json_schema


class FakeAffordanceDetector:
    def __init__(self, use_llm: bool = True) -> None:
        self.use_llm = use_llm

    def detect(self, obj: dict) -> list[dict]:
        return []

    def to_manifest_format(self, affordances: list[dict]) -> dict:
        return {"affordances": [], "affordance_params": {}}


class FakeArenaExporter:
    def __init__(self, config) -> None:
        self.config = config

    def export(self, manifest: dict) -> SimpleNamespace:
        return SimpleNamespace(
            output_dir=self.config.output_dir,
            generated_files=[],
            affordance_count=0,
            task_count=0,
            errors=[],
            success=True,
        )


def test_arena_export_requires_env_vars(load_job_module, monkeypatch) -> None:
    module = load_job_module("arena_export", "arena_export_job.py")

    monkeypatch.delenv("BUCKET", raising=False)
    monkeypatch.delenv("SCENE_ID", raising=False)
    monkeypatch.delenv("ASSETS_PREFIX", raising=False)

    with pytest.raises(SystemExit) as excinfo:
        module.run_from_env()

    assert excinfo.value.code == 1


def test_arena_export_parse_args(load_job_module) -> None:
    module = load_job_module("arena_export", "arena_export_job.py")

    args = module.parse_args(
        [
            "--scene-dir",
            "/tmp/scene",
            "--output-dir",
            "/tmp/out",
            "--no-llm",
            "--enable-hub",
            "--hub-namespace",
            "custom-space",
            "--disable-premium-analytics",
        ]
    )

    assert args.scene_dir.as_posix() == "/tmp/scene"
    assert args.output_dir.as_posix() == "/tmp/out"
    assert args.no_llm is True
    assert args.enable_hub is True
    assert args.hub_namespace == "custom-space"
    assert args.disable_premium_analytics is True


def test_arena_export_minimal_run(load_job_module, tmp_path, monkeypatch) -> None:
    module = load_job_module("arena_export", "arena_export_job.py")

    scene_dir = tmp_path / "scene_123"
    assets_dir = scene_dir / "assets"
    usd_dir = scene_dir / "usd"
    assets_dir.mkdir(parents=True)
    usd_dir.mkdir()
    (usd_dir / "scene.usda").write_text("#usda 1.0")

    manifest = {
        "scene_id": "scene_123",
        "scene": {"environment_type": "kitchen"},
        "objects": [],
    }

    monkeypatch.setattr(module, "AffordanceDetector", FakeAffordanceDetector)
    monkeypatch.setattr(module, "ArenaSceneExporter", FakeArenaExporter)
    monkeypatch.setattr(module, "load_manifest_or_scene_assets", lambda *_: manifest)

    for flag in [
        "PREMIUM_ANALYTICS_AVAILABLE",
        "SIM2REAL_AVAILABLE",
        "EMBODIMENT_TRANSFER_AVAILABLE",
        "TRAJECTORY_OPTIMALITY_AVAILABLE",
        "POLICY_LEADERBOARD_AVAILABLE",
        "TACTILE_SENSOR_AVAILABLE",
        "LANGUAGE_ANNOTATIONS_AVAILABLE",
        "GENERALIZATION_ANALYZER_AVAILABLE",
        "SIM2REAL_VALIDATION_AVAILABLE",
        "AUDIO_NARRATION_AVAILABLE",
    ]:
        monkeypatch.setattr(module, flag, False)

    result = module.run_arena_export(
        scene_dir=scene_dir,
        output_dir=scene_dir,
        use_llm=False,
        enable_hub_registration=False,
        enable_premium_analytics=False,
    )

    schema = {
        "type": "object",
        "required": [
            "success",
            "scene_id",
            "files_generated",
            "errors",
            "premium_analytics_enabled",
        ],
        "properties": {
            "success": {"type": "boolean"},
            "scene_id": {"type": "string"},
            "files_generated": {"type": "array"},
            "errors": {"type": "array"},
            "premium_analytics_enabled": {"type": "boolean"},
        },
    }
    validate_json_schema(result, schema)

    assert result["scene_id"] == "scene_123"
    assert result["success"] is True
    assert result["premium_analytics_enabled"] is False
    assert result["errors"] == []
