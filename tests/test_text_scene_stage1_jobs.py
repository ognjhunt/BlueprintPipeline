from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_text_gen_quality_gate_accepts_zero_collision() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        "text_scene_gen_job_test",
        repo_root / "text-scene-gen-job" / "generate_text_scene.py",
    )

    payload = {
        "scene_id": "scene_zero_collision",
        "quality_gate_report": {
            "metrics": {
                "object_count": 8,
                "collision_rate_pct": 0.0,
                "stability_pct": 99.5,
            }
        },
    }
    assert module._evaluate_package_quality(payload) is True


def test_text_stage1_jobs_integration_writes_canonical_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    gen_module = _load_module(
        "text_scene_gen_job_integration",
        repo_root / "text-scene-gen-job" / "generate_text_scene.py",
    )
    adapter_module = _load_module(
        "text_scene_adapter_job_integration",
        repo_root / "text-scene-adapter-job" / "adapt_text_scene.py",
    )

    gcs_root = tmp_path / "gcs"
    scene_id = "scene_stage1_int"
    request_object = f"scenes/{scene_id}/prompts/scene_request.json"
    textgen_prefix = f"scenes/{scene_id}/textgen"
    assets_prefix = f"scenes/{scene_id}/assets"
    layout_prefix = f"scenes/{scene_id}/layout"
    seg_prefix = f"scenes/{scene_id}/seg"

    request_path = gcs_root / request_object
    request_path.parent.mkdir(parents=True, exist_ok=True)
    request_path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "scene_id": scene_id,
                "source_mode": "text",
                "prompt": "An office desk setup",
                "quality_tier": "standard",
                "seed_count": 1,
                "provider_policy": "openai_primary",
                "fallback": {"allow_image_fallback": True},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(gen_module, "GCS_ROOT", gcs_root)
    monkeypatch.setattr(adapter_module, "GCS_ROOT", gcs_root)

    monkeypatch.setenv("BUCKET", "test-bucket")
    monkeypatch.setenv("SCENE_ID", scene_id)
    monkeypatch.setenv("REQUEST_OBJECT", request_object)
    monkeypatch.setenv("TEXTGEN_PREFIX", textgen_prefix)
    monkeypatch.setenv("TEXT_SEED", "1")
    monkeypatch.setenv("DEFAULT_SOURCE_MODE", "text")
    monkeypatch.setenv("TEXT_GEN_MAX_SEEDS", "16")
    monkeypatch.setenv("TEXT_GEN_STANDARD_PROFILE", "standard_v1")
    monkeypatch.setenv("TEXT_GEN_PREMIUM_PROFILE", "premium_v1")
    monkeypatch.setenv("TEXT_GEN_USE_LLM", "false")
    monkeypatch.delenv("TEXT_GEN_QUALITY_TIER", raising=False)

    assert gen_module.main() == 0
    assert (gcs_root / textgen_prefix / "package.json").is_file()
    assert (gcs_root / textgen_prefix / ".textgen_complete").is_file()
    completion = json.loads((gcs_root / textgen_prefix / ".textgen_complete").read_text(encoding="utf-8"))
    assert completion["generation_mode"] in {"llm", "deterministic_fallback"}
    assert "llm_attempts" in completion
    assert "llm_fallback_used" in completion

    monkeypatch.setenv("ASSETS_PREFIX", assets_prefix)
    monkeypatch.setenv("LAYOUT_PREFIX", layout_prefix)
    monkeypatch.setenv("SEG_PREFIX", seg_prefix)

    assert adapter_module.main() == 0

    assert (gcs_root / assets_prefix / "scene_manifest.json").is_file()
    assert (gcs_root / layout_prefix / "scene_layout_scaled.json").is_file()
    assert (gcs_root / seg_prefix / "inventory.json").is_file()
    assert (gcs_root / assets_prefix / ".regen3d_complete").is_file()
