import importlib.util
import sys
import types
from pathlib import Path

import pytest


def _install_fake_google_genai() -> None:
    google_module = types.ModuleType("google")
    google_module.__path__ = []

    genai_module = types.ModuleType("google.genai")

    class FakeClient:
        def __init__(self, api_key: str = "") -> None:
            self.api_key = api_key

    genai_module.Client = FakeClient
    genai_module.types = types.SimpleNamespace()

    google_module.genai = genai_module

    sys.modules["google"] = google_module
    sys.modules["google.genai"] = genai_module


def _load_scene_generation_module():
    _install_fake_google_genai()
    module_path = Path(__file__).resolve().parents[1] / "scene-generation-job" / "generate_scene_images.py"
    spec = importlib.util.spec_from_file_location("scene_generation_images", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["scene_generation_images"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_generate_scene_batch_propagates_failure(monkeypatch, tmp_path):
    module = _load_scene_generation_module()
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    def fake_prompt(self, archetype, recent_prompts, variation_coverage, target_variations=None):
        return "prompt", ["tag"]

    def fake_generate(self, request, dry_run=False):
        return module.SceneGenerationResult(
            scene_id=request.scene_id,
            archetype=request.archetype,
            success=False,
            error="generation failed",
            error_code=module.SceneGenerationErrorCode.IMAGE_GENERATION_FAILED.value,
            error_context={"scene_id": request.scene_id},
        )

    monkeypatch.setattr(module.PromptDiversifier, "generate_diverse_prompt", fake_prompt)
    monkeypatch.setattr(module.SceneImageGenerator, "generate_scene_image", fake_generate)

    with pytest.raises(module.SceneGenerationBatchError) as excinfo:
        module.generate_scene_batch(
            count=1,
            bucket=None,
            dry_run=False,
            specific_archetypes=["kitchen"],
            output_dir=tmp_path,
        )

    assert excinfo.value.summary["failed"] == 1
