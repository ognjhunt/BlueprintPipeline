import importlib.util
import sys
import types
from pathlib import Path


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


def test_scene_generation_models_use_env_overrides(monkeypatch):
    module = _load_scene_generation_module()
    monkeypatch.setenv("GEMINI_PRO_MODEL", "custom-pro-model")
    monkeypatch.setenv("GEMINI_IMAGE_MODEL", "custom-image-model")
    monkeypatch.setenv("GEMINI_IMAGE_MODEL_FALLBACK", "custom-fallback-model")

    assert module.get_prompt_model() == "custom-pro-model"
    assert module.get_image_model_overrides() == ("custom-image-model", "custom-fallback-model")
    assert module.get_image_model_chain() == ["custom-image-model", "custom-fallback-model"]
