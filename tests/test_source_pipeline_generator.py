from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import tools.source_pipeline.generator as generator_mod
from tools.source_pipeline.generator import (
    LLMPlanResult,
    _build_placement_graph,
    _compute_quality_metrics,
    _default_dims_for_category,
    _extract_room_type,
    _is_articulated,
    _llm_plan_enabled,
    _physics_hints,
    _room_template,
    generate_text_scene_package,
    resolve_provider_chain,
)
from tools.source_pipeline.request import QualityTier


@pytest.fixture(autouse=True)
def _disable_llm_for_stable_tests(monkeypatch) -> None:
    monkeypatch.setenv("TEXT_GEN_USE_LLM", "false")


def test_generate_text_scene_package_produces_valid_structure() -> None:
    result = generate_text_scene_package(
        scene_id="scene_gen_001",
        prompt="A realistic kitchen with counters and mugs",
        quality_tier=QualityTier.STANDARD,
        seed=1,
        provider_policy="openai_primary",
    )

    assert result["schema_version"] == "v1"
    assert result["scene_id"] == "scene_gen_001"
    assert result["seed"] == 1
    assert result["quality_tier"] == "standard"
    assert isinstance(result["objects"], list)
    assert len(result["objects"]) > 0
    assert isinstance(result["placement_graph"], dict)
    assert isinstance(result["physics_hints"], dict)
    assert isinstance(result["provenance"], dict)
    assert isinstance(result["quality_gate_report"], dict)

    # Each object must have required keys
    for obj in result["objects"]:
        assert "id" in obj
        assert "name" in obj
        assert "category" in obj
        assert "sim_role" in obj
        assert "transform" in obj
        assert "dimensions_est" in obj
        assert "physics_hints" in obj
        assert "articulation" in obj


def test_generate_text_scene_package_respects_seed_determinism() -> None:
    kwargs = dict(
        scene_id="scene_det_001",
        prompt="A bedroom with a nightstand",
        quality_tier=QualityTier.STANDARD,
        seed=42,
        provider_policy="openai_primary",
    )

    run1 = generate_text_scene_package(**kwargs)
    run2 = generate_text_scene_package(**kwargs)

    assert run1["objects"] == run2["objects"]
    assert run1["quality_gate_report"]["metrics"] == run2["quality_gate_report"]["metrics"]


def test_generate_text_scene_package_standard_vs_premium_object_count() -> None:
    standard = generate_text_scene_package(
        scene_id="tier_s",
        prompt="An office desk setup",
        quality_tier=QualityTier.STANDARD,
        seed=1,
        provider_policy="openai_primary",
    )
    premium = generate_text_scene_package(
        scene_id="tier_p",
        prompt="An office desk setup",
        quality_tier=QualityTier.PREMIUM,
        seed=1,
        provider_policy="openai_primary",
    )

    assert len(standard["objects"]) == 12
    assert len(premium["objects"]) == 18


def test_generate_text_scene_package_v2_emits_staged_fields(monkeypatch) -> None:
    monkeypatch.setenv("TEXT_PIPELINE_V2_ENABLED", "true")
    monkeypatch.setenv("TEXT_PLACEMENT_STAGED_ENABLED", "true")

    result = generate_text_scene_package(
        scene_id="scene_v2_staged",
        prompt="A cluttered kitchen with mugs, bowls, and tools",
        quality_tier=QualityTier.STANDARD,
        seed=7,
        provider_policy="openai_primary",
    )

    assert len(result["objects"]) >= 30
    assert isinstance(result.get("layout_plan"), dict)
    assert isinstance(result["layout_plan"].get("room_box"), dict)
    assert isinstance(result.get("placement_stages"), list)
    assert [stage.get("stage") for stage in result["placement_stages"]] == ["furniture", "manipulands"]
    assert isinstance(result.get("critic_scores"), list)
    assert len(result["critic_scores"]) == 2
    assert isinstance(result.get("support_surfaces"), list)
    assert isinstance(result.get("faithfulness_report"), dict)
    assert 0.0 <= float(result["faithfulness_report"].get("score", 0.0)) <= 1.0

    for obj in result["objects"]:
        assert obj.get("placement_stage") in {"furniture", "manipulands"}
        if obj.get("sim_role") == "manipulable_object":
            assert obj.get("parent_support_id") is not None
            assert isinstance(obj.get("surface_local_se2"), dict)

def test_extract_room_type_from_prompt_keywords() -> None:
    assert _extract_room_type("A modern kitchen with steel appliances", {}) == "kitchen"
    assert _extract_room_type("cozy bedroom with a queen bed", {}) == "bedroom"
    assert _extract_room_type("the living room has a sofa", {}) == "living_room"
    assert _extract_room_type("a corporate office", {}) == "office"
    assert _extract_room_type("random scene with objects", {}) == "generic_room"


def test_extract_room_type_from_constraints_overrides_prompt() -> None:
    # Constraint should take priority over prompt keywords
    result = _extract_room_type("A kitchen scene", {"room_type": "warehouse"})
    assert result == "warehouse"


def test_extract_room_type_uses_prompt_diversity_archetype_when_room_type_missing() -> None:
    result = _extract_room_type(
        "A manipulation scene",
        {"prompt_diversity": {"dimensions": {"archetype": "lab"}}},
    )
    assert result == "lab"


def test_placement_graph_has_on_relations_for_movable_objects() -> None:
    objects = [
        {"id": "obj_001", "sim_role": "static"},
        {"id": "obj_002", "sim_role": "manipulable_object"},
        {"id": "obj_003", "sim_role": "manipulable_object"},
    ]
    graph = _build_placement_graph(objects)

    assert "relations" in graph
    on_relations = [r for r in graph["relations"] if r["type"] == "on"]
    assert len(on_relations) == 2
    for rel in on_relations:
        assert rel["object_id"] == "obj_001"  # anchor


def test_physics_hints_dynamic_flag_for_manipulable_objects() -> None:
    hints = _physics_hints("mug", "manipulable_object")
    assert hints["dynamic"] is True
    assert hints["mass_kg"] == 0.35

    hints_static = _physics_hints("counter", "static")
    assert hints_static["dynamic"] is False


def test_articulated_objects_marked_as_retrieved() -> None:
    result = generate_text_scene_package(
        scene_id="scene_art",
        prompt="A kitchen with a fridge and cabinets",
        quality_tier=QualityTier.STANDARD,
        seed=1,
        provider_policy="openai_primary",
    )

    for obj in result["objects"]:
        if _is_articulated(obj["sim_role"]):
            assert obj["asset_strategy"] == "retrieved"
            assert obj["articulation"]["backend_hint"] == "particulate_first"
        else:
            assert obj["asset_strategy"] == "generated"


def test_resolve_provider_chain_openai_primary() -> None:
    chain = resolve_provider_chain("openai_primary")
    assert chain == ["openai", "gemini", "anthropic"]


def test_room_template_returns_non_empty_for_known_types() -> None:
    for room in ["kitchen", "living_room", "bedroom", "office", "lab", "warehouse", "bathroom"]:
        template = _room_template(room)
        assert len(template) >= 5
        for name, category, sim_role in template:
            assert isinstance(name, str)
            assert isinstance(category, str)
            assert sim_role in {"static", "articulated_furniture", "articulated_appliance", "manipulable_object"}


def test_room_template_returns_generic_for_unknown_type() -> None:
    template = _room_template("underwater_lab")
    assert len(template) >= 3  # generic has at least table, shelf, chair, container, tool


def test_default_dims_known_categories() -> None:
    mug_dims = _default_dims_for_category("mug")
    assert mug_dims["width"] == 0.08
    assert mug_dims["height"] == 0.10

    unknown_dims = _default_dims_for_category("alien_artifact")
    assert unknown_dims == {"width": 0.35, "height": 0.35, "depth": 0.35}


def test_compute_quality_metrics_empty_objects() -> None:
    metrics = _compute_quality_metrics([])
    assert metrics["object_count"] == 0
    assert metrics["collision_rate_pct"] == 0.0
    assert metrics["stability_pct"] == 100.0


def test_compute_quality_metrics_no_collisions() -> None:
    objects = [
        {
            "transform": {"position": {"x": 0.0, "y": 0.0, "z": 0.0}},
            "dimensions_est": {"width": 0.1, "height": 0.1, "depth": 0.1},
            "articulation": {"required": False},
        },
        {
            "transform": {"position": {"x": 5.0, "y": 0.0, "z": 5.0}},
            "dimensions_est": {"width": 0.1, "height": 0.1, "depth": 0.1},
            "articulation": {"required": False},
        },
    ]
    metrics = _compute_quality_metrics(objects)
    assert metrics["collision_rate_pct"] == 0.0
    assert metrics["mean_collision_depth_mm"] == 0.0


def test_compute_quality_metrics_with_collisions() -> None:
    # Two objects at the same position should collide
    objects = [
        {
            "transform": {"position": {"x": 0.0, "y": 0.0, "z": 0.0}},
            "dimensions_est": {"width": 1.0, "height": 1.0, "depth": 1.0},
            "articulation": {"required": False},
        },
        {
            "transform": {"position": {"x": 0.1, "y": 0.0, "z": 0.0}},
            "dimensions_est": {"width": 1.0, "height": 1.0, "depth": 1.0},
            "articulation": {"required": False},
        },
    ]
    metrics = _compute_quality_metrics(objects)
    assert metrics["collision_rate_pct"] == 100.0  # 1 collision out of 1 pair
    assert metrics["mean_collision_depth_mm"] > 0.0


def test_quality_gate_report_has_cost_field(monkeypatch) -> None:
    monkeypatch.setenv("TEXT_GEN_USE_LLM", "false")
    result = generate_text_scene_package(
        scene_id="scene_cost",
        prompt="A kitchen",
        quality_tier=QualityTier.STANDARD,
        seed=1,
        provider_policy="openai_primary",
    )
    report = result["quality_gate_report"]
    assert "cost" in report
    assert "estimated_cost_usd" in report["cost"]
    assert report["cost"]["llm_calls"] == 0  # templates, no LLM
    assert report["cost"]["estimated_cost_usd"] == 0.0


def test_high_density_constraint_adds_objects() -> None:
    result = generate_text_scene_package(
        scene_id="scene_dense",
        prompt="A kitchen scene",
        quality_tier=QualityTier.STANDARD,
        seed=1,
        provider_policy="openai_primary",
        constraints={"object_density": "high"},
    )
    assert len(result["objects"]) == 18  # 12 + 6


def test_prompt_diversity_complexity_can_raise_target_count() -> None:
    result = generate_text_scene_package(
        scene_id="scene_complexity",
        prompt="A warehouse manipulation scene",
        quality_tier=QualityTier.STANDARD,
        seed=4,
        provider_policy="openai_primary",
        constraints={
            "prompt_diversity": {
                "dimensions": {
                    "archetype": "warehouse",
                    "complexity": "high_density_15_to_22_objects",
                }
            }
        },
    )
    assert len(result["objects"]) >= 16


def test_fallback_composition_varies_by_seed() -> None:
    base_kwargs = dict(
        scene_id="scene_seed_variation",
        prompt="A lab scene with manipulable tools and containers",
        quality_tier=QualityTier.STANDARD,
        provider_policy="openai_primary",
    )
    run1 = generate_text_scene_package(seed=2, **base_kwargs)
    run2 = generate_text_scene_package(seed=3, **base_kwargs)

    names1 = [obj["name"] for obj in run1["objects"]]
    names2 = [obj["name"] for obj in run2["objects"]]
    assert names1 != names2


def test_generate_text_scene_seed_is_stable_across_processes() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    probe = (
        "import json\n"
        "from tools.source_pipeline.generator import generate_text_scene_package\n"
        "from tools.source_pipeline.request import QualityTier\n"
        "pkg=generate_text_scene_package("
        "scene_id='scene_proc',"
        "prompt='An office desk setup',"
        "quality_tier=QualityTier.STANDARD,"
        "seed=7,"
        "provider_policy='openai_primary')\n"
        "print(json.dumps(pkg['objects'][0]['transform']['position'], sort_keys=True))\n"
    )

    out1 = subprocess.check_output([sys.executable, "-c", probe], cwd=repo_root, text=True).strip()
    out2 = subprocess.check_output([sys.executable, "-c", probe], cwd=repo_root, text=True).strip()

    assert json.loads(out1) == json.loads(out2)


def test_llm_plan_enabled_defaults_to_true_when_env_missing(monkeypatch) -> None:
    monkeypatch.delenv("TEXT_GEN_USE_LLM", raising=False)
    assert _llm_plan_enabled() is True


def test_generate_text_scene_package_records_llm_fallback_metadata(monkeypatch) -> None:
    monkeypatch.setenv("TEXT_GEN_USE_LLM", "true")

    def _fake_llm_plan(**_: object) -> LLMPlanResult:
        return LLMPlanResult(
            payload=None,
            provider=None,
            attempts=4,
            failure_reason="openai:TimeoutError",
        )

    monkeypatch.setattr(generator_mod, "_try_llm_plan", _fake_llm_plan)

    result = generate_text_scene_package(
        scene_id="scene_llm_fallback",
        prompt="A dense kitchen scene with articulated cabinets",
        quality_tier=QualityTier.PREMIUM,
        seed=3,
        provider_policy="openai_primary",
    )

    assert result["used_llm"] is False
    assert result["llm_attempts"] == 4
    assert result["llm_failure_reason"] == "openai:TimeoutError"
    assert result["fallback_strategy"] == "deterministic_template"
    report = result["quality_gate_report"]
    assert report["generation_mode"] == "deterministic_fallback"
    assert report["llm_attempts"] == 4


def test_generate_text_scene_package_supports_sage_backend(monkeypatch) -> None:
    monkeypatch.setenv("TEXT_SAGE_ACTION_DEMO_ENABLED", "true")
    result = generate_text_scene_package(
        scene_id="scene_sage",
        prompt="A tabletop pick and place scene",
        quality_tier=QualityTier.STANDARD,
        seed=2,
        provider_policy="openai_primary",
        text_backend="sage",
    )
    assert result["text_backend"] == "sage"
    assert result["provider_used"] == "sage"
    assert result["backend"]["selected"] == "sage"
    assert any(entry.get("name") == "sage" for entry in result["backend"]["backends"])
    assert isinstance(result.get("sage_action_demo"), dict)


def test_generate_text_scene_package_sage_backend_uses_live_server_when_available(monkeypatch) -> None:
    monkeypatch.setenv("SAGE_SERVER_URL", "https://sage.example/v1/refine")
    monkeypatch.setenv("SAGE_TIMEOUT_SECONDS", "30")

    live_base = generate_text_scene_package(
        scene_id="scene_sage_live_base",
        prompt="A compact kitchen workspace",
        quality_tier=QualityTier.STANDARD,
        seed=9,
        provider_policy="openai_primary",
        text_backend="internal",
    )
    live_package = json.loads(json.dumps(live_base))
    live_package["scene_id"] = "scene_sage_live"
    response_payload = {"request_id": "req-123", "package": live_package}

    class _FakeResponse:
        def __init__(self, payload: dict) -> None:
            self._payload = payload

        def read(self) -> bytes:
            return json.dumps(self._payload).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    def _fake_urlopen(req, timeout=0):  # type: ignore[no-untyped-def]
        assert req.full_url == "https://sage.example/v1/refine"
        assert timeout == 30.0
        request_payload = json.loads(req.data.decode("utf-8"))
        assert request_payload["scene_id"] == "scene_sage_live"
        return _FakeResponse(response_payload)

    monkeypatch.setattr(generator_mod.url_request, "urlopen", _fake_urlopen)

    result = generate_text_scene_package(
        scene_id="scene_sage_live",
        prompt="A pick-place scene with a bowl and tray",
        quality_tier=QualityTier.STANDARD,
        seed=9,
        provider_policy="openai_primary",
        text_backend="sage",
    )

    assert result["text_backend"] == "sage"
    assert result["quality_gate_report"]["generation_mode"] == "sage_live"
    assert result["sage"]["live_requested"] is True
    assert result["sage"]["live_used"] is True
    assert result["sage"]["live_invocation"]["enabled"] is True
    assert result["sage"]["live_invocation"]["request_id"] == "req-123"


def test_generate_text_scene_package_sage_backend_falls_back_when_live_unavailable(monkeypatch) -> None:
    monkeypatch.setenv("SAGE_SERVER_URL", "https://sage.example/v1/refine")
    monkeypatch.setenv("SAGE_TIMEOUT_SECONDS", "15")

    def _raise_url_error(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise generator_mod.url_error.URLError("connection refused")

    monkeypatch.setattr(generator_mod.url_request, "urlopen", _raise_url_error)

    result = generate_text_scene_package(
        scene_id="scene_sage_live_fallback",
        prompt="A pick-place scene with a mug and shelf",
        quality_tier=QualityTier.STANDARD,
        seed=10,
        provider_policy="openai_primary",
        text_backend="sage",
    )

    assert result["text_backend"] == "sage"
    assert result["quality_gate_report"]["generation_mode"] == "sage_refined"
    assert result["sage"]["live_requested"] is True
    assert result["sage"]["live_used"] is False


def test_generate_text_scene_package_supports_scenesmith_backend() -> None:
    result = generate_text_scene_package(
        scene_id="scene_scenesmith",
        prompt="A cluttered kitchen with many manipulable items",
        quality_tier=QualityTier.STANDARD,
        seed=5,
        provider_policy="openai_primary",
        text_backend="scenesmith",
    )
    assert result["text_backend"] == "scenesmith"
    assert result["backend"]["selected"] == "scenesmith"
    assert any(entry.get("name") == "scenesmith" for entry in result["backend"]["backends"])
    assert len(result["objects"]) >= 18


def test_generate_text_scene_package_scenesmith_backend_uses_live_server_when_available(monkeypatch) -> None:
    monkeypatch.setenv("SCENESMITH_SERVER_URL", "https://scenesmith.example/v1/generate")
    monkeypatch.setenv("SCENESMITH_TIMEOUT_SECONDS", "45")

    response_payload = {
        "request_id": "ss-req-1",
        "room_type": "kitchen",
        "objects": [
            {
                "id": "base_counter",
                "name": "countertop",
                "category": "counter",
                "sim_role": "static",
                "transform": {"position": {"x": 0.0, "y": 0.0, "z": 0.0}},
            },
            {
                "id": "obj_mug",
                "name": "mug",
                "category": "mug",
                "sim_role": "manipulable_object",
                "transform": {"position": {"x": 0.2, "y": 0.9, "z": 0.1}},
            },
        ],
    }

    class _FakeResponse:
        def __init__(self, payload: dict) -> None:
            self._payload = payload

        def read(self) -> bytes:
            return json.dumps(self._payload).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    def _fake_urlopen(req, timeout=0):  # type: ignore[no-untyped-def]
        assert req.full_url == "https://scenesmith.example/v1/generate"
        assert timeout == 45.0
        request_payload = json.loads(req.data.decode("utf-8"))
        assert request_payload["scene_id"] == "scene_scenesmith_live"
        assert request_payload["pipeline_stages"] == [
            "floor_plan",
            "furniture",
            "wall_mounted",
            "ceiling_mounted",
            "manipuland",
        ]
        return _FakeResponse(response_payload)

    monkeypatch.setattr(generator_mod.url_request, "urlopen", _fake_urlopen)

    result = generate_text_scene_package(
        scene_id="scene_scenesmith_live",
        prompt="A kitchen with dense clutter and task objects",
        quality_tier=QualityTier.STANDARD,
        seed=6,
        provider_policy="openai_primary",
        text_backend="scenesmith",
    )

    assert result["text_backend"] == "scenesmith"
    assert result["provider_used"] == "scenesmith"
    assert result["scenesmith"]["live_requested"] is True
    assert result["scenesmith"]["live_used"] is True
    assert result["scenesmith"]["live_invocation"]["enabled"] is True
    assert result["scenesmith"]["live_invocation"]["request_id"] == "ss-req-1"
    assert result["quality_gate_report"]["generation_mode"] == "scenesmith_live"


def test_generate_text_scene_package_scenesmith_backend_falls_back_when_live_unavailable(monkeypatch) -> None:
    monkeypatch.setenv("SCENESMITH_SERVER_URL", "https://scenesmith.example/v1/generate")
    monkeypatch.setenv("SCENESMITH_TIMEOUT_SECONDS", "30")

    def _raise_url_error(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise generator_mod.url_error.URLError("connection refused")

    monkeypatch.setattr(generator_mod.url_request, "urlopen", _raise_url_error)

    result = generate_text_scene_package(
        scene_id="scene_scenesmith_live_fallback",
        prompt="A kitchen with dense clutter and task objects",
        quality_tier=QualityTier.STANDARD,
        seed=6,
        provider_policy="openai_primary",
        text_backend="scenesmith",
    )

    assert result["text_backend"] == "scenesmith"
    assert result["scenesmith"]["live_requested"] is True
    assert result["scenesmith"]["live_used"] is False


def test_generate_text_scene_package_supports_hybrid_serial_backend() -> None:
    result = generate_text_scene_package(
        scene_id="scene_hybrid",
        prompt="A dense kitchen where a robot should move a bowl onto a shelf",
        quality_tier=QualityTier.PREMIUM,
        seed=7,
        provider_policy="openai_primary",
        text_backend="hybrid_serial",
    )
    assert result["text_backend"] == "hybrid_serial"
    backend_names = [entry.get("name") for entry in result["backend"]["backends"]]
    assert "scenesmith" in backend_names
    assert "sage" in backend_names
    assert "hybrid_serial" in backend_names


def test_generate_text_scene_package_respects_backend_allowlist(monkeypatch) -> None:
    monkeypatch.setenv("TEXT_BACKEND_ALLOWLIST", "internal")
    result = generate_text_scene_package(
        scene_id="scene_backend_allowlist",
        prompt="A room",
        quality_tier=QualityTier.STANDARD,
        seed=1,
        provider_policy="openai_primary",
        text_backend="sage",
    )
    assert result["text_backend"] == "internal"
