from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.source_pipeline.generator import (
    _build_placement_graph,
    _compute_quality_metrics,
    _default_dims_for_category,
    _extract_room_type,
    _is_articulated,
    _physics_hints,
    _room_template,
    generate_text_scene_package,
    resolve_provider_chain,
)
from tools.source_pipeline.request import QualityTier


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
    for room in ["kitchen", "living_room", "bedroom", "office"]:
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


def test_quality_gate_report_has_cost_field() -> None:
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
