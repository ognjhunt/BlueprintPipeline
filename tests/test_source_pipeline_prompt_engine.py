from __future__ import annotations

import json
from pathlib import Path

from tools.source_pipeline.prompt_engine import (
    build_prompt_constraints_metadata,
    compute_novelty_score,
    generate_prompt,
    load_prompt_matrix,
)


def test_load_prompt_matrix_default_schema() -> None:
    matrix = load_prompt_matrix()
    assert matrix["schema_version"] == "v1"
    assert isinstance(matrix["dimensions"], dict)
    assert "archetype" in matrix["dimensions"]


def test_generate_prompt_is_deterministic_without_llm(monkeypatch) -> None:
    monkeypatch.setenv("TEXT_PROMPT_USE_LLM", "false")

    recent = [
        {
            "prompt": "A prior scene prompt",
            "prompt_hash": "abc123",
            "dimensions": {"archetype": "kitchen"},
        }
    ]

    run1 = generate_prompt(
        run_date="2026-02-11",
        slot_index=1,
        provider_policy="openai_primary",
        recent_prompts=recent,
    )
    run2 = generate_prompt(
        run_date="2026-02-11",
        slot_index=1,
        provider_policy="openai_primary",
        recent_prompts=recent,
    )

    assert run1.prompt == run2.prompt
    assert run1.prompt_hash == run2.prompt_hash
    assert run1.dimensions == run2.dimensions
    assert run1.used_llm is False


def test_compute_novelty_score_reports_low_novelty_for_duplicate_text() -> None:
    prompt = "A modern kitchen scene with manipulable mugs"
    novelty = compute_novelty_score(prompt, [prompt])
    assert novelty == 0.0


def test_generate_prompt_sets_novelty_override_when_candidates_exhausted(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TEXT_PROMPT_USE_LLM", "false")

    matrix_payload = {
        "schema_version": "v1",
        "max_candidates": 1,
        "novelty_threshold": 0.95,
        "dedupe_window": 10,
        "coverage_rebalance": False,
        "dimensions": {
            "archetype": [{"value": "kitchen", "weight": 1.0}],
            "task_family": [{"value": "pick_and_place", "weight": 1.0}],
            "style": [{"value": "modern", "weight": 1.0}],
            "complexity": [{"value": "high_density_15_to_22_objects", "weight": 1.0}],
            "manipulation_focus": [{"value": "small_items_on_surfaces", "weight": 1.0}],
        },
    }
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps(matrix_payload), encoding="utf-8")

    first = generate_prompt(
        run_date="2026-02-12",
        slot_index=1,
        provider_policy="openai_primary",
        recent_prompts=[],
        matrix_path=matrix_path,
    )
    second = generate_prompt(
        run_date="2026-02-12",
        slot_index=1,
        provider_policy="openai_primary",
        recent_prompts=[{"prompt": first.prompt, "prompt_hash": first.prompt_hash}],
        matrix_path=matrix_path,
    )

    assert second.novelty_override is True


def test_build_prompt_constraints_metadata_contains_required_fields(monkeypatch) -> None:
    monkeypatch.setenv("TEXT_PROMPT_USE_LLM", "false")

    result = generate_prompt(
        run_date="2026-02-11",
        slot_index=2,
        provider_policy="openai_primary",
        recent_prompts=[],
    )
    metadata = build_prompt_constraints_metadata(result)

    assert "prompt_diversity" in metadata
    payload = metadata["prompt_diversity"]
    assert payload["prompt_hash"] == result.prompt_hash
    assert payload["dimensions"] == result.dimensions
    assert isinstance(payload["tags"], list)
