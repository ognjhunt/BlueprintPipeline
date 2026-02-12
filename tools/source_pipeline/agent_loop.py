from __future__ import annotations

import copy
import random
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .placement_tools import (
    check_physics,
    place_furniture_stage,
    place_manipulands_stage,
)
from .scene_observer import summarize_scene


def _tokenize(text: str) -> set[str]:
    return {tok for tok in re.split(r"[^a-z0-9]+", text.lower()) if len(tok) > 2}


def _simple_collision_rate(objects: Sequence[Mapping[str, Any]], room_box: Optional[Mapping[str, Any]]) -> float:
    if len(objects) < 2:
        return 0.0
    collisions = 0
    pairs = 0
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            pairs += 1
            if not check_physics(candidate_obj=objects[i], placed_objects=[objects[j]], room_box=room_box):
                collisions += 1
    if pairs == 0:
        return 0.0
    return collisions / pairs


def compute_faithfulness_report(prompt: str, objects: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    prompt_tokens = _tokenize(prompt)
    category_tokens: set[str] = set()
    for obj in objects:
        category_tokens.update(_tokenize(str(obj.get("category") or "")))
        category_tokens.update(_tokenize(str(obj.get("name") or "")))

    matched = len(prompt_tokens & category_tokens)
    denom = max(1, len(prompt_tokens))
    lexical_alignment = matched / denom
    category_coverage = min(1.0, len(category_tokens) / max(10.0, len(objects) * 0.8))
    score = max(0.0, min(1.0, 0.65 * lexical_alignment + 0.35 * category_coverage))

    return {
        "schema_version": "v1",
        "lexical_alignment": round(lexical_alignment, 4),
        "category_coverage": round(category_coverage, 4),
        "score": round(score, 4),
        "status": "pass" if score >= 0.8 else "warn",
    }


def _critic_score(
    *,
    prompt: str,
    objects: Sequence[Mapping[str, Any]],
    room_box: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    faithfulness = compute_faithfulness_report(prompt, objects)
    collision_rate = _simple_collision_rate(objects, room_box)
    physics_score = max(0.0, 1.0 - (collision_rate * 2.0))
    semantic_score = faithfulness["score"]
    alignment_score = max(0.0, min(1.0, (semantic_score * 0.7) + (physics_score * 0.3)))
    total_0_10 = (0.45 * semantic_score + 0.35 * physics_score + 0.20 * alignment_score) * 10.0
    return {
        "semantic_plausibility": round(semantic_score * 10.0, 3),
        "physical_feasibility": round(physics_score * 10.0, 3),
        "alignment": round(alignment_score * 10.0, 3),
        "collision_rate": round(collision_rate, 6),
        "total": round(total_0_10, 3),
        "faithfulness_report": faithfulness,
    }


def _repair_jitter(
    *,
    objects: List[Dict[str, Any]],
    stage: str,
    rng: random.Random,
) -> None:
    for obj in objects:
        if stage == "furniture" and str(obj.get("placement_stage")) != "furniture":
            continue
        if stage == "manipulands" and str(obj.get("placement_stage")) != "manipulands":
            continue
        transform = obj.get("transform")
        if not isinstance(transform, dict):
            continue
        position = transform.get("position")
        if not isinstance(position, dict):
            continue
        position["x"] = round(float(position.get("x", 0.0)) + rng.uniform(-0.05, 0.05), 4)
        position["z"] = round(float(position.get("z", 0.0)) + rng.uniform(-0.05, 0.05), 4)


def _stage_threshold(stage: str) -> float:
    if stage == "furniture":
        return 7.6
    return 8.0


def run_two_agent_staged_loop(
    *,
    objects: List[Dict[str, Any]],
    prompt: str,
    room_box: Optional[Mapping[str, Any]],
    rng: random.Random,
    max_iterations: int = 2,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Run a lightweight designer+critic staged loop.

    Returns: (objects, placement_stages, critic_scores, support_surfaces, faithfulness_report)
    """
    working = [copy.deepcopy(obj) for obj in objects]
    placement_stages: List[Dict[str, Any]] = []
    critic_scores: List[Dict[str, Any]] = []
    support_surfaces: List[Dict[str, Any]] = []

    for stage_name in ("furniture", "manipulands"):
        stage_iterations: List[Dict[str, Any]] = []
        accepted_iteration = max_iterations

        for iteration in range(1, max_iterations + 1):
            if stage_name == "furniture":
                place_furniture_stage(objects=working, room_box=room_box, rng=rng)
            else:
                support_surfaces = place_manipulands_stage(objects=working, room_box=room_box, rng=rng)

            critique = _critic_score(prompt=prompt, objects=working, room_box=room_box)
            observation = summarize_scene(objects=working, room_box=room_box)
            stage_iterations.append(
                {
                    "iteration": iteration,
                    "critic": critique,
                    "observation": observation,
                }
            )
            if critique["total"] >= _stage_threshold(stage_name):
                accepted_iteration = iteration
                break

            if iteration < max_iterations:
                _repair_jitter(objects=working, stage=stage_name, rng=rng)

        final_critic = stage_iterations[accepted_iteration - 1]["critic"]
        critic_scores.append({"stage": stage_name, **final_critic})
        placement_stages.append(
            {
                "stage": stage_name,
                "accepted_iteration": accepted_iteration,
                "iterations": stage_iterations,
            }
        )

    final_faithfulness = compute_faithfulness_report(prompt, working)
    return working, placement_stages, critic_scores, support_surfaces, final_faithfulness

