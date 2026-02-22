from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .generator import LLMProviderAttempt, resolve_llm_attempt_chain

DEFAULT_PROMPT_MATRIX_PATH = Path(__file__).resolve().parents[2] / "configs" / "text_prompt_matrix.json"


@dataclass(frozen=True)
class PromptGenerationResult:
    prompt: str
    base_prompt: str
    tags: List[str]
    novelty_score: float
    prompt_hash: str
    used_llm: bool
    llm_attempts: int
    llm_failure_reason: Optional[str]
    llm_provider: Optional[str]
    dimensions: Dict[str, str]
    novelty_override: bool


def load_prompt_matrix(path: Optional[Path] = None) -> Dict[str, Any]:
    matrix_path = path or DEFAULT_PROMPT_MATRIX_PATH
    payload = json.loads(matrix_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "v1":
        raise ValueError(f"Unsupported prompt matrix schema: {payload.get('schema_version')!r}")
    if not isinstance(payload.get("dimensions"), dict):
        raise ValueError("Prompt matrix must define a dimensions object")
    return payload


def _llm_prompt_enabled() -> bool:
    raw = os.getenv("TEXT_PROMPT_USE_LLM", "true").strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def _llm_retry_config() -> Tuple[int, float]:
    attempts_raw = os.getenv("TEXT_PROMPT_LLM_MAX_ATTEMPTS", "3").strip()
    backoff_raw = os.getenv("TEXT_PROMPT_LLM_RETRY_BACKOFF_SECONDS", "2").strip()
    try:
        max_attempts = int(attempts_raw)
    except ValueError:
        max_attempts = 3
    try:
        backoff_seconds = float(backoff_raw)
    except ValueError:
        backoff_seconds = 2.0
    return max(1, max_attempts), max(0.0, backoff_seconds)


def _stable_rng_seed(*parts: str) -> int:
    material = "|".join(parts).encode("utf-8")
    return int.from_bytes(sha256(material).digest()[:8], "big") % (2**32)


def _parse_weighted_option(option: Any) -> Tuple[str, float]:
    if isinstance(option, str):
        return option, 1.0
    if isinstance(option, Mapping):
        value = str(option.get("value", "")).strip()
        weight_raw = option.get("weight", 1.0)
        try:
            weight = float(weight_raw)
        except (TypeError, ValueError):
            weight = 1.0
        return value, weight
    return "", 0.0


def _weighted_choice(rng: random.Random, options: Sequence[Any]) -> str:
    normalized: List[Tuple[str, float]] = []
    total = 0.0
    for option in options:
        value, weight = _parse_weighted_option(option)
        if not value or weight <= 0:
            continue
        normalized.append((value, weight))
        total += weight
    if not normalized or total <= 0:
        raise ValueError("No valid weighted options available")

    pick = rng.uniform(0.0, total)
    cumulative = 0.0
    for value, weight in normalized:
        cumulative += weight
        if pick <= cumulative:
            return value
    return normalized[-1][0]


def _normalize_text_tokens(text: str) -> List[str]:
    normalized = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    tokens = [token for token in normalized.split() if len(token) > 2]
    return tokens


def _jaccard_similarity(a: Sequence[str], b: Sequence[str]) -> float:
    aset = set(a)
    bset = set(b)
    if not aset or not bset:
        return 0.0
    return len(aset & bset) / len(aset | bset)


def _extract_recent_prompt_texts(recent_prompts: Sequence[Any]) -> List[str]:
    texts: List[str] = []
    for entry in recent_prompts:
        if isinstance(entry, str):
            if entry.strip():
                texts.append(entry.strip())
            continue
        if isinstance(entry, Mapping):
            for field in ("prompt", "expanded_prompt", "base_prompt", "prompt_summary"):
                value = entry.get(field)
                if isinstance(value, str) and value.strip():
                    texts.append(value.strip())
                    break
    return texts


def compute_novelty_score(prompt: str, recent_prompts: Sequence[Any]) -> float:
    prompt_tokens = _normalize_text_tokens(prompt)
    if not prompt_tokens:
        return 1.0
    recent_texts = _extract_recent_prompt_texts(recent_prompts)
    if not recent_texts:
        return 1.0

    max_similarity = 0.0
    for recent in recent_texts:
        similarity = _jaccard_similarity(prompt_tokens, _normalize_text_tokens(recent))
        if similarity > max_similarity:
            max_similarity = similarity
    return round(max(0.0, 1.0 - max_similarity), 4)


def _hash_prompt(prompt: str) -> str:
    return sha256(prompt.strip().encode("utf-8")).hexdigest()[:16]


def _extract_recent_hashes(recent_prompts: Sequence[Any]) -> List[str]:
    hashes: List[str] = []
    for entry in recent_prompts:
        if isinstance(entry, Mapping):
            value = entry.get("prompt_hash")
            if isinstance(value, str) and value.strip():
                hashes.append(value.strip())
    return hashes


def _rebalanced_dimension_options(
    *,
    dimension_name: str,
    options: Sequence[Any],
    recent_prompts: Sequence[Any],
    enable_rebalance: bool,
) -> List[Dict[str, Any]]:
    parsed: List[Dict[str, Any]] = []
    for item in options:
        value, weight = _parse_weighted_option(item)
        if value:
            parsed.append({"value": value, "weight": max(0.01, weight)})

    if not enable_rebalance or not parsed:
        return parsed

    counts: Dict[str, int] = {entry["value"]: 0 for entry in parsed}
    for entry in recent_prompts:
        if not isinstance(entry, Mapping):
            continue
        dims = entry.get("dimensions")
        if not isinstance(dims, Mapping):
            continue
        selected = dims.get(dimension_name)
        if isinstance(selected, str) and selected in counts:
            counts[selected] += 1

    if not any(counts.values()):
        return parsed

    average = sum(counts.values()) / max(1, len(counts))
    adjusted: List[Dict[str, Any]] = []
    for entry in parsed:
        value = entry["value"]
        weight = float(entry["weight"])
        observed = counts.get(value, 0)
        if observed > average * 1.25:
            weight *= 0.8
        elif observed < average * 0.75:
            weight *= 1.2
        adjusted.append({"value": value, "weight": round(max(0.01, weight), 4)})
    return adjusted


def _humanize(value: str) -> str:
    return value.replace("_", " ")


def _build_base_prompt(dimensions: Mapping[str, str]) -> str:
    archetype = _humanize(dimensions.get("archetype", "indoor scene"))
    task_family = _humanize(dimensions.get("task_family", "pick and place"))
    style = _humanize(dimensions.get("style", "realistic"))
    complexity = _humanize(dimensions.get("complexity", "medium density"))
    focus = _humanize(dimensions.get("manipulation_focus", "mixed objects"))

    return (
        f"Create a photorealistic {style} {archetype} designed for robot {task_family} tasks. "
        f"Prioritize {focus}, with a {complexity}. Include articulated and static objects, "
        "clear manipulation affordances, stable placement, and realistic clutter suitable for "
        "sim-to-real policy training."
    )


def _provider_name_to_enum(provider_name: str, llm_provider_enum: Any) -> Optional[Any]:
    mapping = {
        "openai": getattr(llm_provider_enum, "OPENAI", None),
        "gemini": getattr(llm_provider_enum, "GEMINI", None),
        "anthropic": getattr(llm_provider_enum, "ANTHROPIC", None),
    }
    return mapping.get(provider_name)


def _expand_prompt_with_llm(
    *,
    base_prompt: str,
    tags: Sequence[str],
    attempt_chain: Sequence[LLMProviderAttempt],
) -> Tuple[str, bool, int, Optional[str], Optional[str]]:
    if not _llm_prompt_enabled():
        return base_prompt, False, 0, "llm_disabled", None

    try:
        from tools.llm_client import create_llm_client, LLMProvider
    except Exception as exc:
        return base_prompt, False, 0, f"llm_client_unavailable:{exc.__class__.__name__}", None

    max_attempts, retry_backoff_seconds = _llm_retry_config()
    effort = os.getenv("TEXT_PROMPT_LLM_REASONING_EFFORT", "high").strip().lower() or "high"
    attempts = 0
    failure_reason: Optional[str] = None

    tag_text = ", ".join(tags)
    system_prompt = (
        "You are generating a rich scene prompt for 3D robotic simulation. "
        "Return strict JSON with one key: prompt (string). "
        "The prompt must be concrete, physically plausible, and include manipulable objects and articulated opportunities."
    )

    for round_idx in range(1, max_attempts + 1):
        for attempt in attempt_chain:
            provider_enum = _provider_name_to_enum(attempt.provider, LLMProvider)
            if provider_enum is None:
                continue
            attempts += 1
            try:
                client_kwargs: Dict[str, Any] = {
                    "provider": provider_enum,
                    "fallback_enabled": False,
                    "reasoning_effort": effort,
                }
                if attempt.model:
                    client_kwargs["model"] = attempt.model
                if attempt.api_key:
                    client_kwargs["api_key"] = attempt.api_key
                if attempt.base_url:
                    client_kwargs["base_url"] = attempt.base_url
                if attempt.default_headers:
                    client_kwargs["default_headers"] = dict(attempt.default_headers)
                client = create_llm_client(**client_kwargs)
                response = client.generate(
                    prompt=(
                        f"{system_prompt}\n\n"
                        f"Base prompt: {base_prompt}\n"
                        f"Diversity tags: {tag_text}\n"
                        "Expand into 2-3 concise paragraphs suitable as scene generation input."
                    ),
                    json_output=True,
                )
                response_text = (response.text or "").strip()
                if not response_text:
                    failure_reason = f"{attempt.provider_name}:empty_response"
                    continue
                payload = json.loads(response_text)
                prompt = payload.get("prompt") if isinstance(payload, dict) else None
                if isinstance(prompt, str) and prompt.strip():
                    return prompt.strip(), True, attempts, None, attempt.provider_name
                failure_reason = f"{attempt.provider_name}:missing_prompt"
            except Exception as exc:
                failure_reason = f"{attempt.provider_name}:{exc.__class__.__name__}"
                continue

        if round_idx < max_attempts and retry_backoff_seconds > 0:
            time.sleep(retry_backoff_seconds * (2 ** round_idx))

    return base_prompt, False, attempts, failure_reason or "llm_generation_failed", None


def generate_prompt(
    *,
    run_date: str,
    slot_index: int,
    provider_policy: str,
    recent_prompts: Sequence[Any],
    matrix_path: Optional[Path] = None,
) -> PromptGenerationResult:
    matrix = load_prompt_matrix(matrix_path)
    dimensions = matrix.get("dimensions") or {}
    if not isinstance(dimensions, Mapping) or not dimensions:
        raise ValueError("Prompt matrix dimensions are missing or invalid")

    max_candidates = max(1, int(matrix.get("max_candidates", 5)))
    novelty_threshold = float(matrix.get("novelty_threshold", 0.35))
    dedupe_window = max(1, int(matrix.get("dedupe_window", 50)))
    coverage_rebalance = bool(matrix.get("coverage_rebalance", True))

    windowed_recent = list(recent_prompts)[-dedupe_window:]
    recent_hashes = set(_extract_recent_hashes(windowed_recent))

    attempt_chain = resolve_llm_attempt_chain(provider_policy)

    best_candidate: Optional[PromptGenerationResult] = None
    best_novelty = -1.0

    for candidate_idx in range(max_candidates):
        rng = random.Random(_stable_rng_seed(run_date, str(slot_index), str(candidate_idx), provider_policy))

        selected_dimensions: Dict[str, str] = {}
        for dimension_name, raw_options in dimensions.items():
            if not isinstance(raw_options, Sequence):
                continue
            options = _rebalanced_dimension_options(
                dimension_name=dimension_name,
                options=raw_options,
                recent_prompts=windowed_recent,
                enable_rebalance=coverage_rebalance,
            )
            selected_dimensions[dimension_name] = _weighted_choice(rng, options)

        tags = [f"{key}:{value}" for key, value in sorted(selected_dimensions.items())]
        base_prompt = _build_base_prompt(selected_dimensions)

        expanded_prompt, used_llm, llm_attempts, llm_failure_reason, llm_provider = _expand_prompt_with_llm(
            base_prompt=base_prompt,
            tags=tags,
            attempt_chain=attempt_chain,
        )

        prompt_hash = _hash_prompt(expanded_prompt)
        novelty_score = compute_novelty_score(expanded_prompt, windowed_recent)
        duplicate = prompt_hash in recent_hashes

        candidate = PromptGenerationResult(
            prompt=expanded_prompt,
            base_prompt=base_prompt,
            tags=tags,
            novelty_score=novelty_score,
            prompt_hash=prompt_hash,
            used_llm=used_llm,
            llm_attempts=llm_attempts,
            llm_failure_reason=llm_failure_reason,
            llm_provider=llm_provider,
            dimensions=selected_dimensions,
            novelty_override=False,
        )

        if not duplicate and novelty_score >= novelty_threshold:
            return candidate

        if not duplicate and novelty_score > best_novelty:
            best_candidate = candidate
            best_novelty = novelty_score

    if best_candidate is not None:
        return PromptGenerationResult(
            prompt=best_candidate.prompt,
            base_prompt=best_candidate.base_prompt,
            tags=best_candidate.tags,
            novelty_score=best_candidate.novelty_score,
            prompt_hash=best_candidate.prompt_hash,
            used_llm=best_candidate.used_llm,
            llm_attempts=best_candidate.llm_attempts,
            llm_failure_reason=best_candidate.llm_failure_reason,
            llm_provider=best_candidate.llm_provider,
            dimensions=best_candidate.dimensions,
            novelty_override=True,
        )

    # Absolute fallback to deterministic base prompt generation if all candidates duplicated.
    rng = random.Random(_stable_rng_seed(run_date, str(slot_index), "fallback", provider_policy))
    selected_dimensions: Dict[str, str] = {}
    for dimension_name, raw_options in dimensions.items():
        if isinstance(raw_options, Sequence):
            selected_dimensions[dimension_name] = _weighted_choice(rng, raw_options)
    base_prompt = _build_base_prompt(selected_dimensions)
    return PromptGenerationResult(
        prompt=base_prompt,
        base_prompt=base_prompt,
        tags=[f"{key}:{value}" for key, value in sorted(selected_dimensions.items())],
        novelty_score=0.0,
        prompt_hash=_hash_prompt(base_prompt),
        used_llm=False,
        llm_attempts=0,
        llm_failure_reason="novelty_generation_exhausted",
        llm_provider=None,
        dimensions=selected_dimensions,
        novelty_override=True,
    )


def build_prompt_constraints_metadata(result: PromptGenerationResult) -> Dict[str, Any]:
    return {
        "prompt_diversity": {
            "prompt_hash": result.prompt_hash,
            "novelty_score": result.novelty_score,
            "novelty_override": result.novelty_override,
            "tags": result.tags,
            "dimensions": result.dimensions,
            "base_prompt": result.base_prompt,
            "used_llm": result.used_llm,
            "llm_attempts": result.llm_attempts,
            "llm_provider": result.llm_provider,
            "llm_failure_reason": result.llm_failure_reason,
        }
    }
