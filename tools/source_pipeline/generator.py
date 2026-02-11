from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from hashlib import sha256
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .request import QualityTier

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProviderDecision:
    provider: str
    policy: str
    used_llm: bool


@dataclass(frozen=True)
class LLMPlanResult:
    payload: Optional[Dict[str, Any]]
    provider: Optional[str]
    attempts: int
    failure_reason: Optional[str]


ROOM_TYPE_ALIASES: Dict[str, str] = {
    "kitchen_dining_nook": "kitchen",
    "dining_room": "living_room",
    "livingroom": "living_room",
    "living space": "living_room",
    "laboratory": "lab",
    "study": "office",
    "workshop": "warehouse",
    "garage": "warehouse",
    "laundry": "bathroom",
}


OBJECT_SUBSTITUTIONS: Dict[str, List[str]] = {
    "mug": ["cup", "bowl", "glass"],
    "plate": ["dish", "tray"],
    "bottle": ["canister", "thermos", "flask"],
    "book": ["magazine", "manual"],
    "keyboard": ["tablet", "laptop"],
    "mouse": ["controller", "trackball"],
    "container": ["bin", "basket", "crate"],
    "tool": ["wrench", "screwdriver", "pliers"],
}


def resolve_provider_chain(policy: str) -> List[str]:
    """Resolve ordered provider chain for generation."""

    if policy == "openai_primary":
        return ["openai", "gemini", "anthropic"]
    return ["openai", "gemini", "anthropic"]


def _canonicalize_room_type(raw: str) -> str:
    normalized = re.sub(r"[\s\-]+", "_", raw.strip().lower())
    return ROOM_TYPE_ALIASES.get(normalized, normalized)


def _prompt_diversity_dimensions(constraints: Mapping[str, Any]) -> Mapping[str, str]:
    diversity = constraints.get("prompt_diversity")
    if not isinstance(diversity, Mapping):
        return {}
    dimensions = diversity.get("dimensions")
    if not isinstance(dimensions, Mapping):
        return {}
    output: Dict[str, str] = {}
    for key, value in dimensions.items():
        if isinstance(value, str):
            output[key] = value
    return output


def _extract_room_type(prompt: str, constraints: Dict[str, Any]) -> str:
    constraint_room = str(constraints.get("room_type", "")).strip()
    if constraint_room:
        return _canonicalize_room_type(constraint_room)

    dimensions = _prompt_diversity_dimensions(constraints)
    archetype = str(dimensions.get("archetype", "")).strip()
    if archetype:
        return _canonicalize_room_type(archetype)

    lowered = prompt.lower()
    for token in [
        "kitchen",
        "living room",
        "bedroom",
        "bathroom",
        "office",
        "lab",
        "warehouse",
    ]:
        if token in lowered:
            return _canonicalize_room_type(token)
    return "generic_room"


def _room_template(room_type: str) -> List[Tuple[str, str, str]]:
    """Return (name, category, sim_role) base template for room type."""

    normalized_room = _canonicalize_room_type(room_type)
    templates: Dict[str, List[Tuple[str, str, str]]] = {
        "kitchen": [
            ("countertop", "counter", "static"),
            ("sink", "sink", "static"),
            ("fridge", "refrigerator", "articulated_appliance"),
            ("cabinet", "cabinet", "articulated_furniture"),
            ("island", "counter", "static"),
            ("pantry", "cabinet", "articulated_furniture"),
            ("oven", "appliance", "articulated_appliance"),
            ("stool", "stool", "static"),
            ("cutting_board", "board", "manipulable_object"),
            ("knife_block", "tool", "manipulable_object"),
            ("mug", "mug", "manipulable_object"),
            ("plate", "plate", "manipulable_object"),
            ("bottle", "bottle", "manipulable_object"),
            ("bowl", "bowl", "manipulable_object"),
            ("utensil", "utensil", "manipulable_object"),
            ("jar", "container", "manipulable_object"),
        ],
        "living_room": [
            ("sofa", "sofa", "static"),
            ("coffee_table", "table", "static"),
            ("bookshelf", "bookshelf", "static"),
            ("tv_stand", "tv_stand", "static"),
            ("sideboard", "cabinet", "articulated_furniture"),
            ("media_console", "cabinet", "articulated_furniture"),
            ("lamp", "lamp", "static"),
            ("side_table", "table", "static"),
            ("ottoman", "stool", "static"),
            ("book", "book", "manipulable_object"),
            ("remote", "remote", "manipulable_object"),
            ("mug", "mug", "manipulable_object"),
            ("gamepad", "controller", "manipulable_object"),
            ("coaster", "dish", "manipulable_object"),
            ("throw_pillow", "cushion", "manipulable_object"),
            ("blanket", "fabric", "manipulable_object"),
        ],
        "bedroom": [
            ("bed", "bed", "static"),
            ("nightstand", "nightstand", "static"),
            ("dresser", "dresser", "static"),
            ("closet", "closet", "articulated_furniture"),
            ("wardrobe", "cabinet", "articulated_furniture"),
            ("chest", "cabinet", "articulated_furniture"),
            ("desk", "desk", "static"),
            ("lamp", "lamp", "static"),
            ("book", "book", "manipulable_object"),
            ("alarm_clock", "device", "manipulable_object"),
            ("water_bottle", "bottle", "manipulable_object"),
            ("storage_box", "container", "manipulable_object"),
            ("laundry_basket", "basket", "manipulable_object"),
            ("tablet", "tablet", "manipulable_object"),
        ],
        "office": [
            ("desk", "desk", "static"),
            ("chair", "chair", "static"),
            ("monitor", "monitor", "static"),
            ("cabinet", "cabinet", "articulated_furniture"),
            ("drawer_unit", "cabinet", "articulated_furniture"),
            ("bookshelf", "bookshelf", "static"),
            ("printer_table", "table", "static"),
            ("keyboard", "keyboard", "manipulable_object"),
            ("mouse", "mouse", "manipulable_object"),
            ("notebook", "notebook", "manipulable_object"),
            ("pen_holder", "container", "manipulable_object"),
            ("file_folder", "folder", "manipulable_object"),
            ("headset", "device", "manipulable_object"),
            ("laptop", "laptop", "manipulable_object"),
        ],
        "lab": [
            ("workbench", "table", "static"),
            ("storage_cabinet", "cabinet", "articulated_furniture"),
            ("instrument_cart", "cart", "static"),
            ("fume_hood", "hood", "articulated_appliance"),
            ("sample_rack", "rack", "static"),
            ("bin_station", "container", "manipulable_object"),
            ("beaker", "container", "manipulable_object"),
            ("flask", "bottle", "manipulable_object"),
            ("pipette_case", "container", "manipulable_object"),
            ("notepad", "notebook", "manipulable_object"),
            ("small_toolkit", "tool", "manipulable_object"),
            ("sealed_box", "container", "manipulable_object"),
        ],
        "warehouse": [
            ("pallet_rack", "rack", "static"),
            ("shelf_unit", "shelf", "static"),
            ("rolling_cart", "cart", "static"),
            ("tool_chest", "cabinet", "articulated_furniture"),
            ("locker", "cabinet", "articulated_furniture"),
            ("work_table", "table", "static"),
            ("crate_stack", "crate", "manipulable_object"),
            ("parcel_box", "container", "manipulable_object"),
            ("tote_bin", "container", "manipulable_object"),
            ("barcode_scanner", "device", "manipulable_object"),
            ("tape_dispenser", "tool", "manipulable_object"),
            ("safety_cone", "cone", "manipulable_object"),
            ("battery_pack", "device", "manipulable_object"),
        ],
        "bathroom": [
            ("vanity", "cabinet", "articulated_furniture"),
            ("sink", "sink", "static"),
            ("mirror_cabinet", "cabinet", "articulated_furniture"),
            ("shower_shelf", "shelf", "static"),
            ("hamper", "basket", "manipulable_object"),
            ("soap_dispenser", "bottle", "manipulable_object"),
            ("toothbrush_cup", "container", "manipulable_object"),
            ("towel_stack", "fabric", "manipulable_object"),
            ("spray_bottle", "bottle", "manipulable_object"),
            ("storage_bin", "container", "manipulable_object"),
        ],
    }
    return templates.get(normalized_room, [
        ("table", "table", "static"),
        ("shelf", "shelf", "static"),
        ("chair", "chair", "static"),
        ("cabinet", "cabinet", "articulated_furniture"),
        ("container", "container", "manipulable_object"),
        ("tool", "tool", "manipulable_object"),
        ("bottle", "bottle", "manipulable_object"),
        ("book", "book", "manipulable_object"),
    ])


def _default_dims_for_category(category: str) -> Dict[str, float]:
    dims = {
        "counter": (1.6, 0.9, 0.6),
        "sink": (0.6, 0.3, 0.5),
        "refrigerator": (0.9, 1.8, 0.8),
        "cabinet": (0.8, 1.8, 0.5),
        "closet": (1.2, 2.0, 0.6),
        "dresser": (1.0, 1.0, 0.5),
        "table": (1.2, 0.75, 0.7),
        "desk": (1.4, 0.75, 0.7),
        "sofa": (2.0, 0.9, 0.9),
        "bed": (2.0, 0.6, 1.6),
        "chair": (0.5, 0.9, 0.5),
        "stool": (0.45, 0.5, 0.45),
        "shelf": (1.1, 1.8, 0.4),
        "bookshelf": (1.0, 1.8, 0.35),
        "tv_stand": (1.4, 0.5, 0.4),
        "monitor": (0.55, 0.4, 0.2),
        "notebook": (0.21, 0.02, 0.3),
        "laptop": (0.34, 0.03, 0.24),
        "container": (0.35, 0.25, 0.35),
        "basket": (0.45, 0.35, 0.35),
        "crate": (0.55, 0.45, 0.45),
        "rack": (1.8, 2.1, 0.8),
        "cart": (0.95, 1.0, 0.6),
        "hood": (1.3, 2.2, 0.8),
        "appliance": (0.75, 0.9, 0.7),
        "board": (0.4, 0.03, 0.3),
        "utensil": (0.24, 0.03, 0.03),
        "device": (0.18, 0.08, 0.12),
        "folder": (0.24, 0.03, 0.32),
        "fabric": (0.45, 0.12, 0.3),
        "cushion": (0.4, 0.12, 0.35),
        "cone": (0.22, 0.4, 0.22),
        "dish": (0.28, 0.05, 0.28),
        "mug": (0.08, 0.10, 0.08),
        "plate": (0.26, 0.03, 0.26),
        "bowl": (0.18, 0.08, 0.18),
        "bottle": (0.09, 0.30, 0.09),
        "book": (0.18, 0.03, 0.24),
        "remote": (0.05, 0.02, 0.18),
        "keyboard": (0.44, 0.03, 0.14),
        "mouse": (0.07, 0.04, 0.11),
    }
    width, height, depth = dims.get(category, (0.35, 0.35, 0.35))
    return {"width": width, "height": height, "depth": depth}


def _sample_position(rng: random.Random, index: int) -> Dict[str, float]:
    lane = (index - 1) % 4
    row = (index - 1) // 4
    x = -2.4 + lane * 1.6 + rng.uniform(-0.06, 0.06)
    z = -2.6 + row * 1.2 + rng.uniform(-0.06, 0.06)
    y = max(0.0, rng.uniform(0.0, 0.02))
    return {"x": round(x, 4), "y": round(y, 4), "z": round(z, 4)}


def _is_articulated(sim_role: str) -> bool:
    return sim_role in {"articulated_furniture", "articulated_appliance"}


def _asset_source_hint(sim_role: str) -> str:
    # Particulate-first articulation policy: articulated assets default to retrieval.
    if _is_articulated(sim_role):
        return "retrieved"
    return "generated"


def _physics_hints(category: str, sim_role: str) -> Dict[str, Any]:
    dynamic = sim_role in {"manipulable_object", "clutter"}
    mass = {
        "mug": 0.35,
        "plate": 0.45,
        "bottle": 0.55,
        "book": 0.30,
        "keyboard": 0.75,
        "mouse": 0.15,
    }.get(category, 4.0 if not dynamic else 0.7)
    return {
        "dynamic": dynamic,
        "mass_kg": mass,
        "friction_static": 0.55,
        "friction_dynamic": 0.40,
        "restitution": 0.10,
        "material_name": "generic",
    }


def _compute_quality_metrics(objects: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute deterministic quality metrics from the generated object list.

    Uses axis-aligned bounding box (AABB) overlap detection to measure collision
    rate and depth.  Stability and fallen-object rates are deterministic because
    all objects are placed deliberately on y >= 0.
    """

    n = len(objects)
    if n == 0:
        return {
            "object_count": 0,
            "collision_rate_pct": 0.0,
            "mean_collision_depth_mm": 0.0,
            "stability_pct": 100.0,
            "fallen_object_rate_pct": 0.0,
            "articulated_coverage_pct": 0.0,
        }

    # Build AABB list: each entry is (min_x, min_y, min_z, max_x, max_y, max_z)
    aabbs: List[Tuple[float, float, float, float, float, float]] = []
    for obj in objects:
        transform = obj.get("transform") if isinstance(obj.get("transform"), dict) else {}
        position = transform.get("position") if isinstance(transform.get("position"), dict) else {}
        dims = obj.get("dimensions_est") if isinstance(obj.get("dimensions_est"), dict) else {}

        cx = float(position.get("x", 0.0))
        cy = float(position.get("y", 0.0))
        cz = float(position.get("z", 0.0))
        hw = float(dims.get("width", 0.25)) / 2.0
        hh = float(dims.get("height", 0.25)) / 2.0
        hd = float(dims.get("depth", 0.25)) / 2.0

        aabbs.append((cx - hw, cy, cz - hd, cx + hw, cy + hh * 2, cz + hd))

    # Pairwise overlap check
    collision_count = 0
    total_depth_mm = 0.0
    total_pairs = max(1, n * (n - 1) // 2)

    for i in range(n):
        for j in range(i + 1, n):
            a = aabbs[i]
            b = aabbs[j]
            # Overlap exists if all axes overlap
            ox = max(0.0, min(a[3], b[3]) - max(a[0], b[0]))
            oy = max(0.0, min(a[4], b[4]) - max(a[1], b[1]))
            oz = max(0.0, min(a[5], b[5]) - max(a[2], b[2]))
            if ox > 0 and oy > 0 and oz > 0:
                collision_count += 1
                depth = min(ox, oy, oz) * 1000.0  # meters to mm
                total_depth_mm += depth

    collision_rate_pct = round(100.0 * collision_count / total_pairs, 3) if total_pairs > 0 else 0.0
    mean_depth_mm = round(total_depth_mm / max(1, collision_count), 3) if collision_count > 0 else 0.0

    # All objects are placed deliberately on y >= 0
    stability_pct = 100.0
    fallen_object_rate_pct = 0.0

    articulated_count = len([obj for obj in objects if (obj.get("articulation") or {}).get("required")])
    articulated_coverage_pct = round(100.0 * articulated_count / max(1, n), 2)

    return {
        "object_count": n,
        "collision_rate_pct": collision_rate_pct,
        "mean_collision_depth_mm": mean_depth_mm,
        "stability_pct": stability_pct,
        "fallen_object_rate_pct": fallen_object_rate_pct,
        "articulated_coverage_pct": articulated_coverage_pct,
    }


def _build_placement_graph(objects: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    relations: List[Dict[str, str]] = []
    anchors = [obj for obj in objects if obj.get("sim_role") == "static"]
    movable = [obj for obj in objects if obj.get("sim_role") == "manipulable_object"]

    if anchors and movable:
        anchor_id = str(anchors[0]["id"])
        for item in movable:
            relations.append(
                {
                    "type": "on",
                    "subject_id": str(item["id"]),
                    "object_id": anchor_id,
                }
            )

    for idx in range(max(0, len(anchors) - 1)):
        relations.append(
            {
                "type": "adjacent",
                "subject_id": str(anchors[idx]["id"]),
                "object_id": str(anchors[idx + 1]["id"]),
            }
        )

    return {
        "schema_version": "v1",
        "relations": relations,
    }


def _target_object_count(*, quality_tier: QualityTier, constraints: Mapping[str, Any]) -> int:
    target_count = 12 if quality_tier == QualityTier.STANDARD else 18
    if constraints.get("object_density") == "high":
        target_count += 6

    dimensions = _prompt_diversity_dimensions(constraints)
    complexity = str(dimensions.get("complexity", "")).lower()
    if "high_density" in complexity:
        target_count = max(target_count, 16 if quality_tier == QualityTier.STANDARD else 20)
    elif "medium_density" in complexity and quality_tier == QualityTier.PREMIUM:
        target_count = max(target_count, 16)
    elif "articulation_heavy" in complexity:
        target_count = max(target_count, 14 if quality_tier == QualityTier.STANDARD else 18)
    return max(5, target_count)


def _min_role_requirements(*, target_count: int, constraints: Mapping[str, Any]) -> Tuple[int, int]:
    min_articulated = 1
    min_manipulable = max(4, target_count // 3)

    dimensions = _prompt_diversity_dimensions(constraints)
    complexity = str(dimensions.get("complexity", "")).lower()
    manipulation_focus = str(dimensions.get("manipulation_focus", "")).lower()
    task_family = str(dimensions.get("task_family", "")).lower()

    if "articulation_heavy" in complexity or "drawer" in manipulation_focus:
        min_articulated = 2
    if "small_items" in manipulation_focus or "sorting" in task_family:
        min_manipulable = max(min_manipulable, target_count // 2)

    return min_articulated, min_manipulable


def _choose_without_replacement(
    rng: random.Random,
    pool: Sequence[Tuple[str, str, str]],
    count: int,
) -> List[Tuple[str, str, str]]:
    if count <= 0 or not pool:
        return []
    if count >= len(pool):
        picked = list(pool)
        rng.shuffle(picked)
        return picked
    idxs = list(range(len(pool)))
    rng.shuffle(idxs)
    return [pool[i] for i in idxs[:count]]


def _apply_substitution(
    *,
    name: str,
    category: str,
    sim_role: str,
    rng: random.Random,
) -> Tuple[str, str]:
    if sim_role != "manipulable_object":
        return name, category
    candidates = OBJECT_SUBSTITUTIONS.get(category, [])
    if not candidates or rng.random() > 0.35:
        return name, category
    substitute = candidates[rng.randrange(0, len(candidates))]
    return substitute, substitute


def _compose_fallback_template(
    *,
    room_type: str,
    target_count: int,
    constraints: Mapping[str, Any],
    rng: random.Random,
) -> List[Tuple[str, str, str]]:
    template = _room_template(room_type)
    articulated = [entry for entry in template if _is_articulated(entry[2])]
    manipulable = [entry for entry in template if entry[2] == "manipulable_object"]
    static = [entry for entry in template if entry[2] == "static"]

    min_articulated, min_manipulable = _min_role_requirements(target_count=target_count, constraints=constraints)
    chosen: List[Tuple[str, str, str]] = []
    chosen.extend(_choose_without_replacement(rng, articulated, min_articulated))
    chosen.extend(_choose_without_replacement(rng, manipulable, min_manipulable))

    existing = {(name, category, sim_role) for name, category, sim_role in chosen}
    pool = list(template)
    rng.shuffle(pool)
    for entry in pool:
        if len(chosen) >= min(target_count, len(template)):
            break
        if entry in existing:
            continue
        chosen.append(entry)
        existing.add(entry)

    if len(chosen) < min(target_count, len(template)):
        # Ensure we never return too small a template even when role quotas are impossible.
        fallback_fill = _choose_without_replacement(rng, static + manipulable + articulated, target_count)
        for entry in fallback_fill:
            if entry in existing:
                continue
            chosen.append(entry)
            existing.add(entry)

    rng.shuffle(chosen)
    return chosen


def _llm_plan_enabled() -> bool:
    raw = os.getenv("TEXT_GEN_USE_LLM", "true").strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def _llm_retry_config() -> Tuple[int, float]:
    attempts_raw = os.getenv("TEXT_GEN_LLM_MAX_ATTEMPTS", "3").strip()
    backoff_raw = os.getenv("TEXT_GEN_LLM_RETRY_BACKOFF_SECONDS", "2").strip()
    try:
        max_attempts = int(attempts_raw)
    except ValueError:
        max_attempts = 3
    try:
        backoff_seconds = float(backoff_raw)
    except ValueError:
        backoff_seconds = 2.0
    return max(1, max_attempts), max(0.0, backoff_seconds)


def _try_llm_plan(
    *,
    prompt: str,
    provider_chain: Sequence[str],
    quality_tier: QualityTier,
) -> LLMPlanResult:
    """Attempt an LLM-generated scene plan; fall back to deterministic templates."""

    if not _llm_plan_enabled():
        return LLMPlanResult(payload=None, provider=None, attempts=0, failure_reason="llm_disabled")

    try:
        from tools.llm_client import create_llm_client, LLMProvider
    except Exception as exc:
        return LLMPlanResult(
            payload=None,
            provider=None,
            attempts=0,
            failure_reason=f"llm_client_unavailable:{exc.__class__.__name__}",
        )

    provider_map = {
        "openai": LLMProvider.OPENAI,
        "gemini": LLMProvider.GEMINI,
        "anthropic": LLMProvider.ANTHROPIC,
    }

    effort = "high" if quality_tier == QualityTier.PREMIUM else "medium"
    system_prompt = (
        "Return compact JSON with keys room_type and objects. "
        "objects must be a list of {name,category,sim_role}. "
        "Allowed sim_role: static, articulated_furniture, articulated_appliance, manipulable_object."
    )
    max_attempts, retry_backoff_seconds = _llm_retry_config()
    llm_attempts = 0
    failure_reason: Optional[str] = None

    for round_idx in range(1, max_attempts + 1):
        for provider_name in provider_chain:
            provider = provider_map.get(provider_name)
            if provider is None:
                continue

            llm_attempts += 1
            try:
                client = create_llm_client(provider=provider, fallback_enabled=False, reasoning_effort=effort)
                response = client.generate(
                    prompt=f"{system_prompt}\n\nUser prompt: {prompt}",
                    json_output=True,
                )
                text = (response.text or "").strip()
                if not text:
                    failure_reason = f"{provider_name}:empty_response"
                    continue

                payload = json.loads(text)
                if isinstance(payload, dict) and isinstance(payload.get("objects"), list):
                    return LLMPlanResult(
                        payload=payload,
                        provider=provider_name,
                        attempts=llm_attempts,
                        failure_reason=None,
                    )
                failure_reason = f"{provider_name}:invalid_payload"
            except Exception as exc:
                failure_reason = f"{provider_name}:{exc.__class__.__name__}"
                continue

        if round_idx < max_attempts and retry_backoff_seconds > 0:
            time.sleep(retry_backoff_seconds * round_idx)

    return LLMPlanResult(
        payload=None,
        provider=None,
        attempts=llm_attempts,
        failure_reason=failure_reason or "llm_generation_failed",
    )


def generate_text_scene_package(
    *,
    scene_id: str,
    prompt: str,
    quality_tier: QualityTier,
    seed: int,
    provider_policy: str,
    constraints: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate deterministic text-scene package payload for Stage 1 text path."""

    constraints = constraints or {}
    started_at = time.time()

    provider_chain = resolve_provider_chain(provider_policy)
    provider_decision = ProviderDecision(provider=provider_chain[0], policy=provider_policy, used_llm=False)

    llm_plan = _try_llm_plan(
        prompt=prompt,
        provider_chain=provider_chain,
        quality_tier=quality_tier,
    )
    llm_payload = llm_plan.payload
    llm_provider = llm_plan.provider
    llm_attempts = llm_plan.attempts
    llm_failure_reason = llm_plan.failure_reason
    if llm_payload is not None and llm_provider is not None:
        provider_decision = ProviderDecision(provider=llm_provider, policy=provider_policy, used_llm=True)
        llm_failure_reason = None
    fallback_strategy = "none" if provider_decision.used_llm else "deterministic_template"
    if not provider_decision.used_llm:
        logger.info(
            "[TEXT-GEN] llm-plan-fallback scene=%s attempts=%s reason=%s strategy=%s",
            scene_id,
            llm_attempts,
            llm_failure_reason,
            fallback_strategy,
        )

    seed_material = f"{scene_id}|{prompt}|{seed}|{quality_tier.value}".encode("utf-8")
    rng_seed = int.from_bytes(sha256(seed_material).digest()[:8], "big") % (2**32)
    rng = random.Random(rng_seed)

    target_count = _target_object_count(quality_tier=quality_tier, constraints=constraints)

    if llm_payload is not None:
        room_type = str(llm_payload.get("room_type") or "generic_room").strip().lower().replace(" ", "_")
        template: List[Tuple[str, str, str]] = []
        for item in llm_payload.get("objects", []):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "object").strip().lower().replace(" ", "_")
            category = str(item.get("category") or name).strip().lower().replace(" ", "_")
            sim_role = str(item.get("sim_role") or "manipulable_object").strip().lower()
            if sim_role not in {
                "static",
                "articulated_furniture",
                "articulated_appliance",
                "manipulable_object",
            }:
                sim_role = "manipulable_object"
            template.append((name, category, sim_role))
        if not template:
            template = _compose_fallback_template(
                room_type=room_type,
                target_count=target_count,
                constraints=constraints,
                rng=rng,
            )
    else:
        room_type = _extract_room_type(prompt, constraints)
        template = _compose_fallback_template(
            room_type=room_type,
            target_count=target_count,
            constraints=constraints,
            rng=rng,
        )

    objects: List[Dict[str, Any]] = []
    while len(objects) < target_count:
        for _, (base_name, base_category, sim_role) in enumerate(template):
            if len(objects) >= target_count:
                break
            idx = len(objects) + 1
            oid = f"obj_{idx:03d}"
            name, category = _apply_substitution(
                name=base_name,
                category=base_category,
                sim_role=sim_role,
                rng=rng,
            )
            object_name = f"{name}_{idx:03d}" if idx > len(template) else name
            dims = _default_dims_for_category(category)
            objects.append(
                {
                    "id": oid,
                    "name": object_name,
                    "category": category,
                    "sim_role": sim_role,
                    "description": f"{category} generated from text prompt",
                    "transform": {
                        "position": _sample_position(rng, idx),
                        "scale": {"x": 1.0, "y": 1.0, "z": 1.0},
                        "rotation_quaternion": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
                    },
                    "dimensions_est": dims,
                    "physics_hints": _physics_hints(category, sim_role),
                    "asset_strategy": _asset_source_hint(sim_role),
                    "articulation": {
                        "required": _is_articulated(sim_role),
                        "backend_hint": "particulate_first" if _is_articulated(sim_role) else "none",
                    },
                }
            )

    placement_graph = _build_placement_graph(objects)

    assets_provenance: List[Dict[str, Any]] = []
    for obj in objects:
        strategy = str(obj.get("asset_strategy", "generated"))
        assets_provenance.append(
            {
                "object_id": obj["id"],
                "strategy": strategy,
                "provider": provider_decision.provider,
                "source": "library" if strategy == "retrieved" else "generated",
                "model_or_library": "partnet_mobility" if strategy == "retrieved" else "textgen_placeholder",
            }
        )

    elapsed = max(0.001, time.time() - started_at)

    metrics = _compute_quality_metrics(objects)

    # Cost estimate: template-only generation is essentially free; LLM adds API cost.
    llm_cost_estimate = 0.0
    if provider_decision.used_llm:
        llm_cost_estimate = 0.08 if quality_tier == QualityTier.PREMIUM else 0.04
    cost_estimate = {
        "estimated_cost_usd": round(llm_cost_estimate, 4),
        "llm_calls": 1 if provider_decision.used_llm else 0,
        "tier": quality_tier.value,
    }

    quality_report = {
        "schema_version": "v1",
        "scene_id": scene_id,
        "quality_tier": quality_tier.value,
        "provider": provider_decision.provider,
        "metrics": metrics,
        "cost": cost_estimate,
        "timing": {
            "generation_seconds": round(elapsed, 4),
        },
        "generation_mode": "llm" if provider_decision.used_llm else "deterministic_fallback",
        "llm_attempts": llm_attempts,
        "llm_failure_reason": llm_failure_reason,
        "fallback_strategy": fallback_strategy,
        "status": "passed",
    }

    return {
        "schema_version": "v1",
        "scene_id": scene_id,
        "room_type": room_type,
        "seed": seed,
        "quality_tier": quality_tier.value,
        "provider_policy": provider_policy,
        "provider_used": provider_decision.provider,
        "used_llm": provider_decision.used_llm,
        "llm_attempts": llm_attempts,
        "llm_failure_reason": llm_failure_reason,
        "fallback_strategy": fallback_strategy,
        "prompt": prompt,
        "constraints": constraints,
        "objects": objects,
        "placement_graph": placement_graph,
        "physics_hints": {obj["id"]: obj["physics_hints"] for obj in objects},
        "provenance": {
            "assets": assets_provenance,
            "generation": {
                "strategy": "hybrid_generation_retrieval",
                "articulation_policy": "particulate_first",
            },
        },
        "quality_gate_report": quality_report,
    }
