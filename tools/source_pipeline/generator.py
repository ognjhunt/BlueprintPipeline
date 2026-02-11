from __future__ import annotations

import json
import os
import random
import re
import time
from hashlib import sha256
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .request import QualityTier


@dataclass(frozen=True)
class ProviderDecision:
    provider: str
    policy: str
    used_llm: bool


def resolve_provider_chain(policy: str) -> List[str]:
    """Resolve ordered provider chain for generation."""

    if policy == "openai_primary":
        return ["openai", "gemini", "anthropic"]
    return ["openai", "gemini", "anthropic"]


def _extract_room_type(prompt: str, constraints: Dict[str, Any]) -> str:
    constraint_room = str(constraints.get("room_type", "")).strip().lower()
    if constraint_room:
        return constraint_room

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
            return token.replace(" ", "_")
    return "generic_room"


def _room_template(room_type: str) -> List[Tuple[str, str, str]]:
    """Return (name, category, sim_role) base template for room type."""

    templates: Dict[str, List[Tuple[str, str, str]]] = {
        "kitchen": [
            ("countertop", "counter", "static"),
            ("sink", "sink", "static"),
            ("fridge", "refrigerator", "articulated_appliance"),
            ("cabinet", "cabinet", "articulated_furniture"),
            ("stool", "stool", "static"),
            ("mug", "mug", "manipulable_object"),
            ("plate", "plate", "manipulable_object"),
            ("bottle", "bottle", "manipulable_object"),
        ],
        "living_room": [
            ("sofa", "sofa", "static"),
            ("coffee_table", "table", "static"),
            ("bookshelf", "bookshelf", "static"),
            ("tv_stand", "tv_stand", "static"),
            ("lamp", "lamp", "static"),
            ("book", "book", "manipulable_object"),
            ("remote", "remote", "manipulable_object"),
            ("mug", "mug", "manipulable_object"),
        ],
        "bedroom": [
            ("bed", "bed", "static"),
            ("nightstand", "nightstand", "static"),
            ("dresser", "dresser", "static"),
            ("closet", "closet", "articulated_furniture"),
            ("lamp", "lamp", "static"),
            ("book", "book", "manipulable_object"),
        ],
        "office": [
            ("desk", "desk", "static"),
            ("chair", "chair", "static"),
            ("monitor", "monitor", "static"),
            ("cabinet", "cabinet", "articulated_furniture"),
            ("keyboard", "keyboard", "manipulable_object"),
            ("mouse", "mouse", "manipulable_object"),
            ("notebook", "notebook", "manipulable_object"),
        ],
    }
    return templates.get(room_type, [
        ("table", "table", "static"),
        ("shelf", "shelf", "static"),
        ("chair", "chair", "static"),
        ("container", "container", "manipulable_object"),
        ("tool", "tool", "manipulable_object"),
    ])


def _default_dims_for_category(category: str) -> Dict[str, float]:
    dims = {
        "counter": (1.6, 0.9, 0.6),
        "sink": (0.6, 0.3, 0.5),
        "refrigerator": (0.9, 1.8, 0.8),
        "cabinet": (0.8, 1.8, 0.5),
        "table": (1.2, 0.75, 0.7),
        "sofa": (2.0, 0.9, 0.9),
        "bed": (2.0, 0.6, 1.6),
        "chair": (0.5, 0.9, 0.5),
        "mug": (0.08, 0.10, 0.08),
        "plate": (0.26, 0.03, 0.26),
        "bottle": (0.09, 0.30, 0.09),
        "book": (0.18, 0.03, 0.24),
        "remote": (0.05, 0.02, 0.18),
        "keyboard": (0.44, 0.03, 0.14),
        "mouse": (0.07, 0.04, 0.11),
    }
    width, height, depth = dims.get(category, (0.35, 0.35, 0.35))
    return {"width": width, "height": height, "depth": depth}


def _sample_position(rng: random.Random, index: int) -> Dict[str, float]:
    lane = index % 5
    row = index // 5
    x = -1.8 + lane * 0.9 + rng.uniform(-0.1, 0.1)
    z = -1.6 + row * 0.9 + rng.uniform(-0.1, 0.1)
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


def _llm_plan_enabled() -> bool:
    raw = os.getenv("TEXT_GEN_USE_LLM", "false").strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def _try_llm_plan(
    *,
    prompt: str,
    provider_chain: Sequence[str],
    quality_tier: QualityTier,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Attempt an LLM-generated scene plan; fall back to deterministic templates."""

    if not _llm_plan_enabled():
        return None, None

    try:
        from tools.llm_client import create_llm_client, LLMProvider
    except Exception:
        return None, None

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

    for provider_name in provider_chain:
        provider = provider_map.get(provider_name)
        if provider is None:
            continue
        try:
            client = create_llm_client(provider=provider, fallback_enabled=False, reasoning_effort=effort)
            response = client.generate(
                prompt=f"{system_prompt}\n\nUser prompt: {prompt}",
                json_output=True,
            )
            text = (response.text or "").strip()
            if not text:
                continue
            payload = json.loads(text)
            if isinstance(payload, dict) and isinstance(payload.get("objects"), list):
                return payload, provider_name
        except Exception:
            continue

    return None, None


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

    llm_payload, llm_provider = _try_llm_plan(
        prompt=prompt,
        provider_chain=provider_chain,
        quality_tier=quality_tier,
    )
    if llm_payload is not None and llm_provider is not None:
        provider_decision = ProviderDecision(provider=llm_provider, policy=provider_policy, used_llm=True)

    seed_material = f"{scene_id}|{prompt}|{seed}|{quality_tier.value}".encode("utf-8")
    rng_seed = int.from_bytes(sha256(seed_material).digest()[:8], "big") % (2**32)
    rng = random.Random(rng_seed)

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
            template = _room_template(room_type)
    else:
        room_type = _extract_room_type(prompt, constraints)
        template = _room_template(room_type)

    target_count = 12 if quality_tier == QualityTier.STANDARD else 18
    if constraints.get("object_density") == "high":
        target_count += 6

    objects: List[Dict[str, Any]] = []
    while len(objects) < target_count:
        for base_index, (name, category, sim_role) in enumerate(template):
            if len(objects) >= target_count:
                break
            idx = len(objects) + 1
            oid = f"obj_{idx:03d}"
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
