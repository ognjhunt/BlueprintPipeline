"""Physics estimation via Gemini with sensible fallbacks."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

try:  # pragma: no cover - optional dependency
    from src.planning import GeminiClient  # type: ignore
except Exception:  # pragma: no cover - fallback for environments without Gemini
    class GeminiClient:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.api_key = None

        def generate(self, *args: Any, **kwargs: Any) -> str:
            raise RuntimeError("Gemini client is unavailable in this environment")

from .physics_defaults import PhysicsDefaults


class PhysicsEstimator:
    """Estimate physics properties for objects using Gemini or defaults."""

    def __init__(
        self,
        gemini_client: Optional[GeminiClient] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.15,
    ) -> None:
        model = model_name or os.getenv("PHYSICS_GEMINI_MODEL", "gemini-3-pro-preview")
        client = gemini_client or GeminiClient(model_name=model)
        self.client: Optional[GeminiClient] = client if getattr(client, "api_key", None) else None
        self.defaults = PhysicsDefaults(gemini_client=self.client, model_name=model, temperature=temperature)
        self.temperature = temperature

    def estimate(
        self, obj: Dict[str, Any], matched: Dict[str, Any], dimensions: Dict[str, float]
    ) -> Optional[Dict[str, Any]]:
        """Estimate physics values for an object using Gemini."""

        priors = self.defaults.generate(obj, matched, dimensions)

        if not self.client:
            return priors

        prompt = self._build_prompt(obj, matched, dimensions)
        schema = self._output_schema()

        try:
            raw = self.client.generate(prompt, response_schema=schema, temperature=self.temperature)
            data = self._parse_json(raw)
        except Exception:
            return priors

        ai_values = self._sanitize(data, dimensions)
        return self.defaults.merge(priors, ai_values)

    def _build_prompt(self, obj: Dict[str, Any], matched: Dict[str, Any], dimensions: Dict[str, float]) -> str:
        material = obj.get("material") or matched.get("material") or matched.get("description", "")
        context = {
            "id": obj.get("id"),
            "name": obj.get("name") or obj.get("label"),
            "category": obj.get("category"),
            "description": obj.get("description") or matched.get("description"),
            "material": material,
            "articulation": obj.get("articulation_type"),
        }

        dims = {
            "width_m": dimensions.get("width"),
            "depth_m": dimensions.get("depth"),
            "height_m": dimensions.get("height"),
        }

        schema = json.dumps(self._output_schema(), indent=2)
        return (
            "You are configuring physics for USD assets in NVIDIA Isaac Sim (PhysX).\n"
            "Provide realistic values tailored to this specific object, not generic placeholders.\n"
            "Use meters and kilograms. Respond with JSON ONLY matching the provided schema.\n\n"
            f"Known dimensions (meters): {json.dumps(dims)}\n"
            f"Object context: {json.dumps(context, ensure_ascii=False)}\n\n"
            "Guidance:\n"
            "- mass_kg must be plausible for the size/material (avoid default values).\n"
            "- friction_static >= friction_dynamic; restitution typically low (0-0.2).\n"
            "- collision_approximation should be a PhysX approximation token like convexHull, convexDecomposition, or boundingCube.\n"
            "- center_of_mass_offset is in meters relative to the geometric center.\n"
            "- If unsure, infer from appearance, category, and size—do not return placeholders.\n\n"
            f"Return JSON that matches this schema exactly:\n{schema}"
        )

    def _output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "mass_kg": {"type": "number"},
                "friction_static": {"type": "number"},
                "friction_dynamic": {"type": "number"},
                "restitution": {"type": "number"},
                "collision_approximation": {"type": "string"},
                "center_of_mass_offset": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                },
            },
            "required": ["mass_kg", "friction_static", "friction_dynamic", "restitution"],
        }

    def _parse_json(self, raw: str) -> Dict[str, Any]:
        text = raw.strip()
        if text.startswith("```"):
            lines = [ln for ln in text.splitlines() if not ln.strip().startswith("```")]
            text = "\n".join(lines)
        return json.loads(text)

    def _sanitize(self, data: Dict[str, Any], dimensions: Dict[str, float]) -> Dict[str, Any]:
        def _num(key: str, default: Optional[float] = None) -> Optional[float]:
            if key in data:
                try:
                    return float(data[key])
                except (TypeError, ValueError):
                    return default
            return default

        config: Dict[str, Any] = {
            "enabled": True,
            "collision_enabled": True,
        }

        mass = _num("mass_kg")
        if mass and mass > 0:
            config["mass_override"] = mass

        static_f = _num("friction_static")
        dynamic_f = _num("friction_dynamic")
        restitution = _num("restitution")

        if static_f is not None and dynamic_f is not None:
            if dynamic_f > static_f:
                dynamic_f = static_f
            config["friction_static"] = static_f
            config["friction_dynamic"] = dynamic_f

        if restitution is not None:
            config["restitution"] = max(0.0, min(restitution, 1.0))

        if "collision_approximation" in data:
            approx = str(data.get("collision_approximation") or "").strip()
            if approx:
                config["collision_approximation"] = approx

        com = data.get("center_of_mass_offset")
        if isinstance(com, (list, tuple)) and len(com) == 3:
            try:
                config["center_of_mass_offset"] = [float(x) for x in com]
            except (TypeError, ValueError):
                pass

        return config


__all__ = ["PhysicsEstimator"]
