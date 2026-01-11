"""Physics priors for assets using Gemini and metadata-driven heuristics."""

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


class PhysicsDefaults:
    """Category-aware physics priors derived from metadata and dimensions."""

    def __init__(
        self,
        gemini_client: Optional[GeminiClient] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.1,
    ) -> None:
        model = model_name or os.getenv("PHYSICS_GEMINI_MODEL", "gemini-3-pro-preview")
        client = gemini_client or GeminiClient(model_name=model)
        self.client: Optional[GeminiClient] = client if getattr(client, "api_key", None) else None
        self.temperature = temperature

    def generate(
        self,
        obj: Dict[str, Any],
        matched: Dict[str, Any],
        dimensions: Dict[str, float],
    ) -> Dict[str, Any]:
        """Produce physics priors using Gemini with metadata fallbacks."""

        response: Dict[str, Any] = {}
        metadata = self._collect_metadata(obj, matched)

        if self.client:
            try:
                prompt = self._build_prompt(metadata, dimensions)
                schema = self._output_schema()
                raw = self.client.generate(
                    prompt, response_schema=schema, temperature=self.temperature
                )
                response = self._parse_json(raw)
            except Exception:
                response = {}

        return self._sanitize(response, metadata, dimensions)

    def merge(self, priors: Dict[str, Any], overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge an overrides dict into priors, preserving prior fallbacks."""

        merged = dict(priors)
        if not overrides:
            return merged

        for key, value in overrides.items():
            if value is None:
                continue
            if key == "center_of_mass_offset" and not self._valid_com(value):
                continue
            merged[key] = value
        return merged

    def _collect_metadata(self, obj: Dict[str, Any], matched: Dict[str, Any]) -> Dict[str, Any]:
        simready = matched.get("simready_metadata") or {}
        candidates = matched.get("candidates") or []
        primary = candidates[0] if candidates else {}
        return {
            "id": obj.get("id"),
            "name": obj.get("name") or obj.get("label"),
            "category": obj.get("category") or matched.get("category"),
            "description": obj.get("description") or matched.get("description"),
            "material": obj.get("material") or matched.get("material") or simready.get("material"),
            "pack": matched.get("pack_name"),
            "asset_name": matched.get("asset_name") or matched.get("usd_path"),
            "keywords": matched.get("keywords") or primary.get("keywords"),
            "simready": simready,
        }

    def _build_prompt(self, metadata: Dict[str, Any], dimensions: Dict[str, float]) -> str:
        dims = {
            "width_m": dimensions.get("width"),
            "depth_m": dimensions.get("depth"),
            "height_m": dimensions.get("height"),
            "volume_m3": self._volume(dimensions),
        }

        schema = json.dumps(self._output_schema(), indent=2)
        return (
            "You help author physics for USD assets in NVIDIA Isaac Sim.\n"
            "Use asset_index metadata (category, materials, SimReady fields) to infer plausible values.\n"
            "Respond with JSON ONLY matching the schema; avoid placeholders or defaults.\n\n"
            f"Asset metadata: {json.dumps(metadata, ensure_ascii=False)}\n"
            f"Dimensions (meters): {json.dumps(dims)}\n"
            "Guidance:\n"
            "- mass_density_kg_m3 must match category/material scale; compute mass from the provided volume.\n"
            "- friction_static >= friction_dynamic; higher for soft/fabric, lower for metal/ceramic.\n"
            "- restitution typically between 0 and 0.3 for household items.\n"
            "- collision_approximation should hint at convexHull/convexDecomposition/boundingCube based on shape complexity.\n"
            "- center_of_mass_offset is relative to the geometric center in meters.\n\n"
            f"Return JSON that matches this schema exactly:\n{schema}"
        )

    def _output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "mass_density_kg_m3": {"type": "number"},
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
            "required": ["mass_density_kg_m3"],
        }

    def _parse_json(self, raw: str) -> Dict[str, Any]:
        text = raw.strip()
        if text.startswith("```"):
            lines = [ln for ln in text.splitlines() if not ln.strip().startswith("```")]
            text = "\n".join(lines)
        return json.loads(text)

    def _sanitize(
        self, data: Dict[str, Any], metadata: Dict[str, Any], dimensions: Dict[str, float]
    ) -> Dict[str, Any]:
        simready = metadata.get("simready") or {}
        density = self._coerce_number(data.get("mass_density_kg_m3"))
        volume = self._volume(dimensions)

        if density is None:
            density = self._fallback_density(metadata, volume)

        mass = None
        if density and density > 0 and volume and volume > 0:
            mass = density * volume
        elif simready.get("mass_kg"):
            mass = self._coerce_number(simready.get("mass_kg"))

        priors: Dict[str, Any] = {
            "enabled": True,
            "collision_enabled": True,
        }
        if mass and mass > 0:
            priors["mass_override"] = mass

        static_f = self._coerce_number(data.get("friction_static"))
        dynamic_f = self._coerce_number(data.get("friction_dynamic"))
        restitution = self._coerce_number(data.get("restitution"))

        if static_f is None or dynamic_f is None:
            static_f, dynamic_f = self._fallback_friction(metadata)

        if static_f is not None:
            priors["friction_static"] = static_f
        if dynamic_f is not None:
            priors["friction_dynamic"] = min(dynamic_f, static_f or dynamic_f)

        if restitution is None:
            restitution = self._fallback_restitution(metadata)
        if restitution is not None:
            priors["restitution"] = max(0.0, min(restitution, 1.0))

        collision_hint = data.get("collision_approximation") or simready.get("collision_approximation")
        if not collision_hint:
            collision_hint = self._fallback_collision_hint(dimensions)
        if collision_hint:
            priors["collision_approximation"] = str(collision_hint)

        com = data.get("center_of_mass_offset") or simready.get("center_of_mass_offset")
        if self._valid_com(com):
            priors["center_of_mass_offset"] = [float(x) for x in com]

        return priors

    def _coerce_number(self, value: Any) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _volume(self, dimensions: Dict[str, float]) -> Optional[float]:
        if not dimensions:
            return None
        w = float(dimensions.get("width", 0) or 0)
        d = float(dimensions.get("depth", 0) or 0)
        h = float(dimensions.get("height", 0) or 0)
        if min(w, d, h) <= 0:
            return None
        return w * d * h

    def _fallback_density(self, metadata: Dict[str, Any], volume: Optional[float]) -> Optional[float]:
        material = (metadata.get("material") or "").lower()
        category = (metadata.get("category") or "").lower()

        # GAP-PHYSICS-005 FIX: Expanded material property database
        material_density = {
            # Original materials
            "wood": 650,
            "oak": 750,
            "metal": 7800,
            "steel": 7850,
            "aluminum": 2700,
            "plastic": 950,
            "fabric": 400,
            "glass": 2500,
            "ceramic": 2400,
            "stone": 2600,
            "rubber": 1200,
            # New materials (P2-5)
            "foam": 200,  # Low-density polyurethane foam
            "cork": 240,  # Natural cork
            "leather": 900,  # Treated leather
            "paper": 700,  # Standard paper/cardboard
            "concrete": 2400,  # Portland cement concrete
            "composite": 1600,  # Carbon fiber reinforced polymer
            "fiberglass": 2000,  # Glass fiber reinforced polymer
            "titanium": 4500,  # Titanium alloy
            "electronics": 1200,  # Mixed electronics components
            "decor": 600,  # Generic decorative materials
            "kitchenware": 1400,  # Mix of stainless steel/ceramic/plastic
        }

        category_density = {
            "chair": 520,
            "table": 680,
            "couch": 450,
            "sofa": 450,
            "bed": 520,
            "cabinet": 720,
            "shelf": 680,
            "appliance": 2200,
            # New categories (P2-5)
            "box": 600,
            "container": 1000,
            "bag": 300,
            "bottle": 1100,
            "cup": 900,
            "utensil": 1200,
            "tool": 2000,
            "decoration": 600,
            "electronics_device": 1300,
            "textile": 350,
        }

        density = None
        for token, value in material_density.items():
            if token in material:
                density = value
                break

        if density is None:
            for token, value in category_density.items():
                if token in category:
                    density = value
                    break

        if density is None and volume:
            if volume < 0.01:
                density = 900
            elif volume < 0.05:
                density = 600
            else:
                density = 1000

        return density

    def _fallback_friction(self, metadata: Dict[str, Any]) -> tuple[Optional[float], Optional[float]]:
        material = (metadata.get("material") or "").lower()
        category = (metadata.get("category") or "").lower()

        # GAP-PHYSICS-005 FIX: Expanded friction properties for new materials
        if any(token in material for token in ("rubber", "fabric", "upholstery")):
            static, dynamic = 0.9, 0.8
        elif any(token in material for token in ("metal", "steel", "aluminum", "titanium")):
            static, dynamic = 0.35, 0.25
        elif "glass" in material or "ceramic" in material:
            static, dynamic = 0.4, 0.3
        elif "wood" in material or "oak" in material:
            static, dynamic = 0.6, 0.45
        elif "plastic" in material:
            static, dynamic = 0.55, 0.4
        elif "foam" in material or "cork" in material:
            static, dynamic = 0.45, 0.35  # Soft, compressible materials
        elif "leather" in material:
            static, dynamic = 0.65, 0.55  # Treated leather
        elif "paper" in material or "cardboard" in material:
            static, dynamic = 0.5, 0.4  # Paper products
        elif "concrete" in material or "stone" in material:
            static, dynamic = 0.7, 0.6  # Rough surfaces
        elif "composite" in material or "fiberglass" in material:
            static, dynamic = 0.45, 0.35  # Smooth composites
        elif "electronics" in material:
            static, dynamic = 0.4, 0.3  # Smooth surfaces
        elif "kitchenware" in material:
            static, dynamic = 0.5, 0.4  # Mixed materials
        elif "appliance" in category or "kitchenware" in category:
            static, dynamic = 0.45, 0.35
        else:
            static, dynamic = 0.5, 0.4

        return static, dynamic

    def _fallback_restitution(self, metadata: Dict[str, Any]) -> Optional[float]:
        material = (metadata.get("material") or "").lower()

        # GAP-PHYSICS-005 FIX: Expanded restitution for new materials
        if any(token in material for token in ("rubber",)):
            return 0.4
        if "plastic" in material:
            return 0.2
        if "metal" in material or "steel" in material or "titanium" in material:
            return 0.15
        if "glass" in material or "ceramic" in material:
            return 0.05
        if "foam" in material or "cork" in material:
            return 0.05  # Low bounciness for soft materials
        if "leather" in material:
            return 0.1  # Low restitution
        if "paper" in material or "cardboard" in material:
            return 0.05
        if "concrete" in material or "stone" in material:
            return 0.2  # Moderate for rough surfaces
        if "composite" in material or "fiberglass" in material:
            return 0.1
        if "electronics" in material:
            return 0.08  # Low bounciness
        if "kitchenware" in material or "aluminum" in material:
            return 0.12
        return 0.1

    def _fallback_collision_hint(self, dimensions: Dict[str, float]) -> str:
        width = float(dimensions.get("width", 0) or 0)
        depth = float(dimensions.get("depth", 0) or 0)
        height = float(dimensions.get("height", 0) or 0)

        largest = max(width, depth, height)
        smallest = min(width, depth, height)

        if smallest <= 0 or largest <= 0:
            return "convexHull"

        aspect_ratio = largest / smallest
        if aspect_ratio > 6:
            return "boundingCube"
        if aspect_ratio > 3:
            return "convexDecomposition"
        return "convexHull"

    def _valid_com(self, value: Any) -> bool:
        return isinstance(value, (list, tuple)) and len(value) == 3 and all(
            isinstance(v, (float, int)) for v in value
        )


__all__ = ["PhysicsDefaults"]
