"""
Asset Matcher - Matches scene plan objects to catalog assets.

This module takes object descriptions from the scene plan and finds
the best matching assets from the indexed catalog using:
- Tag-based search
- Dimension matching
- Embedding similarity (when available)
- Style/attribute matching
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .catalog_builder import AssetCatalog, AssetEntry


@dataclass
class AssetMatch:
    """Represents a matched asset with confidence score."""

    asset_id: str
    asset_path: str
    score: float
    match_reasons: list[str]
    dimensions: Optional[dict[str, float]] = None
    variants: Optional[dict[str, str]] = None


@dataclass
class MatchResult:
    """Result of matching an object to assets."""

    object_id: str
    candidates: list[AssetMatch]
    chosen: Optional[AssetMatch] = None
    warnings: list[str] = None
    debug: Optional[dict[str, Any]] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class AssetMatcher:
    """
    Matches scene plan objects to catalog assets.

    Scoring factors:
    - Category match: +0.3
    - Tag overlap: +0.1 per matching tag (max 0.3)
    - Dimension compatibility: +0.2
    - Style attributes: +0.1 per match
    - Embedding similarity: weighted contribution from text/image embeddings (when available)
    """

    DEFAULT_CATEGORY_SYNONYMS: dict[str, list[str]] = {
        "sofa": ["couch", "loveseat", "sectional"],
        "chair": ["seat", "stool", "armchair", "bench"],
        "table": ["desk", "dining", "workbench"],
        "lamp": ["light", "lighting", "sconce", "chandelier"],
        "appliance": [
            "refrigerator",
            "fridge",
            "stove",
            "oven",
            "microwave",
            "dishwasher",
            "washer",
            "dryer",
        ],
        "cabinet": ["cupboard", "storage", "shelf", "shelving", "bookcase", "bookshelf"],
        "decor": [
            "art",
            "painting",
            "frame",
            "poster",
            "mirror",
            "vase",
            "plant",
            "rug",
            "carpet",
            "curtain",
        ],
        "bed": ["bunk", "mattress", "cot"],
        "sink": ["basin"],
        "bathtub": ["bath", "tub"],
        "trash": ["garbage", "bin", "waste"],
        "tv": ["television", "monitor"],
    }

    def __init__(
        self,
        catalog: AssetCatalog,
        embeddings_db: Optional[Any] = None,
        auto_select_floor: float = 0.35,
        auto_select_ceiling: float = 0.5,
        enable_embedding_similarity: bool = True,
        enable_image_similarity: bool = True,
        embedding_weight: float = 0.2,
        image_similarity_weight: float = 0.15,
        debug: Optional[bool] = None,
    ):
        self.catalog = catalog
        self.embeddings_db = embeddings_db
        self.enable_embedding_similarity = enable_embedding_similarity
        self.enable_image_similarity = enable_image_similarity
        self.embedding_weight = embedding_weight
        self.image_similarity_weight = image_similarity_weight

        if debug is None:
            self.debug = bool(os.getenv("ASSET_MATCH_DEBUG"))
        else:
            self.debug = debug

        self.auto_select_floor = max(0.0, auto_select_floor)
        self.auto_select_ceiling = max(self.auto_select_floor, auto_select_ceiling)

        # Build search indices
        self._build_indices()
        self._category_lookup = self._build_category_lookup()

        self._logger = logging.getLogger(__name__)

    def _build_indices(self) -> None:
        """Build search indices for fast lookup."""
        self.category_index: dict[str, list[AssetEntry]] = {}
        self.tag_index: dict[str, list[AssetEntry]] = {}

        for asset in self.catalog.assets:
            # Category index
            cat = asset.category.lower()
            if cat not in self.category_index:
                self.category_index[cat] = []
            self.category_index[cat].append(asset)

            # Tag index
            for tag in asset.tags:
                tag_lower = tag.lower()
                if tag_lower not in self.tag_index:
                    self.tag_index[tag_lower] = []
                self.tag_index[tag_lower].append(asset)

    def match(
        self,
        object_spec: dict[str, Any],
        top_k: int = 5,
        dimension_tolerance: float = 0.3,
        dimension_filter_ratio: float = 3.0,
    ) -> MatchResult:
        """
        Match an object specification to catalog assets.

        Args:
            object_spec: Object from scene plan with category, description, etc.
            top_k: Number of top candidates to return
            dimension_tolerance: Fraction tolerance for dimension matching
            dimension_filter_ratio: Multiplier threshold for excluding candidates
                that are significantly larger or smaller than estimated dimensions

        Returns:
            MatchResult with ranked candidates
        """
        object_id = object_spec.get("id", "unknown")
        category = object_spec.get("category", "").lower()
        raw_description = object_spec.get("description", "") or ""
        description = raw_description
        synthesized_description = False
        if not description.strip():
            description = self.build_description(object_spec)
            synthesized_description = True
        attributes = object_spec.get("attributes", {})
        est_dims = object_spec.get("estimated_dimensions", {})
        image_query_emb = self._get_image_embedding(object_spec)
        text_query_emb = None
        if self.embeddings_db and description:
            text_query_emb = self.embeddings_db.embed_text(description)

        candidates = []
        warnings = []
        debug_info: dict[str, Any] | None = None

        if self.debug:
            debug_info = {
                "object_id": object_id,
                "input_category": category,
                "used_description": description,
                "description_source": "synthesized" if synthesized_description else "provided",
                "estimated_dimensions": est_dims,
                "auto_select": {
                    "floor": self.auto_select_floor,
                    "ceiling": self.auto_select_ceiling,
                },
            }

        # Get candidate assets
        candidate_assets = self._get_candidates(
            category, description, est_dims, dimension_filter_ratio
        )

        if not candidate_assets:
            warnings.append(f"No assets found for category '{category}'")
            if debug_info is not None:
                debug_info["candidates"] = []
                debug_info["warnings"] = warnings.copy()
                self._logger.info(
                    "[ASSET_MATCH_DEBUG] %s",
                    json.dumps(debug_info, default=str),
                )
            return MatchResult(
                object_id=object_id,
                candidates=[],
                warnings=warnings,
                debug=debug_info,
            )

        # Score each candidate
        for asset in candidate_assets:
            score, reasons = self._score_asset(
                asset,
                category,
                description,
                attributes,
                est_dims,
                dimension_tolerance,
                text_query_emb,
                image_query_emb,
            )

            variants, variant_reasons = self._select_best_variant(asset, attributes)
            reasons.extend(variant_reasons)

            candidates.append(
                AssetMatch(
                    asset_id=asset.asset_id,
                    asset_path=asset.relative_path,
                    score=score,
                    match_reasons=reasons,
                    dimensions=asset.dimensions,
                    variants=variants,
                )
            )

        # Sort by score descending
        candidates.sort(key=lambda x: x.score, reverse=True)

        # Take top k
        candidates = candidates[:top_k]

        # Auto-select best if above threshold
        chosen = None
        auto_select_threshold: float | None = None
        if candidates:
            top_score = candidates[0].score
            auto_select_threshold = candidates[0].score if self.auto_select_ceiling < 0 else self.auto_select_ceiling
            if top_score < auto_select_threshold:
                auto_select_threshold = max(self.auto_select_floor, top_score)

            if top_score >= auto_select_threshold:
                chosen = candidates[0]
                if top_score < self.auto_select_ceiling:
                    warn_msg = (
                        "Auto-selected low-confidence candidate; please review."
                    )
                    warnings.append(warn_msg)
                    self._logger.info(
                        (
                            "Auto-selected low-confidence candidate for object '%s' "
                            "(score=%.3f, floor=%.2f, ceiling=%.2f)"
                        ),
                        object_id,
                        top_score,
                        self.auto_select_floor,
                        self.auto_select_ceiling,
                    )

        if debug_info is not None:
            debug_info["candidates"] = [
                {
                    "asset_id": c.asset_id,
                    "asset_path": c.asset_path,
                    "score": c.score,
                    "reasons": c.match_reasons,
                    "dimensions": c.dimensions,
                    "variants": c.variants,
                }
                for c in candidates
            ]
            debug_info["auto_select"].update(
                {
                    "top_score": candidates[0].score if candidates else None,
                    "threshold_used": auto_select_threshold if candidates else None,
                    "chosen_asset_id": chosen.asset_id if chosen else None,
                }
            )
            if warnings:
                debug_info["warnings"] = warnings.copy()

            self._logger.info(
                "[ASSET_MATCH_DEBUG] %s",
                json.dumps(debug_info, default=str),
            )

        return MatchResult(
            object_id=object_id,
            candidates=candidates,
            chosen=chosen,
            warnings=warnings,
            debug=debug_info,
        )

    def _filter_by_dimensions(
        self,
        candidates: list[AssetEntry],
        est_dims: dict[str, float],
        dimension_filter_ratio: float,
    ) -> list[AssetEntry]:
        """Filter out candidates that are far from estimated dimensions."""
        if not est_dims:
            return candidates

        filtered: list[AssetEntry] = []

        for asset in candidates:
            if not asset.dimensions:
                filtered.append(asset)
                continue

            incompatible = False
            for dim, est_val in est_dims.items():
                if est_val is None or est_val <= 0:
                    continue
                if dim not in asset.dimensions:
                    continue

                asset_val = asset.dimensions[dim]
                if asset_val <= 0:
                    continue

                if (
                    asset_val > est_val * dimension_filter_ratio
                    or asset_val < est_val / dimension_filter_ratio
                ):
                    incompatible = True
                    break

            if not incompatible:
                filtered.append(asset)

        return filtered

    def match_batch(self, objects: list[dict[str, Any]], top_k: int = 5) -> dict[str, MatchResult]:
        """Match multiple objects to assets."""
        results = {}
        for obj in objects:
            result = self.match(obj, top_k=top_k)
            results[obj.get("id", "unknown")] = result
        return results

    def _get_candidates(
        self,
        category: str,
        description: str,
        est_dims: Optional[dict[str, float]] = None,
        dimension_filter_ratio: float = 3.0,
    ) -> list[AssetEntry]:
        """Get candidate assets based on category and description."""
        candidates = set()

        normalized_categories = self._resolve_categories(category, description)

        # First, try normalized category match
        for norm_cat in normalized_categories:
            for asset in self.category_index.get(norm_cat, []):
                candidates.add(asset.asset_id)

        # Also search by tags from description
        desc_words = self._tokenize(description)
        for word in desc_words:
            if word in self.tag_index:
                for asset in self.tag_index[word]:
                    candidates.add(asset.asset_id)

        # If still empty, try broader category matching
        if not candidates and category:
            for cat, assets in self.category_index.items():
                if category in cat or cat in category:
                    for asset in assets:
                        candidates.add(asset.asset_id)

        # If still empty and embeddings are available, use semantic search
        if not candidates and self.embeddings_db and description:
            emb_results = self.embeddings_db.search(description, top_k=10, threshold=0.15)
            for asset_id, _ in emb_results:
                candidates.add(asset_id)

        # Convert to asset entries
        candidate_entries = [asset for asset in self.catalog.assets if asset.asset_id in candidates]

        filtered_candidates = self._filter_by_dimensions(
            candidate_entries, est_dims or {}, dimension_filter_ratio
        )

        if est_dims and candidate_entries and not filtered_candidates:
            self._logger.info(
                "All candidates filtered out due to dimension ratio %.2f",
                dimension_filter_ratio,
            )

        return filtered_candidates

    @classmethod
    def build_description(cls, object_spec: dict[str, Any]) -> str:
        """Construct a useful description from category, type, and properties."""

        def _stringify_property_value(value: Any) -> str:
            if value is None:
                return ""
            if isinstance(value, str):
                return value.strip()
            if isinstance(value, (int, float, bool)):
                return str(value)
            if isinstance(value, dict):
                parts = []
                for key, val in value.items():
                    val_str = _stringify_property_value(val)
                    if val_str:
                        parts.append(f"{key} {val_str}".strip())
                return " ".join(parts)
            if isinstance(value, list):
                return " ".join(filter(None, (_stringify_property_value(v) for v in value)))
            return str(value)

        category = (object_spec.get("category") or "").strip()
        obj_type = (object_spec.get("type") or object_spec.get("subcategory") or "").strip()
        properties = (object_spec.get("properties") or object_spec.get("attributes") or {})

        parts: list[str] = []
        if category:
            parts.append(category)
        if obj_type:
            parts.append(obj_type)

        if isinstance(properties, dict):
            for key, value in properties.items():
                value_text = _stringify_property_value(value)
                if value_text:
                    parts.append(f"{key} {value_text}".strip())

        synonyms = []
        if category:
            synonyms.extend(cls.DEFAULT_CATEGORY_SYNONYMS.get(category.lower(), []))

        description = " ".join(filter(None, parts + synonyms)).strip()
        return description or category or obj_type or "object"

    def _resolve_categories(self, category: str, description: str) -> list[str]:
        """Map scene categories to known catalog categories using synonyms."""
        resolved: list[str] = []

        if category:
            normalized = self._category_lookup.get(category.lower())
            if normalized:
                resolved.append(normalized)

        # Leverage description tokens to map to known categories
        for token in self._tokenize(description):
            normalized = self._category_lookup.get(token)
            if normalized:
                resolved.append(normalized)

        # Fallback: approximate category match when synonyms miss
        if not resolved and category:
            approx = self._find_known_category(category)
            if approx:
                resolved.append(approx)

        # Deduplicate while preserving order
        seen = set()
        unique_resolved = []
        for cat in resolved:
            if cat not in seen:
                seen.add(cat)
                unique_resolved.append(cat)

        return unique_resolved

    def _build_category_lookup(self) -> dict[str, str]:
        """Build lookup table mapping synonyms to known catalog categories."""
        lookup: dict[str, str] = {}

        for canonical, synonyms in self.DEFAULT_CATEGORY_SYNONYMS.items():
            known = self._find_known_category(canonical)
            if not known:
                continue

            lookup.setdefault(canonical, known)
            for syn in synonyms:
                lookup.setdefault(syn.lower(), known)

        # Ensure direct category access
        for cat in self.category_index.keys():
            lookup.setdefault(cat, cat)

        return lookup

    def _find_known_category(self, label: str) -> Optional[str]:
        """Find the closest known category to a label."""
        label_lower = label.lower()
        if label_lower in self.category_index:
            return label_lower

        for cat in self.category_index.keys():
            if label_lower in cat or cat in label_lower:
                return cat

        return None

    def _score_asset(
        self,
        asset: AssetEntry,
        category: str,
        description: str,
        attributes: dict[str, Any],
        est_dims: dict[str, float],
        dim_tolerance: float,
        text_query_emb: Optional[np.ndarray] = None,
        image_query_emb: Optional[np.ndarray] = None,
    ) -> tuple[float, list[str]]:
        """Score an asset for a given object specification."""
        score = 0.0
        reasons = []

        # Category match
        if asset.category.lower() == category:
            score += 0.3
            reasons.append("category_exact_match")
        elif category in asset.category.lower() or asset.category.lower() in category:
            score += 0.15
            reasons.append("category_partial_match")

        # Tag overlap
        desc_words = set(self._tokenize(description))
        asset_tags = set(t.lower() for t in asset.tags)
        overlap = desc_words & asset_tags
        tag_score = min(len(overlap) * 0.1, 0.3)
        if tag_score > 0:
            score += tag_score
            reasons.append(f"tags_matched:{','.join(overlap)}")

        # Dimension matching
        if asset.dimensions and est_dims:
            dim_score = self._score_dimensions(asset.dimensions, est_dims, dim_tolerance)
            if dim_score > 0:
                score += dim_score
                reasons.append("dimensions_compatible")

        # Attribute matching
        if attributes:
            attr_score = self._score_attributes(asset, attributes)
            if attr_score > 0:
                score += attr_score
                reasons.append("attributes_matched")

        # SimReady bonus
        if asset.simready_metadata and asset.simready_metadata.get("is_simready"):
            score += 0.1
            reasons.append("simready")

        # Embedding similarity
        if self.embeddings_db and self.enable_embedding_similarity and text_query_emb is not None:
            asset_emb = self.embeddings_db.get_embedding(asset.asset_id, kind="text")
            if asset_emb is not None:
                similarity = self._cosine_similarity(text_query_emb, asset_emb)
                score += similarity * self.embedding_weight
                reasons.append(f"embedding_similarity:{similarity:.3f}")

        if self.embeddings_db and self.enable_image_similarity and image_query_emb is not None:
            thumb_emb = self.embeddings_db.get_embedding(asset.asset_id, kind="thumbnail")
            if thumb_emb is not None:
                similarity = self._cosine_similarity(image_query_emb, thumb_emb)
                score += similarity * self.image_similarity_weight
                reasons.append(f"image_similarity:{similarity:.3f}")

        return score, reasons

    def _cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        if vec_a is None or vec_b is None:
            return 0.0

        denom = (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)) + 1e-8
        if denom == 0:
            return 0.0

        return float(np.dot(vec_a, vec_b) / denom)

    def _score_dimensions(self, asset_dims: dict[str, float], est_dims: dict[str, float], tolerance: float) -> float:
        """Score dimension compatibility."""
        if not asset_dims or not est_dims:
            return 0.0

        matches = 0
        total = 0

        for dim in ["width", "depth", "height"]:
            if dim in asset_dims and dim in est_dims:
                asset_val = asset_dims[dim]
                est_val = est_dims[dim]

                if est_val > 0:
                    ratio = asset_val / est_val
                    if (1 - tolerance) <= ratio <= (1 + tolerance):
                        matches += 1
                total += 1

        if total == 0:
            return 0.0

        return (matches / total) * 0.2

    def _score_attributes(self, asset: AssetEntry, attributes: dict[str, Any]) -> float:
        """Score attribute matching (material, color, style)."""
        score = 0.0

        # Material matching
        if "material" in attributes:
            material = attributes["material"].lower()
            if any(material in m.lower() for m in asset.materials):
                score += 0.05
            if any(material in t.lower() for t in asset.tags):
                score += 0.05

        # Color matching (from tags)
        if "color" in attributes:
            color = attributes["color"].lower()
            if any(color in t.lower() for t in asset.tags):
                score += 0.05

        # Style matching
        if "style" in attributes:
            style = attributes["style"].lower()
            if any(style in t.lower() for t in asset.tags):
                score += 0.05

        return min(score, 0.2)

    def _get_image_embedding(self, object_spec: dict[str, Any]) -> Optional[np.ndarray]:
        """Extract or normalize a provided image embedding."""
        image_emb = object_spec.get("image_embedding") or object_spec.get("thumbnail_embedding")
        if image_emb is None:
            return None

        arr = np.array(image_emb, dtype=np.float32)
        if arr.ndim != 1:
            return None
        return arr

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into searchable words."""
        import re

        # Split on non-alphanumeric
        words = re.split(r"[^a-zA-Z0-9]+", text.lower())
        # Filter short words
        return [w for w in words if len(w) > 2]

    def _select_best_variant(
        self, asset: AssetEntry, attributes: dict[str, Any]
    ) -> tuple[dict[str, str], list[str]]:
        """
        Choose the best variant selections based on object attributes.

        Prefers variants whose names include attribute values (material/color/style),
        falling back to declared defaults or the first available option.
        """

        variants: dict[str, str] = {}
        reasons: list[str] = []

        if not asset.variant_sets:
            return variants, reasons

        attributes = attributes or {}

        attr_values = {
            k: str(v).lower()
            for k, v in attributes.items()
            if k in {"material", "color", "style"} and isinstance(v, str)
        }

        for vset in asset.variant_sets:
            vset_variants = vset.get("variants") or []
            if not vset_variants:
                continue

            default_variant = vset.get("default") or vset_variants[0]
            best_variant = default_variant
            best_score = -1
            best_matches: list[str] = []

            for variant in vset_variants:
                variant_lower = variant.lower()
                score = 0
                matches: list[str] = []

                for key in ("material", "color", "style"):
                    attr_val = attr_values.get(key)
                    if attr_val and attr_val in variant_lower:
                        score += 1
                        matches.append(f"{key}:{attr_val}")

                if score > best_score:
                    best_score = score
                    best_variant = variant
                    best_matches = matches

            if best_score <= 0:
                best_variant = default_variant
                reasons.append(f"variant_{vset['name']}:default_used:{best_variant}")
            else:
                reasons.append(
                    f"variant_{vset['name']}:matched:{','.join(best_matches)}->{best_variant}"
                )

            variants[vset["name"]] = best_variant

        return variants, reasons

    def _get_default_variants(self, asset: AssetEntry) -> dict[str, str]:
        """Get default variant selections for an asset."""
        variants = {}
        for vset in asset.variant_sets:
            if vset.get("default"):
                variants[vset["name"]] = vset["default"]
            elif vset.get("variants"):
                variants[vset["name"]] = vset["variants"][0]
        return variants

    def to_matched_assets(self, results: dict[str, MatchResult]) -> dict[str, dict[str, Any]]:
        """Convert match results to format expected by downstream compilers."""
        matched = {}
        for obj_id, result in results.items():
            if result.chosen:
                matched[obj_id] = {
                    "chosen_path": result.chosen.asset_path,
                    "candidates": [
                        {
                            "asset_path": c.asset_path,
                            "score": c.score,
                            "dimensions": c.dimensions,
                        }
                        for c in result.candidates
                    ],
                    "variants": result.chosen.variants or {},
                    "pack_name": self.catalog.pack_name,
                }
            else:
                matched[obj_id] = {
                    "chosen_path": "",
                    "candidates": [
                        {
                            "asset_path": c.asset_path,
                            "score": c.score,
                            "dimensions": c.dimensions,
                        }
                        for c in result.candidates
                    ],
                    "variants": {},
                    "pack_name": self.catalog.pack_name,
                    "needs_selection": True,
                }
        return matched
