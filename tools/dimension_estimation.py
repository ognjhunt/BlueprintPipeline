"""
Shared dimension estimation utility with Gemini-first, cache, graceful fallback.

Returns (dimensions, source) tuples for provenance tracking.

Sources:
  - "object_data": dimensions from object metadata
  - "gemini_estimated": Gemini LLM estimation
  - "hardcoded_fallback": default [0.1, 0.1, 0.1]
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_DIMENSIONS = [0.1, 0.1, 0.1]

try:
    from tools.llm_client import create_llm_client
    _HAVE_LLM_CLIENT = True
except ImportError:
    _HAVE_LLM_CLIENT = False
    create_llm_client = None


class DimensionEstimator:
    """Estimate object dimensions with Gemini fallback and caching."""

    def __init__(self):
        self._llm_client = None
        self._llm_init_attempted = False
        self._cache: Dict[str, List[float]] = {}

    def _get_llm_client(self):
        if not self._llm_init_attempted:
            self._llm_init_attempted = True
            if _HAVE_LLM_CLIENT:
                try:
                    self._llm_client = create_llm_client()
                except Exception as exc:
                    logger.debug("LLM client init failed: %s", exc)
                    self._llm_client = None
        return self._llm_client

    def estimate_dimensions(
        self,
        obj: Dict[str, Any],
        fallback: Optional[List[float]] = None,
    ) -> Tuple[List[float], str]:
        """Estimate dimensions for an object dict.

        Tries in order:
          1. obj["dimensions"] or obj["bbox"]
          2. Gemini estimation by name/category
          3. fallback (default [0.1, 0.1, 0.1])

        Returns:
            (dimensions, source) where source is one of
            "object_data", "gemini_estimated", "hardcoded_fallback"
        """
        if fallback is None:
            fallback = list(DEFAULT_DIMENSIONS)

        # 1. Try object data
        dims = obj.get("dimensions") if isinstance(obj, dict) else None
        if isinstance(dims, (list, tuple)) and len(dims) >= 3:
            return list(dims[:3]), "object_data"
        bbox = obj.get("bbox") if isinstance(obj, dict) else None
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 3:
            return list(bbox[:3]), "object_data"

        # 2. Try Gemini
        identifier = None
        if isinstance(obj, dict):
            identifier = (
                obj.get("name")
                or obj.get("object_id")
                or obj.get("class")
                or obj.get("category")
                or obj.get("semantic_class")
                or obj.get("id")
            )
        if identifier:
            gemini_dims = self._estimate_gemini(str(identifier))
            if gemini_dims is not None:
                return gemini_dims, "gemini_estimated"

        # 3. Fallback
        logger.warning(
            "DIMENSION_FALLBACK: Using default dimensions %s for object %s",
            fallback,
            obj.get("id") or obj.get("name") or "unknown" if isinstance(obj, dict) else "unknown",
        )
        return list(fallback), "hardcoded_fallback"

    def _estimate_gemini(self, identifier: str) -> Optional[List[float]]:
        """Estimate dimensions via Gemini, with caching."""
        cache_key = identifier.lower().strip()
        if cache_key in self._cache:
            return list(self._cache[cache_key])

        llm = self._get_llm_client()
        if not llm:
            return None

        prompt = (
            f"Estimate the typical bounding box dimensions [width, depth, height] "
            f"in meters for a '{identifier}' object commonly found in indoor environments. "
            f"Return ONLY a JSON object: {{\"dimensions\": [0.2, 0.2, 0.1]}}"
        )

        _rate_limit_backoffs = [30, 60, 120]
        _attempt = 0
        while True:
            try:
                response = llm.generate(prompt, json_output=True, temperature=0.3)
                data = response.parse_json()
                if isinstance(data, dict) and "dimensions" in data:
                    dims = data["dimensions"]
                    if isinstance(dims, (list, tuple)) and len(dims) >= 3:
                        result = [float(d) for d in dims[:3]]
                        if all(0.001 < d < 10.0 for d in result):
                            self._cache[cache_key] = result
                            logger.info(
                                "GEMINI_DIMENSIONS: Estimated %s for '%s'",
                                result, identifier,
                            )
                            return list(result)
                # Parsed but invalid — no retry needed
                return None
            except Exception as exc:
                exc_str = str(exc).lower()
                is_rate_limit = any(
                    k in exc_str for k in ("429", "quota", "resource_exhausted", "rate")
                )
                if is_rate_limit and _attempt < len(_rate_limit_backoffs):
                    wait = _rate_limit_backoffs[_attempt]
                    logger.warning(
                        "GEMINI_RATE_LIMIT: Hit quota for '%s' — "
                        "retry %d/%d in %ds: %s",
                        identifier, _attempt + 1, len(_rate_limit_backoffs), wait, exc,
                    )
                    time.sleep(wait)
                    _attempt += 1
                    continue
                logger.debug(
                    "Gemini dimension estimation failed for '%s': %s",
                    identifier, exc,
                )
                return None


_estimator: Optional[DimensionEstimator] = None


def get_dimension_estimator() -> DimensionEstimator:
    """Get or create the singleton DimensionEstimator."""
    global _estimator
    if _estimator is None:
        _estimator = DimensionEstimator()
    return _estimator
