from __future__ import annotations

import os
import re
import time
import urllib.parse
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np

from tools.asset_catalog import (
    AssetCatalogClient,
    AssetEmbeddings,
    EmbeddingConfig,
    VectorStoreClient,
    VectorStoreConfig,
)

from .asset_retrieval_rollout import effective_retrieval_mode


_ASSET_EXTENSIONS = {".usd", ".usda", ".usdz"}


def _is_truthy(raw: Optional[str], *, default: bool) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _tokenize(text: str) -> List[str]:
    return [token for token in re.split(r"[^a-z0-9]+", text.lower()) if len(token) >= 2]


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    aset = set(a)
    bset = set(b)
    if not aset or not bset:
        return 0.0
    return float(len(aset & bset)) / float(len(aset | bset))


def _deterministic_embedding(text: str, dimension: int) -> np.ndarray:
    normalized = text.strip().encode("utf-8")
    digest = sha256(normalized).digest()
    seed = int.from_bytes(digest[:8], "big")
    rng = np.random.default_rng(seed)
    return rng.standard_normal(dimension).astype(np.float32)


def _normalize_vector_asset_id(vector_id: str) -> str:
    if vector_id.endswith(":text"):
        return vector_id[:-5]
    if vector_id.endswith(":thumbnail"):
        return vector_id[:-10]
    return vector_id


def _normalize_metadata_path(root: Path, value: str) -> Optional[str]:
    raw = value.strip()
    if not raw:
        return None

    if raw.startswith("gs://"):
        parsed = urllib.parse.urlparse(raw)
        raw = parsed.path.lstrip("/")

    candidate = Path(raw)
    if candidate.is_absolute():
        try:
            return candidate.relative_to(root).as_posix()
        except ValueError:
            return None

    rel = raw.lstrip("/")
    full = root / rel
    if full.is_file():
        return rel
    if full.is_dir():
        for path in sorted(full.rglob("*")):
            if path.is_file() and path.suffix.lower() in _ASSET_EXTENSIONS:
                return path.relative_to(root).as_posix()
        return None
    return None


def _dims_for_object(obj: Mapping[str, Any]) -> Dict[str, float]:
    dims = obj.get("dimensions_est") if isinstance(obj.get("dimensions_est"), Mapping) else {}
    return {
        "width": max(0.02, _safe_float(dims.get("width"), 0.25)),
        "height": max(0.02, _safe_float(dims.get("height"), 0.25)),
        "depth": max(0.02, _safe_float(dims.get("depth"), 0.25)),
    }


@dataclass(frozen=True)
class AssetQuerySpec:
    scene_id: str
    object_id: str
    room_type: str
    category: str
    name: str
    description: str
    sim_role: str
    articulation_required: bool
    dimensions: Dict[str, float]
    query_text: str
    tags: List[str]


@dataclass(frozen=True)
class RetrievalCandidate:
    asset_id: str
    path_rel: str
    semantic_score: float
    lexical_score: float
    class_role_score: float
    dimension_score: float
    total_score: float
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["semantic_score"] = round(float(self.semantic_score), 6)
        payload["lexical_score"] = round(float(self.lexical_score), 6)
        payload["class_role_score"] = round(float(self.class_role_score), 6)
        payload["dimension_score"] = round(float(self.dimension_score), 6)
        payload["total_score"] = round(float(self.total_score), 6)
        return payload


@dataclass(frozen=True)
class RetrievalDecision:
    mode: str
    method: str
    candidate_count: int
    latency_ms: float
    ann_attempted: bool
    ann_candidate_count: int
    ann_latency_ms: float
    ann_error: Optional[str]
    lexical_attempted: bool
    lexical_latency_ms: float
    fallback_reason: Optional[str]
    semantic_score: Optional[float]
    lexical_score: Optional[float]
    selected: Optional[RetrievalCandidate]
    ann_top_candidate: Optional[RetrievalCandidate]
    lexical_candidate: Optional[RetrievalCandidate]
    shadow_agreement: Optional[bool]

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "mode": self.mode,
            "method": self.method,
            "candidate_count": self.candidate_count,
            "latency_ms": round(float(self.latency_ms), 4),
            "ann_attempted": self.ann_attempted,
            "ann_candidate_count": self.ann_candidate_count,
            "ann_latency_ms": round(float(self.ann_latency_ms), 4),
            "ann_error": self.ann_error,
            "lexical_attempted": self.lexical_attempted,
            "lexical_latency_ms": round(float(self.lexical_latency_ms), 4),
            "fallback_reason": self.fallback_reason,
            "semantic_score": round(float(self.semantic_score), 6) if self.semantic_score is not None else None,
            "lexical_score": round(float(self.lexical_score), 6) if self.lexical_score is not None else None,
            "selected": self.selected.to_dict() if self.selected else None,
            "ann_top_candidate": self.ann_top_candidate.to_dict() if self.ann_top_candidate else None,
            "lexical_candidate": self.lexical_candidate.to_dict() if self.lexical_candidate else None,
            "shadow_agreement": self.shadow_agreement,
        }
        return payload

    def to_retrieved_entry(self, *, root: Path) -> Optional[Dict[str, Any]]:
        if self.selected is None:
            return None
        source_path = root / self.selected.path_rel
        if not source_path.is_file():
            return None
        return {
            "path": source_path,
            "path_rel": self.selected.path_rel,
            "score": self.selected.lexical_score,
        }


class AssetRetrievalService:
    def __init__(self, *, root: Path, retrieval_mode: Optional[str] = None):
        self.root = root
        self.mode = retrieval_mode or effective_retrieval_mode(root)
        if self.mode not in {"lexical_primary", "ann_shadow", "ann_primary"}:
            self.mode = "ann_shadow"

        self.ann_enabled = _is_truthy(os.getenv("TEXT_ASSET_ANN_ENABLED"), default=True)
        self.ann_top_k = max(1, _safe_int(os.getenv("TEXT_ASSET_ANN_TOP_K"), 40))
        self.ann_max_rerank = max(1, _safe_int(os.getenv("TEXT_ASSET_ANN_MAX_RERANK"), 20))
        self.ann_min_score = max(0.0, min(1.0, _safe_float(os.getenv("TEXT_ASSET_ANN_MIN_SCORE"), 0.28)))
        self.ann_namespace = (os.getenv("TEXT_ASSET_ANN_NAMESPACE") or os.getenv("VECTOR_STORE_NAMESPACE") or "assets-v1").strip()
        self.lexical_fallback_enabled = _is_truthy(os.getenv("TEXT_ASSET_LEXICAL_FALLBACK_ENABLED"), default=True)
        self.allow_duplicate_paths = _is_truthy(os.getenv("TEXT_ASSET_ALLOW_DUPLICATE_PATHS"), default=False)
        self.embedding_model = (os.getenv("TEXT_ASSET_EMBEDDING_MODEL") or "text-embedding-3-small").strip()
        self.embedding_backend = (os.getenv("TEXT_ASSET_EMBEDDING_BACKEND") or "openai").strip().lower()

        self._vector_client: Optional[VectorStoreClient] = None
        self._embedder: Optional[AssetEmbeddings] = None
        self._catalog_client: Optional[AssetCatalogClient] = None
        self._catalog_cache: Dict[str, Dict[str, Any]] = {}

    def _build_query_spec(self, *, scene_id: str, room_type: str, obj: Mapping[str, Any]) -> AssetQuerySpec:
        category = str(obj.get("category") or obj.get("name") or "object").strip().lower()
        name = str(obj.get("name") or category).strip().lower()
        description = str(obj.get("description") or "").strip()
        sim_role = str(obj.get("sim_role") or "manipulable_object").strip().lower()
        articulation_required = bool((obj.get("articulation") or {}).get("required", False))
        dimensions = _dims_for_object(obj)
        oid = str(obj.get("id") or "")
        tags = [category, name, sim_role, room_type]
        query_text = " ".join(
            part
            for part in [
                f"room:{room_type}",
                f"object:{name}",
                f"category:{category}",
                f"sim_role:{sim_role}",
                "articulation_required" if articulation_required else "",
                description,
            ]
            if part
        )
        return AssetQuerySpec(
            scene_id=scene_id,
            object_id=oid,
            room_type=room_type,
            category=category,
            name=name,
            description=description,
            sim_role=sim_role,
            articulation_required=articulation_required,
            dimensions=dimensions,
            query_text=query_text,
            tags=tags,
        )

    def _vector_store(self) -> VectorStoreClient:
        if self._vector_client is not None:
            return self._vector_client
        provider = (os.getenv("VECTOR_STORE_PROVIDER") or "vertex").strip()
        config = VectorStoreConfig.from_env(
            provider=provider,
            collection=self.ann_namespace,
            namespace=self.ann_namespace,
            dimension=_safe_int(os.getenv("VECTOR_STORE_DIMENSION"), 1536),
        )
        self._vector_client = VectorStoreClient(config)
        return self._vector_client

    def _embedding_client(self) -> AssetEmbeddings:
        if self._embedder is not None:
            return self._embedder
        api_key = (
            os.getenv("OPENAI_API_KEY")
            or os.getenv("TEXT_ASSET_OPENAI_API_KEY")
            or os.getenv("TEXT_GEN_OPENAI_API_KEY")
            or ""
        ).strip()
        config = EmbeddingConfig(
            backend=self.embedding_backend,
            model_name=self.embedding_model,
            api_key=api_key or None,
            dimension=_safe_int(os.getenv("VECTOR_STORE_DIMENSION"), 1536),
        )
        self._embedder = AssetEmbeddings(config=config)
        return self._embedder

    def _embed_query(self, query: str) -> np.ndarray:
        if self.embedding_backend in {"deterministic", "stub"}:
            dimension = max(8, _safe_int(os.getenv("VECTOR_STORE_DIMENSION"), 1536))
            return _deterministic_embedding(query, dimension)
        return self._embedding_client().embed_text(query)

    def _catalog(self) -> AssetCatalogClient:
        if self._catalog_client is None:
            self._catalog_client = AssetCatalogClient()
        return self._catalog_client

    def _metadata_for_asset(self, asset_id: str, seed: Mapping[str, Any]) -> Dict[str, Any]:
        metadata = dict(seed)
        cached = self._catalog_cache.get(asset_id)
        if cached is not None:
            merged = dict(cached)
            merged.update(metadata)
            return merged

        has_direct_path = isinstance(metadata.get("usd_path"), str) and bool(str(metadata.get("usd_path")).strip())
        has_direct_class = isinstance(metadata.get("class_name"), str) and bool(str(metadata.get("class_name")).strip())
        has_direct_roles = isinstance(metadata.get("sim_roles"), list) and len(metadata.get("sim_roles") or []) > 0
        if has_direct_path and has_direct_class and has_direct_roles:
            self._catalog_cache[asset_id] = dict(metadata)
            return metadata

        try:
            doc = self._catalog().fetch_asset_document(asset_id=asset_id)
        except Exception:
            doc = None
        if doc is None:
            self._catalog_cache[asset_id] = dict(metadata)
            return metadata

        enriched = dict(metadata)
        enriched.setdefault("asset_id", doc.asset_id)
        enriched.setdefault("usd_path", doc.usd_path or "")
        enriched.setdefault("gcs_uri", doc.gcs_uri or "")
        enriched.setdefault("class_name", doc.class_name or "")
        enriched.setdefault("description", doc.description or "")
        enriched.setdefault("sim_roles", list(doc.sim_roles or []))
        if isinstance(doc.extra_metadata, Mapping):
            for key, value in doc.extra_metadata.items():
                enriched.setdefault(str(key), value)
        self._catalog_cache[asset_id] = dict(enriched)
        return enriched

    def _candidate_tokens(self, metadata: Mapping[str, Any]) -> List[str]:
        parts: List[str] = []
        for key in ("asset_id", "class_name", "description", "usd_path", "path"):
            value = metadata.get(key)
            if isinstance(value, str):
                parts.extend(_tokenize(value))
        for key in ("sim_roles", "tags"):
            value = metadata.get(key)
            if isinstance(value, list):
                for item in value:
                    parts.extend(_tokenize(str(item)))
        return sorted(set(parts))

    def _class_role_score(self, spec: AssetQuerySpec, metadata: Mapping[str, Any], lexical_score: float) -> float:
        candidate_class = str(metadata.get("class_name") or "").strip().lower()
        class_score = lexical_score
        if candidate_class and (candidate_class == spec.category or candidate_class == spec.name):
            class_score = 1.0

        sim_roles = metadata.get("sim_roles")
        if not isinstance(sim_roles, list) or not sim_roles:
            role_score = 0.6
        else:
            role_score = 1.0 if spec.sim_role in [str(item).strip().lower() for item in sim_roles] else 0.0
        return (0.6 * class_score) + (0.4 * role_score)

    def _dimension_score(self, spec: AssetQuerySpec, metadata: Mapping[str, Any]) -> float:
        dims_raw = metadata.get("dimensions")
        if not isinstance(dims_raw, Mapping):
            dims_raw = metadata.get("dimensions_est")
        if not isinstance(dims_raw, Mapping):
            return 0.5

        width = _safe_float(dims_raw.get("width"), _safe_float(dims_raw.get("width_m"), 0.0))
        height = _safe_float(dims_raw.get("height"), _safe_float(dims_raw.get("height_m"), 0.0))
        depth = _safe_float(dims_raw.get("depth"), _safe_float(dims_raw.get("depth_m"), 0.0))
        if width <= 0 or height <= 0 or depth <= 0:
            return 0.5

        deltas = []
        for key, value in (("width", width), ("height", height), ("depth", depth)):
            target = max(0.02, _safe_float(spec.dimensions.get(key), 0.25))
            deltas.append(abs(value - target) / max(target, value, 1e-6))
        mean_delta = sum(deltas) / len(deltas)
        return max(0.0, min(1.0, 1.0 - mean_delta))

    def _ann_candidates(self, *, spec: AssetQuerySpec, used_paths: set[str]) -> tuple[list[RetrievalCandidate], Optional[str], float]:
        if not self.ann_enabled:
            return [], None, 0.0

        started = time.perf_counter()
        try:
            query_embedding = self._embed_query(spec.query_text)
            records = self._vector_store().query(
                query_embedding,
                top_k=self.ann_top_k,
                namespace=self.ann_namespace,
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - started) * 1000.0
            return [], f"ann_query_failed:{exc}", latency_ms

        query_tokens = _tokenize(spec.query_text)
        ranked: List[RetrievalCandidate] = []
        for record in records:
            asset_id = _normalize_vector_asset_id(str(record.id))
            metadata = self._metadata_for_asset(asset_id, dict(record.metadata or {}))
            metadata.setdefault("asset_id", asset_id)

            path_rel = None
            for key in ("usd_path", "path", "asset_path", "gcs_uri"):
                raw = metadata.get(key)
                if isinstance(raw, str):
                    path_rel = _normalize_metadata_path(self.root, raw)
                    if path_rel:
                        break
            if not path_rel:
                continue
            if (not self.allow_duplicate_paths) and path_rel in used_paths:
                continue

            full_path = self.root / path_rel
            if not full_path.is_file():
                continue

            semantic_score = max(0.0, min(1.0, _safe_float(record.score, 0.0)))
            if semantic_score < self.ann_min_score:
                continue

            candidate_tokens = self._candidate_tokens(metadata)
            lexical_score = _jaccard(query_tokens, candidate_tokens)
            class_role_score = self._class_role_score(spec, metadata, lexical_score)
            dimension_score = self._dimension_score(spec, metadata)
            total_score = (0.7 * semantic_score) + (0.2 * class_role_score) + (0.1 * dimension_score)
            ranked.append(
                RetrievalCandidate(
                    asset_id=asset_id,
                    path_rel=path_rel,
                    semantic_score=semantic_score,
                    lexical_score=lexical_score,
                    class_role_score=class_role_score,
                    dimension_score=dimension_score,
                    total_score=total_score,
                    metadata=metadata,
                )
            )

        ranked.sort(key=lambda item: item.total_score, reverse=True)
        latency_ms = (time.perf_counter() - started) * 1000.0
        return ranked[: self.ann_max_rerank], None, latency_ms

    def _lexical_candidate(
        self,
        *,
        spec: AssetQuerySpec,
        obj: Mapping[str, Any],
        library_entries: List[Dict[str, Any]],
        used_paths: set[str],
        lexical_selector: Callable[[Mapping[str, Any], List[Dict[str, Any]], set[str]], Optional[Dict[str, Any]]],
    ) -> tuple[Optional[RetrievalCandidate], float]:
        started = time.perf_counter()
        chosen = lexical_selector(obj=obj, library_entries=library_entries, used_paths=used_paths)
        latency_ms = (time.perf_counter() - started) * 1000.0
        if chosen is None:
            return None, latency_ms

        path_rel = str(chosen.get("path_rel") or "")
        if not path_rel:
            return None, latency_ms
        if (not self.allow_duplicate_paths) and path_rel in used_paths:
            return None, latency_ms
        source_path = Path(chosen["path"])
        if not source_path.is_file():
            return None, latency_ms

        lexical_score = max(0.0, min(1.0, _safe_float(chosen.get("score"), 0.0)))
        class_role_score = self._class_role_score(spec, {"class_name": spec.category, "sim_roles": [spec.sim_role]}, lexical_score)
        candidate = RetrievalCandidate(
            asset_id=str(chosen.get("asset_id") or f"lexical::{path_rel}"),
            path_rel=path_rel,
            semantic_score=0.0,
            lexical_score=lexical_score,
            class_role_score=class_role_score,
            dimension_score=0.5,
            total_score=lexical_score,
            metadata={"source": "lexical", "path_rel": path_rel},
        )
        return candidate, latency_ms

    def select_asset(
        self,
        *,
        scene_id: str,
        room_type: str,
        obj: Mapping[str, Any],
        library_entries: List[Dict[str, Any]],
        used_paths: set[str],
        lexical_selector: Callable[[Mapping[str, Any], List[Dict[str, Any]], set[str]], Optional[Dict[str, Any]]],
    ) -> RetrievalDecision:
        started = time.perf_counter()
        spec = self._build_query_spec(scene_id=scene_id, room_type=room_type, obj=obj)

        ann_attempted = self.ann_enabled and self.mode in {"ann_shadow", "ann_primary"}
        ann_candidates: List[RetrievalCandidate] = []
        ann_error: Optional[str] = None
        ann_latency_ms = 0.0
        if ann_attempted:
            ann_candidates, ann_error, ann_latency_ms = self._ann_candidates(spec=spec, used_paths=used_paths)
        ann_top = ann_candidates[0] if ann_candidates else None

        lexical_attempted = self.lexical_fallback_enabled
        lexical_candidate: Optional[RetrievalCandidate] = None
        lexical_latency_ms = 0.0
        if lexical_attempted:
            lexical_candidate, lexical_latency_ms = self._lexical_candidate(
                spec=spec,
                obj=obj,
                library_entries=library_entries,
                used_paths=used_paths,
                lexical_selector=lexical_selector,
            )

        selected: Optional[RetrievalCandidate] = None
        method = "none"
        fallback_reason: Optional[str] = None
        shadow_agreement: Optional[bool] = None

        if self.mode == "ann_primary":
            if ann_top is not None:
                selected = ann_top
                method = "ann"
            elif lexical_candidate is not None:
                selected = lexical_candidate
                method = "lexical"
                fallback_reason = "ann_unavailable_or_no_match"
        elif self.mode == "ann_shadow":
            if lexical_candidate is not None:
                selected = lexical_candidate
                method = "lexical"
            elif ann_top is not None:
                selected = ann_top
                method = "ann"
                fallback_reason = "lexical_miss"
            if ann_top is not None and lexical_candidate is not None:
                shadow_agreement = ann_top.path_rel == lexical_candidate.path_rel
        else:
            if lexical_candidate is not None:
                selected = lexical_candidate
                method = "lexical"
            elif ann_top is not None:
                selected = ann_top
                method = "ann"
                fallback_reason = "lexical_miss"

        latency_ms = (time.perf_counter() - started) * 1000.0
        return RetrievalDecision(
            mode=self.mode,
            method=method,
            candidate_count=(len(ann_candidates) + (1 if lexical_candidate is not None else 0)),
            latency_ms=latency_ms,
            ann_attempted=ann_attempted,
            ann_candidate_count=len(ann_candidates),
            ann_latency_ms=ann_latency_ms,
            ann_error=ann_error,
            lexical_attempted=lexical_attempted,
            lexical_latency_ms=lexical_latency_ms,
            fallback_reason=fallback_reason,
            semantic_score=selected.semantic_score if selected is not None else None,
            lexical_score=selected.lexical_score if selected is not None else None,
            selected=selected,
            ann_top_candidate=ann_top,
            lexical_candidate=lexical_candidate,
            shadow_agreement=shadow_agreement,
        )


__all__ = [
    "AssetQuerySpec",
    "RetrievalCandidate",
    "RetrievalDecision",
    "AssetRetrievalService",
]
