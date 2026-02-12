from __future__ import annotations

from pathlib import Path

import numpy as np

from tools.asset_catalog.vector_store import VectorRecord, VectorStoreClient, VectorStoreConfig
from tools.source_pipeline.asset_retrieval import AssetRetrievalService
from tools.source_pipeline.asset_retrieval_rollout import effective_retrieval_mode, update_rollout_state


def _write_usd(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('#usda 1.0\n(\n    defaultPrim = "Root"\n)\ndef Xform "Root" {}\n', encoding="utf-8")


def _lexical_none(*, obj, library_entries, used_paths):  # noqa: ANN001
    return None


def _lexical_first(*, obj, library_entries, used_paths):  # noqa: ANN001
    for item in library_entries:
        if item["path_rel"] not in used_paths:
            return item
    return None


def test_ann_decision_selects_best_candidate(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TEXT_ASSET_ANN_ENABLED", "true")
    monkeypatch.setenv("TEXT_ASSET_ANN_MIN_SCORE", "0.1")
    monkeypatch.setenv("TEXT_ASSET_ANN_TOP_K", "10")

    good_path = tmp_path / "asset-library" / "kitchen" / "mug_good.usd"
    alt_path = tmp_path / "asset-library" / "kitchen" / "chair_alt.usd"
    _write_usd(good_path)
    _write_usd(alt_path)

    vector_client = VectorStoreClient(VectorStoreConfig(provider="in-memory", collection="assets-v1"))
    vector_client.upsert(
        [
            VectorRecord(
                id="asset-good:text",
                embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                metadata={
                    "asset_id": "asset-good",
                    "usd_path": "asset-library/kitchen/mug_good.usd",
                    "class_name": "mug",
                    "sim_roles": ["manipulable_object"],
                    "dimensions": {"width": 0.08, "height": 0.1, "depth": 0.08},
                },
            ),
            VectorRecord(
                id="asset-alt:text",
                embedding=np.array([0.1, 0.9, 0.0], dtype=np.float32),
                metadata={
                    "asset_id": "asset-alt",
                    "usd_path": "asset-library/kitchen/chair_alt.usd",
                    "class_name": "chair",
                    "sim_roles": ["static"],
                },
            ),
        ],
        namespace="assets-v1",
    )

    service = AssetRetrievalService(root=tmp_path, retrieval_mode="ann_primary")
    monkeypatch.setattr(service, "_vector_store", lambda: vector_client)
    monkeypatch.setattr(service, "_embed_query", lambda _: np.array([1.0, 0.0, 0.0], dtype=np.float32))

    decision = service.select_asset(
        scene_id="scene_ann",
        room_type="kitchen",
        obj={
            "id": "obj_001",
            "name": "mug",
            "category": "mug",
            "sim_role": "manipulable_object",
            "description": "ceramic mug near a sink",
            "dimensions_est": {"width": 0.08, "height": 0.1, "depth": 0.08},
        },
        library_entries=[],
        used_paths=set(),
        lexical_selector=_lexical_none,
    )

    assert decision.method == "ann"
    assert decision.selected is not None
    assert decision.selected.path_rel == "asset-library/kitchen/mug_good.usd"
    assert decision.semantic_score is not None and decision.semantic_score >= 0.9


def test_lexical_fallback_engages_when_ann_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TEXT_ASSET_ANN_ENABLED", "true")
    monkeypatch.setenv("TEXT_ASSET_ANN_MIN_SCORE", "0.9")
    monkeypatch.setenv("TEXT_ASSET_LEXICAL_FALLBACK_ENABLED", "true")

    usd_path = tmp_path / "asset-library" / "kitchen" / "plate_001.usd"
    _write_usd(usd_path)

    library_entries = [
        {
            "path": usd_path,
            "path_rel": "asset-library/kitchen/plate_001.usd",
            "tokens": {"plate"},
        }
    ]

    service = AssetRetrievalService(root=tmp_path, retrieval_mode="ann_primary")
    monkeypatch.setattr(
        service,
        "_ann_candidates",
        lambda **_: ([], "ann_query_failed:missing_index", 2.0),
    )

    decision = service.select_asset(
        scene_id="scene_fallback",
        room_type="kitchen",
        obj={
            "id": "obj_plate",
            "name": "plate",
            "category": "plate",
            "sim_role": "manipulable_object",
        },
        library_entries=library_entries,
        used_paths=set(),
        lexical_selector=_lexical_first,
    )

    assert decision.method == "lexical"
    assert decision.fallback_reason == "ann_unavailable_or_no_match"
    assert decision.selected is not None
    assert decision.selected.path_rel.endswith("plate_001.usd")


def test_ann_can_select_with_zero_lexical_overlap(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TEXT_ASSET_ANN_ENABLED", "true")
    monkeypatch.setenv("TEXT_ASSET_ANN_MIN_SCORE", "0.2")

    mesh_path = tmp_path / "asset-library" / "misc" / "mesh_abc.usd"
    _write_usd(mesh_path)

    vector_client = VectorStoreClient(VectorStoreConfig(provider="in-memory", collection="assets-v1"))
    vector_client.upsert(
        [
            VectorRecord(
                id="asset-semantic:text",
                embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                metadata={
                    "asset_id": "asset-semantic",
                    "usd_path": "asset-library/misc/mesh_abc.usd",
                    "class_name": "unknown_item",
                    "sim_roles": ["manipulable_object"],
                },
            )
        ],
        namespace="assets-v1",
    )

    service = AssetRetrievalService(root=tmp_path, retrieval_mode="ann_primary")
    monkeypatch.setattr(service, "_vector_store", lambda: vector_client)
    monkeypatch.setattr(service, "_embed_query", lambda _: np.array([1.0, 0.0, 0.0], dtype=np.float32))

    decision = service.select_asset(
        scene_id="scene_semantic",
        room_type="lab",
        obj={
            "id": "obj_target",
            "name": "beaker",
            "category": "beaker",
            "sim_role": "manipulable_object",
        },
        library_entries=[],
        used_paths=set(),
        lexical_selector=_lexical_none,
    )

    assert decision.method == "ann"
    assert decision.selected is not None
    assert decision.selected.lexical_score < 0.2
    assert decision.semantic_score is not None and decision.semantic_score >= 0.9


def test_rollout_promotes_to_ann_primary_after_passing_windows(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TEXT_ASSET_RETRIEVAL_MODE", "ann_shadow")
    monkeypatch.setenv("TEXT_ASSET_ROLLOUT_MIN_DECISIONS", "2")
    monkeypatch.setenv("TEXT_ASSET_ROLLOUT_MIN_HIT_RATE", "0.95")
    monkeypatch.setenv("TEXT_ASSET_ROLLOUT_MAX_ERROR_RATE", "0.01")
    monkeypatch.setenv("TEXT_ASSET_ROLLOUT_MAX_P95_MS", "400")
    monkeypatch.setenv("TEXT_ASSET_ROLLOUT_MIN_PASSING_WINDOWS", "3")

    window = [
        {"ann_attempted": True, "ann_candidate_count": 1, "ann_error": None, "ann_latency_ms": 120.0},
        {"ann_attempted": True, "ann_candidate_count": 1, "ann_error": None, "ann_latency_ms": 140.0},
    ]

    for _ in range(3):
        update_rollout_state(root=tmp_path, decisions=list(window))

    assert effective_retrieval_mode(tmp_path) == "ann_primary"


def test_ann_clip_rerank_populates_clip_scores(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TEXT_ASSET_ANN_ENABLED", "true")
    monkeypatch.setenv("TEXT_ASSET_ANN_MIN_SCORE", "0.1")
    monkeypatch.setenv("TEXT_ASSET_CLIP_RERANK_ENABLED", "true")

    plain_path = tmp_path / "asset-library" / "kitchen" / "mug_plain.usd"
    ceramic_path = tmp_path / "asset-library" / "kitchen" / "mug_ceramic.usd"
    _write_usd(plain_path)
    _write_usd(ceramic_path)

    vector_client = VectorStoreClient(VectorStoreConfig(provider="in-memory", collection="assets-v1"))
    vector_client.upsert(
        [
            VectorRecord(
                id="asset-plain:text",
                embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                metadata={
                    "asset_id": "asset-plain",
                    "usd_path": "asset-library/kitchen/mug_plain.usd",
                    "class_name": "mug",
                    "description": "plain travel mug",
                    "sim_roles": ["manipulable_object"],
                    "dimensions": {"width": 0.08, "height": 0.1, "depth": 0.08},
                },
            ),
            VectorRecord(
                id="asset-ceramic:text",
                embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                metadata={
                    "asset_id": "asset-ceramic",
                    "usd_path": "asset-library/kitchen/mug_ceramic.usd",
                    "class_name": "ceramic_mug",
                    "description": "ceramic stoneware mug",
                    "sim_roles": ["manipulable_object"],
                    "dimensions": {"width": 0.08, "height": 0.1, "depth": 0.08},
                },
            ),
        ],
        namespace="assets-v1",
    )

    service = AssetRetrievalService(root=tmp_path, retrieval_mode="ann_primary")
    monkeypatch.setattr(service, "_vector_store", lambda: vector_client)
    monkeypatch.setattr(service, "_embed_query", lambda _: np.array([1.0, 0.0, 0.0], dtype=np.float32))

    decision = service.select_asset(
        scene_id="scene_clip",
        room_type="kitchen",
        obj={
            "id": "obj_clip",
            "name": "mug",
            "category": "mug",
            "description": "ceramic stoneware mug next to sink",
            "sim_role": "manipulable_object",
            "dimensions_est": {"width": 0.08, "height": 0.1, "depth": 0.08},
        },
        library_entries=[],
        used_paths=set(),
        lexical_selector=_lexical_none,
    )

    assert decision.method == "ann"
    assert decision.selected is not None
    assert decision.selected.clip_score >= 0.1
    assert decision.ann_candidate_count == 2
    assert decision.ann_top_candidate is not None
    assert decision.ann_top_candidate.clip_score > 0.0


def test_ann_articulated_priority_prefers_articulated_candidate(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TEXT_ASSET_ANN_ENABLED", "true")
    monkeypatch.setenv("TEXT_ASSET_ANN_MIN_SCORE", "0.1")
    monkeypatch.setenv("TEXT_ASSET_CLIP_RERANK_ENABLED", "true")
    monkeypatch.setenv("TEXT_ASSET_ARTICULATED_PRIORITY_ENABLED", "true")

    articulated_path = tmp_path / "asset-library" / "kitchen" / "cabinet_articulated.usd"
    static_path = tmp_path / "asset-library" / "kitchen" / "cabinet_static.usd"
    _write_usd(articulated_path)
    _write_usd(static_path)

    vector_client = VectorStoreClient(VectorStoreConfig(provider="in-memory", collection="assets-v1"))
    vector_client.upsert(
        [
            VectorRecord(
                id="asset-art:text",
                embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                metadata={
                    "asset_id": "asset-art",
                    "usd_path": "asset-library/kitchen/cabinet_articulated.usd",
                    "class_name": "cabinet",
                    "sim_roles": ["articulated_furniture"],
                    "dimensions": {"width": 0.9, "height": 1.8, "depth": 0.5},
                },
            ),
            VectorRecord(
                id="asset-static:text",
                embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                metadata={
                    "asset_id": "asset-static",
                    "usd_path": "asset-library/kitchen/cabinet_static.usd",
                    "class_name": "cabinet",
                    "sim_roles": ["static"],
                    "dimensions": {"width": 0.9, "height": 1.8, "depth": 0.5},
                },
            ),
        ],
        namespace="assets-v1",
    )

    service = AssetRetrievalService(root=tmp_path, retrieval_mode="ann_primary")
    monkeypatch.setattr(service, "_vector_store", lambda: vector_client)
    monkeypatch.setattr(service, "_embed_query", lambda _: np.array([1.0, 0.0, 0.0], dtype=np.float32))

    decision = service.select_asset(
        scene_id="scene_art_priority",
        room_type="kitchen",
        obj={
            "id": "obj_cabinet",
            "name": "cabinet",
            "category": "cabinet",
            "sim_role": "articulated_furniture",
            "dimensions_est": {"width": 0.9, "height": 1.8, "depth": 0.5},
        },
        library_entries=[],
        used_paths=set(),
        lexical_selector=_lexical_none,
    )

    assert decision.method == "ann"
    assert decision.selected is not None
    assert decision.selected.path_rel == "asset-library/kitchen/cabinet_articulated.usd"
