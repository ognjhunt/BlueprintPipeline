from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


def _load_scene_graph_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "tools" / "geniesim_adapter" / "scene_graph.py"
    spec = importlib.util.spec_from_file_location("scene_graph", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load scene_graph module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


scene_graph = _load_scene_graph_module()
GenieSimNode = scene_graph.GenieSimNode
Pose = scene_graph.Pose
RelationInferencer = scene_graph.RelationInferencer
SceneGraphRuntimeConfig = scene_graph.SceneGraphRuntimeConfig


def _node(
    asset_id: str,
    position: list[float],
    size: list[float],
    bp_metadata: dict | None = None,
) -> GenieSimNode:
    return GenieSimNode(
        asset_id=asset_id,
        semantic=asset_id,
        size=size,
        pose=Pose(position=position, orientation=[1.0, 0.0, 0.0, 0.0]),
        task_tag=[],
        usd_path="",
        properties={},
        bp_metadata=bp_metadata or {},
    )


def test_on_in_edges_marked_heuristic_without_physics_metadata() -> None:
    config = SceneGraphRuntimeConfig(
        vertical_proximity_threshold=0.05,
        horizontal_proximity_threshold=0.15,
        alignment_angle_threshold=5.0,
        streaming_batch_size=100,
        enable_physics_validation=True,
        physics_contact_depth_threshold=0.001,
        physics_containment_ratio_threshold=0.8,
        heuristic_confidence_scale=0.5,
        allow_unvalidated_input=True,
        require_physics_validation=False,
    )
    inferencer = RelationInferencer(verbose=False, config=config)

    on_top = _node("box_top", [0.0, 0.0, 0.75], [0.5, 0.5, 0.5])
    on_base = _node("box_base", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    in_item = _node("toy", [5.0, 0.0, 0.0], [0.5, 0.5, 0.5])
    in_container = _node("bin", [5.0, 0.0, 0.0], [2.0, 2.0, 2.0])

    edges = inferencer.infer_relations([on_top, on_base, in_item, in_container])

    on_edge = next(
        edge for edge in edges
        if edge.relation == "on" and edge.source == "box_top" and edge.target == "box_base"
    )
    in_edge = next(
        edge for edge in edges
        if edge.relation == "in" and edge.source == "toy" and edge.target == "bin"
    )

    assert on_edge.metadata.get("inference_method") == "heuristic"
    assert on_edge.confidence == pytest.approx(0.8 * 0.5)
    assert in_edge.metadata.get("inference_method") == "heuristic"
    assert in_edge.confidence == pytest.approx(0.9 * 0.5)


def test_strict_physics_validation_raises_without_isaac(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scene_graph, "HAVE_PIPELINE_CONFIG", False)
    monkeypatch.setattr(scene_graph, "resolve_production_mode", lambda: True)
    monkeypatch.setattr(scene_graph, "_is_isaac_sim_available", lambda: False)
    monkeypatch.delenv("BP_SCENE_GRAPH_ALLOW_HEURISTICS_IN_PROD", raising=False)
    scene_graph._SCENE_GRAPH_CONFIG = None

    with pytest.raises(RuntimeError, match="Isaac Sim"):
        scene_graph._load_scene_graph_runtime_config()
