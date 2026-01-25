from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


class ExportConsistencyError(ValueError):
    """Raised when Genie Sim export artifacts are inconsistent."""


def _load_json(path: Path, context: str) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ExportConsistencyError(f"{context} missing at {path}.") from exc
    except json.JSONDecodeError as exc:
        raise ExportConsistencyError(
            f"Failed to parse {context} JSON at {path}: {exc}."
        ) from exc


def _parse_workspace_bounds(
    bounds: Any,
    *,
    context: str,
    path: Path,
) -> Optional[Tuple[List[float], List[float]]]:
    if bounds is None:
        return None
    if isinstance(bounds, dict):
        try:
            min_pt = [float(bounds["x"][0]), float(bounds["y"][0]), float(bounds["z"][0])]
            max_pt = [float(bounds["x"][1]), float(bounds["y"][1]), float(bounds["z"][1])]
        except (KeyError, TypeError, ValueError) as exc:
            raise ExportConsistencyError(
                f"Invalid workspace_bounds dict in {context} at {path}."
            ) from exc
        return min_pt, max_pt
    if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
        try:
            min_pt = [float(v) for v in bounds[0]]
            max_pt = [float(v) for v in bounds[1]]
        except (TypeError, ValueError) as exc:
            raise ExportConsistencyError(
                f"Invalid workspace_bounds list in {context} at {path}."
            ) from exc
        if len(min_pt) != 3 or len(max_pt) != 3:
            raise ExportConsistencyError(
                f"workspace_bounds must be 3D in {context} at {path}."
            )
        return min_pt, max_pt
    raise ExportConsistencyError(
        f"Unsupported workspace_bounds format in {context} at {path}."
    )


def _position_in_bounds(position: Iterable[float], bounds: Tuple[List[float], List[float]]) -> bool:
    min_pt, max_pt = bounds
    return all(min_pt[i] <= position[i] <= max_pt[i] for i in range(3))


def _normalize_position(value: Any) -> Optional[List[float]]:
    if isinstance(value, (list, tuple)) and len(value) == 3:
        try:
            return [float(v) for v in value]
        except (TypeError, ValueError):
            return None
    return None


def _extract_object_ids_from_entry(entry: Dict[str, Any]) -> Set[str]:
    object_ids: Set[str] = set()
    single_keys = {
        "target_object_id",
        "target_object",
        "target_object_name",
        "goal_region",
        "goal_object_id",
        "goal_object",
        "object_id",
    }
    list_keys = {"objects", "object_ids", "target_objects"}

    for key in single_keys:
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            object_ids.add(value)

    for key in list_keys:
        value = entry.get(key)
        if not isinstance(value, list):
            continue
        for item in value:
            if isinstance(item, str) and item.strip():
                object_ids.add(item)
            elif isinstance(item, dict):
                for id_key in ("object_id", "id", "name", "asset_id"):
                    id_value = item.get(id_key)
                    if isinstance(id_value, str) and id_value.strip():
                        object_ids.add(id_value)

    return object_ids


def _extract_positions(task: Dict[str, Any]) -> List[Tuple[str, List[float]]]:
    positions: List[Tuple[str, List[float]]] = []
    for key in ("target_position", "place_position", "goal_position", "target_positions", "goal_positions"):
        value = task.get(key)
        if isinstance(value, list) and value and isinstance(value[0], (list, tuple)):
            for idx, entry in enumerate(value):
                normalized = _normalize_position(entry)
                if normalized is not None:
                    positions.append((f"{key}[{idx}]", normalized))
            continue
        normalized = _normalize_position(value)
        if normalized is not None:
            positions.append((key, normalized))

    target_objects = task.get("target_objects")
    if isinstance(target_objects, list):
        for idx, obj in enumerate(target_objects):
            if isinstance(obj, dict):
                normalized = _normalize_position(obj.get("position"))
                if normalized is not None:
                    positions.append((f"target_objects[{idx}].position", normalized))

    return positions


def _namespaced_asset_id(scene_id: Optional[str], obj_id: str) -> str:
    if not scene_id or scene_id == "unknown":
        return obj_id
    scene_prefix = f"{scene_id}_obj_"
    if obj_id.startswith(scene_prefix) or obj_id.startswith(f"{scene_id}:"):
        return obj_id
    return f"{scene_id}_obj_{obj_id}"


def _validate_export_consistency_payloads(
    *,
    scene_graph: Dict[str, Any],
    asset_index: Dict[str, Any],
    task_config: Dict[str, Any],
    scene_graph_context: Path,
    asset_index_context: Path,
    task_config_context: Path,
) -> None:

    asset_ids = {
        asset.get("asset_id")
        for asset in asset_index.get("assets", [])
        if isinstance(asset, dict) and asset.get("asset_id")
    }

    missing_scene_assets = {
        node.get("asset_id")
        for node in scene_graph.get("nodes", [])
        if isinstance(node, dict)
        and node.get("asset_id")
        and node.get("asset_id") not in asset_ids
    }

    task_entries: List[Dict[str, Any]] = []
    if isinstance(task_config, dict):
        suggested_tasks = task_config.get("suggested_tasks")
        if isinstance(suggested_tasks, list):
            task_entries.extend(task for task in suggested_tasks if isinstance(task, dict))
        tasks = task_config.get("tasks")
        if isinstance(tasks, list):
            task_entries.extend(task for task in tasks if isinstance(task, dict))

    referenced_ids: Set[str] = set()
    for entry in task_entries:
        referenced_ids.update(_extract_object_ids_from_entry(entry))

    scene_id = scene_graph.get("scene_id") if isinstance(scene_graph, dict) else None

    missing_task_assets = {
        obj_id
        for obj_id in referenced_ids
        if obj_id not in asset_ids
        and _namespaced_asset_id(scene_id, obj_id) not in asset_ids
    }

    workspace_bounds = _parse_workspace_bounds(
        task_config.get("robot_config", {}).get("workspace_bounds")
        if isinstance(task_config, dict)
        else None,
        context="task config",
        path=task_config_context,
    )
    if workspace_bounds is None and isinstance(task_config, dict):
        workspace_bounds = _parse_workspace_bounds(
            task_config.get("workspace_bounds"),
            context="task config",
            path=task_config_context,
        )

    out_of_bounds: List[str] = []
    for idx, task in enumerate(task_entries):
        task_bounds = workspace_bounds
        task_bounds_override = None
        if isinstance(task, dict):
            task_bounds_override = task.get("workspace_bounds") or task.get("robot_config", {}).get(
                "workspace_bounds"
            )
        if task_bounds_override is not None:
            task_bounds = _parse_workspace_bounds(
                task_bounds_override,
                context=f"task[{idx}]",
                path=task_config_context,
            )
        if task_bounds is None:
            continue
        for key, position in _extract_positions(task):
            if not _position_in_bounds(position, task_bounds):
                task_label = task.get("task_type") or task.get("name") or "unknown"
                out_of_bounds.append(
                    f"Task[{idx}] ({task_label}) {key}={position} outside workspace bounds "
                    f"min={task_bounds[0]} max={task_bounds[1]} (task_config: {task_config_context})"
                )

    errors: List[str] = []
    if missing_scene_assets:
        errors.append(
            "Scene graph references missing asset_ids "
            f"{sorted(missing_scene_assets)} (scene_graph: {scene_graph_context}, "
            f"asset_index: {asset_index_context})"
        )
    if missing_task_assets:
        errors.append(
            "Task config references missing object ids "
            f"{sorted(missing_task_assets)} (task_config: {task_config_context}, "
            f"asset_index: {asset_index_context})"
        )
    if out_of_bounds:
        errors.append(
            "Task positions out of workspace bounds:\n  - " + "\n  - ".join(out_of_bounds)
        )

    if errors:
        raise ExportConsistencyError(
            "Genie Sim export consistency validation failed:\n- " + "\n- ".join(errors)
        )


def validate_export_consistency(
    *,
    scene_graph_path: Path,
    asset_index_path: Path,
    task_config_path: Path,
) -> None:
    scene_graph = _load_json(scene_graph_path, "scene graph")
    asset_index = _load_json(asset_index_path, "asset index")
    task_config = _load_json(task_config_path, "task config")
    _validate_export_consistency_payloads(
        scene_graph=scene_graph,
        asset_index=asset_index,
        task_config=task_config,
        scene_graph_context=scene_graph_path,
        asset_index_context=asset_index_path,
        task_config_context=task_config_path,
    )


def validate_export_consistency_data(
    *,
    scene_graph: Dict[str, Any],
    asset_index: Dict[str, Any],
    task_config: Dict[str, Any],
) -> None:
    placeholder = Path("<in-memory>")
    _validate_export_consistency_payloads(
        scene_graph=scene_graph,
        asset_index=asset_index,
        task_config=task_config,
        scene_graph_context=placeholder,
        asset_index_context=placeholder,
        task_config_context=placeholder,
    )
