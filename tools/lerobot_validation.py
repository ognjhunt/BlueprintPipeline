"""LeRobot dataset validation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional, Sequence


EXPECTED_SCHEMA_COLUMNS = {
    "episode_id",
    "frame_index",
    "timestamp",
    "observation",
    "action",
    "reward",
    "done",
    "task_name",
    "task_id",
}


def _load_json(path: Path, label: str, errors: List[str]) -> Optional[Any]:
    if not path.is_file():
        errors.append(f"Missing {label}: {path}")
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        errors.append(f"Invalid JSON in {label} ({path}): {exc}")
        return None


def _resolve_dataset_format(dataset_info: dict[str, Any]) -> Optional[str]:
    return dataset_info.get("format") or dataset_info.get("dataset_type")


def _resolve_dataset_version(dataset_info: dict[str, Any]) -> Optional[str]:
    return dataset_info.get("version") or dataset_info.get("format_version")


def _normalize_value(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    return str(value).strip()


def _is_v3_export(dataset_info: dict[str, Any], info: dict[str, Any], chunk_dir: Path) -> bool:
    export_format = _normalize_value(info.get("export_format") or dataset_info.get("export_format"))
    version = _normalize_value(info.get("version") or dataset_info.get("version"))
    if export_format == "lerobot_v3" or version == "3.0":
        return True
    return (chunk_dir / "episodes.parquet").is_file()


def _count_episode_index(payload: Any) -> Optional[int]:
    if isinstance(payload, dict):
        return len(payload)
    if isinstance(payload, list):
        return len(payload)
    return None


def _validate_schema_columns(path: Path, errors: List[str]) -> None:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError:
        return
    try:
        schema = pq.read_schema(path)
    except Exception as exc:
        errors.append(f"Unable to read parquet schema at {path}: {exc}")
        return
    column_names = set(schema.names)
    missing = sorted(EXPECTED_SCHEMA_COLUMNS - column_names)
    if missing:
        errors.append(
            f"Parquet schema at {path} missing columns: {', '.join(missing)}"
        )


def _select_parquet_for_schema(files: Sequence[Path]) -> Optional[Path]:
    for candidate in files:
        if candidate.is_file():
            return candidate
    return None


def validate_lerobot_dataset(lerobot_dir: Path) -> List[str]:
    """Validate LeRobot dataset layout and metadata."""
    errors: List[str] = []
    lerobot_dir = Path(lerobot_dir)

    dataset_info_path = lerobot_dir / "dataset_info.json"
    dataset_info = _load_json(dataset_info_path, "dataset_info.json", errors)

    info_path = lerobot_dir / "meta" / "info.json"
    info = _load_json(info_path, "lerobot/meta/info.json", errors)

    if isinstance(dataset_info, dict) and isinstance(info, dict):
        dataset_format = _normalize_value(_resolve_dataset_format(dataset_info))
        info_format = _normalize_value(info.get("format"))
        if not dataset_format:
            errors.append("dataset_info.json missing format/dataset_type")
        if not info_format:
            errors.append("lerobot/meta/info.json missing format")
        if dataset_format and info_format and dataset_format != info_format:
            errors.append(
                f"Format mismatch: dataset_info.json has '{dataset_format}' "
                f"but lerobot/meta/info.json has '{info_format}'"
            )

        dataset_version = _normalize_value(_resolve_dataset_version(dataset_info))
        info_version = _normalize_value(info.get("version"))
        if not dataset_version:
            errors.append("dataset_info.json missing version/format_version")
        if not info_version:
            errors.append("lerobot/meta/info.json missing version")
        if dataset_version and info_version and dataset_version != info_version:
            errors.append(
                f"Version mismatch: dataset_info.json has '{dataset_version}' "
                f"but lerobot/meta/info.json has '{info_version}'"
            )

    data_dir = lerobot_dir / "data" / "chunk-000"
    if not data_dir.is_dir():
        errors.append(f"Missing LeRobot data directory: {data_dir}")
        return errors

    is_v3 = False
    if isinstance(dataset_info, dict) and isinstance(info, dict):
        is_v3 = _is_v3_export(dataset_info, info, data_dir)

    parquet_files: List[Path] = []
    if is_v3:
        episodes_parquet = data_dir / "episodes.parquet"
        if not episodes_parquet.is_file():
            errors.append(f"Missing v3 episodes parquet: {episodes_parquet}")
        else:
            parquet_files.append(episodes_parquet)
    else:
        parquet_files = list(data_dir.glob("episode_*.parquet"))
        if not parquet_files:
            errors.append(f"Missing v2 episode parquet files under {data_dir}")

    if is_v3:
        episode_index_path = lerobot_dir / "meta" / "episode_index.json"
        episode_index = _load_json(episode_index_path, "lerobot/meta/episode_index.json", errors)
        if isinstance(info, dict):
            expected_episodes = info.get("total_episodes")
        else:
            expected_episodes = None
        actual_episodes = _count_episode_index(episode_index)
        if expected_episodes is not None and actual_episodes is not None:
            if int(expected_episodes) != actual_episodes:
                errors.append(
                    "Episode index count mismatch: "
                    f"info.json has {expected_episodes} but episode_index.json has {actual_episodes}"
                )

    parquet_to_check = _select_parquet_for_schema(parquet_files)
    if parquet_to_check:
        _validate_schema_columns(parquet_to_check, errors)

    return errors
