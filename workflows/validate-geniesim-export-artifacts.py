#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import jsonschema

from tools.validation.geniesim_export import ExportConsistencyError, validate_export_consistency


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _validate_schema(payload_path: Path, schema_path: Path) -> None:
    schema = _load_json(schema_path)
    payload = _load_json(payload_path)
    jsonschema.validate(instance=payload, schema=schema)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Genie Sim export artifacts.")
    parser.add_argument("--scene-graph", required=True, type=Path)
    parser.add_argument("--asset-index", required=True, type=Path)
    parser.add_argument("--task-config", required=True, type=Path)
    parser.add_argument("--schema-dir", required=True, type=Path)
    args = parser.parse_args()

    required_files = {
        "scene_graph": args.scene_graph,
        "asset_index": args.asset_index,
        "task_config": args.task_config,
    }
    missing = {name: path for name, path in required_files.items() if not path.is_file()}
    if missing:
        missing_list = ", ".join(f"{name} ({path})" for name, path in missing.items())
        raise SystemExit(f"Missing required export artifacts: {missing_list}")

    schema_dir = args.schema_dir
    schemas = {
        "scene_graph": schema_dir / "scene_graph.schema.json",
        "asset_index": schema_dir / "asset_index.schema.json",
        "task_config": schema_dir / "task_config.schema.json",
    }
    missing_schemas = {name: path for name, path in schemas.items() if not path.is_file()}
    if missing_schemas:
        missing_list = ", ".join(f"{name} ({path})" for name, path in missing_schemas.items())
        raise SystemExit(f"Missing schema files: {missing_list}")

    try:
        _validate_schema(args.scene_graph, schemas["scene_graph"])
        _validate_schema(args.asset_index, schemas["asset_index"])
        _validate_schema(args.task_config, schemas["task_config"])
        validate_export_consistency(
            scene_graph_path=args.scene_graph,
            asset_index_path=args.asset_index,
            task_config_path=args.task_config,
        )
    except jsonschema.ValidationError as exc:
        raise SystemExit(f"Schema validation failed: {exc.message}") from exc
    except ExportConsistencyError as exc:
        raise SystemExit(str(exc)) from exc

    report = {
        "status": "passed",
        "artifacts": {name: str(path) for name, path in required_files.items()},
        "schemas": {name: str(path) for name, path in schemas.items()},
        "checks": [
            "jsonschema(scene_graph)",
            "jsonschema(asset_index)",
            "jsonschema(task_config)",
            "export_consistency(scene_graph, asset_index, task_config)",
        ],
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
