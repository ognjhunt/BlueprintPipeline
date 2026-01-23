from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = REPO_ROOT / "fixtures" / "contracts"


def load_schema(schema_name: str) -> Dict[str, Any]:
    schema_path = SCHEMA_DIR / schema_name
    return json.loads(schema_path.read_text())


def validate_json_schema(payload: Any, schema: Dict[str, Any]) -> None:
    try:
        import jsonschema  # type: ignore
    except ImportError:
        _validate_minimal_schema(payload, schema, path="$")
    else:
        jsonschema.validate(instance=payload, schema=schema)


def _validate_minimal_schema(payload: Any, schema: Dict[str, Any], path: str) -> None:
    any_of = schema.get("anyOf") or schema.get("oneOf")
    if any_of:
        errors = []
        for idx, option in enumerate(any_of):
            try:
                _validate_minimal_schema(payload, option, path)
            except ValueError as exc:
                errors.append(f"Option {idx}: {exc}")
            else:
                return
        raise ValueError(f"{path}: payload did not match anyOf/oneOf: {errors}")
    for option in schema.get("allOf", []):
        _validate_minimal_schema(payload, option, path)
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        last_error: Optional[Exception] = None
        for candidate in schema_type:
            candidate_schema = dict(schema)
            candidate_schema["type"] = candidate
            try:
                _validate_minimal_schema(payload, candidate_schema, path)
            except ValueError as exc:
                last_error = exc
                continue
            else:
                return
        if last_error is not None:
            raise last_error
        return
    if schema_type == "object":
        if not isinstance(payload, dict):
            raise ValueError(f"{path}: expected object")
        required = schema.get("required", [])
        for key in required:
            if key not in payload:
                raise ValueError(f"{path}: missing required field '{key}'")
        properties = schema.get("properties", {})
        for key, prop_schema in properties.items():
            if key in payload:
                _validate_minimal_schema(payload[key], prop_schema, f"{path}.{key}")
    elif schema_type == "array":
        if not isinstance(payload, list):
            raise ValueError(f"{path}: expected array")
        min_items = schema.get("minItems")
        max_items = schema.get("maxItems")
        if min_items is not None and len(payload) < min_items:
            raise ValueError(f"{path}: expected at least {min_items} items")
        if max_items is not None and len(payload) > max_items:
            raise ValueError(f"{path}: expected at most {max_items} items")
        items_schema = schema.get("items")
        if items_schema:
            for idx, item in enumerate(payload):
                _validate_minimal_schema(item, items_schema, f"{path}[{idx}]")
    elif schema_type == "string":
        if not isinstance(payload, str):
            raise ValueError(f"{path}: expected string")
        enum = schema.get("enum")
        if enum and payload not in enum:
            raise ValueError(f"{path}: value '{payload}' not in enum {enum}")
    elif schema_type == "integer":
        if not isinstance(payload, int):
            raise ValueError(f"{path}: expected integer")
    elif schema_type == "number":
        if not isinstance(payload, (int, float)):
            raise ValueError(f"{path}: expected number")
    elif schema_type == "boolean":
        if not isinstance(payload, bool):
            raise ValueError(f"{path}: expected boolean")
    elif schema_type == "null":
        if payload is not None:
            raise ValueError(f"{path}: expected null")
