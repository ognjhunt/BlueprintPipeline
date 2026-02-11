from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


_VALID_MODES = {"lexical_primary", "ann_shadow", "ann_primary"}


def _is_truthy(raw: str | None, *, default: bool) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _safe_int(raw: str | None, default: int) -> int:
    try:
        if raw is None:
            return default
        return int(raw)
    except (TypeError, ValueError):
        return default


def _safe_float(raw: str | None, default: float) -> float:
    try:
        if raw is None:
            return default
        return float(raw)
    except (TypeError, ValueError):
        return default


def _percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    p = max(0.0, min(1.0, p))
    idx = int((len(sorted_values) - 1) * p)
    return float(sorted_values[idx])


def _state_prefix() -> str:
    return (os.getenv("TEXT_ASSET_ROLLOUT_STATE_PREFIX") or "automation/asset_retrieval_rollout").strip().strip("/")


def _state_path(root: Path) -> Path:
    return root / _state_prefix() / "state.json"


def _default_mode() -> str:
    raw = (os.getenv("TEXT_ASSET_RETRIEVAL_MODE") or "ann_shadow").strip().lower()
    if raw in _VALID_MODES:
        return raw
    return "ann_shadow"


def _load_state(root: Path) -> Dict[str, Any]:
    state_file = _state_path(root)
    if not state_file.is_file():
        return {
            "schema_version": "v1",
            "retrieval_mode": _default_mode(),
            "requested_mode": _default_mode(),
            "total_decisions": 0,
            "consecutive_passing_windows": 0,
            "last_window": {},
            "last_updated_at": "",
        }
    try:
        payload = json.loads(state_file.read_text(encoding="utf-8"))
    except Exception:
        payload = {}

    if not isinstance(payload, dict):
        payload = {}
    payload.setdefault("schema_version", "v1")
    payload.setdefault("retrieval_mode", _default_mode())
    payload.setdefault("requested_mode", _default_mode())
    payload.setdefault("total_decisions", 0)
    payload.setdefault("consecutive_passing_windows", 0)
    payload.setdefault("last_window", {})
    payload.setdefault("last_updated_at", "")
    return payload


def _write_state(root: Path, payload: Mapping[str, Any]) -> None:
    state_file = _state_path(root)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(dict(payload), indent=2), encoding="utf-8")


def effective_retrieval_mode(root: Path) -> str:
    requested_mode = _default_mode()
    state = _load_state(root)
    mode = str(state.get("retrieval_mode") or requested_mode).strip().lower()
    if mode not in _VALID_MODES:
        return requested_mode
    return mode


def update_rollout_state(*, root: Path, decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
    state = _load_state(root)
    now = datetime.now(timezone.utc).isoformat()
    requested_mode = _default_mode()
    active_mode = str(state.get("retrieval_mode") or requested_mode).strip().lower()
    if active_mode not in _VALID_MODES:
        active_mode = requested_mode

    state["requested_mode"] = requested_mode
    state["retrieval_mode"] = active_mode

    decision_count = len(decisions)
    ann_attempted = [item for item in decisions if bool(item.get("ann_attempted"))]
    ann_attempted_count = len(ann_attempted)
    ann_hit_count = len([item for item in ann_attempted if int(item.get("ann_candidate_count") or 0) > 0])
    ann_error_count = len([item for item in ann_attempted if bool(item.get("ann_error"))])
    ann_latencies = [float(item.get("ann_latency_ms") or 0.0) for item in ann_attempted if float(item.get("ann_latency_ms") or 0.0) > 0]

    ann_hit_rate = float(ann_hit_count) / max(1, ann_attempted_count)
    ann_error_rate = float(ann_error_count) / max(1, ann_attempted_count)
    ann_latency_p95_ms = _percentile(ann_latencies, 0.95)

    min_decisions = max(1, _safe_int(os.getenv("TEXT_ASSET_ROLLOUT_MIN_DECISIONS"), 500))
    min_hit_rate = _safe_float(os.getenv("TEXT_ASSET_ROLLOUT_MIN_HIT_RATE"), 0.95)
    max_error_rate = _safe_float(os.getenv("TEXT_ASSET_ROLLOUT_MAX_ERROR_RATE"), 0.01)
    max_p95_ms = _safe_float(os.getenv("TEXT_ASSET_ROLLOUT_MAX_P95_MS"), 400.0)
    min_passing_windows = max(1, _safe_int(os.getenv("TEXT_ASSET_ROLLOUT_MIN_PASSING_WINDOWS"), 3))

    window_pass = (
        ann_attempted_count >= min_decisions
        and ann_hit_rate >= min_hit_rate
        and ann_error_rate <= max_error_rate
        and ann_latency_p95_ms <= max_p95_ms
    )

    consecutive = int(state.get("consecutive_passing_windows") or 0)
    promoted = False
    if active_mode == "ann_shadow":
        if window_pass:
            consecutive += 1
        else:
            consecutive = 0
        if consecutive >= min_passing_windows:
            promoted = True
            active_mode = "ann_primary"
    elif active_mode != "ann_primary":
        consecutive = 0

    state["retrieval_mode"] = active_mode
    state["consecutive_passing_windows"] = consecutive
    state["total_decisions"] = int(state.get("total_decisions") or 0) + decision_count
    state["last_window"] = {
        "decision_count": decision_count,
        "ann_attempted_count": ann_attempted_count,
        "ann_hit_count": ann_hit_count,
        "ann_error_count": ann_error_count,
        "ann_hit_rate": round(ann_hit_rate, 6),
        "ann_error_rate": round(ann_error_rate, 6),
        "ann_latency_p95_ms": round(ann_latency_p95_ms, 4),
        "window_pass": bool(window_pass),
    }
    state["last_updated_at"] = now
    state["promoted_to_ann_primary"] = promoted

    _write_state(root, state)
    return {
        "schema_version": "v1",
        "requested_mode": requested_mode,
        "active_mode": active_mode,
        "promoted_to_ann_primary": promoted,
        "consecutive_passing_windows": consecutive,
        "last_window": state["last_window"],
        "last_updated_at": now,
    }


__all__ = [
    "effective_retrieval_mode",
    "update_rollout_state",
]

