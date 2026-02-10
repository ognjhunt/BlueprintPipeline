#!/usr/bin/env python3
"""Evaluate nightly canary stability for required robots."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_REQUIRED_ROBOTS = ("franka", "ur5e", "ur10")


def _utc_today() -> date:
    return datetime.now(timezone.utc).date()


def _safe_date(value: Any) -> Optional[date]:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return date.fromisoformat(value.strip()[:10])
    except ValueError:
        return None


def _safe_status(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"pass", "passed", "success", "ok", "true", "1"}
    return False


@dataclass
class DayStatus:
    date: date
    robots: Dict[str, bool]


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _parse_history_entries(payload: Any) -> List[DayStatus]:
    records: List[DayStatus] = []
    if isinstance(payload, dict):
        payload = payload.get("history") or payload.get("runs") or []
    if not isinstance(payload, list):
        return records

    daily: Dict[date, Dict[str, bool]] = defaultdict(dict)
    for item in payload:
        if not isinstance(item, dict):
            continue
        record_date = _safe_date(item.get("date") or item.get("run_date") or item.get("timestamp"))
        if record_date is None:
            continue

        robots = item.get("robots")
        if isinstance(robots, dict):
            for robot, status in robots.items():
                robot_key = str(robot).strip().lower()
                if not robot_key:
                    continue
                current = daily[record_date].get(robot_key)
                status_bool = _safe_status(status)
                daily[record_date][robot_key] = bool(current) and status_bool if current is not None else status_bool
            continue

        robot = str(item.get("robot") or item.get("robot_type") or "").strip().lower()
        if robot:
            status_bool = _safe_status(item.get("status", item.get("passed", item.get("success"))))
            current = daily[record_date].get(robot)
            daily[record_date][robot] = bool(current) and status_bool if current is not None else status_bool

    for record_date in sorted(daily.keys()):
        records.append(DayStatus(date=record_date, robots=dict(sorted(daily[record_date].items()))))
    return records


def evaluate_stability(
    history: Iterable[DayStatus],
    *,
    required_robots: Iterable[str],
    days: int,
    reference_date: Optional[date] = None,
) -> Dict[str, Any]:
    required = [robot.strip().lower() for robot in required_robots if robot.strip()]
    normalized_history = list(history)
    if reference_date is None:
        latest = max((entry.date for entry in normalized_history), default=_utc_today())
        reference_date = latest

    window_dates = [reference_date - timedelta(days=offset) for offset in range(max(0, days))]
    by_date = {entry.date: entry.robots for entry in normalized_history}

    per_robot: Dict[str, Dict[str, Any]] = {}
    stable = True
    for robot in required:
        missing_dates: List[str] = []
        failing_dates: List[str] = []
        passing_days = 0
        for day in window_dates:
            statuses = by_date.get(day)
            if not statuses or robot not in statuses:
                missing_dates.append(day.isoformat())
                stable = False
                continue
            if statuses[robot]:
                passing_days += 1
            else:
                failing_dates.append(day.isoformat())
                stable = False
        per_robot[robot] = {
            "passing_days": passing_days,
            "required_days": len(window_dates),
            "missing_dates": missing_dates,
            "failing_dates": failing_dates,
            "stable": not missing_dates and not failing_dates and passing_days == len(window_dates),
        }

    return {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "reference_date": reference_date.isoformat(),
        "window_days": len(window_dates),
        "required_robots": required,
        "stable_7_day": stable,
        "per_robot": per_robot,
        "window_dates": [day.isoformat() for day in window_dates],
    }


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Evaluate 7-day nightly canary stability.")
    parser.add_argument(
        "--history",
        type=Path,
        default=repo_root / "analysis_outputs" / "canary_history.json",
        help="JSON file containing canary history records.",
    )
    parser.add_argument(
        "--required-robots",
        default="franka,ur5e,ur10",
        help="Comma-separated required robot types.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Consecutive day window required for stability.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "analysis_outputs" / "canary_stability_gate.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when stability gate fails.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    history_path = args.history.expanduser().resolve()
    payload = _read_json(history_path) if history_path.is_file() else []
    history = _parse_history_entries(payload)

    required_robots = [item.strip() for item in args.required_robots.split(",") if item.strip()]
    if not required_robots:
        required_robots = list(DEFAULT_REQUIRED_ROBOTS)

    result = evaluate_stability(
        history,
        required_robots=required_robots,
        days=max(1, int(args.days)),
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(f"[canary-stability-gate] wrote {output}")

    if args.strict and not bool(result.get("stable_7_day")):
        print("[canary-stability-gate] gate failed")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
