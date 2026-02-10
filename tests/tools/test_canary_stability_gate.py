from __future__ import annotations

from datetime import date, timedelta

from tools.quality_gates.canary_stability_gate import DayStatus, evaluate_stability


def test_canary_stability_gate_passes_with_full_7_day_coverage() -> None:
    ref = date(2026, 2, 10)
    history = []
    for offset in range(7):
        day = ref - timedelta(days=offset)
        history.append(
            DayStatus(
                date=day,
                robots={"franka": True, "ur5e": True, "ur10": True},
            )
        )

    result = evaluate_stability(
        history,
        required_robots=("franka", "ur5e", "ur10"),
        days=7,
        reference_date=ref,
    )

    assert result["stable_7_day"] is True
    assert result["per_robot"]["franka"]["stable"] is True
    assert result["per_robot"]["ur5e"]["stable"] is True
    assert result["per_robot"]["ur10"]["stable"] is True


def test_canary_stability_gate_fails_when_robot_missing_day() -> None:
    ref = date(2026, 2, 10)
    history = []
    for offset in range(7):
        day = ref - timedelta(days=offset)
        statuses = {"franka": True, "ur5e": True, "ur10": True}
        if offset == 3:
            statuses.pop("ur10")
        history.append(DayStatus(date=day, robots=statuses))

    result = evaluate_stability(
        history,
        required_robots=("franka", "ur5e", "ur10"),
        days=7,
        reference_date=ref,
    )

    assert result["stable_7_day"] is False
    assert result["per_robot"]["ur10"]["stable"] is False
    assert result["per_robot"]["ur10"]["missing_dates"] == ["2026-02-07"]
