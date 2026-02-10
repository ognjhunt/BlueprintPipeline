from __future__ import annotations

from tools.articulation.label_hints import (
    infer_primary_joint_type,
    parse_label_articulation_hint,
)


def test_parse_label_articulation_hint_detects_drawer() -> None:
    hint = parse_label_articulation_hint("desk with drawers")
    assert hint["is_articulated"] is True
    assert "drawer" in hint["parts"]
    assert "prismatic" in hint["joint_types"]
    assert hint["confidence"] >= 0.9


def test_parse_label_articulation_hint_detects_multiple_parts() -> None:
    hint = parse_label_articulation_hint("cabinet with doors and knobs")
    assert hint["is_articulated"] is True
    assert "door" in hint["parts"]
    assert "knob" in hint["parts"]
    assert "revolute" in hint["joint_types"]
    assert "continuous" in hint["joint_types"]


def test_parse_label_articulation_hint_non_articulated_label() -> None:
    hint = parse_label_articulation_hint("ceramic bowl")
    assert hint["is_articulated"] is False
    assert hint["confidence"] == 0.0
    assert hint["parts"] == []


def test_infer_primary_joint_type_prefers_prismatic_over_revolute_on_tie() -> None:
    hint = parse_label_articulation_hint("desk with drawers and side door")
    primary = infer_primary_joint_type(hint)
    assert primary == "prismatic"
