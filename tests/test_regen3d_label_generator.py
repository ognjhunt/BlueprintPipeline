from __future__ import annotations

from tools.regen3d_runner.gemini_label_generator import _post_process_labels


def test_post_process_labels_quality_filters_tiny_parts_and_caps() -> None:
    raw = [
        "stacked washer and dryer",
        "stacked washer dryer",
        "control knob",
        "digital display panel",
        "woven basket",
        "detergent bottle",
        "detergent bottles",
        "floor",
        "wall",
    ]
    processed = _post_process_labels(raw, max_labels=4, quality_mode="quality")
    assert len(processed) == 4
    joined = " ".join(processed).lower()
    assert "knob" not in joined
    assert "display panel" not in joined
    assert any("stacked washer" in label.lower() for label in processed)


def test_post_process_labels_compat_keeps_nonempty_labels() -> None:
    raw = ["control knob", "control knob", "washer", "floor"]
    processed = _post_process_labels(raw, max_labels=10, quality_mode="compat")
    # compat mode still deduplicates, but keeps labels that quality mode might prune.
    assert "control knob" in processed
    assert "washer" in processed
