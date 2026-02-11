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
    assert len(processed) == 3
    joined = " ".join(processed).lower()
    assert "knob" not in joined
    assert "display panel" not in joined
    assert any("stacked washer" in label.lower() for label in processed)
    assert "floor" not in joined
    assert "wall" not in joined


def test_post_process_labels_compat_keeps_nonempty_labels() -> None:
    raw = ["control knob", "control knob", "washer", "floor"]
    processed = _post_process_labels(raw, max_labels=10, quality_mode="compat")
    # compat mode still deduplicates, but keeps labels that quality mode might prune.
    assert "control knob" in processed
    assert "washer" in processed


def test_post_process_labels_quality_filters_structural_and_plural_duplicates() -> None:
    raw = [
        "detergent bottle",
        "detergent bottles",
        "woven baskets",
        "woven basket",
        "window curtain",
        "floor",
        "wall",
    ]
    processed = _post_process_labels(raw, max_labels=10, quality_mode="quality")
    assert "detergent bottle" in processed
    assert any("woven basket" in label.lower() for label in processed)
    assert all("floor" not in label.lower() for label in processed)
    assert all("wall" not in label.lower() for label in processed)
    assert all("curtain" not in label.lower() for label in processed)
    assert len([label for label in processed if "detergent" in label.lower()]) == 1
    assert len([label for label in processed if "basket" in label.lower()]) == 1
