from __future__ import annotations

from pathlib import Path

import pytest

from tools.validation import (
    InputValidator,
    PATTERN_ALPHANUMERIC,
    ValidationError,
    sanitize_path,
    sanitize_string,
    validate_category,
    validate_dimensions,
    validate_enum,
    validate_numeric,
    validate_object_id,
    validate_quaternion,
    validate_rgb_color,
    validate_rotation_matrix,
    validate_url,
)


pytestmark = pytest.mark.usefixtures("add_repo_to_path")


def test_sanitize_string_removes_control_and_validates_pattern() -> None:
    assert (
        sanitize_string("Valid\n", max_length=10, allow_pattern=PATTERN_ALPHANUMERIC)
        == "Valid"
    )

    with pytest.raises(ValidationError, match="Expected string"):
        sanitize_string(123)

    with pytest.raises(ValidationError, match="maximum length"):
        sanitize_string("toolong", max_length=3)

    with pytest.raises(ValidationError, match="invalid characters"):
        sanitize_string("bad!", allow_pattern=PATTERN_ALPHANUMERIC)


def test_sanitize_path_validates_bounds_and_existence(tmp_path: Path) -> None:
    file_path = tmp_path / "file.txt"
    file_path.write_text("ok")

    resolved = sanitize_path("file.txt", allowed_parent=tmp_path, must_exist=True)
    assert resolved == file_path.resolve()

    with pytest.raises(ValidationError, match="Path traversal"):
        sanitize_path("../outside.txt", allowed_parent=tmp_path)

    with pytest.raises(ValidationError, match="does not exist"):
        sanitize_path("missing.txt", allowed_parent=tmp_path, must_exist=True)


def test_validate_numeric_bounds_and_type() -> None:
    assert validate_numeric(5, min_value=1, max_value=10) == 5

    with pytest.raises(ValidationError, match="Expected numeric"):
        validate_numeric("bad")

    with pytest.raises(ValidationError, match="less than minimum"):
        validate_numeric(0, min_value=1)

    with pytest.raises(ValidationError, match="exceeds maximum"):
        validate_numeric(11, max_value=10)


def test_validate_category_and_dimensions() -> None:
    assert validate_category("Cup", strict=True) == "cup"

    with pytest.raises(ValidationError, match="allowed list"):
        validate_category("not_allowed", allowed_categories={"cup"}, strict=True)

    dims = validate_dimensions({"width": 1.0, "height": 2.0, "depth": 3.0})
    assert dims["width"] == 1.0

    with pytest.raises(ValidationError, match="Missing required dimension keys"):
        validate_dimensions({"width": 1.0})

    with pytest.raises(ValidationError, match="Expected dict"):
        validate_dimensions(["bad"])  # type: ignore[list-item]


def test_validate_rgb_color() -> None:
    assert validate_rgb_color([0.1, 0.2, 0.3]) == [0.1, 0.2, 0.3]

    with pytest.raises(ValidationError, match="3 components"):
        validate_rgb_color([0.1, 0.2])

    with pytest.raises(ValidationError, match="exceeds maximum"):
        validate_rgb_color([0.1, 0.2, 1.5])


def test_validate_quaternion_and_rotation_matrix_errors() -> None:
    with pytest.raises(ValidationError, match="Expected quaternion"):
        validate_quaternion("bad")  # type: ignore[arg-type]

    with pytest.raises(ValidationError, match="too small"):
        validate_quaternion([0.0, 0.0, 0.0, 0.0])

    with pytest.raises(ValidationError, match="deviates from 1.0"):
        validate_quaternion([2.0, 0.0, 0.0, 0.0], auto_normalize=False)

    assert validate_quaternion({"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}) == [
        1.0,
        0.0,
        0.0,
        0.0,
    ]

    with pytest.raises(ValidationError, match="must be 3x3"):
        validate_rotation_matrix([[1.0, 0.0], [0.0, 1.0]])

    with pytest.raises(ValidationError, match="Rotation matrix invalid"):
        validate_rotation_matrix(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.1],
                [0.0, 0.0, 0.9],
            ],
            auto_fix=False,
        )


def test_validate_enum_and_url() -> None:
    assert validate_enum("a", {"a", "b"}) == "a"

    with pytest.raises(ValidationError, match="Invalid value"):
        validate_enum("c", {"a", "b"})

    assert validate_url("https://example.com") == "https://example.com"

    with pytest.raises(ValidationError, match="Invalid URL scheme"):
        validate_url("ftp://example.com")

    with pytest.raises(ValidationError, match="URL must have a domain"):
        validate_url("http:///missing")


def test_input_validator_rules_and_errors() -> None:
    validator = InputValidator()
    validator.add_rule("object_id", validate_object_id)

    result = validator.validate({"object_id": "asset_1", "extra": 5})
    assert result == {"object_id": "asset_1", "extra": 5}

    validator.add_rule("bad", lambda value: 1 / 0)
    with pytest.raises(ValidationError, match="Validation failed"):
        validator.validate({"bad": "value"})
