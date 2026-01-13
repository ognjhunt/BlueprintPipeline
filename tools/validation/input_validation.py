"""
Input validation and sanitization utilities.

Provides comprehensive input validation to prevent:
- Path traversal attacks
- Command injection
- XSS attacks
- JSON injection
- Invalid data types and ranges
"""

from __future__ import annotations

import logging
import re
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, List, Optional, Pattern, Union

logger = logging.getLogger(__name__)


class ValidationError(ValueError):
    """Raised when input validation fails."""

    def __init__(self, message: str, field: Optional[str] = None, value: Any = None):
        self.field = field
        self.value = value
        super().__init__(f"{field}: {message}" if field else message)


# Common validation patterns
PATTERN_ALPHANUMERIC = re.compile(r'^[a-zA-Z0-9_\-]+$')
PATTERN_ALPHANUMERIC_SPACE = re.compile(r'^[a-zA-Z0-9_\-\s]+$')
PATTERN_ALPHANUMERIC_EXTENDED = re.compile(r'^[a-zA-Z0-9_\-\.\s]+$')
PATTERN_OBJECT_ID = re.compile(r'^[a-zA-Z0-9_\-]+$')
PATTERN_SCENE_ID = re.compile(r'^[a-zA-Z0-9_\-]+$')
PATTERN_UUID = re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
    re.IGNORECASE
)


def sanitize_string(
    s: str,
    max_length: int = 256,
    allow_pattern: Optional[Pattern] = None,
    field_name: Optional[str] = None,
) -> str:
    """
    Sanitize and validate a string input.

    Args:
        s: String to sanitize
        max_length: Maximum allowed length
        allow_pattern: Regex pattern for allowed characters
        field_name: Field name for error messages

    Returns:
        Sanitized string

    Raises:
        ValidationError: If validation fails

    Example:
        # Allow only alphanumeric and common punctuation
        name = sanitize_string(
            user_input,
            max_length=100,
            allow_pattern=PATTERN_ALPHANUMERIC_EXTENDED,
            field_name="object_name",
        )
    """
    if not isinstance(s, str):
        raise ValidationError(
            f"Expected string, got {type(s).__name__}",
            field=field_name,
            value=s,
        )

    # Truncate if too long
    if len(s) > max_length:
        raise ValidationError(
            f"String exceeds maximum length {max_length} (got {len(s)})",
            field=field_name,
            value=s[:50] + "...",
        )

    # Remove control characters
    sanitized = ''.join(c for c in s if c.isprintable())

    # Check against allowed pattern
    if allow_pattern and not allow_pattern.match(sanitized):
        raise ValidationError(
            f"String contains invalid characters (allowed: {allow_pattern.pattern})",
            field=field_name,
            value=sanitized[:50],
        )

    return sanitized


def sanitize_path(
    path: Union[str, Path],
    allowed_parent: Union[str, Path],
    must_exist: bool = False,
    field_name: Optional[str] = None,
) -> Path:
    """
    Validate and sanitize a file path to prevent path traversal.

    Args:
        path: Path to validate
        allowed_parent: Parent directory that path must be within
        must_exist: If True, path must exist
        field_name: Field name for error messages

    Returns:
        Validated absolute Path

    Raises:
        ValidationError: If path is outside allowed_parent or doesn't exist

    Example:
        # Ensure path is within assets directory
        asset_path = sanitize_path(
            user_provided_path,
            allowed_parent="/data/assets",
            must_exist=True,
            field_name="asset_path",
        )
    """
    if not isinstance(path, (str, Path)):
        raise ValidationError(
            f"Expected str or Path, got {type(path).__name__}",
            field=field_name,
            value=path,
        )

    allowed_parent = Path(allowed_parent).resolve()
    path = Path(path)

    # Resolve to absolute path
    if path.is_absolute():
        resolved = path.resolve()
    else:
        resolved = (allowed_parent / path).resolve()

    # Check if path is within allowed parent
    try:
        resolved.relative_to(allowed_parent)
    except ValueError:
        raise ValidationError(
            f"Path traversal detected: path must be within {allowed_parent}",
            field=field_name,
            value=str(path),
        )

    # Check existence if required
    if must_exist and not resolved.exists():
        raise ValidationError(
            f"Path does not exist: {resolved}",
            field=field_name,
            value=str(path),
        )

    return resolved


def validate_numeric(
    value: Union[int, float],
    min_value: Optional[Union[int, float]] = None,
    max_value: Optional[Union[int, float]] = None,
    field_name: Optional[str] = None,
) -> Union[int, float]:
    """
    Validate a numeric value is within allowed range.

    Args:
        value: Numeric value to validate
        min_value: Minimum allowed value (inclusive)
        max_value: Maximum allowed value (inclusive)
        field_name: Field name for error messages

    Returns:
        Validated value

    Raises:
        ValidationError: If value is out of range

    Example:
        width = validate_numeric(
            obj_data.get("width"),
            min_value=0.0,
            max_value=10.0,
            field_name="width",
        )
    """
    if not isinstance(value, (int, float)):
        raise ValidationError(
            f"Expected numeric type, got {type(value).__name__}",
            field=field_name,
            value=value,
        )

    if min_value is not None and value < min_value:
        raise ValidationError(
            f"Value {value} is less than minimum {min_value}",
            field=field_name,
            value=value,
        )

    if max_value is not None and value > max_value:
        raise ValidationError(
            f"Value {value} exceeds maximum {max_value}",
            field=field_name,
            value=value,
        )

    return value


def validate_object_id(obj_id: str) -> str:
    """
    Validate object ID format.

    Args:
        obj_id: Object ID to validate

    Returns:
        Validated object ID

    Raises:
        ValidationError: If object ID is invalid

    Example:
        obj_id = validate_object_id(user_input)
    """
    return sanitize_string(
        obj_id,
        max_length=128,
        allow_pattern=PATTERN_OBJECT_ID,
        field_name="object_id",
    )


def validate_scene_id(scene_id: str) -> str:
    """
    Validate scene ID format.

    Args:
        scene_id: Scene ID to validate

    Returns:
        Validated scene ID

    Raises:
        ValidationError: If scene ID is invalid
    """
    return sanitize_string(
        scene_id,
        max_length=128,
        allow_pattern=PATTERN_SCENE_ID,
        field_name="scene_id",
    )


def validate_category(category: str) -> str:
    """
    Validate object category.

    Args:
        category: Category string to validate

    Returns:
        Validated category

    Raises:
        ValidationError: If category is invalid
    """
    return sanitize_string(
        category,
        max_length=64,
        allow_pattern=PATTERN_ALPHANUMERIC,
        field_name="category",
    )


def validate_description(description: str) -> str:
    """
    Validate and sanitize object description.

    Args:
        description: Description text to validate

    Returns:
        Sanitized description

    Raises:
        ValidationError: If description is invalid
    """
    return sanitize_string(
        description,
        max_length=1024,
        allow_pattern=PATTERN_ALPHANUMERIC_EXTENDED,
        field_name="description",
    )


def validate_dimensions(dimensions: Dict[str, float]) -> Dict[str, float]:
    """
    Validate object dimensions.

    Args:
        dimensions: Dictionary with width, height, depth

    Returns:
        Validated dimensions

    Raises:
        ValidationError: If dimensions are invalid

    Example:
        dims = validate_dimensions({
            "width": 1.5,
            "height": 2.0,
            "depth": 0.8,
        })
    """
    if not isinstance(dimensions, dict):
        raise ValidationError(
            f"Expected dict, got {type(dimensions).__name__}",
            field="dimensions",
            value=dimensions,
        )

    required_keys = {"width", "height", "depth"}
    missing_keys = required_keys - set(dimensions.keys())
    if missing_keys:
        raise ValidationError(
            f"Missing required dimension keys: {missing_keys}",
            field="dimensions",
            value=dimensions,
        )

    validated = {}
    for key in required_keys:
        value = dimensions[key]
        validated[key] = validate_numeric(
            value,
            min_value=0.001,  # Minimum 1mm
            max_value=100.0,  # Maximum 100m
            field_name=f"dimensions.{key}",
        )

    return validated


def validate_rgb_color(color: Union[List[float], tuple]) -> List[float]:
    """
    Validate RGB color values.

    Args:
        color: RGB color as list or tuple of 3 floats (0.0-1.0)

    Returns:
        Validated RGB color

    Raises:
        ValidationError: If color is invalid

    Example:
        color = validate_rgb_color([0.5, 0.2, 0.8])
    """
    if not isinstance(color, (list, tuple)):
        raise ValidationError(
            f"Expected list or tuple, got {type(color).__name__}",
            field="color",
            value=color,
        )

    if len(color) != 3:
        raise ValidationError(
            f"RGB color must have 3 components, got {len(color)}",
            field="color",
            value=color,
        )

    validated = []
    for i, component in enumerate(color):
        validated.append(validate_numeric(
            component,
            min_value=0.0,
            max_value=1.0,
            field_name=f"color[{i}]",
        ))

    return validated


def validate_enum(
    value: str,
    allowed_values: set[str],
    field_name: Optional[str] = None,
) -> str:
    """
    Validate value is in allowed enum set.

    Args:
        value: Value to validate
        allowed_values: Set of allowed values
        field_name: Field name for error messages

    Returns:
        Validated value

    Raises:
        ValidationError: If value not in allowed set

    Example:
        robot_type = validate_enum(
            user_input,
            {"franka", "ur10", "fetch"},
            field_name="robot_type",
        )
    """
    if not isinstance(value, str):
        raise ValidationError(
            f"Expected string, got {type(value).__name__}",
            field=field_name,
            value=value,
        )

    if value not in allowed_values:
        raise ValidationError(
            f"Invalid value '{value}'. Allowed: {sorted(allowed_values)}",
            field=field_name,
            value=value,
        )

    return value


def validate_url(url: str, allowed_schemes: Optional[set[str]] = None) -> str:
    """
    Validate URL format and scheme.

    Args:
        url: URL to validate
        allowed_schemes: Set of allowed schemes (default: http, https)

    Returns:
        Validated URL

    Raises:
        ValidationError: If URL is invalid
    """
    from urllib.parse import urlparse

    if not isinstance(url, str):
        raise ValidationError(
            f"Expected string, got {type(url).__name__}",
            field="url",
            value=url,
        )

    if allowed_schemes is None:
        allowed_schemes = {"http", "https"}

    try:
        parsed = urlparse(url)
    except Exception as e:
        raise ValidationError(
            f"Invalid URL format: {e}",
            field="url",
            value=url,
        )

    if parsed.scheme not in allowed_schemes:
        raise ValidationError(
            f"Invalid URL scheme '{parsed.scheme}'. Allowed: {allowed_schemes}",
            field="url",
            value=url,
        )

    if not parsed.netloc:
        raise ValidationError(
            "URL must have a domain",
            field="url",
            value=url,
        )

    return url


class InputValidator:
    """
    Reusable input validator with configurable rules.

    Example:
        validator = InputValidator()
        validator.add_rule("object_id", validate_object_id)
        validator.add_rule("dimensions", validate_dimensions)

        # Validate all fields
        validated_data = validator.validate({
            "object_id": "table_001",
            "dimensions": {"width": 1.0, "height": 0.8, "depth": 0.6},
        })
    """

    def __init__(self):
        self.rules: Dict[str, Callable[[Any], Any]] = {}

    def add_rule(self, field: str, validator: Callable[[Any], Any]) -> None:
        """Add validation rule for a field."""
        self.rules[field] = validator

    def validate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate all fields in data using registered rules.

        Args:
            data: Dictionary of data to validate

        Returns:
            Dictionary of validated data

        Raises:
            ValidationError: If any field fails validation
        """
        validated = {}

        for field, value in data.items():
            if field in self.rules:
                try:
                    validated[field] = self.rules[field](value)
                except ValidationError:
                    raise
                except Exception as e:
                    raise ValidationError(
                        f"Validation failed: {e}",
                        field=field,
                        value=value,
                    )
            else:
                # No rule registered, pass through
                validated[field] = value

        return validated
