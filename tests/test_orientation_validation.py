import logging
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.validation import validate_quaternion, validate_rotation_matrix

pytestmark = pytest.mark.usefixtures("add_repo_to_path")


def test_validate_quaternion_normalizes_and_warns(caplog) -> None:
    caplog.set_level(logging.WARNING)

    normalized = validate_quaternion([2.0, 0.0, 0.0, 0.0], field_name="test_quaternion")

    assert normalized == [1.0, 0.0, 0.0, 0.0]
    assert "Normalized quaternion" in caplog.text


def test_validate_rotation_matrix_svd_fix_warns(caplog) -> None:
    caplog.set_level(logging.WARNING)

    matrix = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.1],
        [0.0, 0.0, 0.9],
    ])

    fixed = validate_rotation_matrix(matrix, field_name="test_rotation_matrix")

    assert np.allclose(fixed.T @ fixed, np.eye(3), atol=1e-6)
    assert np.isclose(np.linalg.det(fixed), 1.0, atol=1e-6)
    assert "Adjusted rotation matrix" in caplog.text
