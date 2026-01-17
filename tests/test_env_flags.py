from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.config.env_flags import env_flag


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1", True),
        ("true", True),
        ("TRUE", True),
        (" yes ", True),
        ("\ton\n", True),
        ("y", True),
        ("0", False),
        ("false", False),
        ("no", False),
        ("off", False),
        ("", False),
        ("   ", False),
    ],
)
def test_env_flag_truthy_falsey(value, expected):
    assert env_flag(value) is expected


def test_env_flag_handles_none_with_default():
    assert env_flag(None) is False
    assert env_flag(None, default=True) is True
