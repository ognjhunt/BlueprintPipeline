#!/usr/bin/env python3
"""Tests for Genie Sim client auth headers in local/mock modes."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GENIESIM_EXPORT_DIR = REPO_ROOT / "genie-sim-export-job"

sys.path.insert(0, str(GENIESIM_EXPORT_DIR))

from geniesim_client import GenieSimClient


def test_mock_mode_has_no_auth_header(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GENIESIM_MOCK_MODE", "true")
    client = GenieSimClient(mock_mode=True, validate_on_init=False)

    try:
        session = client.session
        assert "Authorization" not in session.headers
    finally:
        client.close()


def test_local_grpc_has_no_auth_header(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GENIESIM_MOCK_MODE", "false")
    client = GenieSimClient(mock_mode=False, validate_on_init=False)

    try:
        session = client.session
        assert "Authorization" not in session.headers
    finally:
        client.close()
