"""Shared configuration helpers for Genie Sim adapters and clients."""

from __future__ import annotations

import os
from typing import Mapping, Optional

GENIESIM_HOST_ENV = "GENIESIM_HOST"
GENIESIM_PORT_ENV = "GENIESIM_PORT"

DEFAULT_GENIESIM_HOST = "localhost"
DEFAULT_GENIESIM_PORT = 50051


def get_geniesim_host(env: Optional[Mapping[str, str]] = None) -> str:
    """Return the configured Genie Sim host."""
    source = env or os.environ
    return source.get(GENIESIM_HOST_ENV, DEFAULT_GENIESIM_HOST)


def get_geniesim_port(env: Optional[Mapping[str, str]] = None) -> int:
    """Return the configured Genie Sim gRPC port."""
    source = env or os.environ
    port_value = source.get(GENIESIM_PORT_ENV, str(DEFAULT_GENIESIM_PORT))
    return int(port_value)
