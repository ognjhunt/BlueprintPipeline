"""Shared configuration helpers for Genie Sim adapters and clients."""

from __future__ import annotations

import os
from typing import Mapping, Optional

GENIESIM_HOST_ENV = "GENIESIM_HOST"
GENIESIM_PORT_ENV = "GENIESIM_PORT"
GENIESIM_TLS_CERT_ENV = "GENIESIM_TLS_CERT"
GENIESIM_TLS_KEY_ENV = "GENIESIM_TLS_KEY"
GENIESIM_TLS_CA_ENV = "GENIESIM_TLS_CA"
GENIESIM_AUTH_TOKEN_ENV = "GENIESIM_AUTH_TOKEN"
GENIESIM_AUTH_TOKEN_PATH_ENV = "GENIESIM_AUTH_TOKEN_PATH"
GENIESIM_AUTH_CERT_ENV = "GENIESIM_AUTH_CERT"
GENIESIM_AUTH_KEY_ENV = "GENIESIM_AUTH_KEY"
GENIESIM_CIRCUIT_BREAKER_FAILURE_THRESHOLD_ENV = "GENIESIM_CIRCUIT_BREAKER_FAILURE_THRESHOLD"
GENIESIM_CIRCUIT_BREAKER_BACKOFF_SECONDS_ENV = "GENIESIM_CIRCUIT_BREAKER_BACKOFF_SECONDS"

DEFAULT_GENIESIM_HOST = "localhost"
DEFAULT_GENIESIM_PORT = 50051
DEFAULT_CIRCUIT_BREAKER_FAILURE_THRESHOLD = 3
DEFAULT_CIRCUIT_BREAKER_BACKOFF_SECONDS = 2.0


def get_geniesim_host(env: Optional[Mapping[str, str]] = None) -> str:
    """Return the configured Genie Sim host."""
    source = env or os.environ
    return source.get(GENIESIM_HOST_ENV, DEFAULT_GENIESIM_HOST)


def get_geniesim_port(env: Optional[Mapping[str, str]] = None) -> int:
    """Return the configured Genie Sim gRPC port."""
    source = env or os.environ
    port_value = source.get(GENIESIM_PORT_ENV, str(DEFAULT_GENIESIM_PORT))
    return int(port_value)


def get_geniesim_tls_cert_path(env: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """Return the configured TLS client cert path."""
    source = env or os.environ
    return source.get(GENIESIM_TLS_CERT_ENV)


def get_geniesim_tls_key_path(env: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """Return the configured TLS client key path."""
    source = env or os.environ
    return source.get(GENIESIM_TLS_KEY_ENV)


def get_geniesim_tls_ca_path(env: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """Return the configured TLS CA bundle path."""
    source = env or os.environ
    return source.get(GENIESIM_TLS_CA_ENV)


def get_geniesim_auth_token(env: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """Return the configured auth token."""
    source = env or os.environ
    return source.get(GENIESIM_AUTH_TOKEN_ENV)


def get_geniesim_auth_token_path(env: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """Return the configured auth token path."""
    source = env or os.environ
    return source.get(GENIESIM_AUTH_TOKEN_PATH_ENV)


def get_geniesim_auth_cert_path(env: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """Return the configured auth cert path."""
    source = env or os.environ
    return source.get(GENIESIM_AUTH_CERT_ENV)


def get_geniesim_auth_key_path(env: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """Return the configured auth key path."""
    source = env or os.environ
    return source.get(GENIESIM_AUTH_KEY_ENV)


def get_geniesim_circuit_breaker_failure_threshold(env: Optional[Mapping[str, str]] = None) -> int:
    """Return the circuit breaker failure threshold."""
    source = env or os.environ
    threshold = source.get(
        GENIESIM_CIRCUIT_BREAKER_FAILURE_THRESHOLD_ENV,
        str(DEFAULT_CIRCUIT_BREAKER_FAILURE_THRESHOLD),
    )
    return int(threshold)


def get_geniesim_circuit_breaker_backoff_seconds(env: Optional[Mapping[str, str]] = None) -> float:
    """Return the circuit breaker backoff window in seconds."""
    source = env or os.environ
    backoff = source.get(
        GENIESIM_CIRCUIT_BREAKER_BACKOFF_SECONDS_ENV,
        str(DEFAULT_CIRCUIT_BREAKER_BACKOFF_SECONDS),
    )
    return float(backoff)
