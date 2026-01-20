"""Shared configuration helpers for Genie Sim adapters and clients."""

from __future__ import annotations

import os
from typing import Mapping, Optional

from tools.config.env import parse_float_env, parse_int_env
from tools.config.production_mode import resolve_env_with_legacy
GENIESIM_HOST_ENV = "GENIESIM_HOST"
GENIESIM_PORT_ENV = "GENIESIM_PORT"
GENIESIM_TLS_CERT_ENV = "GENIESIM_TLS_CERT"
GENIESIM_TLS_KEY_ENV = "GENIESIM_TLS_KEY"
GENIESIM_TLS_CA_ENV = "GENIESIM_TLS_CA"
GENIESIM_AUTH_TOKEN_ENV = "GENIESIM_AUTH_TOKEN"
GENIESIM_AUTH_TOKEN_PATH_ENV = "GENIESIM_AUTH_TOKEN_PATH"
GENIESIM_AUTH_CERT_ENV = "GENIESIM_AUTH_CERT"
GENIESIM_AUTH_KEY_ENV = "GENIESIM_AUTH_KEY"
GENIESIM_GRPC_TIMEOUT_S_ENV = "GENIESIM_GRPC_TIMEOUT_S"
# Legacy alias for GENIESIM_GRPC_TIMEOUT_S_ENV.
GENIESIM_TIMEOUT_LEGACY_ENV = "GENIESIM_TIMEOUT"
GENIESIM_CIRCUIT_BREAKER_FAILURE_THRESHOLD_ENV = "GENIESIM_CIRCUIT_BREAKER_FAILURE_THRESHOLD"
GENIESIM_CIRCUIT_BREAKER_SUCCESS_THRESHOLD_ENV = "GENIESIM_CIRCUIT_BREAKER_SUCCESS_THRESHOLD"
GENIESIM_CIRCUIT_BREAKER_RECOVERY_TIMEOUT_S_ENV = "GENIESIM_CIRCUIT_BREAKER_RECOVERY_TIMEOUT_S"
GENIESIM_CIRCUIT_BREAKER_BACKOFF_SECONDS_ENV = "GENIESIM_CIRCUIT_BREAKER_BACKOFF_SECONDS"
GENIESIM_TASK_CONFIDENCE_THRESHOLD_ENV = "GENIESIM_TASK_CONFIDENCE_THRESHOLD"
GENIESIM_TASK_SIZE_SMALL_THRESHOLD_ENV = "GENIESIM_TASK_SIZE_SMALL_THRESHOLD"
GENIESIM_TASK_SIZE_LARGE_THRESHOLD_ENV = "GENIESIM_TASK_SIZE_LARGE_THRESHOLD"
GENIESIM_TASK_MAX_PER_OBJECT_ENV = "GENIESIM_TASK_MAX_PER_OBJECT"

DEFAULT_GENIESIM_HOST = "localhost"
DEFAULT_GENIESIM_PORT = 50051
DEFAULT_GENIESIM_GRPC_TIMEOUT_S = 30.0
DEFAULT_CIRCUIT_BREAKER_FAILURE_THRESHOLD = 3
DEFAULT_CIRCUIT_BREAKER_SUCCESS_THRESHOLD = 2
DEFAULT_CIRCUIT_BREAKER_RECOVERY_TIMEOUT_S = 30.0
DEFAULT_CIRCUIT_BREAKER_BACKOFF_SECONDS = 2.0
DEFAULT_TASK_CONFIDENCE_THRESHOLD = 0.8
DEFAULT_TASK_SIZE_SMALL_THRESHOLD = 0.05
DEFAULT_TASK_SIZE_LARGE_THRESHOLD = 0.3
DEFAULT_TASK_MAX_PER_OBJECT = 3


def get_geniesim_host(env: Optional[Mapping[str, str]] = None) -> str:
    """Return the configured Genie Sim host."""
    source = env or os.environ
    return source.get(GENIESIM_HOST_ENV, DEFAULT_GENIESIM_HOST)


def get_geniesim_port(env: Optional[Mapping[str, str]] = None) -> int:
    """Return the configured Genie Sim gRPC port."""
    source = env or os.environ
    return parse_int_env(
        source.get(GENIESIM_PORT_ENV),
        default=DEFAULT_GENIESIM_PORT,
        min_value=1,
        max_value=65535,
        name=GENIESIM_PORT_ENV,
    )


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


def get_geniesim_grpc_timeout_s(env: Optional[Mapping[str, str]] = None) -> float:
    """Return the configured gRPC timeout in seconds (legacy: GENIESIM_TIMEOUT)."""
    source = env or os.environ
    value, _ = resolve_env_with_legacy(
        canonical_names=(GENIESIM_GRPC_TIMEOUT_S_ENV,),
        legacy_names=(GENIESIM_TIMEOUT_LEGACY_ENV,),
        env=source,
        preferred_name=GENIESIM_GRPC_TIMEOUT_S_ENV,
    )
    return parse_float_env(
        value,
        default=DEFAULT_GENIESIM_GRPC_TIMEOUT_S,
        min_value=0.0,
        name=GENIESIM_GRPC_TIMEOUT_S_ENV,
    )


def get_geniesim_circuit_breaker_failure_threshold(env: Optional[Mapping[str, str]] = None) -> int:
    """Return the circuit breaker failure threshold."""
    source = env or os.environ
    return parse_int_env(
        source.get(GENIESIM_CIRCUIT_BREAKER_FAILURE_THRESHOLD_ENV),
        default=DEFAULT_CIRCUIT_BREAKER_FAILURE_THRESHOLD,
        min_value=0,
        name=GENIESIM_CIRCUIT_BREAKER_FAILURE_THRESHOLD_ENV,
    )


def get_geniesim_circuit_breaker_success_threshold(env: Optional[Mapping[str, str]] = None) -> int:
    """Return the circuit breaker success threshold."""
    source = env or os.environ
    return parse_int_env(
        source.get(GENIESIM_CIRCUIT_BREAKER_SUCCESS_THRESHOLD_ENV),
        default=DEFAULT_CIRCUIT_BREAKER_SUCCESS_THRESHOLD,
        min_value=0,
        name=GENIESIM_CIRCUIT_BREAKER_SUCCESS_THRESHOLD_ENV,
    )


def get_geniesim_circuit_breaker_recovery_timeout_s(env: Optional[Mapping[str, str]] = None) -> float:
    """Return the circuit breaker recovery timeout in seconds."""
    source = env or os.environ
    value = source.get(GENIESIM_CIRCUIT_BREAKER_RECOVERY_TIMEOUT_S_ENV)
    if value is None:
        value = source.get(GENIESIM_CIRCUIT_BREAKER_BACKOFF_SECONDS_ENV)
    return parse_float_env(
        value,
        default=DEFAULT_CIRCUIT_BREAKER_RECOVERY_TIMEOUT_S,
        min_value=0.0,
        name=GENIESIM_CIRCUIT_BREAKER_RECOVERY_TIMEOUT_S_ENV,
    )


def get_geniesim_circuit_breaker_backoff_seconds(env: Optional[Mapping[str, str]] = None) -> float:
    """Return the circuit breaker backoff window in seconds."""
    source = env or os.environ
    return parse_float_env(
        source.get(GENIESIM_CIRCUIT_BREAKER_BACKOFF_SECONDS_ENV),
        default=DEFAULT_CIRCUIT_BREAKER_BACKOFF_SECONDS,
        min_value=0.0,
        name=GENIESIM_CIRCUIT_BREAKER_BACKOFF_SECONDS_ENV,
    )


def get_geniesim_task_confidence_threshold(env: Optional[Mapping[str, str]] = None) -> float:
    """Return the affordance confidence threshold for task priority boosts."""
    source = env or os.environ
    return parse_float_env(
        source.get(GENIESIM_TASK_CONFIDENCE_THRESHOLD_ENV),
        default=DEFAULT_TASK_CONFIDENCE_THRESHOLD,
        min_value=0.0,
        max_value=1.0,
        name=GENIESIM_TASK_CONFIDENCE_THRESHOLD_ENV,
    )


def get_geniesim_task_size_small_threshold(env: Optional[Mapping[str, str]] = None) -> float:
    """Return the size threshold below which objects are considered small."""
    source = env or os.environ
    return parse_float_env(
        source.get(GENIESIM_TASK_SIZE_SMALL_THRESHOLD_ENV),
        default=DEFAULT_TASK_SIZE_SMALL_THRESHOLD,
        min_value=0.0,
        name=GENIESIM_TASK_SIZE_SMALL_THRESHOLD_ENV,
    )


def get_geniesim_task_size_large_threshold(env: Optional[Mapping[str, str]] = None) -> float:
    """Return the size threshold above which objects are considered large."""
    source = env or os.environ
    return parse_float_env(
        source.get(GENIESIM_TASK_SIZE_LARGE_THRESHOLD_ENV),
        default=DEFAULT_TASK_SIZE_LARGE_THRESHOLD,
        min_value=0.0,
        name=GENIESIM_TASK_SIZE_LARGE_THRESHOLD_ENV,
    )


def get_geniesim_task_max_per_object(env: Optional[Mapping[str, str]] = None) -> int:
    """Return the max number of tasks to keep per object."""
    source = env or os.environ
    return parse_int_env(
        source.get(GENIESIM_TASK_MAX_PER_OBJECT_ENV),
        default=DEFAULT_TASK_MAX_PER_OBJECT,
        min_value=1,
        name=GENIESIM_TASK_MAX_PER_OBJECT_ENV,
    )
