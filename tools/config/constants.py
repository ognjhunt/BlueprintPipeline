"""Shared configuration defaults for timeouts and quality thresholds."""

# Default request timeout for long-running inference or large payload transfers.
DEFAULT_HTTP_TIMEOUT_S = 300

# Default timeout for subprocesses that may download/load models or run GPU checks.
DEFAULT_PROCESS_TIMEOUT_S = 300

# Default quality thresholds used when configuration files/env overrides are unset.
DEFAULT_COLLISION_FREE_RATE_MIN = 0.90
DEFAULT_QUALITY_SCORE_MIN = 0.90
DEFAULT_AVERAGE_QUALITY_SCORE_MIN = 0.90
DEFAULT_SENSOR_CAPTURE_RATE_MIN = 0.95
DEFAULT_PHYSICS_VALIDATION_RATE_MIN = 0.95
