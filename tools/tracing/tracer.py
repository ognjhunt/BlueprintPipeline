"""OpenTelemetry distributed tracing for BlueprintPipeline.

Provides distributed tracing capabilities to track requests across services
and understand cross-service dependencies and performance bottlenecks.

Key features:
- Automatic trace propagation across services
- Integration with Cloud Trace (GCP) and Jaeger
- Minimal overhead in FREE path when disabled
- Correlation with existing metrics and logs
"""

from __future__ import annotations

import functools
import logging
import os
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional, TypeVar, cast

from tools.config.env import parse_bool_env
from tools.tracing.correlation import ensure_request_id

logger = logging.getLogger(__name__)

# Type variable for decorated functions
F = TypeVar("F", bound=Callable[..., Any])

# Global tracer instance
_tracer: Optional["TracerWrapper"] = None
_tracing_enabled: bool = False


class TracerWrapper:
    """Wrapper for OpenTelemetry tracer with graceful degradation."""

    def __init__(self, service_name: str = "blueprint-pipeline"):
        """Initialize tracer wrapper.

        Args:
            service_name: Name of the service for tracing
        """
        self.service_name = service_name
        self.enabled = False
        self.tracer = None
        self.trace_module = None

        try:
            # Import OpenTelemetry
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.sdk.resources import Resource

            self.trace_module = trace

            # Create resource with service name
            resource = Resource.create(
                {
                    "service.name": service_name,
                    "service.version": os.getenv("SERVICE_VERSION", "unknown"),
                    "deployment.environment": os.getenv("PIPELINE_ENV", "local"),
                }
            )

            # Create tracer provider
            provider = TracerProvider(resource=resource)

            # Configure exporter based on environment
            exporter = self._configure_exporter()
            if exporter:
                processor = BatchSpanProcessor(exporter)
                provider.add_span_processor(processor)

            # Set global tracer provider
            trace.set_tracer_provider(provider)

            # Get tracer
            self.tracer = trace.get_tracer(__name__)
            self.enabled = True

            logger.info(f"OpenTelemetry tracing initialized for {service_name}")

        except ImportError:
            logger.debug(
                "OpenTelemetry not installed. Tracing disabled. "
                "Install with: pip install opentelemetry-api opentelemetry-sdk"
            )
        except Exception as e:
            logger.warning(f"Failed to initialize tracing: {e}")

    def _configure_exporter(self):
        """Configure trace exporter based on environment."""
        # Try Cloud Trace (GCP) first
        if os.getenv("GCP_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT"):
            try:
                from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

                project_id = os.getenv("GCP_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")
                logger.info(f"Using Cloud Trace exporter (project: {project_id})")
                return CloudTraceSpanExporter(project_id=project_id)
            except ImportError:
                logger.debug(
                    "Cloud Trace exporter not available. "
                    "Install with: pip install opentelemetry-exporter-gcp-trace"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize Cloud Trace exporter: {e}")

        # Try Jaeger
        jaeger_endpoint = os.getenv("JAEGER_ENDPOINT")
        if jaeger_endpoint:
            try:
                from opentelemetry.exporter.jaeger.thrift import JaegerExporter

                logger.info(f"Using Jaeger exporter (endpoint: {jaeger_endpoint})")
                return JaegerExporter(
                    agent_host_name=jaeger_endpoint.split(":")[0],
                    agent_port=int(jaeger_endpoint.split(":")[1]) if ":" in jaeger_endpoint else 6831,
                )
            except ImportError:
                logger.debug(
                    "Jaeger exporter not available. "
                    "Install with: pip install opentelemetry-exporter-jaeger"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize Jaeger exporter: {e}")

        # Try OTLP (standard protocol)
        otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        if otlp_endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

                logger.info(f"Using OTLP exporter (endpoint: {otlp_endpoint})")
                return OTLPSpanExporter(endpoint=otlp_endpoint)
            except ImportError:
                logger.debug(
                    "OTLP exporter not available. "
                    "Install with: pip install opentelemetry-exporter-otlp"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize OTLP exporter: {e}")

        # Console exporter for local development
        if os.getenv("PIPELINE_ENV") == "local" and os.getenv("DEBUG_TRACING") == "true":
            try:
                from opentelemetry.sdk.trace.export import ConsoleSpanExporter

                logger.info("Using Console exporter for local debugging")
                return ConsoleSpanExporter()
            except Exception as e:
                logger.debug(f"Console exporter failed: {e}")

        logger.info("No trace exporter configured. Tracing disabled.")
        return None

    @contextmanager
    def start_span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        kind: Optional[str] = None,
    ):
        """Start a new span.

        Args:
            name: Span name
            attributes: Optional span attributes
            kind: Span kind (CLIENT, SERVER, INTERNAL, etc.)

        Yields:
            Span context
        """
        if not self.enabled or not self.tracer or not self.trace_module:
            # No-op context manager when tracing is disabled
            yield None
            return

        # Map kind string to SpanKind enum
        span_kind = None
        if kind:
            from opentelemetry.trace import SpanKind

            kind_map = {
                "CLIENT": SpanKind.CLIENT,
                "SERVER": SpanKind.SERVER,
                "INTERNAL": SpanKind.INTERNAL,
                "PRODUCER": SpanKind.PRODUCER,
                "CONSUMER": SpanKind.CONSUMER,
            }
            span_kind = kind_map.get(kind.upper())

        with self.tracer.start_as_current_span(name, kind=span_kind) as span:
            # Set attributes
            if attributes and span.is_recording():
                for key, value in attributes.items():
                    # Convert value to string if needed
                    if isinstance(value, (list, dict)):
                        value = str(value)
                    span.set_attribute(key, value)

            yield span

    def get_current_span(self):
        """Get the current active span."""
        if not self.enabled or not self.trace_module:
            return None

        return self.trace_module.get_current_span()

    def set_attribute(self, key: str, value: Any):
        """Set attribute on current span."""
        span = self.get_current_span()
        if span and span.is_recording():
            if isinstance(value, (list, dict)):
                value = str(value)
            span.set_attribute(key, value)

    def set_error(self, error: Exception):
        """Record an error on the current span."""
        span = self.get_current_span()
        if span and span.is_recording():
            from opentelemetry.trace import Status, StatusCode

            span.set_status(Status(StatusCode.ERROR, str(error)))
            span.record_exception(error)

    def inject_context(self) -> Dict[str, str]:
        """Inject trace context for propagation.

        Returns:
            Dictionary of trace context headers
        """
        if not self.enabled or not self.trace_module:
            return {}

        try:
            from opentelemetry.propagate import inject

            carrier: Dict[str, str] = {}
            inject(carrier)
            return carrier
        except Exception as e:
            logger.debug(f"Failed to inject trace context: {e}")
            return {}

    def extract_context(self, carrier: Dict[str, str]):
        """Extract trace context from headers.

        Args:
            carrier: Dictionary of trace context headers
        """
        if not self.enabled or not self.trace_module:
            return None

        try:
            from opentelemetry.propagate import extract

            return extract(carrier)
        except Exception as e:
            logger.debug(f"Failed to extract trace context: {e}")
            return None


def init_tracing(
    service_name: str = "blueprint-pipeline",
    enabled: Optional[bool] = None,
) -> bool:
    """Initialize distributed tracing.

    Args:
        service_name: Name of the service
        enabled: Explicitly enable/disable tracing (auto-detected if None)

    Returns:
        True if tracing was successfully initialized
    """
    global _tracer, _tracing_enabled

    # Auto-detect if not explicitly set
    if enabled is None:
        # Enable in production or when explicitly configured
        enabled = (
            os.getenv("PIPELINE_ENV") == "production"
            or parse_bool_env(os.getenv("ENABLE_TRACING"), default=False)
            or bool(os.getenv("GCP_PROJECT_ID"))
            or bool(os.getenv("JAEGER_ENDPOINT"))
            or bool(os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"))
        )

    _tracing_enabled = enabled

    if not enabled:
        logger.debug("Tracing disabled")
        return False

    _tracer = TracerWrapper(service_name=service_name)
    return _tracer.enabled


def get_tracer() -> TracerWrapper:
    """Get the global tracer instance.

    Returns:
        TracerWrapper instance (may be disabled)
    """
    global _tracer, _tracing_enabled

    if _tracer is None:
        # Initialize with auto-detection
        init_tracing()

    # Return a no-op wrapper if tracing is disabled
    if _tracer is None or not _tracing_enabled:
        _tracer = TracerWrapper()

    return _tracer


@contextmanager
def trace_job(
    job_name: str,
    scene_id: Optional[str] = None,
    **attributes: Any,
):
    """Context manager to trace a job execution.

    Args:
        job_name: Name of the job
        scene_id: Optional scene identifier
        **attributes: Additional attributes to set

    Example:
        with trace_job("regen3d-job", scene_id="scene_001", step="physics"):
            # Process scene
            pass
    """
    tracer = get_tracer()

    request_id = ensure_request_id()
    attrs = {
        "job.name": job_name,
        "pipeline.step": job_name.replace("-job", ""),
        "request.id": request_id,
    }
    if scene_id:
        attrs["scene.id"] = scene_id
    attrs.update(attributes)

    with tracer.start_span(
        f"job:{job_name}",
        attributes=attrs,
        kind="INTERNAL",
    ) as span:
        yield span


def trace_function(
    name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
) -> Callable[[F], F]:
    """Decorator to trace a function.

    Args:
        name: Optional span name (defaults to function name)
        attributes: Optional static attributes

    Example:
        @trace_function(attributes={"component": "physics"})
        def process_physics(obj_id: str):
            pass
    """

    def decorator(func: F) -> F:
        span_name = name or f"{func.__module__}.{func.__name__}"

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_tracer()

            if not tracer.enabled:
                # Fast path when tracing is disabled
                return func(*args, **kwargs)

            attrs = attributes.copy() if attributes else {}
            attrs["function.name"] = func.__name__
            attrs["function.module"] = func.__module__

            with tracer.start_span(span_name, attributes=attrs, kind="INTERNAL"):
                return func(*args, **kwargs)

        return cast(F, wrapper)

    return decorator


def set_trace_attribute(key: str, value: Any):
    """Set an attribute on the current span.

    Args:
        key: Attribute key
        value: Attribute value

    Example:
        with trace_job("simready-job"):
            set_trace_attribute("object.id", "mug_0")
            set_trace_attribute("physics.mode", "deterministic")
    """
    tracer = get_tracer()
    tracer.set_attribute(key, value)


def set_trace_error(error: Exception):
    """Record an error on the current span.

    Args:
        error: Exception to record

    Example:
        with trace_job("usd-job"):
            try:
                process()
            except Exception as e:
                set_trace_error(e)
                raise
    """
    tracer = get_tracer()
    tracer.set_error(error)


def get_current_span():
    """Get the current active span.

    Returns:
        Current span or None
    """
    tracer = get_tracer()
    return tracer.get_current_span()


def inject_trace_context() -> Dict[str, str]:
    """Inject trace context for cross-service propagation.

    Returns:
        Dictionary of trace context headers

    Example:
        # Service A
        headers = inject_trace_context()
        response = requests.post(url, headers=headers)

        # Service B
        context = extract_trace_context(request.headers)
        with tracer.start_span("process", context=context):
            pass
    """
    tracer = get_tracer()
    carrier = tracer.inject_context()
    carrier["x-request-id"] = ensure_request_id()
    return carrier


def extract_trace_context(carrier: Dict[str, str]):
    """Extract trace context from headers.

    Args:
        carrier: Dictionary of trace context headers

    Returns:
        Extracted context

    Example:
        context = extract_trace_context(request.headers)
        with tracer.start_span("process", context=context):
            pass
    """
    tracer = get_tracer()
    return tracer.extract_context(carrier)
