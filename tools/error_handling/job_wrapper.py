"""
Shared job wrapper for publishing failures to the dead letter queue.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

from .dead_letter import DeadLetterMessage, get_dead_letter_queue
from .errors import ErrorContext, PipelineError

logger = logging.getLogger(__name__)

# Import tracing (graceful degradation if not available)
try:
    from tools.tracing import trace_job, set_trace_attribute, set_trace_error
    TRACING_AVAILABLE = True
except ImportError:
    TRACING_AVAILABLE = False
    logger.debug("Tracing not available")


def _build_context(
    *,
    scene_id: Optional[str],
    job_type: str,
    step: str,
    input_params: Optional[Dict[str, Any]],
) -> ErrorContext:
    context = ErrorContext(scene_id=scene_id or "", step=step or "")
    if job_type:
        context.additional["job_type"] = job_type
    if input_params is not None:
        context.additional["input_params"] = input_params
    return context


def _ensure_context(
    pipeline_error: PipelineError,
    *,
    scene_id: Optional[str],
    job_type: str,
    step: str,
    input_params: Optional[Dict[str, Any]],
) -> None:
    context = pipeline_error.context or ErrorContext()
    if scene_id and not context.scene_id:
        context.scene_id = scene_id
    if step and not context.step:
        context.step = step
    if job_type and "job_type" not in context.additional:
        context.additional["job_type"] = job_type
    if input_params is not None and "input_params" not in context.additional:
        context.additional["input_params"] = input_params
    pipeline_error.context = context


def publish_failure(
    exc: Exception,
    *,
    scene_id: Optional[str],
    job_type: str,
    step: str,
    input_params: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Publish a job failure to the dead letter queue.

    Returns the DLQ message_id if available.
    """
    if isinstance(exc, PipelineError):
        pipeline_error = exc
        _ensure_context(
            pipeline_error,
            scene_id=scene_id,
            job_type=job_type,
            step=step,
            input_params=input_params,
        )
    else:
        context = _build_context(
            scene_id=scene_id,
            job_type=job_type,
            step=step,
            input_params=input_params,
        )
        pipeline_error = PipelineError.from_exception(exc, context=context)

    message = DeadLetterMessage.from_pipeline_error(
        pipeline_error,
        original_payload=input_params or {},
    )
    if scene_id and not message.scene_id:
        message.scene_id = scene_id
    if job_type and not message.job_type:
        message.job_type = job_type
    if step and not message.step:
        message.step = step

    logger.info(
        "Publishing failure to DLQ (job_type=%s, step=%s, scene_id=%s)",
        job_type,
        step,
        scene_id,
    )
    dlq = get_dead_letter_queue()
    try:
        message_id = dlq.publish(message)
        logger.info("DLQ publish attempt complete (message_id=%s)", message_id)
        return message_id
    except Exception:  # pragma: no cover - DLQ failures should not mask original error
        logger.exception("Failed to publish dead letter message")
        return None


def run_job_with_dead_letter_queue(
    job_fn: Callable[[], Optional[int]],
    *,
    scene_id: Optional[str],
    job_type: str,
    step: str,
    input_params: Optional[Dict[str, Any]] = None,
    failure_exit_code: int = 1,
) -> int:
    """
    Run a job entrypoint and publish failures to the DLQ.

    Includes distributed tracing when enabled.
    """
    # Context manager for tracing (no-op if tracing not available)
    if TRACING_AVAILABLE:
        trace_context = trace_job(
            job_name=job_type,
            scene_id=scene_id,
            step=step,
            **(input_params or {}),
        )
    else:
        from contextlib import nullcontext
        trace_context = nullcontext()

    with trace_context:
        # Set trace attributes
        if TRACING_AVAILABLE and input_params:
            for key, value in input_params.items():
                # Only set simple types as attributes
                if isinstance(value, (str, int, float, bool)):
                    set_trace_attribute(f"input.{key}", value)

        try:
            result = job_fn()

            # Mark success in trace
            if TRACING_AVAILABLE:
                set_trace_attribute("job.status", "success")
                set_trace_attribute("job.exit_code", result if isinstance(result, int) else 0)

            if isinstance(result, int):
                return result
            return 0

        except SystemExit as exc:
            if exc.code not in (0, None):
                # Mark failure in trace
                if TRACING_AVAILABLE:
                    set_trace_attribute("job.status", "failure")
                    set_trace_attribute("job.exit_code", exc.code)
                    set_trace_error(RuntimeError(f"Job exited with code {exc.code}"))

                publish_failure(
                    RuntimeError(f"Job exited with code {exc.code}"),
                    scene_id=scene_id,
                    job_type=job_type,
                    step=step,
                    input_params=input_params,
                )
            raise

        except Exception as exc:
            # Mark error in trace
            if TRACING_AVAILABLE:
                set_trace_attribute("job.status", "error")
                set_trace_attribute("error.type", type(exc).__name__)
                set_trace_attribute("error.message", str(exc))
                set_trace_error(exc)

            publish_failure(
                exc,
                scene_id=scene_id,
                job_type=job_type,
                step=step,
                input_params=input_params,
            )
            return failure_exit_code
