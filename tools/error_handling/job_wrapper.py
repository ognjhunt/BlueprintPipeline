"""
Shared job wrapper for publishing failures to the dead letter queue.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

from .dead_letter import DeadLetterMessage, get_dead_letter_queue
from .errors import ErrorContext, PipelineError

logger = logging.getLogger(__name__)


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

    dlq = get_dead_letter_queue()
    try:
        return dlq.publish(message)
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
    """
    try:
        result = job_fn()
        if isinstance(result, int):
            return result
        return 0
    except SystemExit as exc:
        if exc.code not in (0, None):
            publish_failure(
                RuntimeError(f"Job exited with code {exc.code}"),
                scene_id=scene_id,
                job_type=job_type,
                step=step,
                input_params=input_params,
            )
        raise
    except Exception as exc:
        publish_failure(
            exc,
            scene_id=scene_id,
            job_type=job_type,
            step=step,
            input_params=input_params,
        )
        return failure_exit_code
