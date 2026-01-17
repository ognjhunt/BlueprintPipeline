"""
Partial failure handling utilities.

Provides utilities for handling partial failures in batch operations,
ensuring successful items are saved even when some items fail.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar

from tools.utils.atomic_write import write_json_atomic

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class PartialFailureResult(Generic[R]):
    """Result from an operation with partial failures."""
    successful: List[R] = field(default_factory=list)
    failed: List[Dict[str, Any]] = field(default_factory=list)
    total_attempted: int = 0
    duration_seconds: float = 0.0

    @property
    def success_count(self) -> int:
        """Number of successful items."""
        return len(self.successful)

    @property
    def failure_count(self) -> int:
        """Number of failed items."""
        return len(self.failed)

    @property
    def success_rate(self) -> float:
        """Success rate (0.0 to 1.0)."""
        if self.total_attempted == 0:
            return 0.0
        return self.success_count / self.total_attempted

    @property
    def all_succeeded(self) -> bool:
        """True if all items succeeded."""
        return self.failure_count == 0

    @property
    def all_failed(self) -> bool:
        """True if all items failed."""
        return self.success_count == 0

    def meets_threshold(self, min_success_rate: float = 0.5) -> bool:
        """Check if success rate meets minimum threshold."""
        return self.success_rate >= min_success_rate

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "total_attempted": self.total_attempted,
            "success_rate": self.success_rate,
            "duration_seconds": self.duration_seconds,
            "failed_items": self.failed,
        }


def process_with_partial_failure(
    items: List[T],
    process_fn: Callable[[T], R],
    min_success_rate: float = 0.5,
    save_progress: bool = True,
    progress_file: Optional[Path] = None,
    item_id_fn: Optional[Callable[[T], str]] = None,
) -> PartialFailureResult[R]:
    """
    Process items with partial failure handling.

    Continues processing even when individual items fail, and saves
    successful results. Raises exception only if success rate is below
    minimum threshold.

    Args:
        items: List of items to process
        process_fn: Function to process each item
        min_success_rate: Minimum success rate (0.0-1.0) to avoid raising exception
        save_progress: If True, save successful items incrementally
        progress_file: File to save progress (required if save_progress=True)
        item_id_fn: Function to extract item ID for logging

    Returns:
        PartialFailureResult with successful and failed items

    Raises:
        PartialFailureError: If success rate is below min_success_rate

    Example:
        # Generate episodes with partial failure handling
        result = process_with_partial_failure(
            tasks,
            generate_episode,
            min_success_rate=0.5,
            save_progress=True,
            progress_file=Path("progress.json"),
            item_id_fn=lambda t: t["task_id"],
        )

        print(f"Success: {result.success_count}/{result.total_attempted}")
        print(f"Failed: {result.failure_count}")

        # Access successful results
        for episode in result.successful:
            export_episode(episode)

        # Review failures
        for failure in result.failed:
            print(f"Failed {failure['item_id']}: {failure['error']}")
    """
    start_time = time.time()
    result = PartialFailureResult()
    result.total_attempted = len(items)

    logger.info(
        f"Processing {len(items)} items with partial failure handling "
        f"(min_success_rate: {min_success_rate})"
    )

    for i, item in enumerate(items):
        # Get item ID for logging
        item_id = item_id_fn(item) if item_id_fn else f"item_{i}"

        try:
            # Process item
            processed = process_fn(item)
            result.successful.append(processed)

            logger.debug(
                f"Successfully processed {item_id} "
                f"({i + 1}/{len(items)})"
            )

            # Save progress incrementally
            if save_progress and progress_file:
                _save_progress(progress_file, result)

        except Exception as e:
            # Record failure
            failure_info = {
                "item_index": i,
                "item_id": item_id,
                "item": item,
                "error": str(e),
                "error_type": type(e).__name__,
                "timestamp": time.time(),
            }
            result.failed.append(failure_info)

            logger.warning(
                f"Failed to process {item_id}: {e}",
                extra=failure_info,
            )

    result.duration_seconds = time.time() - start_time

    # Log summary
    logger.info(
        f"Partial failure processing complete: "
        f"{result.success_count}/{result.total_attempted} successful "
        f"({result.success_rate * 100:.1f}% success rate) "
        f"in {result.duration_seconds:.2f}s"
    )

    # Check if we meet minimum success rate
    if not result.meets_threshold(min_success_rate):
        error_msg = (
            f"Success rate {result.success_rate:.1%} below minimum "
            f"{min_success_rate:.1%} ({result.success_count}/{result.total_attempted} succeeded)"
        )
        logger.error(error_msg)
        raise PartialFailureError(error_msg, result)

    return result


def _save_progress(progress_file: Path, result: PartialFailureResult) -> None:
    """Save progress to file."""
    try:
        write_json_atomic(progress_file, result.to_dict(), indent=2)

    except Exception as e:
        logger.warning(f"Failed to save progress: {e}")


class PartialFailureError(Exception):
    """Raised when success rate is below minimum threshold."""

    def __init__(self, message: str, result: PartialFailureResult):
        self.result = result
        super().__init__(message)


class PartialFailureHandler:
    """
    Reusable partial failure handler with configurable behavior.

    Example:
        handler = PartialFailureHandler(
            min_success_rate=0.5,
            save_successful=True,
            output_dir=Path("outputs"),
        )

        # Process batches
        for batch in batches:
            result = handler.process_batch(
                batch,
                process_fn=generate_episode,
            )

            # Handle successful items
            for item in result.successful:
                export_item(item)

        # Get overall statistics
        stats = handler.get_stats()
        print(f"Total success rate: {stats['overall_success_rate']:.1%}")
    """

    def __init__(
        self,
        min_success_rate: float = 0.5,
        save_successful: bool = True,
        output_dir: Optional[Path] = None,
        failure_report_path: Optional[Path] = None,
    ):
        self.min_success_rate = min_success_rate
        self.save_successful = save_successful
        self.output_dir = output_dir
        self.failure_report_path = failure_report_path

        # Statistics
        self._total_attempted = 0
        self._total_successful = 0
        self._total_failed = 0
        self._all_failures: List[Dict[str, Any]] = []

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

    def process_batch(
        self,
        items: List[T],
        process_fn: Callable[[T], R],
        item_id_fn: Optional[Callable[[T], str]] = None,
        batch_name: Optional[str] = None,
    ) -> PartialFailureResult[R]:
        """
        Process a batch with partial failure handling.

        Args:
            items: List of items to process
            process_fn: Function to process each item
            item_id_fn: Function to extract item ID
            batch_name: Name for this batch (for logging)

        Returns:
            PartialFailureResult
        """
        batch_name = batch_name or f"batch_{self._total_attempted}"

        logger.info(f"Processing {batch_name}: {len(items)} items")

        # Determine progress file
        progress_file = None
        if self.save_successful and self.output_dir:
            progress_file = self.output_dir / f"{batch_name}_progress.json"

        # Process with partial failure handling
        result = process_with_partial_failure(
            items,
            process_fn,
            min_success_rate=self.min_success_rate,
            save_progress=self.save_successful,
            progress_file=progress_file,
            item_id_fn=item_id_fn,
        )

        # Update statistics
        self._total_attempted += result.total_attempted
        self._total_successful += result.success_count
        self._total_failed += result.failure_count
        self._all_failures.extend(result.failed)

        # Write failure report
        if self.failure_report_path and result.failed:
            self._write_failure_report()

        return result

    def _write_failure_report(self) -> None:
        """Write failure report to file."""
        try:
            self.failure_report_path.parent.mkdir(parents=True, exist_ok=True)

            report = {
                "total_failures": self._total_failed,
                "total_attempted": self._total_attempted,
                "failure_rate": self._total_failed / self._total_attempted if self._total_attempted > 0 else 0,
                "failures": self._all_failures,
            }

            with open(self.failure_report_path, 'w') as f:
                json.dump(report, f, indent=2)

            logger.info(f"Wrote failure report to {self.failure_report_path}")

        except Exception as e:
            logger.error(f"Failed to write failure report: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get overall processing statistics."""
        return {
            "total_attempted": self._total_attempted,
            "total_successful": self._total_successful,
            "total_failed": self._total_failed,
            "overall_success_rate": (
                self._total_successful / self._total_attempted
                if self._total_attempted > 0
                else 0.0
            ),
            "failures": self._all_failures,
        }

    def reset(self) -> None:
        """Reset statistics."""
        self._total_attempted = 0
        self._total_successful = 0
        self._total_failed = 0
        self._all_failures.clear()


def save_successful_items(
    items: List[R],
    output_path: Path,
    format: str = "json",
) -> None:
    """
    Save successful items to file.

    Args:
        items: List of successful items
        output_path: Output file path
        format: Output format ("json" or "jsonl")

    Example:
        # Save successful episodes
        save_successful_items(
            result.successful,
            Path("outputs/successful_episodes.json"),
        )
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        with open(output_path, 'w') as f:
            # Convert items to dict if they have to_dict method
            serializable = []
            for item in items:
                if hasattr(item, 'to_dict'):
                    serializable.append(item.to_dict())
                elif hasattr(item, '__dict__'):
                    serializable.append(vars(item))
                else:
                    serializable.append(item)

            json.dump(serializable, f, indent=2)

    elif format == "jsonl":
        with open(output_path, 'w') as f:
            for item in items:
                if hasattr(item, 'to_dict'):
                    line = json.dumps(item.to_dict())
                elif hasattr(item, '__dict__'):
                    line = json.dumps(vars(item))
                else:
                    line = json.dumps(item)
                f.write(line + '\n')

    else:
        raise ValueError(f"Unsupported format: {format}")

    logger.info(
        f"Saved {len(items)} successful items to {output_path}"
    )
