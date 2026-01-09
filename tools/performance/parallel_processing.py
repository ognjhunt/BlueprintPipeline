"""
Parallel processing utilities for independent operations.

Provides utilities for parallelizing independent operations like:
- Object processing
- API calls
- File I/O
- Physics estimation
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class ParallelResult(Generic[T]):
    """Result from parallel processing."""
    successful: List[T] = field(default_factory=list)
    failed: List[Dict[str, Any]] = field(default_factory=list)
    total: int = 0
    duration: float = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total == 0:
            return 0.0
        return len(self.successful) / self.total

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate."""
        return 1.0 - self.success_rate


def process_parallel_threaded(
    items: List[T],
    process_fn: Callable[[T], R],
    max_workers: int = 10,
    timeout: Optional[float] = None,
    on_error: Optional[Callable[[T, Exception], None]] = None,
) -> ParallelResult[R]:
    """
    Process items in parallel using threads (for I/O-bound tasks).

    Args:
        items: List of items to process
        process_fn: Function to apply to each item
        max_workers: Maximum number of worker threads
        timeout: Timeout in seconds for each item
        on_error: Callback for handling errors (receives item and exception)

    Returns:
        ParallelResult with successful and failed items

    Example:
        # Process objects in parallel
        def estimate_physics(obj):
            return gemini_api.estimate(obj)

        results = process_parallel_threaded(
            objects,
            estimate_physics,
            max_workers=20,
            timeout=10.0,
        )

        print(f"Success: {len(results.successful)}/{results.total}")
        for failure in results.failed:
            print(f"Failed: {failure['item_index']} - {failure['error']}")
    """
    start_time = time.time()
    result = ParallelResult()
    result.total = len(items)

    logger.info(
        f"Processing {len(items)} items in parallel "
        f"(max_workers: {max_workers}, timeout: {timeout}s)"
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_item = {
            executor.submit(process_fn, item): (i, item)
            for i, item in enumerate(items)
        }

        # Collect results as they complete
        for future in as_completed(future_to_item, timeout=timeout):
            item_index, item = future_to_item[future]

            try:
                item_result = future.result(timeout=timeout)
                result.successful.append(item_result)

                logger.debug(f"Processed item {item_index + 1}/{len(items)}")

            except Exception as e:
                error_info = {
                    "item_index": item_index,
                    "item": item,
                    "error": str(e),
                    "error_type": type(e).__name__,
                }
                result.failed.append(error_info)

                logger.warning(
                    f"Failed to process item {item_index}: {e}",
                    extra=error_info,
                )

                if on_error:
                    try:
                        on_error(item, e)
                    except Exception as callback_error:
                        logger.error(
                            f"Error callback failed: {callback_error}"
                        )

    result.duration = time.time() - start_time

    logger.info(
        f"Parallel processing complete: "
        f"{len(result.successful)}/{result.total} successful "
        f"({result.success_rate * 100:.1f}%) "
        f"in {result.duration:.2f}s"
    )

    return result


def process_parallel_multiprocess(
    items: List[T],
    process_fn: Callable[[T], R],
    max_workers: int = 4,
    timeout: Optional[float] = None,
    on_error: Optional[Callable[[T, Exception], None]] = None,
) -> ParallelResult[R]:
    """
    Process items in parallel using processes (for CPU-bound tasks).

    Args:
        items: List of items to process
        process_fn: Function to apply to each item (must be picklable)
        max_workers: Maximum number of worker processes
        timeout: Timeout in seconds for each item
        on_error: Callback for handling errors

    Returns:
        ParallelResult with successful and failed items

    Example:
        # CPU-intensive processing
        def compute_embeddings(text):
            return model.encode(text)

        results = process_parallel_multiprocess(
            descriptions,
            compute_embeddings,
            max_workers=4,
        )
    """
    start_time = time.time()
    result = ParallelResult()
    result.total = len(items)

    logger.info(
        f"Processing {len(items)} items with multiprocessing "
        f"(max_workers: {max_workers})"
    )

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {
            executor.submit(process_fn, item): (i, item)
            for i, item in enumerate(items)
        }

        for future in as_completed(future_to_item, timeout=timeout):
            item_index, item = future_to_item[future]

            try:
                item_result = future.result(timeout=timeout)
                result.successful.append(item_result)

            except Exception as e:
                error_info = {
                    "item_index": item_index,
                    "item": item,
                    "error": str(e),
                    "error_type": type(e).__name__,
                }
                result.failed.append(error_info)

                logger.warning(f"Failed to process item {item_index}: {e}")

                if on_error:
                    try:
                        on_error(item, e)
                    except Exception as callback_error:
                        logger.error(f"Error callback failed: {callback_error}")

    result.duration = time.time() - start_time

    logger.info(
        f"Multiprocess processing complete: "
        f"{len(result.successful)}/{result.total} successful "
        f"in {result.duration:.2f}s"
    )

    return result


def process_in_batches(
    items: List[T],
    process_fn: Callable[[List[T]], List[R]],
    batch_size: int = 10,
    max_workers: int = 4,
) -> ParallelResult[R]:
    """
    Process items in batches in parallel.

    Useful when the processing function can handle multiple items more
    efficiently than processing them one at a time.

    Args:
        items: List of items to process
        process_fn: Function that processes a batch of items
        batch_size: Number of items per batch
        max_workers: Maximum number of worker threads

    Returns:
        ParallelResult with successful and failed items

    Example:
        # Batch API calls
        def batch_estimate_physics(objects):
            return gemini_api.batch_estimate(objects)

        results = process_in_batches(
            objects,
            batch_estimate_physics,
            batch_size=10,
            max_workers=5,
        )
    """
    start_time = time.time()
    result = ParallelResult()
    result.total = len(items)

    # Create batches
    batches = [
        items[i:i + batch_size]
        for i in range(0, len(items), batch_size)
    ]

    logger.info(
        f"Processing {len(items)} items in {len(batches)} batches "
        f"(batch_size: {batch_size}, max_workers: {max_workers})"
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {
            executor.submit(process_fn, batch): (i, batch)
            for i, batch in enumerate(batches)
        }

        for future in as_completed(future_to_batch):
            batch_index, batch = future_to_batch[future]

            try:
                batch_results = future.result()
                result.successful.extend(batch_results)

                logger.debug(
                    f"Processed batch {batch_index + 1}/{len(batches)} "
                    f"({len(batch)} items)"
                )

            except Exception as e:
                # Mark all items in batch as failed
                for item_index, item in enumerate(batch):
                    error_info = {
                        "item_index": batch_index * batch_size + item_index,
                        "item": item,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "batch_index": batch_index,
                    }
                    result.failed.append(error_info)

                logger.warning(
                    f"Failed to process batch {batch_index}: {e}"
                )

    result.duration = time.time() - start_time

    logger.info(
        f"Batch processing complete: "
        f"{len(result.successful)}/{result.total} successful "
        f"in {result.duration:.2f}s"
    )

    return result


class ParallelProcessor:
    """
    Reusable parallel processor with configurable settings.

    Example:
        processor = ParallelProcessor(
            max_workers=20,
            use_processes=False,
            timeout=10.0,
        )

        # Process multiple batches
        results1 = processor.process(objects1, estimate_physics)
        results2 = processor.process(objects2, estimate_physics)

        # Get statistics
        stats = processor.get_stats()
        print(f"Total processed: {stats['total_items']}")
    """

    def __init__(
        self,
        max_workers: int = 10,
        use_processes: bool = False,
        timeout: Optional[float] = None,
        batch_size: Optional[int] = None,
    ):
        self.max_workers = max_workers
        self.use_processes = use_processes
        self.timeout = timeout
        self.batch_size = batch_size

        self._total_items = 0
        self._successful_items = 0
        self._failed_items = 0
        self._total_duration = 0.0

    def process(
        self,
        items: List[T],
        process_fn: Callable[[T], R],
        on_error: Optional[Callable[[T, Exception], None]] = None,
    ) -> ParallelResult[R]:
        """Process items using configured settings."""
        if self.batch_size:
            # Convert single-item function to batch function
            def batch_fn(batch):
                if self.use_processes:
                    inner_result = process_parallel_multiprocess(
                        batch,
                        process_fn,
                        max_workers=self.max_workers,
                        timeout=self.timeout,
                        on_error=on_error,
                    )
                else:
                    inner_result = process_parallel_threaded(
                        batch,
                        process_fn,
                        max_workers=self.max_workers,
                        timeout=self.timeout,
                        on_error=on_error,
                    )
                return inner_result.successful

            result = process_in_batches(
                items,
                batch_fn,
                batch_size=self.batch_size,
                max_workers=self.max_workers,
            )
        elif self.use_processes:
            result = process_parallel_multiprocess(
                items,
                process_fn,
                max_workers=self.max_workers,
                timeout=self.timeout,
                on_error=on_error,
            )
        else:
            result = process_parallel_threaded(
                items,
                process_fn,
                max_workers=self.max_workers,
                timeout=self.timeout,
                on_error=on_error,
            )

        # Update statistics
        self._total_items += result.total
        self._successful_items += len(result.successful)
        self._failed_items += len(result.failed)
        self._total_duration += result.duration

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            "total_items": self._total_items,
            "successful_items": self._successful_items,
            "failed_items": self._failed_items,
            "success_rate": (
                self._successful_items / self._total_items
                if self._total_items > 0
                else 0.0
            ),
            "total_duration": self._total_duration,
            "avg_items_per_second": (
                self._successful_items / self._total_duration
                if self._total_duration > 0
                else 0.0
            ),
        }

    def reset_stats(self) -> None:
        """Reset statistics."""
        self._total_items = 0
        self._successful_items = 0
        self._failed_items = 0
        self._total_duration = 0.0
