"""Parallel Scene Processing for BlueprintPipeline.

Enable parallel processing of multiple scenes to maximize throughput
while respecting resource limits and API rate limits.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class SceneStatus(str, Enum):
    """Scene processing status."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class SceneResult:
    """Result of processing a single scene."""
    scene_id: str
    status: SceneStatus
    duration_seconds: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "status": self.status.value,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class BatchResult:
    """Result of batch processing."""
    total: int = 0
    success: int = 0
    failed: int = 0
    cancelled: int = 0
    total_duration_seconds: float = 0.0
    results: List[SceneResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "success": self.success,
            "failed": self.failed,
            "cancelled": self.cancelled,
            "total_duration_seconds": self.total_duration_seconds,
            "success_rate": self.success / self.total if self.total > 0 else 0,
            "results": [r.to_dict() for r in self.results],
        }


class ParallelPipelineRunner:
    """Run pipeline on multiple scenes in parallel.

    Features:
    - Concurrent processing with configurable limits
    - Rate limiting for API calls
    - Error handling and retry logic
    - Progress tracking and monitoring
    - Graceful cancellation

    Example:
        # Create runner
        runner = ParallelPipelineRunner(max_concurrent=5)

        # Define processing function
        async def process_scene(scene_id: str) -> Dict[str, Any]:
            # Run pipeline for scene
            return {"status": "success"}

        # Process batch
        results = await runner.process_batch(
            scene_ids=["scene_001", "scene_002", "scene_003"],
            process_fn=process_scene
        )

        print(f"Success rate: {results.success / results.total:.1%}")
    """

    def __init__(
        self,
        max_concurrent: int = 5,
        retry_attempts: int = 3,
        retry_delay: float = 5.0,
        rate_limit_per_second: float = 10.0,
        enable_logging: bool = True,
    ):
        """Initialize parallel runner.

        Args:
            max_concurrent: Maximum number of concurrent scenes
            retry_attempts: Number of retry attempts for failed scenes
            retry_delay: Delay between retries in seconds
            rate_limit_per_second: Maximum API calls per second
            enable_logging: Whether to log progress
        """
        self.max_concurrent = max_concurrent
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.rate_limit_per_second = rate_limit_per_second
        self.enable_logging = enable_logging

        # Concurrency control
        self.semaphore = asyncio.Semaphore(max_concurrent)

        # Rate limiting
        self.rate_limiter = asyncio.Semaphore(int(rate_limit_per_second))
        self._rate_limit_reset_task: Optional[asyncio.Task] = None

        # Progress tracking
        self.total_scenes = 0
        self.completed_scenes = 0
        self.failed_scenes = 0

        # Cancellation
        self._cancelled = False

    def log(self, msg: str) -> None:
        """Log if enabled."""
        if self.enable_logging:
            logger.info(f"[PARALLEL] {msg}")

    async def _reset_rate_limiter(self) -> None:
        """Reset rate limiter every second."""
        while not self._cancelled:
            await asyncio.sleep(1.0)
            # Release all permits
            for _ in range(int(self.rate_limit_per_second)):
                try:
                    self.rate_limiter.release()
                except ValueError:
                    # Already at max
                    break

    async def process_batch(
        self,
        scene_ids: List[str],
        process_fn: Callable[[str], Any],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> BatchResult:
        """Process multiple scenes in parallel.

        Args:
            scene_ids: List of scene IDs to process
            process_fn: Async function to process each scene
            progress_callback: Optional callback for progress updates

        Returns:
            BatchResult with processing results

        Example:
            async def my_process_fn(scene_id: str) -> Dict[str, Any]:
                # Run pipeline
                return {"status": "success"}

            results = await runner.process_batch(
                scene_ids=["scene_001", "scene_002"],
                process_fn=my_process_fn
            )
        """
        start_time = time.time()

        self.total_scenes = len(scene_ids)
        self.completed_scenes = 0
        self.failed_scenes = 0
        self._cancelled = False

        self.log(f"Starting batch processing: {self.total_scenes} scenes")

        # Start rate limiter reset task
        self._rate_limit_reset_task = asyncio.create_task(self._reset_rate_limiter())

        # Create tasks for all scenes
        tasks = [
            self._process_scene_with_retry(
                scene_id=scene_id,
                process_fn=process_fn,
                progress_callback=progress_callback,
            )
            for scene_id in scene_ids
        ]

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Stop rate limiter
        self._cancelled = True
        if self._rate_limit_reset_task:
            self._rate_limit_reset_task.cancel()
            try:
                await self._rate_limit_reset_task
            except asyncio.CancelledError:
                pass

        # Build batch result
        batch_result = BatchResult(
            total=len(scene_ids),
            total_duration_seconds=time.time() - start_time,
        )

        for result in results:
            if isinstance(result, Exception):
                # Unexpected exception
                batch_result.failed += 1
                batch_result.results.append(SceneResult(
                    scene_id="unknown",
                    status=SceneStatus.FAILED,
                    error=str(result),
                ))
            elif isinstance(result, SceneResult):
                batch_result.results.append(result)
                if result.status == SceneStatus.SUCCESS:
                    batch_result.success += 1
                elif result.status == SceneStatus.FAILED:
                    batch_result.failed += 1
                elif result.status == SceneStatus.CANCELLED:
                    batch_result.cancelled += 1

        self.log(
            f"Batch complete: {batch_result.success} success, "
            f"{batch_result.failed} failed, {batch_result.cancelled} cancelled "
            f"in {batch_result.total_duration_seconds:.2f}s"
        )

        return batch_result

    async def _process_scene_with_retry(
        self,
        scene_id: str,
        process_fn: Callable[[str], Any],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> SceneResult:
        """Process a single scene with retry logic.

        Args:
            scene_id: Scene identifier
            process_fn: Processing function
            progress_callback: Optional progress callback

        Returns:
            SceneResult
        """
        attempts = 0
        last_error = None

        while attempts < self.retry_attempts:
            if self._cancelled:
                return SceneResult(
                    scene_id=scene_id,
                    status=SceneStatus.CANCELLED,
                )

            attempts += 1

            try:
                result = await self._process_scene(
                    scene_id=scene_id,
                    process_fn=process_fn,
                    progress_callback=progress_callback,
                )

                if result.status == SceneStatus.SUCCESS:
                    return result

                last_error = result.error

            except Exception as e:
                last_error = str(e)
                self.log(f"Scene {scene_id} attempt {attempts} failed: {e}")

            # Wait before retry
            if attempts < self.retry_attempts:
                await asyncio.sleep(self.retry_delay * attempts)

        # All retries failed
        self.failed_scenes += 1
        return SceneResult(
            scene_id=scene_id,
            status=SceneStatus.FAILED,
            error=f"Failed after {self.retry_attempts} attempts: {last_error}",
        )

    async def _process_scene(
        self,
        scene_id: str,
        process_fn: Callable[[str], Any],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> SceneResult:
        """Process a single scene.

        Args:
            scene_id: Scene identifier
            process_fn: Processing function
            progress_callback: Optional progress callback

        Returns:
            SceneResult
        """
        # Acquire semaphore (limit concurrent scenes)
        async with self.semaphore:
            start_time = time.time()

            self.log(f"Processing scene: {scene_id}")

            try:
                # Rate limit
                async with self.rate_limiter:
                    pass

                # Call processing function
                if asyncio.iscoroutinefunction(process_fn):
                    result_data = await process_fn(scene_id)
                else:
                    # Wrap sync function
                    result_data = await asyncio.to_thread(process_fn, scene_id)

                # Success
                self.completed_scenes += 1
                duration = time.time() - start_time

                self.log(f"Scene {scene_id} completed in {duration:.2f}s")

                # Progress callback
                if progress_callback:
                    progress_callback(self.completed_scenes, self.total_scenes)

                return SceneResult(
                    scene_id=scene_id,
                    status=SceneStatus.SUCCESS,
                    duration_seconds=duration,
                    metadata=result_data if isinstance(result_data, dict) else {},
                )

            except Exception as e:
                # Failure
                duration = time.time() - start_time

                self.log(f"Scene {scene_id} failed: {e}")

                return SceneResult(
                    scene_id=scene_id,
                    status=SceneStatus.FAILED,
                    duration_seconds=duration,
                    error=str(e),
                )

    def cancel(self) -> None:
        """Cancel batch processing.

        Gracefully stops processing new scenes and waits for
        currently running scenes to complete.
        """
        self.log("Cancelling batch processing...")
        self._cancelled = True

    async def process_batch_with_dependencies(
        self,
        scenes: Dict[str, List[str]],
        process_fn: Callable[[str], Any],
    ) -> BatchResult:
        """Process scenes with dependencies.

        Args:
            scenes: Dict mapping scene_id -> list of dependencies
            process_fn: Processing function

        Returns:
            BatchResult

        Example:
            # scene_002 depends on scene_001
            scenes = {
                "scene_001": [],
                "scene_002": ["scene_001"],
                "scene_003": ["scene_001"],
            }

            results = await runner.process_batch_with_dependencies(
                scenes=scenes,
                process_fn=my_process_fn
            )
        """
        # Topological sort
        from collections import defaultdict, deque

        # Build dependency graph
        in_degree = defaultdict(int)
        graph = defaultdict(list)

        for scene_id, deps in scenes.items():
            if not deps:
                in_degree[scene_id] = 0
            for dep in deps:
                graph[dep].append(scene_id)
                in_degree[scene_id] += 1

        # Find scenes with no dependencies
        queue = deque([s for s in scenes if in_degree[s] == 0])

        # Process in batches
        start_time = time.time()
        all_results = []
        completed = set()

        while queue:
            # Get current batch (all scenes with satisfied dependencies)
            current_batch = list(queue)
            queue.clear()

            # Process current batch in parallel
            batch_result = await self.process_batch(
                scene_ids=current_batch,
                process_fn=process_fn,
            )

            all_results.extend(batch_result.results)

            # Mark successful scenes as completed
            for result in batch_result.results:
                if result.status == SceneStatus.SUCCESS:
                    completed.add(result.scene_id)

                    # Add dependent scenes to queue if all dependencies satisfied
                    for dependent in graph[result.scene_id]:
                        in_degree[dependent] -= 1
                        if in_degree[dependent] == 0:
                            # All dependencies satisfied
                            deps_satisfied = all(
                                dep in completed
                                for dep in scenes[dependent]
                            )
                            if deps_satisfied:
                                queue.append(dependent)

        # Build final result
        final_result = BatchResult(
            total=len(scenes),
            total_duration_seconds=time.time() - start_time,
            results=all_results,
        )

        for result in all_results:
            if result.status == SceneStatus.SUCCESS:
                final_result.success += 1
            elif result.status == SceneStatus.FAILED:
                final_result.failed += 1
            elif result.status == SceneStatus.CANCELLED:
                final_result.cancelled += 1

        return final_result


# Convenience function for running batches
async def run_parallel_batch(
    scene_ids: List[str],
    process_fn: Callable[[str], Any],
    max_concurrent: int = 5,
    **kwargs
) -> BatchResult:
    """Convenience function to run a parallel batch.

    Args:
        scene_ids: List of scene IDs
        process_fn: Processing function
        max_concurrent: Max concurrent scenes
        **kwargs: Additional runner arguments

    Returns:
        BatchResult

    Example:
        async def my_process(scene_id: str):
            # Process scene
            return {"status": "done"}

        results = await run_parallel_batch(
            scene_ids=["scene_001", "scene_002"],
            process_fn=my_process,
            max_concurrent=3
        )
    """
    runner = ParallelPipelineRunner(max_concurrent=max_concurrent, **kwargs)
    return await runner.process_batch(scene_ids, process_fn)
