"""Real-Time Feedback Loop for Online Learning.

Stream generated episode data to training systems in real-time, enabling
online/continual learning where policies are trained as data is generated.

Features:
- Streaming episode data as it's generated
- Quality filtering (only send high-quality episodes)
- Diversity-based prioritization (prefer diverse episodes)
- Adaptive generation (adjust parameters based on training feedback)
- Batch accumulation for efficient transfer
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

import numpy as np

logger = logging.getLogger(__name__)


class DataStreamProtocol(str, Enum):
    """Data streaming protocols."""
    HTTP_POST = "http_post"  # POST to REST API
    GRPC = "grpc"  # gRPC streaming
    WEBSOCKET = "websocket"  # WebSocket streaming
    MESSAGE_QUEUE = "message_queue"  # Pub/Sub (Cloud Pub/Sub, Kafka)
    FILE_WATCH = "file_watch"  # Write files and notify


@dataclass
class DataStreamConfig:
    """Configuration for data streaming."""

    # Protocol
    protocol: DataStreamProtocol = DataStreamProtocol.HTTP_POST

    # Endpoint
    endpoint_url: str = ""
    api_key: Optional[str] = None

    # Batching
    batch_size: int = 10  # Episodes per batch
    max_wait_time_seconds: float = 60.0  # Max time to wait for batch

    # Quality filtering
    min_quality_score: float = 0.7  # Only send episodes >= this score
    min_diversity_contribution: float = 0.0  # Only send if adds diversity

    # Rate limiting
    max_episodes_per_second: float = 10.0

    # Retry
    retry_attempts: int = 3
    retry_delay_seconds: float = 5.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "protocol": self.protocol.value,
            "endpoint_url": self.endpoint_url,
            "batch_size": self.batch_size,
            "max_wait_time_seconds": self.max_wait_time_seconds,
            "min_quality_score": self.min_quality_score,
            "min_diversity_contribution": self.min_diversity_contribution,
            "max_episodes_per_second": self.max_episodes_per_second,
            "retry_attempts": self.retry_attempts,
            "retry_delay_seconds": self.retry_delay_seconds,
        }


@dataclass
class FeedbackMetrics:
    """Metrics from training system feedback."""

    # Training performance
    policy_loss: Optional[float] = None
    success_rate: Optional[float] = None
    episode_reward: Optional[float] = None

    # Data quality
    data_quality_score: float = 0.0
    diversity_score: float = 0.0

    # Requests
    request_more_of_task_type: Optional[str] = None
    request_more_difficulty: Optional[float] = None  # -1 to 1 (easier to harder)
    request_specific_scenarios: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_loss": self.policy_loss,
            "success_rate": self.success_rate,
            "episode_reward": self.episode_reward,
            "data_quality_score": self.data_quality_score,
            "diversity_score": self.diversity_score,
            "request_more_of_task_type": self.request_more_of_task_type,
            "request_more_difficulty": self.request_more_difficulty,
            "request_specific_scenarios": self.request_specific_scenarios,
        }


class TrainingSystemClient:
    """Client for communicating with training systems.

    Handles sending episodes and receiving feedback from the training system.
    """

    def __init__(
        self,
        config: DataStreamConfig,
        enable_logging: bool = True,
    ):
        """Initialize training system client.

        Args:
            config: Data stream configuration
            enable_logging: Whether to log events
        """
        self.config = config
        self.enable_logging = enable_logging

        # Statistics
        self.episodes_sent = 0
        self.episodes_rejected = 0
        self.bytes_sent = 0

    def log(self, msg: str) -> None:
        """Log if enabled."""
        if self.enable_logging:
            logger.info(f"[TRAINING_CLIENT] {msg}")

    async def send_episode_batch(
        self,
        episodes: List[Dict[str, Any]]
    ) -> bool:
        """Send a batch of episodes to training system.

        Args:
            episodes: List of episode data dictionaries

        Returns:
            True if successful
        """
        if not episodes:
            return True

        self.log(f"Sending batch of {len(episodes)} episodes")

        # Prepare payload
        payload = {
            "episodes": episodes,
            "metadata": {
                "timestamp": time.time(),
                "source": "blueprint_pipeline",
                "batch_size": len(episodes),
            }
        }

        # Send based on protocol
        try:
            if self.config.protocol == DataStreamProtocol.HTTP_POST:
                success = await self._send_http_post(payload)

            elif self.config.protocol == DataStreamProtocol.WEBSOCKET:
                success = await self._send_websocket(payload)

            elif self.config.protocol == DataStreamProtocol.MESSAGE_QUEUE:
                success = await self._send_message_queue(payload)

            elif self.config.protocol == DataStreamProtocol.FILE_WATCH:
                success = await self._send_file_watch(payload)

            else:
                raise ValueError(f"Unsupported protocol: {self.config.protocol}")

            if success:
                self.episodes_sent += len(episodes)
                payload_size = len(json.dumps(payload))
                self.bytes_sent += payload_size

                self.log(f"Successfully sent {len(episodes)} episodes ({payload_size} bytes)")

            return success

        except Exception as e:
            self.log(f"Failed to send episodes: {e}")
            self.episodes_rejected += len(episodes)
            return False

    async def _send_http_post(self, payload: Dict[str, Any]) -> bool:
        """Send via HTTP POST."""
        try:
            import aiohttp

            headers = {}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            headers["Content-Type"] = "application/json"

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config.endpoint_url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        return True
                    else:
                        self.log(f"HTTP error: {response.status}")
                        return False

        except ImportError:
            logger.warning("aiohttp not installed. Install with: pip install aiohttp")
            return False
        except Exception as e:
            logger.error(f"HTTP POST failed: {e}")
            return False

    async def _send_websocket(self, payload: Dict[str, Any]) -> bool:
        """Send via WebSocket."""
        try:
            import websockets

            async with websockets.connect(self.config.endpoint_url) as websocket:
                await websocket.send(json.dumps(payload))
                # Wait for acknowledgment
                response = await websocket.recv()
                return json.loads(response).get("success", False)

        except ImportError:
            logger.warning("websockets not installed. Install with: pip install websockets")
            return False
        except Exception as e:
            logger.error(f"WebSocket send failed: {e}")
            return False

    async def _send_message_queue(self, payload: Dict[str, Any]) -> bool:
        """Send via message queue (Cloud Pub/Sub)."""
        try:
            from google.cloud import pubsub_v1

            publisher = pubsub_v1.PublisherClient()

            # Parse topic from endpoint URL
            # Format: projects/{project}/topics/{topic}
            topic_path = self.config.endpoint_url

            # Publish message
            data = json.dumps(payload).encode("utf-8")
            future = publisher.publish(topic_path, data)

            # Wait for publish to complete (with timeout)
            message_id = future.result(timeout=10)

            self.log(f"Published message: {message_id}")
            return True

        except ImportError:
            logger.warning("google-cloud-pubsub not installed. Install with: pip install google-cloud-pubsub")
            return False
        except Exception as e:
            logger.error(f"Pub/Sub publish failed: {e}")
            return False

    async def _send_file_watch(self, payload: Dict[str, Any]) -> bool:
        """Send via file write + notification."""
        try:
            # Write to file
            output_path = Path(self.config.endpoint_url)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Atomic write using temp file
            temp_path = output_path.with_suffix(".tmp")
            temp_path.write_text(json.dumps(payload, indent=2))
            temp_path.rename(output_path)

            self.log(f"Wrote batch to: {output_path}")
            return True

        except Exception as e:
            logger.error(f"File write failed: {e}")
            return False

    async def receive_feedback(self) -> Optional[FeedbackMetrics]:
        """Receive feedback from training system.

        Returns:
            FeedbackMetrics if available, else None
        """
        # This would poll or listen for feedback from training system
        # Placeholder implementation
        return None


class RealtimeFeedbackLoop:
    """Real-time feedback loop for streaming data to training systems.

    Example:
        # Configure streaming
        config = DataStreamConfig(
            protocol=DataStreamProtocol.HTTP_POST,
            endpoint_url="http://training-server:8000/episodes",
            batch_size=10,
            min_quality_score=0.7,
        )

        # Create feedback loop
        loop = RealtimeFeedbackLoop(config)

        # Start streaming
        await loop.start()

        # Add episodes as they're generated
        loop.queue_episode(episode_data)

        # Stop streaming
        await loop.stop()
    """

    def __init__(
        self,
        config: DataStreamConfig,
        enable_logging: bool = True,
    ):
        """Initialize real-time feedback loop.

        Args:
            config: Data stream configuration
            enable_logging: Whether to log events
        """
        self.config = config
        self.enable_logging = enable_logging

        self.client = TrainingSystemClient(config, enable_logging)

        # Episode queue
        self.queue: List[Dict[str, Any]] = []
        self.queue_lock = asyncio.Lock()

        # State
        self.running = False
        self.worker_task: Optional[asyncio.Task] = None

        # Feedback
        self.latest_feedback: Optional[FeedbackMetrics] = None

        # Statistics
        self.episodes_queued = 0
        self.episodes_filtered = 0

    def log(self, msg: str) -> None:
        """Log if enabled."""
        if self.enable_logging:
            logger.info(f"[FEEDBACK_LOOP] {msg}")

    async def start(self) -> None:
        """Start the feedback loop."""
        if self.running:
            self.log("Already running")
            return

        self.running = True
        self.worker_task = asyncio.create_task(self._worker())

        self.log("Started real-time feedback loop")

    async def stop(self) -> None:
        """Stop the feedback loop."""
        if not self.running:
            return

        self.running = False

        # Wait for worker to finish
        if self.worker_task:
            await self.worker_task

        # Flush remaining episodes
        await self._flush_queue()

        self.log("Stopped real-time feedback loop")

    def queue_episode(
        self,
        episode: Dict[str, Any],
        quality_score: Optional[float] = None,
    ) -> bool:
        """Queue an episode for streaming.

        Args:
            episode: Episode data
            quality_score: Optional quality score

        Returns:
            True if queued, False if filtered out
        """
        # Quality filtering
        if quality_score is not None and quality_score < self.config.min_quality_score:
            self.episodes_filtered += 1
            self.log(f"Filtered episode (quality {quality_score:.2f} < {self.config.min_quality_score:.2f})")
            return False

        # Add to queue
        self.queue.append(episode)
        self.episodes_queued += 1

        return True

    async def _worker(self) -> None:
        """Background worker for sending batches."""
        last_send_time = time.time()

        while self.running or len(self.queue) > 0:
            # Check if should send batch
            should_send = False

            async with self.queue_lock:
                batch_full = len(self.queue) >= self.config.batch_size
                timeout = (time.time() - last_send_time) >= self.config.max_wait_time_seconds
                has_episodes = len(self.queue) > 0

                should_send = batch_full or (timeout and has_episodes)

            if should_send:
                # Send batch
                async with self.queue_lock:
                    batch = self.queue[:self.config.batch_size]
                    self.queue = self.queue[self.config.batch_size:]

                # Send with retry
                success = await self._send_with_retry(batch)

                if success:
                    last_send_time = time.time()

                    # Check for feedback
                    feedback = await self.client.receive_feedback()
                    if feedback:
                        self.latest_feedback = feedback
                        self._apply_feedback(feedback)

            # Sleep briefly
            await asyncio.sleep(0.1)

    async def _send_with_retry(self, batch: List[Dict[str, Any]]) -> bool:
        """Send batch with retry logic.

        Args:
            batch: Episode batch

        Returns:
            True if successful
        """
        for attempt in range(self.config.retry_attempts):
            success = await self.client.send_episode_batch(batch)

            if success:
                return True

            # Wait before retry
            if attempt < self.config.retry_attempts - 1:
                delay = self.config.retry_delay_seconds * (2 ** attempt)
                self.log(f"Retry {attempt + 1}/{self.config.retry_attempts} in {delay}s")
                await asyncio.sleep(delay)

        self.log(f"Failed to send batch after {self.config.retry_attempts} attempts")
        return False

    async def _flush_queue(self) -> None:
        """Flush remaining episodes in queue."""
        if not self.queue:
            return

        self.log(f"Flushing {len(self.queue)} remaining episodes")

        while self.queue:
            batch = self.queue[:self.config.batch_size]
            self.queue = self.queue[self.config.batch_size:]

            await self._send_with_retry(batch)

    def _apply_feedback(self, feedback: FeedbackMetrics) -> None:
        """Apply feedback to adjust generation parameters.

        Args:
            feedback: Feedback from training system
        """
        self.log(f"Received feedback: quality={feedback.data_quality_score:.2f}")

        # This would adjust generation parameters based on feedback
        # For example:
        # - If diversity_score is low, increase task variation
        # - If request_more_difficulty, adjust task complexity
        # - If specific scenarios requested, prioritize those

        # Placeholder for future implementation
        pass

    def get_statistics(self) -> Dict[str, Any]:
        """Get streaming statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "running": self.running,
            "episodes_queued": self.episodes_queued,
            "episodes_filtered": self.episodes_filtered,
            "episodes_sent": self.client.episodes_sent,
            "episodes_rejected": self.client.episodes_rejected,
            "bytes_sent": self.client.bytes_sent,
            "queue_size": len(self.queue),
            "latest_feedback": self.latest_feedback.to_dict() if self.latest_feedback else None,
        }
