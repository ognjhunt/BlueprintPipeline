"""
Dead Letter Queue Implementation for BlueprintPipeline.

Handles failed pipeline jobs by storing them for later analysis,
retry, or manual intervention.

Supports multiple backends:
- GCS (Google Cloud Storage) - for persistent storage
- Pub/Sub - for event-driven processing
- Local file system - for development
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .errors import PipelineError, ErrorContext

logger = logging.getLogger(__name__)


@dataclass
class DeadLetterMessage:
    """A message in the dead letter queue."""

    # Unique identifier
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Original job information
    scene_id: str = ""
    job_type: str = ""
    step: str = ""

    # Error information
    error_type: str = ""
    error_message: str = ""
    error_details: Dict[str, Any] = field(default_factory=dict)
    traceback: Optional[str] = None

    # Context
    attempt_count: int = 1
    max_attempts: int = 3
    first_failure_time: str = field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z"
    )
    last_failure_time: str = field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z"
    )

    # Original payload
    original_payload: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Processing status
    status: str = "pending"  # pending, retrying, resolved, abandoned

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "scene_id": self.scene_id,
            "job_type": self.job_type,
            "step": self.step,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "error_details": self.error_details,
            "traceback": self.traceback,
            "attempt_count": self.attempt_count,
            "max_attempts": self.max_attempts,
            "first_failure_time": self.first_failure_time,
            "last_failure_time": self.last_failure_time,
            "original_payload": self.original_payload,
            "metadata": self.metadata,
            "status": self.status,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeadLetterMessage":
        return cls(**data)

    @classmethod
    def from_pipeline_error(
        cls,
        error: PipelineError,
        original_payload: Optional[Dict[str, Any]] = None,
    ) -> "DeadLetterMessage":
        """Create a dead letter message from a PipelineError."""
        context = error.context or ErrorContext()

        return cls(
            scene_id=context.scene_id or "",
            job_type=context.additional.get("job_type", "unknown"),
            step=context.step or "",
            error_type=type(error).__name__,
            error_message=error.message,
            error_details=error.to_dict(),
            traceback=error.traceback_str,
            attempt_count=context.attempt,
            max_attempts=context.max_attempts,
            original_payload=original_payload or {},
            metadata={
                "category": error.category.value,
                "severity": error.severity.value,
                "retryable": error.retryable,
            },
        )


class DeadLetterQueue(ABC):
    """Abstract base class for dead letter queue implementations."""

    @abstractmethod
    def publish(self, message: DeadLetterMessage) -> str:
        """Publish a message to the dead letter queue."""
        pass

    @abstractmethod
    def get_pending(self, limit: int = 100) -> List[DeadLetterMessage]:
        """Get pending messages for retry."""
        pass

    @abstractmethod
    def mark_resolved(self, message_id: str) -> bool:
        """Mark a message as resolved."""
        pass

    @abstractmethod
    def mark_abandoned(self, message_id: str, reason: str = "") -> bool:
        """Mark a message as abandoned (no more retries)."""
        pass

    @abstractmethod
    def update_status(
        self,
        message_id: str,
        status: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update message status."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        pass


class GCSDeadLetterQueue(DeadLetterQueue):
    """
    Dead letter queue backed by Google Cloud Storage.

    Messages are stored as JSON files in GCS for durability
    and easy inspection.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "dead-letter",
        project_id: Optional[str] = None,
    ):
        self.bucket_name = bucket
        self.prefix = prefix.strip("/")
        self.project_id = project_id or os.getenv("PROJECT_ID")

        # Import GCS client
        try:
            from google.cloud import storage
            self._client = storage.Client(project=self.project_id)
            self._bucket = self._client.bucket(self.bucket_name)
        except ImportError:
            logger.warning("google-cloud-storage not installed, using mock")
            self._client = None
            self._bucket = None

    def _get_blob_path(self, message_id: str, status: str = "pending") -> str:
        """Get the blob path for a message."""
        return f"{self.prefix}/{status}/{message_id}.json"

    def publish(self, message: DeadLetterMessage) -> str:
        """Publish a message to GCS dead letter queue."""
        if not self._bucket:
            logger.warning(f"DLQ not available, logging message: {message.message_id}")
            logger.error(f"Dead letter: {message.to_json()}")
            return message.message_id

        blob_path = self._get_blob_path(message.message_id, message.status)
        blob = self._bucket.blob(blob_path)
        blob.upload_from_string(
            message.to_json(),
            content_type="application/json",
        )

        logger.info(f"Published to DLQ: {message.message_id} at {blob_path}")
        return message.message_id

    def get_pending(self, limit: int = 100) -> List[DeadLetterMessage]:
        """Get pending messages from GCS."""
        if not self._bucket:
            return []

        messages = []
        prefix = f"{self.prefix}/pending/"

        blobs = self._bucket.list_blobs(prefix=prefix, max_results=limit)
        for blob in blobs:
            try:
                content = blob.download_as_string()
                data = json.loads(content)
                messages.append(DeadLetterMessage.from_dict(data))
            except Exception as e:
                logger.error(f"Failed to load DLQ message {blob.name}: {e}")

        return messages

    def mark_resolved(self, message_id: str) -> bool:
        """Mark a message as resolved by moving it."""
        return self._move_message(message_id, "pending", "resolved")

    def mark_abandoned(self, message_id: str, reason: str = "") -> bool:
        """Mark a message as abandoned."""
        success = self._move_message(message_id, "pending", "abandoned")
        if success and reason:
            # Update metadata with reason
            self.update_status(message_id, "abandoned", {"abandon_reason": reason})
        return success

    def _move_message(self, message_id: str, from_status: str, to_status: str) -> bool:
        """Move a message from one status folder to another."""
        if not self._bucket:
            return False

        try:
            from_path = self._get_blob_path(message_id, from_status)
            to_path = self._get_blob_path(message_id, to_status)

            source_blob = self._bucket.blob(from_path)
            if not source_blob.exists():
                logger.warning(f"Message not found: {from_path}")
                return False

            # Copy to new location
            self._bucket.copy_blob(source_blob, self._bucket, to_path)

            # Delete from old location
            source_blob.delete()

            logger.info(f"Moved DLQ message {message_id}: {from_status} -> {to_status}")
            return True

        except Exception as e:
            logger.error(f"Failed to move DLQ message {message_id}: {e}")
            return False

    def update_status(
        self,
        message_id: str,
        status: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update message status and metadata."""
        if not self._bucket:
            return False

        try:
            # Find the message in any status folder
            for current_status in ["pending", "retrying", "resolved", "abandoned"]:
                blob_path = self._get_blob_path(message_id, current_status)
                blob = self._bucket.blob(blob_path)

                if blob.exists():
                    content = blob.download_as_string()
                    data = json.loads(content)

                    # Update
                    data["status"] = status
                    data["last_failure_time"] = datetime.utcnow().isoformat() + "Z"
                    if metadata:
                        data["metadata"].update(metadata)

                    # Write back
                    blob.upload_from_string(
                        json.dumps(data, indent=2),
                        content_type="application/json",
                    )

                    # Move if status changed
                    if current_status != status:
                        self._move_message(message_id, current_status, status)

                    return True

            logger.warning(f"Message not found for update: {message_id}")
            return False

        except Exception as e:
            logger.error(f"Failed to update DLQ message {message_id}: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics from GCS."""
        if not self._bucket:
            return {"error": "GCS not available"}

        stats = {
            "pending": 0,
            "retrying": 0,
            "resolved": 0,
            "abandoned": 0,
        }

        for status in stats.keys():
            prefix = f"{self.prefix}/{status}/"
            blobs = list(self._bucket.list_blobs(prefix=prefix))
            stats[status] = len(blobs)

        stats["total"] = sum(stats.values())
        return stats


class PubSubDeadLetterQueue(DeadLetterQueue):
    """
    Dead letter queue backed by Google Cloud Pub/Sub.

    Good for event-driven retry processing.
    """

    def __init__(
        self,
        project_id: str,
        topic_name: str = "blueprint-dead-letter",
        subscription_name: str = "blueprint-dead-letter-sub",
    ):
        self.project_id = project_id
        self.topic_name = topic_name
        self.subscription_name = subscription_name

        try:
            from google.cloud import pubsub_v1
            self._publisher = pubsub_v1.PublisherClient()
            self._subscriber = pubsub_v1.SubscriberClient()
            self._topic_path = self._publisher.topic_path(project_id, topic_name)
            self._subscription_path = self._subscriber.subscription_path(
                project_id, subscription_name
            )
        except ImportError:
            logger.warning("google-cloud-pubsub not installed")
            self._publisher = None
            self._subscriber = None

        self._stats = {
            "published": 0,
            "resolved": 0,
            "abandoned": 0,
        }
        self._stats_lock = threading.Lock()

    def publish(self, message: DeadLetterMessage) -> str:
        """Publish message to Pub/Sub topic."""
        if not self._publisher:
            logger.warning(f"Pub/Sub not available, logging: {message.message_id}")
            logger.error(f"Dead letter: {message.to_json()}")
            return message.message_id

        try:
            data = message.to_json().encode("utf-8")
            future = self._publisher.publish(
                self._topic_path,
                data,
                message_id=message.message_id,
                scene_id=message.scene_id,
                job_type=message.job_type,
                error_type=message.error_type,
            )
            future.result()  # Wait for publish

            with self._stats_lock:
                self._stats["published"] += 1

            logger.info(f"Published to Pub/Sub DLQ: {message.message_id}")
            return message.message_id

        except Exception as e:
            logger.error(f"Failed to publish to Pub/Sub: {e}")
            raise

    def get_pending(self, limit: int = 100) -> List[DeadLetterMessage]:
        """Pull pending messages from Pub/Sub subscription."""
        if not self._subscriber:
            return []

        try:
            response = self._subscriber.pull(
                request={
                    "subscription": self._subscription_path,
                    "max_messages": limit,
                }
            )

            messages = []
            for received in response.received_messages:
                try:
                    data = json.loads(received.message.data.decode("utf-8"))
                    message = DeadLetterMessage.from_dict(data)
                    message.metadata["ack_id"] = received.ack_id
                    messages.append(message)
                except Exception as e:
                    logger.error(f"Failed to parse Pub/Sub message: {e}")

            return messages

        except Exception as e:
            logger.error(f"Failed to pull from Pub/Sub: {e}")
            return []

    def mark_resolved(self, message_id: str) -> bool:
        """Acknowledge message in Pub/Sub."""
        # Note: In Pub/Sub, we need the ack_id, not message_id
        # This is typically done during processing
        with self._stats_lock:
            self._stats["resolved"] += 1
        return True

    def mark_abandoned(self, message_id: str, reason: str = "") -> bool:
        """Mark message as abandoned (acknowledge but don't retry)."""
        with self._stats_lock:
            self._stats["abandoned"] += 1
        return True

    def update_status(
        self,
        message_id: str,
        status: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update is not applicable for Pub/Sub (fire and forget)."""
        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        with self._stats_lock:
            return dict(self._stats)


class LocalDeadLetterQueue(DeadLetterQueue):
    """
    Dead letter queue backed by local file system.

    For development and testing only.
    """

    def __init__(self, directory: str = "./dead_letter"):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

        # Create status directories
        for status in ["pending", "retrying", "resolved", "abandoned"]:
            (self.directory / status).mkdir(exist_ok=True)

    def _get_path(self, message_id: str, status: str = "pending") -> Path:
        return self.directory / status / f"{message_id}.json"

    def publish(self, message: DeadLetterMessage) -> str:
        path = self._get_path(message.message_id, message.status)
        path.write_text(message.to_json())
        logger.info(f"Published to local DLQ: {path}")
        return message.message_id

    def get_pending(self, limit: int = 100) -> List[DeadLetterMessage]:
        messages = []
        pending_dir = self.directory / "pending"

        for path in list(pending_dir.glob("*.json"))[:limit]:
            try:
                data = json.loads(path.read_text())
                messages.append(DeadLetterMessage.from_dict(data))
            except Exception as e:
                logger.error(f"Failed to load {path}: {e}")

        return messages

    def mark_resolved(self, message_id: str) -> bool:
        return self._move_message(message_id, "pending", "resolved")

    def mark_abandoned(self, message_id: str, reason: str = "") -> bool:
        return self._move_message(message_id, "pending", "abandoned")

    def _move_message(self, message_id: str, from_status: str, to_status: str) -> bool:
        from_path = self._get_path(message_id, from_status)
        to_path = self._get_path(message_id, to_status)

        if not from_path.exists():
            return False

        from_path.rename(to_path)
        return True

    def update_status(
        self,
        message_id: str,
        status: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        for current_status in ["pending", "retrying", "resolved", "abandoned"]:
            path = self._get_path(message_id, current_status)
            if path.exists():
                data = json.loads(path.read_text())
                data["status"] = status
                if metadata:
                    data["metadata"].update(metadata)
                path.write_text(json.dumps(data, indent=2))

                if current_status != status:
                    self._move_message(message_id, current_status, status)
                return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        stats = {}
        for status in ["pending", "retrying", "resolved", "abandoned"]:
            status_dir = self.directory / status
            stats[status] = len(list(status_dir.glob("*.json")))
        stats["total"] = sum(stats.values())
        return stats


def get_dead_letter_queue(
    backend: str = "auto",
    **kwargs,
) -> DeadLetterQueue:
    """
    Factory function to get the appropriate DLQ backend.

    Args:
        backend: "gcs", "pubsub", "local", or "auto"
        **kwargs: Backend-specific configuration

    Returns:
        DeadLetterQueue instance
    """
    if backend == "auto":
        # Try to detect environment
        if os.getenv("BUCKET"):
            backend = "gcs"
        elif os.getenv("PUBSUB_TOPIC"):
            backend = "pubsub"
        else:
            backend = "local"

    if backend == "gcs":
        return GCSDeadLetterQueue(
            bucket=kwargs.get("bucket", os.getenv("BUCKET", "")),
            prefix=kwargs.get("prefix", "dead-letter"),
            project_id=kwargs.get("project_id"),
        )
    elif backend == "pubsub":
        return PubSubDeadLetterQueue(
            project_id=kwargs.get("project_id", os.getenv("PROJECT_ID", "")),
            topic_name=kwargs.get("topic_name", "blueprint-dead-letter"),
        )
    else:
        return LocalDeadLetterQueue(
            directory=kwargs.get("directory", "./dead_letter"),
        )
