from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from google.api_core import exceptions as gcs_exceptions
from google.cloud import storage

DEFAULT_TTL_SECONDS = 30 * 60
DEFAULT_HEARTBEAT_SECONDS = 5 * 60


def _default_owner_id() -> str:
    host = socket.gethostname()
    pid = os.getpid()
    token = uuid.uuid4().hex[:8]
    return f"{host}:{pid}:{token}"


def _parse_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass
class GCSLock:
    bucket_name: str
    object_name: str
    ttl_seconds: int = DEFAULT_TTL_SECONDS
    heartbeat_seconds: int = DEFAULT_HEARTBEAT_SECONDS
    owner_id: str = field(default_factory=_default_owner_id)
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))

    _client: storage.Client = field(init=False, repr=False)
    _generation: Optional[int] = field(init=False, default=None, repr=False)
    _metageneration: Optional[int] = field(init=False, default=None, repr=False)
    _heartbeat_thread: Optional[threading.Thread] = field(init=False, default=None, repr=False)
    _stop_event: Optional[threading.Event] = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self._client = storage.Client()

    @property
    def gcs_uri(self) -> str:
        return f"gs://{self.bucket_name}/{self.object_name}"

    def acquire(self, *, wait_timeout: float = 0.0, retry_interval: float = 5.0) -> bool:
        deadline = time.time() + wait_timeout
        while True:
            if self._try_acquire():
                self._start_heartbeat()
                return True
            if wait_timeout <= 0 or time.time() >= deadline:
                return False
            time.sleep(retry_interval)

    def refresh(self) -> bool:
        if self._generation is None:
            return False
        bucket = self._client.bucket(self.bucket_name)
        blob = bucket.blob(self.object_name)
        try:
            blob.reload()
        except gcs_exceptions.NotFound:
            self.logger.warning("GCS lock missing during refresh: %s", self.gcs_uri)
            return False
        if not self._is_owner(blob):
            self.logger.warning("GCS lock ownership mismatch during refresh: %s", self.gcs_uri)
            return False
        now = time.time()
        metadata = self._build_metadata(now)
        blob.metadata = metadata
        try:
            blob.patch(if_metageneration_match=blob.metageneration)
            blob.reload()
            self._generation = blob.generation
            self._metageneration = blob.metageneration
            return True
        except gcs_exceptions.PreconditionFailed:
            self.logger.warning("GCS lock refresh precondition failed: %s", self.gcs_uri)
            return False

    def release(self) -> None:
        self._stop_heartbeat()
        if self._generation is None:
            return
        bucket = self._client.bucket(self.bucket_name)
        blob = bucket.blob(self.object_name)
        try:
            blob.delete(if_generation_match=self._generation)
            self.logger.info("Released GCS lock: %s", self.gcs_uri)
        except gcs_exceptions.NotFound:
            self.logger.warning("GCS lock already removed: %s", self.gcs_uri)
        except gcs_exceptions.PreconditionFailed:
            self.logger.warning("GCS lock changed before release: %s", self.gcs_uri)

    def __enter__(self) -> "GCSLock":
        acquired = self.acquire()
        if not acquired:
            raise RuntimeError(f"Failed to acquire GCS lock {self.gcs_uri}")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

    def _try_acquire(self) -> bool:
        bucket = self._client.bucket(self.bucket_name)
        blob = bucket.blob(self.object_name)
        now = time.time()
        metadata = self._build_metadata(now)
        blob.metadata = metadata
        payload = json.dumps(metadata, sort_keys=True)
        try:
            blob.upload_from_string(payload, content_type="application/json", if_generation_match=0)
            blob.reload()
            self._generation = blob.generation
            self._metageneration = blob.metageneration
            self.logger.info("Acquired GCS lock: %s", self.gcs_uri)
            return True
        except (gcs_exceptions.PreconditionFailed, gcs_exceptions.Conflict):
            if self._evict_if_expired(blob):
                return self._try_acquire()
            self.logger.info("GCS lock already held: %s", self.gcs_uri)
            return False

    def _build_metadata(self, now: float) -> dict[str, str]:
        expires_at = now + self.ttl_seconds
        return {
            "owner_id": self.owner_id,
            "created_at": f"{now:.3f}",
            "expires_at": f"{expires_at:.3f}",
            "heartbeat_at": f"{now:.3f}",
            "ttl_seconds": str(self.ttl_seconds),
        }

    def _evict_if_expired(self, blob) -> bool:
        try:
            if not blob.exists():
                return False
            blob.reload()
        except gcs_exceptions.NotFound:
            return False
        if not self._is_expired(blob):
            return False
        try:
            blob.delete(if_generation_match=blob.generation)
            self.logger.warning("Evicted stale GCS lock: %s", self.gcs_uri)
            return True
        except gcs_exceptions.NotFound:
            return True
        except gcs_exceptions.PreconditionFailed:
            return False

    def _is_expired(self, blob) -> bool:
        metadata = blob.metadata or {}
        expires_at = _parse_float(metadata.get("expires_at"))
        if expires_at is None:
            created_at = _parse_float(metadata.get("created_at"))
            if created_at is None and blob.time_created:
                created_at = blob.time_created.timestamp()
            if created_at is not None:
                expires_at = created_at + self.ttl_seconds
        if expires_at is None:
            return False
        return time.time() >= expires_at

    def _is_owner(self, blob) -> bool:
        metadata = blob.metadata or {}
        return metadata.get("owner_id") == self.owner_id

    def _start_heartbeat(self) -> None:
        if self.heartbeat_seconds <= 0:
            return
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            return
        self._stop_event = threading.Event()
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()

    def _stop_heartbeat(self) -> None:
        if self._stop_event is None:
            return
        self._stop_event.set()
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=2)

    def _heartbeat_loop(self) -> None:
        interval = max(1, int(self.heartbeat_seconds))
        while self._stop_event and not self._stop_event.wait(interval):
            if not self.refresh():
                self.logger.warning("Failed to refresh GCS lock heartbeat: %s", self.gcs_uri)
