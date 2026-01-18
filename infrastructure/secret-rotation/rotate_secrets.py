"""Rotate Secret Manager secrets on a schedule."""

from __future__ import annotations

import logging
import os
import secrets
from datetime import datetime, timezone

from secret_manager import update_secret

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


def _parse_secret_ids(raw_value: str) -> list[str]:
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def _generate_secret_value(byte_length: int) -> str:
    return secrets.token_urlsafe(byte_length)


def main() -> None:
    project_id = os.getenv("GCP_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        raise RuntimeError("GCP project is required via GCP_PROJECT/GOOGLE_CLOUD_PROJECT")

    raw_secret_ids = os.getenv("ROTATION_SECRET_IDS", "")
    secret_ids = _parse_secret_ids(raw_secret_ids)
    if not secret_ids:
        raise RuntimeError("ROTATION_SECRET_IDS must list at least one secret")

    byte_length = int(os.getenv("ROTATION_BYTE_LENGTH", "32"))
    rotation_reason = os.getenv("ROTATION_REASON", "scheduled")
    rotation_actor = os.getenv("ROTATION_ACTOR", "cloud-scheduler")

    for secret_id in secret_ids:
        new_value = _generate_secret_value(byte_length)
        update_secret(
            secret_id=secret_id,
            secret_value=new_value,
            project_id=project_id,
            rotation_reason=rotation_reason,
            rotation_actor=rotation_actor,
            rotation_metadata={
                "rotation_source": "cloud-run-job",
                "rotation_time": datetime.now(timezone.utc).isoformat(),
            },
        )
        logger.info("Rotated secret: %s", secret_id)


if __name__ == "__main__":
    main()
