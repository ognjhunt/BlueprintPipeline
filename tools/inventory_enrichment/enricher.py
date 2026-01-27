"""Inventory enrichment interface and implementations."""

from __future__ import annotations

import copy
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import requests
import jsonschema

from tools.config.production_mode import resolve_production_mode
from tools.error_handling.retry import NonRetryableError, RetryableError, retry_with_backoff
from tools.secrets import get_secret_or_env, SecretIds

logger = logging.getLogger(__name__)


class InventoryEnrichmentError(Exception):
    """Raised when inventory enrichment fails."""


class InventoryEnrichmentHTTPError(InventoryEnrichmentError, RetryableError):
    """Raised when the enrichment HTTP request fails."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class InventoryEnrichmentAuthError(InventoryEnrichmentHTTPError, NonRetryableError):
    """Raised when authentication to the enrichment service fails."""


class InventoryEnrichmentValidationError(InventoryEnrichmentError, NonRetryableError):
    """Raised when the enrichment response fails schema validation."""


class InventoryEnricher(ABC):
    """Interface for inventory enrichment providers."""

    @abstractmethod
    def enrich(self, inventory: Dict[str, Any]) -> Dict[str, Any]:
        """Return an enriched inventory payload."""


@dataclass
class InventoryEnrichmentConfig:
    """Configuration for inventory enrichment."""

    mode: str
    endpoint: Optional[str] = None


REQUEST_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["inventory", "requested_at"],
    "properties": {
        "inventory": {"type": "object"},
        "requested_at": {"type": "string"},
    },
    "additionalProperties": False,
}


RESPONSE_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["request_id", "provider", "enrichment"],
    "properties": {
        "request_id": {"type": "string"},
        "provider": {"type": "string"},
        "enrichment": {"type": "object"},
    },
    "additionalProperties": True,
}


class MockInventoryEnricher(InventoryEnricher):
    """Offline-friendly mock enricher used for tests."""

    def enrich(self, inventory: Dict[str, Any]) -> Dict[str, Any]:
        enriched = copy.deepcopy(inventory)
        metadata = enriched.get("metadata") or {}
        metadata["inventory_enrichment"] = {
            "provider": "mock",
            "status": "stub",
            "enriched_at": datetime.utcnow().isoformat() + "Z",
            "note": "Mock inventory enrichment (offline mode).",
        }
        enriched["metadata"] = metadata
        return enriched


class ExternalInventoryEnricher(InventoryEnricher):
    """Stub implementation for external inventory enrichment services."""

    _timeout_seconds = 10

    def __init__(self, api_key: str, endpoint: str):
        if not api_key:
            raise InventoryEnrichmentError("Inventory enrichment API key is missing")
        if not endpoint:
            raise InventoryEnrichmentError("Inventory enrichment endpoint is missing")
        self.api_key = api_key
        self.endpoint = endpoint

    def enrich(self, inventory: Dict[str, Any]) -> Dict[str, Any]:
        requested_at = datetime.utcnow().isoformat() + "Z"
        payload = {
            "inventory": inventory,
            "requested_at": requested_at,
        }

        try:
            jsonschema.validate(instance=payload, schema=REQUEST_SCHEMA)
        except jsonschema.ValidationError as exc:
            raise InventoryEnrichmentValidationError(
                f"Request payload failed schema validation: {exc.message}"
            ) from exc

        @retry_with_backoff(
            max_retries=3,
            base_delay=1.0,
            retryable_exceptions={
                InventoryEnrichmentHTTPError,
                requests.Timeout,
                requests.ConnectionError,
            },
        )
        def _post_request() -> Dict[str, Any]:
            response = requests.post(
                self.endpoint,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self._timeout_seconds,
            )

            if response.status_code in {401, 403}:
                raise InventoryEnrichmentAuthError(
                    f"Inventory enrichment auth failed with status {response.status_code}",
                    status_code=response.status_code,
                )

            if response.status_code >= 400:
                raise InventoryEnrichmentHTTPError(
                    f"Inventory enrichment request failed with status {response.status_code}",
                    status_code=response.status_code,
                )

            try:
                response_payload = response.json()
            except ValueError as exc:
                raise InventoryEnrichmentValidationError(
                    "Inventory enrichment response is not valid JSON"
                ) from exc

            try:
                jsonschema.validate(instance=response_payload, schema=RESPONSE_SCHEMA)
            except jsonschema.ValidationError as exc:
                raise InventoryEnrichmentValidationError(
                    f"Inventory enrichment response failed schema validation: {exc.message}"
                ) from exc

            return response_payload

        response_payload = _post_request()

        enriched = copy.deepcopy(inventory)
        metadata = enriched.get("metadata") or {}
        metadata["inventory_enrichment"] = {
            "provider": response_payload["provider"],
            "request_id": response_payload["request_id"],
            "received_at": datetime.utcnow().isoformat() + "Z",
            "endpoint": self.endpoint,
            "status": "success",
            "data": response_payload["enrichment"],
        }
        enriched["metadata"] = metadata
        return enriched


def _resolve_config(mode: Optional[str] = None) -> InventoryEnrichmentConfig:
    env_mode = os.getenv("INVENTORY_ENRICHMENT_MODE")
    resolved_mode = (mode or env_mode or "mock").strip().lower()
    if not mode and env_mode is None and _is_production_env():
        resolved_mode = "external"
    endpoint = os.getenv("INVENTORY_ENRICHMENT_ENDPOINT")
    return InventoryEnrichmentConfig(mode=resolved_mode, endpoint=endpoint)


def _is_production_env() -> bool:
    return resolve_production_mode()


def get_inventory_enricher(mode: Optional[str] = None) -> InventoryEnricher:
    """Factory for inventory enrichment providers."""
    config = _resolve_config(mode)
    production_mode = _is_production_env()
    mode_source = (
        "argument"
        if mode
        else "environment"
        if os.getenv("INVENTORY_ENRICHMENT_MODE") is not None
        else "production-default"
        if production_mode
        else "default"
    )
    logger.info(
        "Resolved inventory enrichment mode",
        extra={
            "inventory_enrichment_mode": config.mode,
            "inventory_enrichment_mode_source": mode_source,
            "inventory_enrichment_production": production_mode,
            "inventory_enrichment_has_endpoint": bool(config.endpoint),
        },
    )

    api_key = None
    if config.mode == "external":
        api_key = get_secret_or_env(
            SecretIds.INVENTORY_ENRICHMENT_API_KEY,
            env_var="INVENTORY_ENRICHMENT_API_KEY",
            fallback_to_env=not production_mode,
        )
        if production_mode:
            missing = []
            if not config.endpoint:
                missing.append("INVENTORY_ENRICHMENT_ENDPOINT")
            if not api_key:
                missing.append("INVENTORY_ENRICHMENT_API_KEY (Secret Manager or env var)")
            if missing:
                missing_list = ", ".join(missing)
                raise InventoryEnrichmentError(
                    "Production inventory enrichment requires missing configuration: "
                    f"{missing_list}"
                )

    if config.mode in {"mock", "stub", "offline"}:
        return MockInventoryEnricher()
    if config.mode == "external":
        if not api_key:
            raise InventoryEnrichmentError(
                "Inventory enrichment API key not found in Secret Manager or env var"
            )
        if not config.endpoint:
            raise InventoryEnrichmentError("INVENTORY_ENRICHMENT_ENDPOINT is required")
        return ExternalInventoryEnricher(api_key=api_key, endpoint=config.endpoint)

    raise InventoryEnrichmentError(f"Unknown inventory enrichment mode: {config.mode}")


def enrich_inventory_file(
    inventory_path: Path,
    output_path: Optional[Path] = None,
    mode: Optional[str] = None,
) -> Path:
    """Enrich an inventory.json file and write the enriched output."""
    if not inventory_path.is_file():
        raise InventoryEnrichmentError(f"Inventory file not found: {inventory_path}")

    output_path = output_path or inventory_path.with_name("inventory_enriched.json")

    inventory = json.loads(inventory_path.read_text())
    enricher = get_inventory_enricher(mode=mode)
    enriched = enricher.enrich(inventory)
    output_path.write_text(json.dumps(enriched, indent=2))

    logger.info("Wrote enriched inventory to %s", output_path)
    return output_path
