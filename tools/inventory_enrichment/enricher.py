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

from tools.secrets import get_secret_or_env, SecretIds

logger = logging.getLogger(__name__)


class InventoryEnrichmentError(Exception):
    """Raised when inventory enrichment fails."""


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

    def __init__(self, api_key: str, endpoint: str):
        if not api_key:
            raise InventoryEnrichmentError("Inventory enrichment API key is missing")
        if not endpoint:
            raise InventoryEnrichmentError("Inventory enrichment endpoint is missing")
        self.api_key = api_key
        self.endpoint = endpoint

    def enrich(self, inventory: Dict[str, Any]) -> Dict[str, Any]:
        enriched = copy.deepcopy(inventory)
        metadata = enriched.get("metadata") or {}
        metadata["inventory_enrichment"] = {
            "provider": "external",
            "status": "stub",
            "endpoint": self.endpoint,
            "enriched_at": datetime.utcnow().isoformat() + "Z",
        }
        enriched["metadata"] = metadata
        return enriched


def _resolve_config(mode: Optional[str] = None) -> InventoryEnrichmentConfig:
    resolved_mode = (mode or os.getenv("INVENTORY_ENRICHMENT_MODE", "mock")).strip().lower()
    endpoint = os.getenv("INVENTORY_ENRICHMENT_ENDPOINT")
    return InventoryEnrichmentConfig(mode=resolved_mode, endpoint=endpoint)


def get_inventory_enricher(mode: Optional[str] = None) -> InventoryEnricher:
    """Factory for inventory enrichment providers."""
    config = _resolve_config(mode)

    if config.mode in {"mock", "stub", "offline"}:
        return MockInventoryEnricher()
    if config.mode == "external":
        api_key = get_secret_or_env(
            SecretIds.INVENTORY_ENRICHMENT_API_KEY,
            env_var="INVENTORY_ENRICHMENT_API_KEY",
        )
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
