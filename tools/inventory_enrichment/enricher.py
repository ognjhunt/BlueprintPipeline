"""Inventory enrichment interface and implementations."""

from __future__ import annotations

import copy
import json
import logging
import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import jsonschema

from tools.config.production_mode import resolve_production_mode
from tools.error_handling.retry import NonRetryableError, RetryableError, retry_with_backoff
from tools.secret_store import get_secret_or_env, SecretIds

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


class GeminiInventoryEnricher(InventoryEnricher):
    """Enriches inventory objects using Gemini for semantic metadata."""

    def __init__(self, api_key: str, model: str = "gemini-3-pro-preview"):
        if not api_key:
            raise InventoryEnrichmentError("GEMINI_API_KEY is required for Gemini enrichment")
        self.api_key = api_key
        self.model = model

    def _build_prompt(self, objects: List[Dict[str, Any]], environment_type: str) -> str:
        obj_descriptions = []
        for obj in objects:
            dims = obj.get("bounds", {}).get("size", [0, 0, 0])
            obj_descriptions.append(
                f'- id: "{obj["id"]}", category: "{obj.get("category", "unknown")}", '
                f"dimensions: {dims[0]:.3f} x {dims[1]:.3f} x {dims[2]:.3f} m"
            )
        obj_block = "\n".join(obj_descriptions)

        return f"""You are a robotics simulation expert. Given these objects from a {environment_type} scene, provide detailed semantic enrichment for each object.

Objects:
{obj_block}

For EACH object, return a JSON array where each element has:
- "id": the object id (string)
- "description": 1-2 sentence description of the real-world object (string)
- "primary_material": main material (e.g. "wood", "metal", "fabric", "plastic", "ceramic", "glass") (string)
- "secondary_materials": list of other materials present (list of strings)
- "typical_mass_kg": realistic mass in kg for this object at these dimensions (float)
- "surface_finish": e.g. "smooth", "rough", "polished", "textured", "soft" (string)
- "is_graspable": whether a robot gripper could pick this up (boolean)
- "is_articulated": whether it has moving parts like doors/drawers (boolean)
- "affordances": list of interactions possible (e.g. ["open", "place_on", "sit_on"]) (list of strings)
- "semantic_tags": list of descriptive tags for scene understanding (list of strings)
- "sim_notes": any notes relevant for physics simulation (string)

Return ONLY a valid JSON array. No markdown, no explanation."""

    def _call_gemini(self, prompt: str) -> List[Dict[str, Any]]:
        """Call Gemini API and parse the response."""
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)
        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=4096),
                temperature=0.2,
                response_mime_type="application/json",
            ),
        )

        text = response.text.strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            if text.endswith("```"):
                text = text.rsplit("```", 1)[0]
            text = text.strip()

        result = json.loads(text)
        if isinstance(result, dict) and "objects" in result:
            result = result["objects"]
        if not isinstance(result, list):
            raise InventoryEnrichmentValidationError(
                f"Expected JSON array from Gemini, got {type(result).__name__}"
            )
        return result

    def enrich(self, inventory: Dict[str, Any]) -> Dict[str, Any]:
        objects = inventory.get("objects", [])
        environment_type = inventory.get("environment_type", "unknown")

        if not objects:
            logger.warning("No objects in inventory to enrich")
            enriched = copy.deepcopy(inventory)
            metadata = enriched.get("metadata") or {}
            metadata["inventory_enrichment"] = {
                "provider": "gemini",
                "status": "skipped",
                "enriched_at": datetime.utcnow().isoformat() + "Z",
                "note": "No objects to enrich.",
            }
            enriched["metadata"] = metadata
            return enriched

        prompt = self._build_prompt(objects, environment_type)
        gemini_results = self._call_gemini(prompt)

        # Index Gemini results by object id
        results_by_id = {r["id"]: r for r in gemini_results if "id" in r}

        enriched = copy.deepcopy(inventory)
        enriched_count = 0
        for obj in enriched.get("objects", []):
            oid = obj["id"]
            if oid in results_by_id:
                gemini_data = results_by_id[oid]
                # Merge Gemini fields into object (don't overwrite existing keys like bounds)
                for key in (
                    "description", "primary_material", "secondary_materials",
                    "typical_mass_kg", "surface_finish", "is_graspable",
                    "is_articulated", "affordances", "semantic_tags", "sim_notes",
                ):
                    if key in gemini_data:
                        obj[key] = gemini_data[key]
                # Replace generic short_description with Gemini description
                if "description" in gemini_data and obj.get("short_description", "").startswith("Object "):
                    obj["short_description"] = gemini_data["description"]
                enriched_count += 1
            else:
                logger.warning("No Gemini enrichment for object %s", oid)

        metadata = enriched.get("metadata") or {}
        metadata["inventory_enrichment"] = {
            "provider": "gemini",
            "model": self.model,
            "request_id": str(uuid.uuid4()),
            "status": "success",
            "enriched_at": datetime.utcnow().isoformat() + "Z",
            "enriched_count": enriched_count,
            "total_objects": len(objects),
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
    if config.mode == "gemini":
        gemini_key = os.getenv("GEMINI_API_KEY", "")
        if not gemini_key:
            logger.warning("GEMINI_API_KEY not set; falling back to mock enricher")
            return MockInventoryEnricher()
        gemini_model = os.getenv("GEMINI_MODEL", "gemini-3-pro-preview")
        return GeminiInventoryEnricher(api_key=gemini_key, model=gemini_model)
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
