"""
Google Secret Manager integration for secure credential management.

Replaces plain-text API keys in environment variables with secure secret storage.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class SecretManagerError(Exception):
    """Raised when secret access fails."""
    pass


@lru_cache(maxsize=128)
def get_secret(
    secret_id: str,
    project_id: Optional[str] = None,
    version: str = "latest",
) -> str:
    """
    Fetch secret from Google Secret Manager.

    Secrets are cached in memory for the lifetime of the process to avoid
    repeated API calls.

    Args:
        secret_id: Secret identifier (e.g., "genie-sim-api-key")
        project_id: GCP project ID (defaults to current project)
        version: Secret version (default: "latest")

    Returns:
        Secret value as string

    Raises:
        SecretManagerError: If secret cannot be accessed

    Example:
        api_key = get_secret("genie-sim-api-key")

        # Or with explicit project
        api_key = get_secret(
            secret_id="genie-sim-api-key",
            project_id="my-project-123",
        )
    """
    try:
        from google.cloud import secretmanager
    except ImportError:
        raise SecretManagerError(
            "google-cloud-secret-manager not installed. "
            "Install with: pip install google-cloud-secret-manager"
        )

    # Get project ID from environment if not provided
    if project_id is None:
        project_id = os.getenv("GCP_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
        if not project_id:
            raise SecretManagerError(
                "Project ID not provided and GCP_PROJECT/GOOGLE_CLOUD_PROJECT "
                "environment variable not set"
            )

    try:
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{project_id}/secrets/{secret_id}/versions/{version}"

        logger.debug(f"Fetching secret: {secret_id} (version: {version})")

        response = client.access_secret_version(request={"name": name})
        secret_value = response.payload.data.decode("UTF-8")

        logger.info(f"Successfully fetched secret: {secret_id}")
        return secret_value

    except Exception as e:
        raise SecretManagerError(
            f"Failed to access secret '{secret_id}': {e}"
        ) from e


def get_secret_or_env(
    secret_id: str,
    env_var: str,
    project_id: Optional[str] = None,
    fallback_to_env: bool = True,
) -> Optional[str]:
    """
    Get secret from Secret Manager, with optional fallback to environment variable.

    This is useful during migration from env vars to Secret Manager.

    Args:
        secret_id: Secret identifier in Secret Manager
        env_var: Environment variable name as fallback
        project_id: GCP project ID (defaults to current project)
        fallback_to_env: If True, fall back to env var if secret not found

    Returns:
        Secret value, or None if not found

    Example:
        # Try Secret Manager first, fall back to env var
        api_key = get_secret_or_env(
            secret_id="gemini-api-key",
            env_var="GEMINI_API_KEY",
        )
    """
    # Try Secret Manager first
    try:
        return get_secret(secret_id, project_id=project_id)
    except SecretManagerError as e:
        if fallback_to_env:
            logger.warning(
                f"Failed to fetch secret '{secret_id}': {e}. "
                f"Falling back to environment variable '{env_var}'"
            )
            return os.getenv(env_var)
        else:
            raise


class SecretCache:
    """
    Centralized cache for application secrets.

    Example:
        secrets = SecretCache()
        secrets.load("genie-sim-api-key")
        secrets.load("gemini-api-key")

        # Access secrets
        api_key = secrets.get("genie-sim-api-key")
    """

    def __init__(self, project_id: Optional[str] = None):
        self.project_id = project_id
        self._cache: Dict[str, str] = {}
        self._loaded = False

    def load(self, secret_id: str, version: str = "latest") -> None:
        """Load a secret into the cache."""
        try:
            value = get_secret(
                secret_id=secret_id,
                project_id=self.project_id,
                version=version,
            )
            self._cache[secret_id] = value
            logger.debug(f"Loaded secret: {secret_id}")
        except SecretManagerError as e:
            logger.error(f"Failed to load secret '{secret_id}': {e}")
            raise

    def load_all(self, secret_ids: list[str]) -> None:
        """Load multiple secrets into the cache."""
        for secret_id in secret_ids:
            try:
                self.load(secret_id)
            except SecretManagerError:
                # Log already happened in load()
                pass

    def get(self, secret_id: str, default: Optional[str] = None) -> Optional[str]:
        """Get a secret from the cache."""
        return self._cache.get(secret_id, default)

    def get_or_load(self, secret_id: str) -> str:
        """Get secret from cache, loading if not already cached."""
        if secret_id not in self._cache:
            self.load(secret_id)
        return self._cache[secret_id]

    def clear(self) -> None:
        """Clear all cached secrets."""
        self._cache.clear()
        logger.debug("Cleared secret cache")


# Singleton instance for application-wide use
_global_secret_cache: Optional[SecretCache] = None


def get_global_secret_cache(project_id: Optional[str] = None) -> SecretCache:
    """
    Get the global secret cache instance.

    Args:
        project_id: GCP project ID (only used on first call)

    Returns:
        Global SecretCache instance

    Example:
        # Initialize once at app startup
        cache = get_global_secret_cache(project_id="my-project")
        cache.load_all([
            "genie-sim-api-key",
            "gemini-api-key",
            "openai-api-key",
        ])

        # Use anywhere in the application
        cache = get_global_secret_cache()
        api_key = cache.get("genie-sim-api-key")
    """
    global _global_secret_cache

    if _global_secret_cache is None:
        _global_secret_cache = SecretCache(project_id=project_id)

    return _global_secret_cache


# Pre-defined secret IDs for BlueprintPipeline
class SecretIds:
    """Standard secret IDs used in BlueprintPipeline."""
    GEMINI_API_KEY = "gemini-api-key"
    OPENAI_API_KEY = "openai-api-key"
    ANTHROPIC_API_KEY = "anthropic-api-key"
    PARTICULATE_API_KEY = "particulate-api-key"
    INVENTORY_ENRICHMENT_API_KEY = "inventory-enrichment-api-key"


def load_pipeline_secrets(project_id: Optional[str] = None) -> SecretCache:
    """
    Load all standard BlueprintPipeline secrets.

    Args:
        project_id: GCP project ID

    Returns:
        SecretCache with all loaded secrets

    Example:
        # At application startup
        secrets = load_pipeline_secrets()

        # Access secrets
        gemini_key = secrets.get(SecretIds.GEMINI_API_KEY)
    """
    cache = get_global_secret_cache(project_id=project_id)

    secret_ids = [
        SecretIds.GEMINI_API_KEY,
        SecretIds.OPENAI_API_KEY,
        SecretIds.ANTHROPIC_API_KEY,
        SecretIds.PARTICULATE_API_KEY,
        SecretIds.INVENTORY_ENRICHMENT_API_KEY,
    ]

    for secret_id in secret_ids:
        try:
            cache.load(secret_id)
        except SecretManagerError as e:
            # Log but don't fail - some secrets may be optional
            logger.warning(f"Could not load secret '{secret_id}': {e}")

    return cache


def create_secret(
    secret_id: str,
    secret_value: str,
    project_id: Optional[str] = None,
) -> None:
    """
    Create a new secret in Secret Manager.

    Args:
        secret_id: Secret identifier
        secret_value: Secret value to store
        project_id: GCP project ID

    Raises:
        SecretManagerError: If secret creation fails

    Example:
        # Create a new secret
        create_secret(
            secret_id="my-api-key",
            secret_value="sk-1234567890",
            project_id="my-project-123",
        )
    """
    try:
        from google.cloud import secretmanager
    except ImportError:
        raise SecretManagerError(
            "google-cloud-secret-manager not installed"
        )

    if project_id is None:
        project_id = os.getenv("GCP_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
        if not project_id:
            raise SecretManagerError("Project ID not provided")

    try:
        client = secretmanager.SecretManagerServiceClient()
        parent = f"projects/{project_id}"

        # Create secret
        secret = client.create_secret(
            request={
                "parent": parent,
                "secret_id": secret_id,
                "secret": {"replication": {"automatic": {}}},
            }
        )

        logger.info(f"Created secret: {secret.name}")

        # Add secret version with value
        client.add_secret_version(
            request={
                "parent": secret.name,
                "payload": {"data": secret_value.encode("UTF-8")},
            }
        )

        logger.info(f"Added value to secret: {secret_id}")

    except Exception as e:
        raise SecretManagerError(
            f"Failed to create secret '{secret_id}': {e}"
        ) from e


def update_secret(
    secret_id: str,
    secret_value: str,
    project_id: Optional[str] = None,
) -> None:
    """
    Update an existing secret with a new value.

    Creates a new version of the secret.

    Args:
        secret_id: Secret identifier
        secret_value: New secret value
        project_id: GCP project ID

    Raises:
        SecretManagerError: If update fails

    Example:
        update_secret(
            secret_id="my-api-key",
            secret_value="sk-new-value",
        )
    """
    try:
        from google.cloud import secretmanager
    except ImportError:
        raise SecretManagerError(
            "google-cloud-secret-manager not installed"
        )

    if project_id is None:
        project_id = os.getenv("GCP_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
        if not project_id:
            raise SecretManagerError("Project ID not provided")

    try:
        client = secretmanager.SecretManagerServiceClient()
        parent = f"projects/{project_id}/secrets/{secret_id}"

        # Add new version
        response = client.add_secret_version(
            request={
                "parent": parent,
                "payload": {"data": secret_value.encode("UTF-8")},
            }
        )

        logger.info(f"Updated secret: {secret_id} (version: {response.name})")

    except Exception as e:
        raise SecretManagerError(
            f"Failed to update secret '{secret_id}': {e}"
        ) from e
