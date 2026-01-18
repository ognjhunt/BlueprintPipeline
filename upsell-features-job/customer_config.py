#!/usr/bin/env python3
"""
Customer Configuration Service.

This module provides Firestore-based customer configuration management for:
1. Bundle tier derivation from customer account
2. Dynamic feature flags per customer
3. Scene configuration overrides
4. Usage tracking and quota management

The service integrates with the pipeline to automatically determine the correct
bundle tier and features for each customer's scenes.

Usage:
    from customer_config import CustomerConfigService

    service = CustomerConfigService()
    config = service.get_scene_config("scene_id")
    tier = config.bundle_tier
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.gcs_upload import calculate_md5_base64, verify_blob_upload

# Ensure repo root is in path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Try to import Firestore
try:
    from google.cloud import firestore
    HAVE_FIRESTORE = True
except ImportError:
    HAVE_FIRESTORE = False
    firestore = None

# Try to import GCS
try:
    from google.cloud import storage
    HAVE_GCS = True
except ImportError:
    HAVE_GCS = False
    storage = None


# =============================================================================
# Constants
# =============================================================================

FIRESTORE_COLLECTION_CUSTOMERS = "customers"
FIRESTORE_COLLECTION_SCENES = "scenes"
FIRESTORE_COLLECTION_USAGE = "usage_tracking"
FIRESTORE_COLLECTION_FEATURE_FLAGS = "feature_flags"

# Default tier when no customer config is found
DEFAULT_BUNDLE_TIER = "standard"


class BundleTier(str, Enum):
    """Available bundle tiers."""
    STANDARD = "standard"
    PRO = "pro"
    ENTERPRISE = "enterprise"
    FOUNDATION = "foundation"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CustomerConfig:
    """Customer configuration from Firestore."""
    customer_id: str
    bundle_tier: BundleTier
    organization_name: str = ""
    email: str = ""

    # Feature overrides (per-customer customization)
    feature_overrides: Dict[str, Any] = field(default_factory=dict)

    # Quota and limits
    monthly_scene_quota: int = -1  # -1 = unlimited
    scenes_generated_this_month: int = 0

    # Audio/narration settings
    audio_narration_enabled: bool = False
    subtitle_generation_enabled: bool = False
    preferred_tts_voice: str = "en-US-Neural2-D"
    preferred_language: str = "en-US"

    # Support tier
    support_tier: str = "basic"  # basic, priority, dedicated

    # Contract dates
    contract_start: Optional[datetime] = None
    contract_end: Optional[datetime] = None

    # Metadata
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def is_active(self) -> bool:
        """Check if customer contract is active."""
        now = datetime.now(timezone.utc)
        if self.contract_end and self.contract_end < now:
            return False
        return True

    def has_quota_remaining(self) -> bool:
        """Check if customer has scene quota remaining."""
        if self.monthly_scene_quota < 0:
            return True  # Unlimited
        return self.scenes_generated_this_month < self.monthly_scene_quota

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["bundle_tier"] = self.bundle_tier.value
        if self.contract_start:
            data["contract_start"] = self.contract_start.isoformat()
        if self.contract_end:
            data["contract_end"] = self.contract_end.isoformat()
        if self.created_at:
            data["created_at"] = self.created_at.isoformat()
        if self.updated_at:
            data["updated_at"] = self.updated_at.isoformat()
        return data


@dataclass
class SceneConfig:
    """Scene-level configuration."""
    scene_id: str
    customer_id: str
    bundle_tier: BundleTier

    # Feature settings for this scene
    audio_narration_enabled: bool = False
    subtitle_generation_enabled: bool = False

    # Data pack settings
    data_pack_tier: str = "core"  # core, plus, full

    # Episode settings
    episodes_per_variation: int = 10
    max_variations: int = 250

    # Quality settings
    min_quality_score: float = 0.7

    # Robot settings
    robot_type: str = "franka"

    # VLA models to generate
    vla_models: List[str] = field(default_factory=list)

    # Language settings
    num_language_variations: int = 0

    # Advanced features
    sim2real_enabled: bool = False
    contact_rich_enabled: bool = False
    tactile_enabled: bool = False

    # Custom overrides
    custom_settings: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["bundle_tier"] = self.bundle_tier.value
        return data


@dataclass
class UsageRecord:
    """Usage tracking record."""
    customer_id: str
    scene_id: str
    action: str  # scene_generated, episodes_generated, upsell_applied, etc.
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)

    # Resource usage
    episodes_generated: int = 0
    compute_minutes: float = 0.0
    storage_gb: float = 0.0

    # Billing info
    billable: bool = True
    estimated_cost: float = 0.0


# =============================================================================
# Customer Config Service
# =============================================================================

class CustomerConfigService:
    """
    Service for managing customer configuration from Firestore.

    Provides automatic bundle tier derivation, feature flag management,
    and usage tracking for the pipeline.
    """

    def __init__(
        self,
        project_id: Optional[str] = None,
        verbose: bool = True,
    ):
        self.verbose = verbose
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT_ID")
        self.db = None
        self.storage_client = None

        # Initialize Firestore
        if HAVE_FIRESTORE and firestore is not None:
            try:
                self.db = firestore.Client(project=self.project_id)
                self.log("Firestore connection established")
            except Exception as e:
                self.log(f"WARNING: Firestore unavailable: {e}", "WARNING")
        else:
            self.log("WARNING: Firestore not installed", "WARNING")

        # Initialize GCS (for reading scene configs from bucket)
        if HAVE_GCS and storage is not None:
            try:
                self.storage_client = storage.Client(project=self.project_id)
            except Exception as e:
                self.log(f"WARNING: GCS unavailable: {e}", "WARNING")

        # Local cache for configs
        self._customer_cache: Dict[str, CustomerConfig] = {}
        self._scene_cache: Dict[str, SceneConfig] = {}

    def log(self, msg: str, level: str = "INFO") -> None:
        if self.verbose:
            print(f"[CUSTOMER-CONFIG] [{level}] {msg}")

    # =========================================================================
    # Customer Management
    # =========================================================================

    def get_customer_config(self, customer_id: str) -> Optional[CustomerConfig]:
        """
        Get customer configuration from Firestore.

        Args:
            customer_id: Customer ID

        Returns:
            CustomerConfig if found, None otherwise
        """
        # Check cache first
        if customer_id in self._customer_cache:
            return self._customer_cache[customer_id]

        if not self.db:
            self.log(f"No database connection, using default config for {customer_id}")
            return self._create_default_customer_config(customer_id)

        try:
            doc_ref = self.db.collection(FIRESTORE_COLLECTION_CUSTOMERS).document(customer_id)
            doc = doc_ref.get()

            if not doc.exists:
                self.log(f"Customer {customer_id} not found, using defaults")
                return self._create_default_customer_config(customer_id)

            data = doc.to_dict()
            config = self._parse_customer_config(customer_id, data)

            # Cache the config
            self._customer_cache[customer_id] = config

            return config

        except Exception as e:
            self.log(f"Error fetching customer config: {e}", "ERROR")
            return self._create_default_customer_config(customer_id)

    def _parse_customer_config(self, customer_id: str, data: Dict[str, Any]) -> CustomerConfig:
        """Parse customer config from Firestore data."""
        # Parse bundle tier
        tier_str = data.get("bundle_tier", DEFAULT_BUNDLE_TIER)
        try:
            bundle_tier = BundleTier(tier_str)
        except ValueError:
            bundle_tier = BundleTier.STANDARD

        # Parse dates
        contract_start = None
        contract_end = None
        created_at = None
        updated_at = None

        if data.get("contract_start"):
            if hasattr(data["contract_start"], 'timestamp'):
                contract_start = datetime.fromtimestamp(
                    data["contract_start"].timestamp(), tz=timezone.utc
                )
            elif isinstance(data["contract_start"], str):
                contract_start = datetime.fromisoformat(
                    data["contract_start"].replace("Z", "+00:00")
                )

        if data.get("contract_end"):
            if hasattr(data["contract_end"], 'timestamp'):
                contract_end = datetime.fromtimestamp(
                    data["contract_end"].timestamp(), tz=timezone.utc
                )
            elif isinstance(data["contract_end"], str):
                contract_end = datetime.fromisoformat(
                    data["contract_end"].replace("Z", "+00:00")
                )

        return CustomerConfig(
            customer_id=customer_id,
            bundle_tier=bundle_tier,
            organization_name=data.get("organization_name", ""),
            email=data.get("email", ""),
            feature_overrides=data.get("feature_overrides", {}),
            monthly_scene_quota=data.get("monthly_scene_quota", -1),
            scenes_generated_this_month=data.get("scenes_generated_this_month", 0),
            audio_narration_enabled=data.get("audio_narration_enabled", False),
            subtitle_generation_enabled=data.get("subtitle_generation_enabled", False),
            preferred_tts_voice=data.get("preferred_tts_voice", "en-US-Neural2-D"),
            preferred_language=data.get("preferred_language", "en-US"),
            support_tier=data.get("support_tier", "basic"),
            contract_start=contract_start,
            contract_end=contract_end,
            created_at=created_at,
            updated_at=updated_at,
        )

    def _create_default_customer_config(self, customer_id: str) -> CustomerConfig:
        """Create a default customer config."""
        return CustomerConfig(
            customer_id=customer_id,
            bundle_tier=BundleTier.STANDARD,
            organization_name="",
            email="",
        )

    def save_customer_config(self, config: CustomerConfig) -> bool:
        """Save customer configuration to Firestore."""
        if not self.db:
            self.log("No database connection, cannot save config", "ERROR")
            return False

        try:
            doc_ref = self.db.collection(FIRESTORE_COLLECTION_CUSTOMERS).document(
                config.customer_id
            )

            data = config.to_dict()
            data["updated_at"] = firestore.SERVER_TIMESTAMP

            doc_ref.set(data, merge=True)

            # Update cache
            self._customer_cache[config.customer_id] = config

            self.log(f"Saved config for customer {config.customer_id}")
            return True

        except Exception as e:
            self.log(f"Error saving customer config: {e}", "ERROR")
            return False

    # =========================================================================
    # Scene Configuration
    # =========================================================================

    def get_scene_config(
        self,
        scene_id: str,
        customer_id: Optional[str] = None,
        bucket: Optional[str] = None,
    ) -> SceneConfig:
        """
        Get scene configuration.

        Derives configuration from:
        1. Scene-specific Firestore document (if exists)
        2. Scene config file in GCS bucket (if exists)
        3. Customer configuration (based on bundle tier)
        4. Default values

        Args:
            scene_id: Scene ID
            customer_id: Customer ID (optional, will look up from scene)
            bucket: GCS bucket to check for scene config

        Returns:
            SceneConfig with all settings
        """
        # Check cache
        if scene_id in self._scene_cache:
            return self._scene_cache[scene_id]

        # Try to load scene config from Firestore
        scene_data = self._load_scene_from_firestore(scene_id)

        # If no customer_id, try to get from scene data
        if not customer_id and scene_data:
            customer_id = scene_data.get("customer_id")

        # Try to load from GCS bucket
        if not scene_data and bucket:
            scene_data = self._load_scene_from_gcs(scene_id, bucket)

        # Get customer config to derive defaults
        customer_config = None
        if customer_id:
            customer_config = self.get_customer_config(customer_id)

        # Build scene config
        config = self._build_scene_config(scene_id, scene_data, customer_config)

        # Cache it
        self._scene_cache[scene_id] = config

        return config

    def _load_scene_from_firestore(self, scene_id: str) -> Optional[Dict[str, Any]]:
        """Load scene config from Firestore."""
        if not self.db:
            return None

        try:
            doc_ref = self.db.collection(FIRESTORE_COLLECTION_SCENES).document(scene_id)
            doc = doc_ref.get()

            if doc.exists:
                return doc.to_dict()
            return None

        except Exception as e:
            self.log(f"Error loading scene from Firestore: {e}", "WARNING")
            return None

    def _load_scene_from_gcs(
        self,
        scene_id: str,
        bucket: str,
    ) -> Optional[Dict[str, Any]]:
        """Load scene config from GCS bucket."""
        if not self.storage_client:
            return None

        try:
            bucket_obj = self.storage_client.bucket(bucket)
            blob = bucket_obj.blob(f"scenes/{scene_id}/config.json")

            if blob.exists():
                content = blob.download_as_text()
                return json.loads(content)
            return None

        except Exception as e:
            self.log(f"Error loading scene from GCS: {e}", "WARNING")
            return None

    def _build_scene_config(
        self,
        scene_id: str,
        scene_data: Optional[Dict[str, Any]],
        customer_config: Optional[CustomerConfig],
    ) -> SceneConfig:
        """Build scene configuration from various sources."""
        scene_data = scene_data or {}

        # Determine bundle tier (scene override > customer config > default)
        tier_str = scene_data.get("bundle_tier")
        if tier_str:
            try:
                bundle_tier = BundleTier(tier_str)
            except ValueError:
                bundle_tier = BundleTier.STANDARD
        elif customer_config:
            bundle_tier = customer_config.bundle_tier
        else:
            bundle_tier = BundleTier.STANDARD

        # Get customer ID
        customer_id = scene_data.get("customer_id", "")
        if not customer_id and customer_config:
            customer_id = customer_config.customer_id

        # Derive data pack tier from bundle tier
        data_pack_tier = self._get_data_pack_tier(bundle_tier)

        # Derive episodes per variation
        episodes_per_variation = self._get_episodes_per_variation(bundle_tier)

        # Derive max variations
        max_variations = self._get_max_variations(bundle_tier)

        # Derive quality score
        min_quality_score = self._get_min_quality_score(bundle_tier)

        # Derive VLA models
        vla_models = self._get_vla_models(bundle_tier)

        # Derive language variations
        num_language_variations = self._get_language_variations(bundle_tier)

        # Audio/narration settings
        audio_enabled = scene_data.get("audio_narration_enabled", False)
        subtitle_enabled = scene_data.get("subtitle_generation_enabled", False)

        # If customer has it enabled and tier supports it, enable by default
        if customer_config:
            if customer_config.audio_narration_enabled and bundle_tier in [
                BundleTier.PRO, BundleTier.ENTERPRISE, BundleTier.FOUNDATION
            ]:
                audio_enabled = True
            if customer_config.subtitle_generation_enabled and bundle_tier in [
                BundleTier.PRO, BundleTier.ENTERPRISE, BundleTier.FOUNDATION
            ]:
                subtitle_enabled = True

        # Advanced features based on tier
        sim2real = bundle_tier in [BundleTier.ENTERPRISE, BundleTier.FOUNDATION]
        contact_rich = bundle_tier in [BundleTier.ENTERPRISE, BundleTier.FOUNDATION]
        tactile = bundle_tier in [BundleTier.ENTERPRISE, BundleTier.FOUNDATION]

        # Override from scene data
        sim2real = scene_data.get("sim2real_enabled", sim2real)
        contact_rich = scene_data.get("contact_rich_enabled", contact_rich)
        tactile = scene_data.get("tactile_enabled", tactile)

        return SceneConfig(
            scene_id=scene_id,
            customer_id=customer_id,
            bundle_tier=bundle_tier,
            audio_narration_enabled=audio_enabled,
            subtitle_generation_enabled=subtitle_enabled,
            data_pack_tier=data_pack_tier,
            episodes_per_variation=scene_data.get(
                "episodes_per_variation", episodes_per_variation
            ),
            max_variations=scene_data.get("max_variations", max_variations),
            min_quality_score=scene_data.get("min_quality_score", min_quality_score),
            robot_type=scene_data.get("robot_type", "franka"),
            vla_models=scene_data.get("vla_models", vla_models),
            num_language_variations=scene_data.get(
                "num_language_variations", num_language_variations
            ),
            sim2real_enabled=sim2real,
            contact_rich_enabled=contact_rich,
            tactile_enabled=tactile,
            custom_settings=scene_data.get("custom_settings", {}),
        )

    def _get_data_pack_tier(self, bundle_tier: BundleTier) -> str:
        """Derive data pack tier from bundle tier."""
        mapping = {
            BundleTier.STANDARD: "core",
            BundleTier.PRO: "plus",
            BundleTier.ENTERPRISE: "full",
            BundleTier.FOUNDATION: "full",
        }
        return mapping.get(bundle_tier, "core")

    def _get_episodes_per_variation(self, bundle_tier: BundleTier) -> int:
        """Derive episodes per variation from bundle tier."""
        mapping = {
            BundleTier.STANDARD: 10,
            BundleTier.PRO: 10,
            BundleTier.ENTERPRISE: 10,
            BundleTier.FOUNDATION: 25,
        }
        return mapping.get(bundle_tier, 10)

    def _get_max_variations(self, bundle_tier: BundleTier) -> int:
        """Derive max variations from bundle tier."""
        mapping = {
            BundleTier.STANDARD: 250,
            BundleTier.PRO: 500,
            BundleTier.ENTERPRISE: 1000,
            BundleTier.FOUNDATION: 2000,
        }
        return mapping.get(bundle_tier, 250)

    def _get_min_quality_score(self, bundle_tier: BundleTier) -> float:
        """Derive minimum quality score from bundle tier."""
        mapping = {
            BundleTier.STANDARD: 0.7,
            BundleTier.PRO: 0.8,
            BundleTier.ENTERPRISE: 0.85,
            BundleTier.FOUNDATION: 0.9,
        }
        return mapping.get(bundle_tier, 0.7)

    def _get_vla_models(self, bundle_tier: BundleTier) -> List[str]:
        """Derive VLA models from bundle tier."""
        mapping = {
            BundleTier.STANDARD: [],
            BundleTier.PRO: ["openvla", "smolvla"],
            BundleTier.ENTERPRISE: ["openvla", "pi0", "smolvla", "groot"],
            BundleTier.FOUNDATION: ["openvla", "pi0", "smolvla", "groot"],
        }
        return mapping.get(bundle_tier, [])

    def _get_language_variations(self, bundle_tier: BundleTier) -> int:
        """Derive language variations from bundle tier."""
        mapping = {
            BundleTier.STANDARD: 0,
            BundleTier.PRO: 10,
            BundleTier.ENTERPRISE: 15,
            BundleTier.FOUNDATION: 20,
        }
        return mapping.get(bundle_tier, 0)

    def save_scene_config(self, config: SceneConfig, bucket: Optional[str] = None) -> bool:
        """
        Save scene configuration.

        Saves to both Firestore and GCS bucket (if provided).
        """
        success = True

        # Save to Firestore
        if self.db:
            try:
                doc_ref = self.db.collection(FIRESTORE_COLLECTION_SCENES).document(
                    config.scene_id
                )
                doc_ref.set(config.to_dict(), merge=True)
                self.log(f"Saved scene config to Firestore: {config.scene_id}")
            except Exception as e:
                self.log(f"Error saving to Firestore: {e}", "ERROR")
                success = False

        # Save to GCS bucket
        if bucket and self.storage_client:
            try:
                bucket_obj = self.storage_client.bucket(bucket)
                blob = bucket_obj.blob(f"scenes/{config.scene_id}/config.json")
                payload_json = json.dumps(config.to_dict(), indent=2)
                payload_bytes = payload_json.encode("utf-8")
                blob.upload_from_string(
                    payload_json,
                    content_type="application/json",
                )
                verified, failure_reason = verify_blob_upload(
                    blob,
                    gcs_uri=f"gs://{bucket}/scenes/{config.scene_id}/config.json",
                    expected_size=len(payload_bytes),
                    expected_md5=calculate_md5_base64(payload_bytes),
                )
                if not verified:
                    self.log(
                        f"Error verifying GCS upload for {config.scene_id}: {failure_reason}",
                        "ERROR",
                    )
                    success = False
                self.log(f"Saved scene config to GCS: {config.scene_id}")
            except Exception as e:
                self.log(f"Error saving to GCS: {e}", "ERROR")
                success = False

        # Update cache
        self._scene_cache[config.scene_id] = config

        return success

    # =========================================================================
    # Feature Flags
    # =========================================================================

    def get_feature_flags(self, customer_id: str) -> Dict[str, bool]:
        """
        Get dynamic feature flags for a customer.

        Feature flags allow enabling/disabling features without code changes.
        """
        # Default flags
        flags = {
            "audio_narration": False,
            "subtitle_generation": False,
            "vla_finetuning": False,
            "sim2real": False,
            "contact_rich": False,
            "tactile": False,
            "multi_robot": False,
            "deformable": False,
            "bimanual": False,
            "dwm_conditioning": False,
            "streaming_export": False,
        }

        if not self.db:
            return flags

        try:
            # Get customer-specific flags
            doc_ref = self.db.collection(FIRESTORE_COLLECTION_FEATURE_FLAGS).document(
                customer_id
            )
            doc = doc_ref.get()

            if doc.exists:
                customer_flags = doc.to_dict()
                flags.update(customer_flags)

            # Get global flags (can override customer flags)
            global_ref = self.db.collection(FIRESTORE_COLLECTION_FEATURE_FLAGS).document(
                "_global"
            )
            global_doc = global_ref.get()

            if global_doc.exists:
                global_flags = global_doc.to_dict()
                # Global flags only disable, not enable
                for key, value in global_flags.items():
                    if value is False:
                        flags[key] = False

            return flags

        except Exception as e:
            self.log(f"Error fetching feature flags: {e}", "WARNING")
            return flags

    def set_feature_flag(
        self,
        customer_id: str,
        flag_name: str,
        value: bool,
    ) -> bool:
        """Set a feature flag for a customer."""
        if not self.db:
            return False

        try:
            doc_ref = self.db.collection(FIRESTORE_COLLECTION_FEATURE_FLAGS).document(
                customer_id
            )
            doc_ref.set({flag_name: value}, merge=True)
            return True
        except Exception as e:
            self.log(f"Error setting feature flag: {e}", "ERROR")
            return False

    # =========================================================================
    # Usage Tracking
    # =========================================================================

    def record_usage(self, record: UsageRecord) -> bool:
        """
        Record usage for billing and analytics.

        Args:
            record: Usage record to store

        Returns:
            True if successful
        """
        if not self.db:
            self.log("No database, skipping usage recording", "WARNING")
            return False

        try:
            # Create document ID
            doc_id = f"{record.customer_id}_{record.scene_id}_{record.action}_{int(record.timestamp.timestamp())}"

            doc_ref = self.db.collection(FIRESTORE_COLLECTION_USAGE).document(doc_id)

            data = {
                "customer_id": record.customer_id,
                "scene_id": record.scene_id,
                "action": record.action,
                "timestamp": record.timestamp,
                "details": record.details,
                "episodes_generated": record.episodes_generated,
                "compute_minutes": record.compute_minutes,
                "storage_gb": record.storage_gb,
                "billable": record.billable,
                "estimated_cost": record.estimated_cost,
            }

            doc_ref.set(data)

            # Also increment customer usage counters
            self._increment_customer_usage(record)

            return True

        except Exception as e:
            self.log(f"Error recording usage: {e}", "ERROR")
            return False

    def _increment_customer_usage(self, record: UsageRecord) -> None:
        """Increment customer usage counters."""
        if not self.db:
            return

        try:
            customer_ref = self.db.collection(FIRESTORE_COLLECTION_CUSTOMERS).document(
                record.customer_id
            )

            if record.action == "scene_generated":
                customer_ref.update({
                    "scenes_generated_this_month": firestore.Increment(1),
                    "total_scenes_generated": firestore.Increment(1),
                })

            if record.episodes_generated > 0:
                customer_ref.update({
                    "total_episodes_generated": firestore.Increment(
                        record.episodes_generated
                    ),
                })

        except Exception as e:
            self.log(f"Error incrementing usage counters: {e}", "WARNING")

    def get_customer_usage(
        self,
        customer_id: str,
        days: int = 30,
    ) -> Dict[str, Any]:
        """Get usage summary for a customer."""
        if not self.db:
            return {"error": "No database connection"}

        try:
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)

            query = (
                self.db.collection(FIRESTORE_COLLECTION_USAGE)
                .where("customer_id", "==", customer_id)
                .where("timestamp", ">=", cutoff)
            )

            docs = list(query.stream())

            # Aggregate usage
            total_episodes = 0
            total_compute = 0.0
            total_storage = 0.0
            total_cost = 0.0
            scenes = set()
            actions = {}

            for doc in docs:
                data = doc.to_dict()
                total_episodes += data.get("episodes_generated", 0)
                total_compute += data.get("compute_minutes", 0)
                total_storage += data.get("storage_gb", 0)
                total_cost += data.get("estimated_cost", 0)
                scenes.add(data.get("scene_id", ""))

                action = data.get("action", "unknown")
                actions[action] = actions.get(action, 0) + 1

            return {
                "customer_id": customer_id,
                "period_days": days,
                "total_scenes": len(scenes),
                "total_episodes": total_episodes,
                "total_compute_minutes": total_compute,
                "total_storage_gb": total_storage,
                "estimated_cost": total_cost,
                "action_counts": actions,
            }

        except Exception as e:
            self.log(f"Error fetching usage: {e}", "ERROR")
            return {"error": str(e)}

    def reset_monthly_quotas(self) -> int:
        """
        Reset monthly scene quotas for all customers.

        Should be called on the first of each month.

        Returns:
            Number of customers reset
        """
        if not self.db:
            return 0

        try:
            customers = self.db.collection(FIRESTORE_COLLECTION_CUSTOMERS).stream()
            count = 0

            for doc in customers:
                doc.reference.update({
                    "scenes_generated_this_month": 0,
                    "quota_reset_at": firestore.SERVER_TIMESTAMP,
                })
                count += 1

            self.log(f"Reset quotas for {count} customers")
            return count

        except Exception as e:
            self.log(f"Error resetting quotas: {e}", "ERROR")
            return 0


# =============================================================================
# Helper Functions
# =============================================================================

def get_bundle_tier_for_scene(
    scene_id: str,
    customer_id: Optional[str] = None,
    bucket: Optional[str] = None,
) -> str:
    """
    Convenience function to get bundle tier for a scene.

    This is the main entry point for the pipeline to determine
    what tier a scene should be processed with.

    Args:
        scene_id: Scene ID
        customer_id: Optional customer ID
        bucket: Optional GCS bucket

    Returns:
        Bundle tier string (standard, pro, enterprise, foundation)
    """
    service = CustomerConfigService(verbose=False)
    config = service.get_scene_config(scene_id, customer_id, bucket)
    return config.bundle_tier.value


def get_scene_features(
    scene_id: str,
    customer_id: Optional[str] = None,
    bucket: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get all features enabled for a scene.

    Returns a dictionary of feature settings for use in the pipeline.
    """
    service = CustomerConfigService(verbose=False)
    config = service.get_scene_config(scene_id, customer_id, bucket)
    return config.to_dict()


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Customer configuration management"
    )

    subparsers = parser.add_subparsers(dest="command")

    # Get customer config
    get_parser = subparsers.add_parser("get-customer", help="Get customer config")
    get_parser.add_argument("--customer-id", required=True)

    # Get scene config
    scene_parser = subparsers.add_parser("get-scene", help="Get scene config")
    scene_parser.add_argument("--scene-id", required=True)
    scene_parser.add_argument("--customer-id")
    scene_parser.add_argument("--bucket")

    # Get usage
    usage_parser = subparsers.add_parser("usage", help="Get customer usage")
    usage_parser.add_argument("--customer-id", required=True)
    usage_parser.add_argument("--days", type=int, default=30)

    # Create sample customer
    create_parser = subparsers.add_parser("create-sample", help="Create sample customer")
    create_parser.add_argument("--customer-id", required=True)
    create_parser.add_argument("--tier", choices=["standard", "pro", "enterprise", "foundation"], default="standard")

    args = parser.parse_args()

    service = CustomerConfigService()

    if args.command == "get-customer":
        config = service.get_customer_config(args.customer_id)
        if config:
            print(json.dumps(config.to_dict(), indent=2, default=str))
        else:
            print("Customer not found")

    elif args.command == "get-scene":
        config = service.get_scene_config(
            args.scene_id,
            args.customer_id,
            args.bucket,
        )
        print(json.dumps(config.to_dict(), indent=2, default=str))

    elif args.command == "usage":
        usage = service.get_customer_usage(args.customer_id, args.days)
        print(json.dumps(usage, indent=2, default=str))

    elif args.command == "create-sample":
        config = CustomerConfig(
            customer_id=args.customer_id,
            bundle_tier=BundleTier(args.tier),
            organization_name=f"Sample Organization ({args.tier})",
            email=f"{args.customer_id}@example.com",
            audio_narration_enabled=args.tier in ["pro", "enterprise", "foundation"],
            subtitle_generation_enabled=args.tier in ["pro", "enterprise", "foundation"],
            monthly_scene_quota=10 if args.tier == "standard" else -1,
            support_tier="basic" if args.tier == "standard" else "priority",
        )
        if service.save_customer_config(config):
            print(f"Created customer: {args.customer_id} with tier: {args.tier}")
        else:
            print("Failed to create customer")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
