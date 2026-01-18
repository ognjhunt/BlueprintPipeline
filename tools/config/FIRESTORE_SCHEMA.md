# Firestore Schema Documentation

This document defines the Firestore database structure used by the BlueprintPipeline for customer configuration and feature management.

## Overview

The BlueprintPipeline uses Google Cloud Firestore to manage:
- **Customer configurations**: Bundle tiers, feature flags, quotas
- **Scene configurations**: Per-scene overrides and settings
- **Feature flags**: Dynamic feature enablement per customer
- **Usage tracking**: Billable events and resource consumption

## Collections

### customers

Stores customer account information and configuration.

**Document ID**: `customer_id` (string, required)

**Fields**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `schema_version` | integer | Yes | Schema version for this document (start at `1`) |
| `bundle_tier` | string | Yes | Tier: "standard", "pro", "enterprise", or "foundation" |
| `organization_name` | string | No | Customer's organization name |
| `email` | string | No | Primary contact email |
| `monthly_scene_quota` | integer | No | Max scenes per month (-1 = unlimited) |
| `scenes_generated_this_month` | integer | No | Counter for current month |
| `total_scenes_generated` | integer | No | Lifetime counter |
| `total_episodes_generated` | integer | No | Lifetime episodes count |
| `audio_narration_enabled` | boolean | No | Whether audio narration is enabled |
| `subtitle_generation_enabled` | boolean | No | Whether subtitle generation is enabled |
| `preferred_tts_voice` | string | No | Google Cloud TTS voice ID (e.g., "en-US-Neural2-D") |
| `preferred_language` | string | No | Language code (e.g., "en-US") |
| `support_tier` | string | No | Support level: "basic", "priority", "dedicated" |
| `contract_start` | timestamp | No | Contract start date |
| `contract_end` | timestamp | No | Contract end date |
| `feature_overrides` | map | No | Per-customer feature flag overrides |
| `created_at` | timestamp | No | Account creation timestamp (server-generated) |
| `updated_at` | timestamp | No | Last update timestamp (server-generated) |
| `quota_reset_at` | timestamp | No | When monthly quotas were last reset |

**Indexes**:
- `bundle_tier` (for querying by tier)
- `contract_end` (for checking active contracts)

**Example**:
```json
{
  "schema_version": 1,
  "bundle_tier": "pro",
  "organization_name": "AI Research Lab",
  "email": "contact@ailab.edu",
  "monthly_scene_quota": 100,
  "scenes_generated_this_month": 45,
  "support_tier": "priority",
  "contract_start": "2025-01-01T00:00:00Z",
  "contract_end": "2026-01-01T00:00:00Z",
  "audio_narration_enabled": true,
  "subtitle_generation_enabled": true,
  "preferred_tts_voice": "en-US-Neural2-D",
  "preferred_language": "en-US",
  "feature_overrides": {
    "tactile_sensors": true
  },
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2025-01-10T14:22:00Z"
}
```

---

### scenes

Stores scene-level configurations and settings.

**Document ID**: `scene_id` (string, required)

**Fields**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `schema_version` | integer | Yes | Schema version for this document (start at `1`) |
| `customer_id` | string | Yes | Reference to customer document |
| `bundle_tier` | string | No | Override bundle tier for this scene (inherits from customer if not set) |
| `robot_type` | string | No | Robot type (e.g., "franka", "ur10") |
| `data_pack_tier` | string | No | Data tier: "core", "plus", "full" |
| `episodes_per_variation` | integer | No | Episodes to generate per variation |
| `max_variations` | integer | No | Maximum scene variations to generate |
| `min_quality_score` | float | No | Minimum quality threshold (0.0-1.0) |
| `audio_narration_enabled` | boolean | No | Enable audio narration for this scene |
| `subtitle_generation_enabled` | boolean | No | Enable subtitle generation |
| `vla_models` | array | No | VLA models to generate (e.g., ["openvla", "pi0"]) |
| `num_language_variations` | integer | No | Language variations to generate |
| `sim2real_enabled` | boolean | No | Enable sim-to-real features |
| `contact_rich_enabled` | boolean | No | Enable contact-rich simulation |
| `tactile_enabled` | boolean | No | Enable tactile sensor simulation |
| `custom_settings` | map | No | Custom per-scene settings |
| `created_at` | timestamp | No | Scene creation timestamp |
| `updated_at` | timestamp | No | Last update timestamp |

**Indexes**:
- `customer_id` (for querying scenes by customer)
- `bundle_tier` (for queries by tier)

**Example**:
```json
{
  "schema_version": 1,
  "customer_id": "lab_001",
  "robot_type": "franka",
  "data_pack_tier": "plus",
  "episodes_per_variation": 50,
  "max_variations": 100,
  "min_quality_score": 0.85,
  "vla_models": ["openvla", "pi0"],
  "sim2real_enabled": true,
  "contact_rich_enabled": true,
  "tactile_enabled": false,
  "custom_settings": {
    "lighting_variation": "high",
    "physics_profile": "manipulation_contact_rich"
  },
  "created_at": "2025-01-05T09:15:00Z",
  "updated_at": "2025-01-10T16:45:00Z"
}
```

---

### feature_flags

Stores dynamic feature flags for customers and global overrides.

**Document ID**: `customer_id` or `"_global"` for global flags

**Fields** (all boolean):

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | integer | Schema version for this document (start at `1`) |
| `audio_narration` | boolean | Enable audio narration |
| `subtitle_generation` | boolean | Enable subtitle generation |
| `vla_finetuning` | boolean | Enable VLA fine-tuning data |
| `sim2real` | boolean | Enable sim-to-real features |
| `contact_rich` | boolean | Enable contact-rich manipulation |
| `tactile` | boolean | Enable tactile sensor integration |
| `multi_robot` | boolean | Enable multi-robot scenes |
| `deformable` | boolean | Enable deformable object simulation |
| `bimanual` | boolean | Enable bimanual manipulation |
| `dwm_conditioning` | boolean | Enable DWM conditioning |
| `streaming_export` | boolean | Enable streaming data export |

**Special Document**: `"_global"` contains global flag overrides
- Can disable features globally (security policy)
- Cannot enable features globally (customer tier determines enablement)

**Example (customer-specific)**:
```json
{
  "schema_version": 1,
  "audio_narration": true,
  "subtitle_generation": true,
  "vla_finetuning": true,
  "sim2real": true,
  "tactile": true
}
```

**Example (global, disables features)**:
```json
{
  "_comment": "Global policy - tactile sensor disabled due to beta status",
  "schema_version": 1,
  "tactile": false,
  "streaming_export": false
}
```

---

### usage_tracking

Records all billable actions and resource consumption.

**Document ID**: Auto-generated (based on `customer_id_scene_id_action_timestamp`)

**Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | integer | Schema version for this document (start at `1`) |
| `customer_id` | string | Customer ID (indexed) |
| `scene_id` | string | Scene ID (indexed) |
| `action` | string | Action type: "scene_generated", "episodes_generated", "upsell_applied", etc. |
| `timestamp` | timestamp | When the action occurred (indexed) |
| `details` | map | Additional context for the action |
| `episodes_generated` | integer | Number of episodes (if applicable) |
| `compute_minutes` | float | Compute time used (in minutes) |
| `storage_gb` | float | Storage consumed (in GB) |
| `billable` | boolean | Whether this action is billable |
| `estimated_cost` | float | Estimated cost in USD |

**Indexes**:
- Composite: `(customer_id, timestamp)` - for usage queries
- Composite: `(scene_id, timestamp)` - for per-scene tracking

**Example**:
```json
{
  "schema_version": 1,
  "customer_id": "lab_001",
  "scene_id": "scene_kitchen_v2",
  "action": "episodes_generated",
  "timestamp": "2025-01-10T14:30:00Z",
  "episodes_generated": 100,
  "compute_minutes": 45.5,
  "storage_gb": 2.3,
  "billable": true,
  "estimated_cost": 15.50,
  "details": {
    "batch_size": 10,
    "quality_level": "high"
  }
}
```

---

## Bundle Tier Specifications

Tiers determine default values for scene configuration:

| Aspect | Standard | Pro | Enterprise | Foundation |
|--------|----------|-----|------------|-----------|
| Data Tier | core | plus | full | full |
| Episodes/Var | 10 | 10 | 10 | 25 |
| Max Variations | 250 | 500 | 1,000 | 2,000 |
| Min Quality | 0.70 | 0.80 | 0.85 | 0.90 |
| VLA Models | none | openvla, smolvla | openvla, pi0, smolvla, groot | all |
| Language Vars | 0 | 10 | 15 | 20 |
| Audio/Subtitles | No | Yes | Yes | Yes |
| Sim2Real | No | No | Yes | Yes |
| Contact Rich | No | No | Yes | Yes |
| Tactile | No | No | Yes | Yes |

---

## Access Control

### Service Account Permissions

Required IAM roles for BlueprintPipeline service account:
- `roles/datastore.user` - Read/write to Firestore
- `roles/datastore.importExportAdmin` - Backup/restore capabilities (optional)

### Field-Level Security

Firestore Security Rules example:

```
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    // Customers collection
    match /customers/{customerId} {
      allow read, write: if request.auth.uid == customerId || isAdmin();
      allow list: if isAdmin();
    }

    // Scenes collection
    match /scenes/{sceneId} {
      allow read: if request.auth.uid == resource.data.customer_id || isAdmin();
      allow write: if request.auth.uid == resource.data.customer_id || isAdmin();
    }

    // Feature flags
    match /feature_flags/{document=**} {
      allow read: if isAdmin() || request.auth.uid == document;
      allow write: if isAdmin();
    }

    // Usage tracking
    match /usage_tracking/{document=**} {
      allow read: if isAdmin();
      allow write: if isAdmin() || request.auth.uid == resource.data.customer_id;
    }

    function isAdmin() {
      return request.auth.token.admin == true;
    }
  }
}
```

---

## Migration Guide

### From JSON Files to Firestore

1. Export existing customer configurations as JSON
2. Create documents in `customers` collection
3. Create corresponding `scenes` documents
4. Initialize feature flags from customer tier
5. Verify all references are updated

### Backward Compatibility

The `CustomerConfigService` class supports both Firestore and local JSON files:
- If Firestore unavailable, falls back to defaults
- JSON config files can still override via environment variables
- Graceful degradation if services unavailable

---

## Best Practices

1. **Always validate before saving**: Use `DataValidation` utilities
2. **Use timestamps**: Always include `created_at` and `updated_at` for auditability
3. **Batch operations**: Use batch writes for multiple updates
4. **Cache selectively**: Customer configs are cached; invalidate on updates
5. **Monitor usage**: Use `usage_tracking` for billing and alerts
6. **Feature rollout**: Use feature flags for gradual feature enablement

---

## Troubleshooting

### Issue: Customer config not found

**Solution**: Check if `customer_id` exists in `customers` collection. Falls back to default bundle tier.

### Issue: Scene config inherits wrong tier

**Solution**: Explicitly set `bundle_tier` in scene document or update parent customer tier.

### Issue: Feature flag not taking effect

**Solution**: Clear service cache with `CustomerConfigService.get_customer_config.cache_clear()` after updates.

### Issue: Usage tracking missing

**Solution**: Ensure Firestore write permissions and check for errors in pipeline logs.
