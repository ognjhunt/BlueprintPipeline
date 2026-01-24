# Configuration Reference

This document indexes configuration files and environment variables used across the BlueprintPipeline. It summarizes purpose, schema/fields, defaults, and validation expectations, and cross-references runtime-impacting docs.

Genie Sim configuration is local-only: workflow settings and environment variables assume a local gRPC server running alongside the pipeline jobs.

## Configuration precedence & validation entrypoints

- **Environment variables override JSON/YAML defaults** and are the primary runtime override mechanism; see the centralized list in `tools/config/ENVIRONMENT_VARIABLES.md` for defaults and supported values.【F:tools/config/ENVIRONMENT_VARIABLES.md†L1-L18】
- **Pipeline defaults** live in `tools/config/pipeline_config.json`, including episode generation, video, physics, domain randomization, and resource allocation settings.【F:tools/config/pipeline_config.json†L1-L136】
- **Firestore configuration schema** defines customer/scene fields like `bundle_tier`, `min_quality_score`, and `custom_settings` (used to override pipeline behavior).【F:tools/config/FIRESTORE_SCHEMA.md†L9-L138】
- **Runtime validation references**:
  - Scene manifests must conform to the schema in `tools/scene_manifest/manifest_schema.json`; missing `version` fails QA validation (see `docs/SCENE_MANIFEST.md`).【F:docs/SCENE_MANIFEST.md†L1-L22】
  - Production validation runs enforce strict environment flags and quality thresholds (see `docs/PRODUCTION_E2E_VALIDATION.md`).【F:docs/PRODUCTION_E2E_VALIDATION.md†L1-L33】

## policy_configs/

Global policy configuration files consumed by pipeline jobs.

| File | Purpose | Schema/fields | Defaults | Validation expectations |
| --- | --- | --- | --- | --- |
| `policy_configs/adaptive_timeouts.yaml` | Adaptive timeout defaults for workflows. | `bundle_tier`, `scene_complexity`, and `default_timeout_seconds` (seconds).【F:policy_configs/adaptive_timeouts.yaml†L1-L19】 | Standard/pro/enterprise tier defaults and fallback `default_timeout_seconds` are defined in the file.【F:policy_configs/adaptive_timeouts.yaml†L8-L19】 | Workflows that accept tier/complexity inputs should align with these defaults (documented in workflow timeout policy).【F:workflows/TIMEOUT_AND_RETRY_POLICY.md†L20-L40】 |
| `policy_configs/environment_policies.json` | Environment and policy templates for scene/policy selection. | Top-level `version`, `description`, and `environments` map with fields like `display_name`, `asset_pack_priority`, `default_policies`, and `placement_region_templates`.【F:policy_configs/environment_policies.json†L1-L28】 | Defaults are per-environment entries listed in the file (e.g., kitchen, warehouse).【F:policy_configs/environment_policies.json†L6-L46】 | Validate shape by ensuring required environment keys exist and match consumer expectations; use job-level tests/QA since there is no explicit schema file. |
| `policy_configs/physics_profiles.json` | Physics profile presets for simulation fidelity vs. speed. | `profiles` map with `simulation`, `contact`, and `randomization` fields per profile (e.g., `manipulation_contact_rich`).【F:policy_configs/physics_profiles.json†L1-L33】 | Per-profile defaults (dt, substeps, solver settings) are defined inline in the file.【F:policy_configs/physics_profiles.json†L8-L66】 | Ensure profile names align with scene/customer config (see `custom_settings.physics_profile`).【F:tools/config/FIRESTORE_SCHEMA.md†L92-L113】 |
| `policy_configs/planning_defaults.yaml` | Motion planning timing and validation thresholds. | `motion_planner.timing` steps and `validation.thresholds/requirements/retries`.【F:policy_configs/planning_defaults.yaml†L1-L52】 | Default timings and thresholds listed in the file (e.g., collision limits, quality score, max retries).【F:policy_configs/planning_defaults.yaml†L1-L52】 | Validate updates via downstream pipeline QA and any planning validators consuming these thresholds. |
| `policy_configs/retry_policy.yaml` | Shared retry/backoff defaults for workflows and local pipeline tooling. | `max_retries`, `base_delay_seconds`, `max_delay_seconds`, `backoff_factor`.【F:policy_configs/retry_policy.yaml†L1-L5】 | Canonical retry defaults are defined inline (5 retries, 1s base, 60s max, x2 backoff).【F:policy_configs/retry_policy.yaml†L2-L5】 | Keep workflow retry settings aligned unless an override is documented in `workflows/TIMEOUT_AND_RETRY_POLICY.md`. |
| `policy_configs/robot_embodiments.json` | Robot metadata (DOFs, assets, kinematics). | `robots` map with `num_dofs`, `gripper_dofs`, `ee_frame`, `default_joint_pos`, `assets`, and `kinematics`.【F:policy_configs/robot_embodiments.json†L1-L31】 | Defaults per robot (e.g., `franka`, `ur10`, `fetch`) defined in the file.【F:policy_configs/robot_embodiments.json†L2-L43】 | Ensure referenced assets exist and match supported robots in pipeline config (`robot_config.supported_robots`).【F:tools/config/pipeline_config.json†L88-L113】 |

### Job-specific policy copy

- `replicator-job/policy_configs/environment_policies.json` mirrors the global environment policy schema for the replicator job; keep fields consistent with the global version (same `version`, `environments`, and `policies` structure).【F:replicator-job/policy_configs/environment_policies.json†L1-L24】

## tools/config/

Shared configuration helpers and schemas.

| File | Purpose | Schema/fields | Defaults | Validation expectations |
| --- | --- | --- | --- | --- |
| `tools/config/pipeline_config.json` | Canonical pipeline defaults (episodes, video, physics, domain randomization, resources, quality gates). | Sections: `episode_generation`, `video`, `physics`, `physics_profiles`, `domain_randomization`, `resource_allocation`, `quality_gates`, etc.【F:tools/config/pipeline_config.json†L1-L171】 | Defaults for episode counts, video resolution, physics timestep, domain randomization intensity, resource allocation, etc.【F:tools/config/pipeline_config.json†L7-L171】 | Validate via config loaders and any schema-aware tooling; environment variables can override values at runtime.【F:tools/config/ENVIRONMENT_VARIABLES.md†L1-L18】 |
| `tools/config/ENVIRONMENT_VARIABLES.md` | Central index of runtime environment variables and defaults. | Sections for `BP_*`, `PIPELINE_*`, production flags, quality overrides, etc.【F:tools/config/ENVIRONMENT_VARIABLES.md†L1-L155】 | Defaults listed in each table (e.g., `PIPELINE_RETENTION_DAYS=30`).【F:tools/config/ENVIRONMENT_VARIABLES.md†L45-L63】 | Ensure environment variables conform to type and range constraints described in the doc; production mode flags are resolved in priority order as documented.【F:tools/config/ENVIRONMENT_VARIABLES.md†L65-L101】 |
| `tools/config/FIRESTORE_SCHEMA.md` | Firestore schema for customer/scene configuration and feature flags. | Collections `customers`, `scenes`, `feature_flags` with field types and required flags.【F:tools/config/FIRESTORE_SCHEMA.md†L15-L142】 | Defaults are implicit; unset fields inherit customer defaults or pipeline defaults where applicable.【F:tools/config/FIRESTORE_SCHEMA.md†L76-L113】 | Validate Firestore documents against the documented schema; use `bundle_tier`, `min_quality_score`, and feature flags to drive runtime behavior.【F:tools/config/FIRESTORE_SCHEMA.md†L31-L142】 |

## workflows/

Workflow definitions and trigger scripts that orchestrate pipeline jobs.

| File/Location | Purpose | Schema/fields | Defaults | Validation expectations |
| --- | --- | --- | --- | --- |
| `workflows/*-pipeline.yaml` | Cloud Workflows definitions for orchestration. | Workflow YAML defines parameters, job invocations, retries, and timeouts.【F:workflows/README.md†L1-L17】 | Timeouts and retry policies are standardized in `TIMEOUT_AND_RETRY_POLICY.md` (with adaptive timeout alignment to `policy_configs/adaptive_timeouts.yaml`).【F:workflows/TIMEOUT_AND_RETRY_POLICY.md†L1-L40】 | Validate by running workflows in staging; ensure timeout settings are consistent with adaptive defaults and job requirements.【F:workflows/TIMEOUT_AND_RETRY_POLICY.md†L20-L60】 |
| `workflows/*-poller.yaml` | Poller workflows (e.g., Genie Sim import poller). | YAML workflows that poll for markers and invoke downstream steps.【F:workflows/README.md†L8-L13】 | Defaults include polling intervals and retry rules in `TIMEOUT_AND_RETRY_POLICY.md`.【F:workflows/TIMEOUT_AND_RETRY_POLICY.md†L132-L170】 | Validate that marker paths and retry policies align with runtime artifacts (see production validation runbook).【F:docs/PRODUCTION_E2E_VALIDATION.md†L35-L121】 |
| `workflows/setup-*.sh` | Trigger setup scripts. | Shell scripts expect environment variables and credentials (see env var doc).【F:workflows/README.md†L8-L21】 | Defaults depend on the environment variables exported by the operator.【F:tools/config/ENVIRONMENT_VARIABLES.md†L1-L18】 | Validate by running setup scripts with correct credentials and verifying triggers are created. |

## k8s/

Kubernetes manifests for deploying pipeline jobs or infrastructure.

| File/Location | Purpose | Schema/fields | Defaults | Validation expectations |
| --- | --- | --- | --- | --- |
| `k8s/*.yaml` | Kubernetes manifests (jobs, namespace setup). | Standard Kubernetes YAML for jobs, deployments, and namespaces.【F:k8s/README.md†L1-L9】 | Defaults are embedded in each manifest (image, env vars, resource requests). | Validate via `kubectl apply` and `kubectl diff`, ensuring cluster credentials are set (`KUBECONFIG`).【F:k8s/README.md†L9-L17】 |

## Job-specific config folders

| File/Location | Purpose | Schema/fields | Defaults | Validation expectations |
| --- | --- | --- | --- | --- |
| `scene-generation-job/scheduler_config.yaml` | Cloud Scheduler config for daily scene generation. | Scheduler metadata, `schedule`, `timezone`, `paused`, HTTP target, and env overrides (`SCENES_PER_RUN`, `BUCKET`).【F:scene-generation-job/scheduler_config.yaml†L1-L73】 | Scheduler is **paused by default** (`paused: true`); schedule defaults to 8:00 AM PT.【F:scene-generation-job/scheduler_config.yaml†L27-L41】 | Validate by applying via `gcloud scheduler` and confirming the job runs with provided env overrides.【F:scene-generation-job/scheduler_config.yaml†L6-L20】 |
| `scene-generation-job/archetypes/archetype_config.json` | Scene archetype catalog for scene generation. | `archetypes` map with fields like `name`, `marketplace_category`, `sub_types`, `policy_targets`, and `weight`.【F:scene-generation-job/archetypes/archetype_config.json†L1-L40】 | Defaults per archetype are defined in the file (e.g., kitchen/grocery weights).【F:scene-generation-job/archetypes/archetype_config.json†L8-L74】 | Validate by ensuring archetype keys match supported categories and policy targets exist in `policy_configs/environment_policies.json`.【F:scene-generation-job/archetypes/archetype_config.json†L38-L74】 |
| `genie-sim-import-job/quality_config.json` | Default quality threshold for Genie Sim imports. | `default_min_quality_score`, `min_allowed`, `max_allowed`, and override note via `MIN_QUALITY_SCORE`.【F:genie-sim-import-job/quality_config.json†L1-L5】 | Default minimum quality score is `0.85` with allowable range `[0.0, 1.0]`.【F:genie-sim-import-job/quality_config.json†L1-L5】 | Validate that overrides stay within `min_allowed`/`max_allowed` and match production validation requirements (`MIN_QUALITY_SCORE >= 0.85` in prod).【F:genie-sim-import-job/quality_config.json†L1-L5】【F:docs/PRODUCTION_E2E_VALIDATION.md†L37-L52】 |

## Runtime-impacting configuration cross-references

- **Scene manifest compliance**: Scene assembly and downstream jobs rely on `scene_manifest.json` matching the schema; missing `version` fails QA validation (see `docs/SCENE_MANIFEST.md`).【F:docs/SCENE_MANIFEST.md†L1-L24】
- **Production validation**: Production E2E runs require strict environment flags (e.g., `DATA_QUALITY_LEVEL=production`, `REQUIRE_REAL_PHYSICS=true`) and minimum quality scores to pass validation gates; these settings should be aligned with job configs and env overrides.【F:docs/PRODUCTION_E2E_VALIDATION.md†L21-L90】
