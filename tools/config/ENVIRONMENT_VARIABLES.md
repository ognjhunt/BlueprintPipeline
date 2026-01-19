# Environment Variables Documentation

Complete reference for all environment variables used in the BlueprintPipeline.

## Overview

The pipeline uses environment variables for:
- Configuration overrides (BP_ prefix)
- Service credentials (keys, OAuth tokens)
- Feature flags (ENABLE_ prefix)
- Service endpoints (URL_ prefix)
- Debugging and logging

Environment variables take precedence over JSON config files, allowing runtime customization without redeployment.

## Canonical Boolean Values

Boolean environment variables are parsed by the shared helper in `tools/config/env.py`. The canonical accepted values
are listed below (case-insensitive, surrounding whitespace ignored):

**Truthy values**: `1`, `true`, `yes`, `y`, `on`, `json`  
**Falsey values**: `0`, `false`, `no`, `off`, `plain`, `text`

When a value is unset or unrecognized, the caller's explicit default is used (for example `None`, `False`, or `True`).

## Job-Level Documentation

Job-specific READMEs should link back to this centralized environment variable list for shared configuration details.
Relevant job-level documentation includes:

- [`dream2flow-preparation-job/README.md`](../../dream2flow-preparation-job/README.md)
- [`dwm-preparation-job/README.md`](../../dwm-preparation-job/README.md)
- [`episode-generation-job/README.md`](../../episode-generation-job/README.md)
- [`genie-sim-export-job/README.md`](../../genie-sim-export-job/README.md)
- [`genie-sim-gpu-job/README.md`](../../genie-sim-gpu-job/README.md)
- [`genie-sim-import-job/README.md`](../../genie-sim-import-job/README.md)
- [`genie-sim-local-job/README.md`](../../genie-sim-local-job/README.md)
- [`replicator-job/README.md`](../../replicator-job/README.md)
- [`scene-generation-job/README.md`](../../scene-generation-job/README.md)
- [`simready-job/README.md`](../../simready-job/README.md)
- [`smart-placement-engine-job/README.md`](../../smart-placement-engine-job/README.md)
- [`variation-asset-pipeline-job/README.md`](../../variation-asset-pipeline-job/README.md)

## Configuration Overrides (BP_ Prefix)

### BP_SPLIT_* - Dataset Split Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `BP_SPLIT_SEED` | int | 42 | Seed for reproducible train/val/test splits |
| `BP_SPLIT_STRATEGY` | str | "random" | Split strategy: "random", "scene", or "task" |
| `BP_SPLIT_TRAIN_RATIO` | float | 0.8 | Training set ratio |
| `BP_SPLIT_VAL_RATIO` | float | 0.1 | Validation set ratio |
| `BP_SPLIT_TEST_RATIO` | float | 0.1 | Test set ratio |

**Example**:
```bash
# Use a custom seed for reproducibility
export BP_SPLIT_SEED=1337

# Use scene-level splits (all episodes from same scene in same split)
export BP_SPLIT_STRATEGY=scene

# Custom split ratios
export BP_SPLIT_TRAIN_RATIO=0.7
export BP_SPLIT_VAL_RATIO=0.15
export BP_SPLIT_TEST_RATIO=0.15
```

---

## Pipeline Controls

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `PIPELINE_SEED` | int | unset | Global seed applied to Python, NumPy, and supported ML frameworks |
| `PIPELINE_RETENTION_DAYS` | int | 30 | Default retention window (days) for intermediate pipeline artifacts |
| `PIPELINE_INPUT_RETENTION_DAYS` | int | 90 | Retention window (days) for input artifacts |
| `PIPELINE_INTERMEDIATE_RETENTION_DAYS` | int | 30 | Retention window (days) for intermediate artifacts |
| `PIPELINE_OUTPUT_RETENTION_DAYS` | int | 365 | Retention window (days) for output artifacts |
| `PIPELINE_LOG_RETENTION_DAYS` | int | 180 | Retention window (days) for pipeline logs |
| `PIPELINE_RETENTION_DRY_RUN` | bool | false | Log retention deletions without removing files |

**Example**:
```bash
# Ensure deterministic pipeline outputs
export PIPELINE_SEED=1234
```

---

## LLM Client Controls

Controls for rate limiting, concurrency, and caching in `tools/llm_client`.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `LLM_RATE_LIMIT_QPS` | float | unset | Global requests-per-second limit applied to all LLM providers. |
| `LLM_RATE_LIMIT_QPS_GEMINI` | float | unset | Provider-specific QPS limit for Gemini (overrides global). |
| `LLM_RATE_LIMIT_QPS_ANTHROPIC` | float | unset | Provider-specific QPS limit for Anthropic (overrides global). |
| `LLM_RATE_LIMIT_QPS_OPENAI` | float | unset | Provider-specific QPS limit for OpenAI (overrides global). |
| `LLM_MAX_CONCURRENCY` | int | unset | Max in-process concurrent LLM requests across providers (queued when saturated). |
| `LLM_CACHE_ENABLED` | bool | true | Enable prompt+settings response caching when TTL is set. |
| `LLM_CACHE_TTL_SECONDS` | int | 0 | Cache TTL in seconds; set >0 to enable caching. |

**Example**:
```bash
# Limit Gemini to 2 QPS and cap concurrency at 4 requests
export LLM_RATE_LIMIT_QPS_GEMINI=2
export LLM_MAX_CONCURRENCY=4

# Enable caching for 60 seconds
export LLM_CACHE_TTL_SECONDS=60
```

---

## Production Mode Resolution

Production mode is resolved through a shared helper (`tools/config/production_mode.py`). It treats the following
environment variables as production indicators, in the order below. The first match that evaluates to production
enables production mode; there is no explicit "false override" once any flag is set to a production value.

**Canonical flags (preferred)**:
1. `PIPELINE_ENV=production` or `PIPELINE_ENV=prod`
2. `PRODUCTION_MODE=1|true|yes`
3. `SIMREADY_PRODUCTION_MODE=1|true|yes`

**Legacy compatibility flags (still honored)**:
- `DATA_QUALITY_LEVEL=production`
- `ISAAC_SIM_REQUIRED=1|true|yes`
- `REQUIRE_REAL_PHYSICS=1|true|yes`
- `PRODUCTION=1|true|yes`
- `LABS_STAGING=1|true|yes`

**Example**:
```bash
# Canonical production indicator
export PIPELINE_ENV=production

# Legacy compatibility (still supported)
export DATA_QUALITY_LEVEL=production
```

---

### BP_QUALITY_* - Quality Gate Configuration

Quality validation thresholds can be overridden via environment variables.

Format: `BP_QUALITY_SECTION_KEY=value`

**Physics Thresholds**:
| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `BP_QUALITY_PHYSICS_MASS_MIN_KG` | float | 0.01 | Minimum object mass |
| `BP_QUALITY_PHYSICS_MASS_MAX_KG` | float | 500.0 | Maximum object mass |
| `BP_QUALITY_PHYSICS_FRICTION_MIN` | float | 0.0 | Minimum friction coefficient |
| `BP_QUALITY_PHYSICS_FRICTION_MAX` | float | 2.0 | Maximum friction coefficient |

**Episode Thresholds**:
| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `BP_QUALITY_EPISODES_COLLISION_FREE_RATE_MIN` | float | 0.90 | Min collision-free rate |
| `BP_QUALITY_EPISODES_QUALITY_SCORE_MIN` | float | 0.90 | Min quality score |
| `BP_QUALITY_EPISODES_QUALITY_PASS_RATE_MIN` | float | 0.70 | Min pass rate |
| `BP_QUALITY_EPISODES_MIN_EPISODES_REQUIRED` | int | 3 | Min episodes to validate |
| `BP_QUALITY_EPISODES_TIER_THRESHOLDS` | json | unset | JSON mapping of tier thresholds (core/plus/full). See example below. |

Tier overrides are merged by data pack tier (`DATA_PACK_TIER` or per-scene `data_pack_tier`). In production, episode thresholds must meet the production floor even if tier overrides are lower.

**Simulation Thresholds**:
| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `BP_QUALITY_SIMULATION_MIN_STABLE_STEPS` | int | 10 | Min stable simulation steps |
| `BP_QUALITY_SIMULATION_MAX_PENETRATION_DEPTH_M` | float | 0.01 | Max penetration depth |
| `BP_QUALITY_SIMULATION_PHYSICS_STABILITY_TIMEOUT_S` | float | 30.0 | Physics timeout |

**Example**:
```bash
# Stricter quality requirements
export BP_QUALITY_EPISODES_QUALITY_SCORE_MIN=0.95
export BP_QUALITY_EPISODES_COLLISION_FREE_RATE_MIN=0.95

# Tier overrides (JSON)
export BP_QUALITY_EPISODES_TIER_THRESHOLDS='{"core":{"collision_free_rate_min":0.90,"quality_score_min":0.90,"quality_pass_rate_min":0.70,"min_episodes_required":3},"plus":{"collision_free_rate_min":0.90,"quality_score_min":0.90,"quality_pass_rate_min":0.70,"min_episodes_required":4},"full":{"collision_free_rate_min":0.90,"quality_score_min":0.90,"quality_pass_rate_min":0.70,"min_episodes_required":5}}'

# Relaxed physics constraints
export BP_QUALITY_PHYSICS_FRICTION_MAX=5.0
```

---

### BP_PIPELINE_* - Pipeline Configuration

Pipeline execution parameters:

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `BP_PIPELINE_EPISODE_GENERATION_EPISODES_PER_TASK` | int | 10 | Episodes per task |
| `BP_PIPELINE_EPISODE_GENERATION_NUM_VARIATIONS` | int | 5 | Scene variations |
| `BP_PIPELINE_EPISODE_GENERATION_MAX_PARALLEL_EPISODES` | int | 8 | Parallel episode jobs |
| `BP_PIPELINE_EPISODE_GENERATION_EPISODE_TIMEOUT_SECONDS` | int | 300 | Episode timeout (seconds) |
| `BP_PIPELINE_VIDEO_FPS` | int | 30 | Video framerate |
| `BP_PIPELINE_VIDEO_RESOLUTION_WIDTH` | int | 640 | Video width (pixels) |
| `BP_PIPELINE_VIDEO_RESOLUTION_HEIGHT` | int | 480 | Video height (pixels) |
| `BP_PIPELINE_PHYSICS_TIMESTEP_HZ` | int | 120 | Physics simulation rate |
| `BP_PIPELINE_PHYSICS_SOLVER_ITERATIONS` | int | 4 | Physics solver iterations |
| `BP_PIPELINE_RESOURCE_ALLOCATION_GPU_MEMORY_FRACTION` | float | 0.8 | GPU memory fraction |
| `BP_PIPELINE_RESOURCE_ALLOCATION_NUM_CPU_WORKERS` | int | 4 | CPU worker threads |
| `BP_PIPELINE_RESOURCE_ALLOCATION_NUM_GPU_WORKERS` | int | 1 | GPU worker processes |

**Example**:
```bash
# High-throughput processing
export BP_PIPELINE_EPISODE_GENERATION_MAX_PARALLEL_EPISODES=32

# High quality video
export BP_PIPELINE_VIDEO_RESOLUTION_WIDTH=1280
export BP_PIPELINE_VIDEO_RESOLUTION_HEIGHT=720
export BP_PIPELINE_VIDEO_FPS=60

# More resources
export BP_PIPELINE_RESOURCE_ALLOCATION_NUM_CPU_WORKERS=16
export BP_PIPELINE_RESOURCE_ALLOCATION_GPU_MEMORY_FRACTION=0.9
```

---

### BP_SCENE_GRAPH_* - Scene Graph Configuration

Scene graph relation inference thresholds and streaming options.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `BP_SCENE_GRAPH_VERTICAL_PROXIMITY_THRESHOLD` | float | 0.05 | Vertical proximity threshold (meters) for inferring "on" relations |
| `BP_SCENE_GRAPH_HORIZONTAL_PROXIMITY_THRESHOLD` | float | 0.15 | Horizontal proximity threshold (meters) for inferring "adjacent" relations |
| `BP_SCENE_GRAPH_ALIGNMENT_ANGLE_THRESHOLD` | float | 5.0 | Alignment threshold (degrees) for inferring "aligned" relations |
| `BP_SCENE_GRAPH_STREAMING_BATCH_SIZE` | int | 100 | Batch size for streaming scene manifest parsing |

**Example**:
```bash
# Tune relation inference sensitivity
export BP_SCENE_GRAPH_VERTICAL_PROXIMITY_THRESHOLD=0.03
export BP_SCENE_GRAPH_HORIZONTAL_PROXIMITY_THRESHOLD=0.10
export BP_SCENE_GRAPH_ALIGNMENT_ANGLE_THRESHOLD=3.0
export BP_SCENE_GRAPH_STREAMING_BATCH_SIZE=200
```

---

### BP_ENABLE_CONFIG_AUDIT

Enable configuration audit trail logging.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `BP_ENABLE_CONFIG_AUDIT` | bool | "0" | Enable config source tracking |

When enabled, `ConfigLoader.dump_audit_trail()` shows where each config value came from.

**Example**:
```bash
export BP_ENABLE_CONFIG_AUDIT=1
# Now audit trails are tracked for debugging
```

---

## Simready Physics Configuration

Environment variables that control simready physics estimation behavior.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SIMREADY_PHYSICS_MODE` | str | "auto" | Physics estimation mode: "auto", "gemini", or "deterministic". Production workflows set this to "deterministic" to avoid Gemini dependencies. |
| `SIMREADY_ALLOW_DETERMINISTIC_PHYSICS` | bool | "0" | When `SIMREADY_PHYSICS_MODE=auto`, allow deterministic physics when Gemini is unavailable. |
| `SIMREADY_PRODUCTION_MODE` | bool | "0" | Force production-mode behavior (Secret Manager required for Gemini). |
| `SIMREADY_ALLOW_HEURISTIC_FALLBACK` | bool | "0" | Allow heuristic-only fallback when Gemini is unavailable in non-production runs. |
| `SIMREADY_FALLBACK_MIN_COVERAGE` | float | 0.6 | Minimum coverage ratio required for deterministic fallback physics. |
| `SIMREADY_NON_LLM_MIN_QUALITY` | float | 0.85 | Minimum quality threshold for non-LLM physics estimation. |

**Example**:
```bash
# Force deterministic physics in production workflows
export SIMREADY_PHYSICS_MODE=deterministic

# Allow deterministic fallback in auto mode
export SIMREADY_ALLOW_DETERMINISTIC_PHYSICS=1
```

---

## Credentials & Authentication

### Google Cloud

| Variable | Type | Required | Description |
|----------|------|----------|-------------|
| `GOOGLE_CLOUD_PROJECT` | str | Yes (alt: GOOGLE_CLOUD_PROJECT_ID) | GCP project ID |
| `GOOGLE_CLOUD_PROJECT_ID` | str | Yes (alt: GOOGLE_CLOUD_PROJECT) | GCP project ID |
| `GOOGLE_APPLICATION_CREDENTIALS` | path | Yes | Path to service account JSON key |

### Firebase / Firestore

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `FIRESTORE_PROJECT` | str | `blueprint-8c1ca` | Firestore project ID for pipeline configuration and metadata |
| `BUCKET` | str | `blueprint-8c1ca.appspot.com` | GCS bucket used for scene assets and pipeline artifacts (not the Firebase Storage upload target) |
| `FIREBASE_STORAGE_BUCKET` | str | None | Firebase Storage bucket used for Firebase upload jobs (e.g., `blueprint-8c1ca.appspot.com`) |
| `FIREBASE_SERVICE_ACCOUNT_JSON` | str | None | JSON payload for Firebase service account credentials |
| `FIREBASE_SERVICE_ACCOUNT_PATH` | path | None | Path to Firebase service account JSON credentials |
| `FIREBASE_UPLOAD_PREFIX` | str | `datasets` | Firebase upload prefix for episode artifacts |
| `FIREBASE_EPISODE_PREFIX` | str | `datasets` | Firebase upload prefix for Genie Sim episode artifacts |

Scene asset input path example (GCS):
```
gs://blueprint-8c1ca.appspot.com/scenes/{SCENE_ID}/images/room.png
```

**Example**:
```bash
export GOOGLE_CLOUD_PROJECT=my-blueprint-project
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
```

### Service Keys

| Variable | Type | Required | Description |
|----------|------|----------|-------------|
| `GEMINI_API_KEY` | str | No | Google Gemini key for LLM features (dev fallback only; production requires Secret Manager) |
| `HF_TOKEN` | str | No | HuggingFace token for model downloads |
| `MESHY_API_KEY` | str | No | Meshy key for 3D generation |
| `INVENTORY_ENRICHMENT_API_KEY` | str | No | Inventory enrichment key (dev fallback; Secret Manager preferred) |

---

## Feature Flags (ENABLE_ Prefix)

Feature flags allow opt-in/opt-out of capabilities.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ENABLE_PREMIUM_ANALYTICS` | bool | "0" | Enable premium analytics |
| `ENABLE_MULTI_ROBOT` | bool | "0" | Enable multi-robot scenes |
| `ENABLE_CUROBO` | bool | "1" | Enable cuRobo trajectory planning |
| `ENABLE_TACTILE_SIMULATION` | bool | "0" | Enable tactile sensor sim |
| `ENABLE_DEFORMABLE_OBJECTS` | bool | "0" | Enable deformable object sim |
| `ENABLE_STREAMING_EXPORT` | bool | "0" | Enable streaming data export |
| `ENABLE_VLA_FINETUNING` | bool | "0" | Enable VLA fine-tuning data |
| `ENABLE_SIM2REAL_VALIDATION` | bool | "0" | Enable sim-to-real checks |
| `ENABLE_DWM_CONDITIONING` | bool | "0" | Enable DWM-style conditioning |
| `ENABLE_DREAM2FLOW` | bool | "0" | Enable Dream2Flow preparation and inference workflows |
| `ENABLE_AUDIO_NARRATION` | bool | "0" | Enable audio narration |
| `ENABLE_SUBTITLE_GENERATION` | bool | "0" | Enable subtitle generation |
| `ENABLE_INVENTORY_ENRICHMENT` | bool | "0" | Enable inventory enrichment before replicator generation |
| `ENABLE_FIREBASE_UPLOAD` | bool | "0" | Enable Firebase Storage uploads for episode artifacts |

**Example**:
```bash
# Enable all advanced features
export ENABLE_MULTI_ROBOT=1
export ENABLE_TACTILE_SIMULATION=1
export ENABLE_DEFORMABLE_OBJECTS=1
export ENABLE_VLA_FINETUNING=1
export ENABLE_SIM2REAL_VALIDATION=1

# Or disable a known problematic feature
export ENABLE_CUROBO=0  # Use fallback planner
```

---

## Service Endpoints

Genie Sim runs locally using the gRPC host/port configuration below for client-server communication.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `GENIE_SIM_GRPC_PORT` | int | 50051 | Genie Sim gRPC port (local) |
| `GENIESIM_ENV` | str | `development` | Environment toggle for Genie Sim integrations (`production` disables mock/fallback behavior). |
| `GENIESIM_HOST` | str | `localhost` | Genie Sim gRPC host (local framework) |
| `GENIESIM_PORT` | int | 50051 | Genie Sim gRPC port (local framework). |
| `GENIESIM_ROOT` | path | `/opt/geniesim` | Genie Sim repository root (local framework) |
| `ISAACSIM_REQUIRED` | bool | false | Require Isaac Sim + Genie Sim installs when using the local framework (enforces `GENIESIM_ROOT` and `ISAAC_SIM_PATH/python.sh`). |
| `CUROBO_REQUIRED` | bool | false | Require cuRobo planning support when using the local framework. |
| `GENIESIM_MOCK_MODE` | bool | false | Enable mock mode for Genie Sim clients in non-production runs (requires `ALLOW_GENIESIM_MOCK=1` or an explicit code flag; ignored in production). |
| `ALLOW_GENIESIM_MOCK` | bool | 0 | Allow Genie Sim mock mode in non-production environments (`1` to enable; production always disables mock mode). |
| `GENIESIM_ALLOW_LINEAR_FALLBACK` | bool | unset | Allow linear interpolation fallback when cuRobo is unavailable (`1` to enable, `0` to disable). In non-production, the local framework auto-enables this fallback if cuRobo is missing and this variable is unset; in production, cuRobo is required and the framework fails fast. |
| `ISAAC_SIM_ENDPOINT` | url | `http://localhost:8011` | Isaac Sim endpoint |
| `ISAAC_SIM_PATH` | path | `/isaac-sim` | Isaac Sim installation path (local framework) |
| `OMNIVERSE_HOST` | str | `localhost` | Omniverse host used to resolve USD asset paths |
| `OMNIVERSE_PATH_ROOT` | path | (contextual) | Omniverse path root for USD assets (e.g., `NVIDIA/Assets/Isaac` or `NVIDIA/Robots`) |
| `PARTICULATE_HEALTHCHECK_HOST` | str | `localhost` | Hostname used for the particulate service health check |
| `PARTICULATE_SERVICE_PORT` | int | 5000 | Particulate service port |
| `INVENTORY_ENRICHMENT_ENDPOINT` | url | unset | Inventory enrichment service endpoint |

**Example**:
```bash
export GENIESIM_HOST=localhost
export GENIESIM_PORT=50051
export OMNIVERSE_HOST=localhost
export OMNIVERSE_PATH_ROOT=NVIDIA/Assets/Isaac
export PARTICULATE_HEALTHCHECK_HOST=localhost
```

**Production Example**:
```bash
export GENIESIM_ENV=production
export PIPELINE_ENV=production
export ISAACSIM_REQUIRED=true
export CUROBO_REQUIRED=true
```

For production asset indexing/embedding flows, ensure a valid embedding provider is configured
(`OPENAI_API_KEY` or `QWEN_API_KEY`/`DASHSCOPE_API_KEY`, plus the corresponding embedding model).

---

## Health Checks

Shared health probe settings used by webhook/services.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `HEALTH_PROBE_TIMEOUT_S` | float | 2.0 | Default timeout (seconds) for dependency health probes |

**Example**:
```bash
# Allow a longer probe window for slow startups
export HEALTH_PROBE_TIMEOUT_S=5.0
```

---

## Data & Storage

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DATA_OUTPUT_DIR` | path | `./outputs` | Output directory for exported data |
| `DATA_CACHE_DIR` | path | `./cache` | Cache directory for intermediate data |
| `GCS_BUCKET_DATA` | str | None | GCS bucket for data storage |
| `GCS_BUCKET_MODELS` | str | None | GCS bucket for model storage |
| `GCS_BUCKET_SCENES` | str | None | GCS bucket for scene configs |

**Example**:
```bash
export DATA_OUTPUT_DIR=/mnt/data/outputs
export GCS_BUCKET_DATA=my-project-data
export GCS_BUCKET_MODELS=my-project-models
```

---

## Logging & Debugging

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `LOG_LEVEL` | str | "INFO" | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `PARTICULATE_DEBUG` | bool | "0" | Enable Particulate service debug mode (requires `PARTICULATE_DEBUG_TOKEN` for /debug) |
| `PARTICULATE_DEBUG_TOKEN` | str | None | Shared secret required to access the Particulate `/debug` endpoint |
| `BLUEPRINT_DEBUG` | bool | "0" | Enable debug logging throughout pipeline |
| `ISAAC_SIM_DEBUG` | bool | "0" | Enable Isaac Sim debug output |

**Example**:
```bash
# Debug mode for troubleshooting
export LOG_LEVEL=DEBUG
export BLUEPRINT_DEBUG=1
export PARTICULATE_DEBUG=1
export PARTICULATE_DEBUG_TOKEN=<shared-secret>

# Production - disable debug
export PARTICULATE_DEBUG=0
export LOG_LEVEL=WARNING
```

---

## Robot & Environment

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DEFAULT_ROBOT` | str | "franka" | Default robot type |
| `ROBOT_CONFIG_DIR` | path | `./configs/robots` | Robot config directory |
| `SCENE_TEMPLATE_DIR` | path | `./configs/scenes` | Scene template directory |
| `OBJECT_ASSET_DIR` | path | `./assets/objects` | Object asset directory |

---

## Development & Testing

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `USE_MOCK_DATA` | bool | "0" | Use mock data instead of simulation |
| `MOCK_PHYSICS` | bool | "0" | Use mock physics (no Isaac Sim) |
| `SKIP_QUALITY_GATES` | bool | "0" | Skip quality validation (⚠️ dev only) |
| `DRY_RUN` | bool | "0" | Dry run (no output generated) |
| `PYTEST_VERBOSE` | bool | "0" | Verbose test output |
| `INVENTORY_ENRICHMENT_MODE` | str | "mock" | Inventory enrichment mode (`mock` or `external`) |

---

## Common Combinations

### Development Environment

```bash
# Local development with mock data
export LOG_LEVEL=DEBUG
export USE_MOCK_DATA=1
export MOCK_PHYSICS=1
export BP_ENABLE_CONFIG_AUDIT=1
export PARTICULATE_DEBUG=1
export PARTICULATE_DEBUG_TOKEN=<shared-secret>
```

### Production Environment

```bash
# Production configuration
export LOG_LEVEL=WARNING
export PARTICULATE_DEBUG=0
export ENABLE_CUROBO=1
export BP_PIPELINE_RESOURCE_ALLOCATION_NUM_CPU_WORKERS=16
export BP_PIPELINE_RESOURCE_ALLOCATION_NUM_GPU_WORKERS=4
export GOOGLE_CLOUD_PROJECT=prod-blueprint-pipeline
```

### High-Throughput Processing

```bash
# Optimize for throughput
export BP_PIPELINE_EPISODE_GENERATION_MAX_PARALLEL_EPISODES=32
export BP_PIPELINE_RESOURCE_ALLOCATION_NUM_CPU_WORKERS=32
export BP_PIPELINE_RESOURCE_ALLOCATION_NUM_GPU_WORKERS=8
export BP_PIPELINE_RESOURCE_ALLOCATION_GPU_MEMORY_FRACTION=0.95
```

### Research Lab (All Features)

```bash
# Enable all research features
export ENABLE_MULTI_ROBOT=1
export ENABLE_TACTILE_SIMULATION=1
export ENABLE_DEFORMABLE_OBJECTS=1
export ENABLE_VLA_FINETUNING=1
export ENABLE_SIM2REAL_VALIDATION=1
export BP_QUALITY_EPISODES_QUALITY_SCORE_MIN=0.95
export BP_SPLIT_STRATEGY=scene
```

---

## Priority Order

Configuration values are applied in this order (later overrides earlier):

1. **Code defaults** (hardcoded in dataclass)
2. **JSON config files** (`pipeline_config.json`, `quality_config.json`)
3. **Firestore** (if available and connected)
4. **Environment variables** (BP_* and other prefixes)
5. **Programmatic overrides** (passed to ConfigLoader)

---

## Validation & Errors

All environment variables are validated when:
- Pipeline starts up
- Configuration is loaded
- New configuration is applied

Common errors:

| Error | Cause | Solution |
|-------|-------|----------|
| `Invalid BP_SPLIT_SEED: not an integer` | Non-integer split seed | Ensure `BP_SPLIT_SEED=42` (no quotes) |
| `Split ratios must sum to 1.0` | Ratios don't sum | Adjust `BP_SPLIT_*_RATIO` values |
| `Unknown split strategy` | Invalid strategy | Use "random", "scene", or "task" |
| `GPU memory fraction must be 0-1` | Invalid fraction | Ensure `0.0 <= value <= 1.0` |

---

## Deprecations & Migration

### Old (Deprecated)

```bash
# Old environment variable names (no longer supported)
export QUALITY_SCORE_MIN=0.85  # Use BP_QUALITY_EPISODES_QUALITY_SCORE_MIN instead
export SPLIT_SEED=42  # Use BP_SPLIT_SEED instead
```

### New (Recommended)

```bash
# New standardized names with BP_ prefix
export BP_QUALITY_EPISODES_QUALITY_SCORE_MIN=0.85
export BP_SPLIT_SEED=42
```

---

## Tips & Best Practices

1. **Use .env files**: Create `.env` file for development configurations
2. **Document custom values**: Comment why you're overriding defaults
3. **Use feature flags wisely**: Don't enable experimental features in production
4. **Validate startup**: Check logs for `[CONFIG]` messages showing loaded values
5. **Use audit trail**: Enable `BP_ENABLE_CONFIG_AUDIT` when debugging config issues
6. **Secure credentials**: Never commit service keys; use GCP Secret Manager
7. **Monitor changes**: Track config changes in version control when possible

---

## Troubleshooting

### Configuration not taking effect

1. Check variable name: Must use exact BP_* prefix and UPPERCASE
2. Verify export: Use `env | grep BP_` to confirm variable is set
3. Check precedence: Firestore/JSON configs might override env vars
4. Restart process: Changes only apply on pipeline startup
5. Check validation: Look for error messages in startup logs

### Invalid configuration values

1. Review error message: States what's invalid and expected range
2. Use correct types: Integer vs float vs string (no quotes around numbers)
3. Validate ranges: Check min/max values in this documentation
4. Test locally: Run `tools/config/__init__.py` to test configurations

### Getting current values

```python
from tools.config import load_pipeline_config, ConfigLoader

# Load current configuration
config = load_pipeline_config()

# View split configuration
print(f"Split seed: {config.models.get_model('placement_engine').default_model}")

# Check audit trail (if enabled)
if audit := ConfigLoader.dump_audit_trail("pipeline"):
    print(audit)
```
