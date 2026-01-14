# Environment Variables Documentation

Complete reference for all environment variables used in the BlueprintPipeline.

## Overview

The pipeline uses environment variables for:
- Configuration overrides (BP_ prefix)
- API credentials (API keys, OAuth tokens)
- Feature flags (ENABLE_ prefix)
- Service endpoints (URL_ prefix)
- Debugging and logging

Environment variables take precedence over JSON config files, allowing runtime customization without redeployment.

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
| `BP_QUALITY_EPISODES_COLLISION_FREE_RATE_MIN` | float | 0.80 | Min collision-free rate |
| `BP_QUALITY_EPISODES_QUALITY_SCORE_MIN` | float | 0.85 | Min quality score |
| `BP_QUALITY_EPISODES_QUALITY_PASS_RATE_MIN` | float | 0.50 | Min pass rate |
| `BP_QUALITY_EPISODES_MIN_EPISODES_REQUIRED` | int | 1 | Min episodes to validate |

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

## Credentials & Authentication

### Google Cloud

| Variable | Type | Required | Description |
|----------|------|----------|-------------|
| `GOOGLE_CLOUD_PROJECT` | str | Yes (alt: GOOGLE_CLOUD_PROJECT_ID) | GCP project ID |
| `GOOGLE_CLOUD_PROJECT_ID` | str | Yes (alt: GOOGLE_CLOUD_PROJECT) | GCP project ID |
| `GOOGLE_APPLICATION_CREDENTIALS` | path | Yes | Path to service account JSON key |

**Example**:
```bash
export GOOGLE_CLOUD_PROJECT=my-blueprint-project
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
```

### API Keys

| Variable | Type | Required | Description |
|----------|------|----------|-------------|
| `GEMINI_API_KEY` | str | No | Google Gemini API key for LLM features (dev fallback only; production requires Secret Manager) |
| `HF_TOKEN` | str | No | HuggingFace API token for model downloads |
| `MESHY_API_KEY` | str | No | Meshy API key for 3D generation |

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
| `ENABLE_AUDIO_NARRATION` | bool | "0" | Enable audio narration |
| `ENABLE_SUBTITLE_GENERATION` | bool | "0" | Enable subtitle generation |

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
| `GENIESIM_HOST` | str | `localhost` | Genie Sim gRPC host (local framework) |
| `GENIESIM_PORT` | int | 50051 | Genie Sim gRPC port (local framework) |
| `GENIESIM_ROOT` | path | `/opt/geniesim` | Genie Sim repository root (local framework) |
| `ISAAC_SIM_ENDPOINT` | url | `http://localhost:8011` | Isaac Sim endpoint |
| `ISAAC_SIM_PATH` | path | `/isaac-sim` | Isaac Sim installation path (local framework) |
| `PARTICULATE_SERVICE_PORT` | int | 5000 | Particulate service port |

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
| `PARTICULATE_DEBUG` | bool | "1" | Enable Particulate service debug mode (⚠️ disables in prod) |
| `BLUEPRINT_DEBUG` | bool | "0" | Enable debug logging throughout pipeline |
| `ISAAC_SIM_DEBUG` | bool | "0" | Enable Isaac Sim debug output |

**Example**:
```bash
# Debug mode for troubleshooting
export LOG_LEVEL=DEBUG
export BLUEPRINT_DEBUG=1

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
6. **Secure credentials**: Never commit API keys; use GCP Secret Manager
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
