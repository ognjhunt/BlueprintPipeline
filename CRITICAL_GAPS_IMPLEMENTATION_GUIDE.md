# Critical Gaps Implementation Guide

This guide provides step-by-step instructions for applying all 10 critical gap fixes (GAP-EH-001 through GAP-PERF-002) to the BlueprintPipeline codebase.

**Date:** January 9, 2026
**Status:** ✅ All infrastructure created - Ready for integration
**Priority:** P0 - Critical for production reliability

---

## Executive Summary

All critical infrastructure has been created and is ready for integration:

### ✅ Completed Infrastructure

1. **Unified Retry & Circuit Breaker** - `tools/external_services/service_client.py`
2. **Timeout Handling** - `tools/error_handling/timeout.py`
3. **Input Validation** - `tools/validation/input_validation.py`
4. **Configuration Schemas** - `tools/validation/config_schemas.py`
5. **Secret Manager Integration** - `tools/secrets/secret_manager.py`
6. **Streaming JSON Parser** - `tools/performance/streaming_json.py`
7. **Parallel Processing** - `tools/performance/parallel_processing.py`
8. **Partial Failure Handling** - `tools/error_handling/partial_failure.py`
9. **Enhanced Failure Markers** - `tools/workflow/failure_markers.py`

---

## Implementation Steps by Priority

### Priority 1: External Service Calls (GAP-EH-001, GAP-EH-002)

**Impact:** Prevents transient failures from causing job failures
**Effort:** 2-3 hours
**Files to Update:**
- `genie-sim-export-job/geniesim_client.py`
- `simready-job/prepare_simready_assets.py`
- `regen3d-job/regen3d_adapter_job.py`
- `episode-generation-job/task_specifier.py`

**Instructions:**

#### 1.1 Update Genie Sim Client

**File:** `genie-sim-export-job/geniesim_client.py`

Initialize with local gRPC configuration:
```python
def __init__(self, ...):
    # Local framework operation uses gRPC endpoints
    self.grpc_host = os.getenv("GENIESIM_HOST", "localhost")
    self.grpc_port = int(os.getenv("GENIESIM_PORT", "50051"))
    self.local_endpoint = f"{self.grpc_host}:{self.grpc_port}"

    # No API key needed for local operation
    self.api_key = None
    self.endpoint = self.local_endpoint
    ...
```

Validate local gRPC endpoint:
```python
def _validate_local_endpoint(self) -> None:
    """Check that local gRPC endpoint is reachable."""
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5)
    result = sock.connect_ex((self.grpc_host, self.grpc_port))
    sock.close()

    if result != 0:
        raise GenieSimAPIError(f"Cannot connect to {self.local_endpoint}")
    response = self.session.post(f"{self.api_url}/jobs", json=config)
    response.raise_for_status()
    return response.json()["job_id"]
```

#### 1.2 Update Gemini API Calls

**File:** `simready-job/prepare_simready_assets.py`

Add import:
```python
from tools.external_services import create_gemini_client
```

Initialize client:
```python
gemini_client = create_gemini_client()
```

Wrap Gemini calls:
```python
# Before (no retry):
response = genai.GenerativeModel('gemini-1.5-pro').generate_content(prompt)

# After (with retry + circuit breaker):
response = gemini_client.call(
    func=lambda: genai.GenerativeModel('gemini-1.5-pro').generate_content(prompt),
    timeout=30.0,
    operation_name="estimate_physics",
)
```

#### 1.3 Update Particulate Service Calls

**File:** Any file calling Particulate service

```python
from tools.external_services import create_particulate_client

particulate_client = create_particulate_client()

# Wrap calls
result = particulate_client.call(
    func=lambda: requests.post(endpoint, json=data),
    timeout=15.0,
)
```

---

### Priority 2: Timeout Handling for Isaac Sim (GAP-EH-005)

**Impact:** Prevents hung processes from blocking jobs indefinitely
**Effort:** 1 hour
**Files to Update:**
- `episode-generation-job/isaac_sim_integration.py`
- `episode-generation-job/sensor_data_capture.py`

**Instructions:**

#### 2.1 Add Timeout to Physics Steps

**File:** `episode-generation-job/isaac_sim_integration.py`

Add import:
```python
from tools.error_handling import timeout_thread, TimeoutManager
```

Initialize timeout manager:
```python
class IsaacSimIntegration:
    def __init__(self, ...):
        ...
        self.timeout_manager = TimeoutManager(default_timeout=60.0)
        self.timeout_manager.set("physics_step", 10.0)
        self.timeout_manager.set("render", 30.0)
```

Wrap physics step:
```python
def step(self) -> PhysicsStepResult:
    """Step physics simulation with timeout protection."""
    if self._use_real_physics:
        with self.timeout_manager("physics_step"):
            self._physics_context.step(render=True)
    ...
```

#### 2.2 Add Timeout to Rendering

```python
def render(self) -> np.ndarray:
    """Render frame with timeout protection."""
    with self.timeout_manager("render"):
        return self._render_impl()
```

---

### Priority 3: Input Validation (GAP-SEC-002)

**Impact:** Prevents injection attacks and invalid data
**Effort:** 2 hours
**Files to Update:**
- `tools/geniesim_adapter/scene_graph.py`
- `tools/geniesim_adapter/task_config.py`
- All user input processing

**Instructions:**

#### 3.1 Validate Scene Graph Objects

**File:** `tools/geniesim_adapter/scene_graph.py`

Add import:
```python
from tools.validation import (
    validate_object_id,
    validate_category,
    validate_description,
    validate_dimensions,
    sanitize_path,
)
```

---

## Genie Sim Export SLI/SLOs and Error Budgets

This section defines the service-level indicators (SLIs), service-level objectives (SLOs), and error budgets that the Genie Sim export pipeline enforces and alerts on.

### SLIs

1. **Export Job Duration (Genie Sim Export Job):** Time from job start to completion for the export job.
2. **Submission Job Duration (Genie Sim Submit/GPU Job):** Time from submission job start to completion.
3. **Genie Sim Generation Duration:** End-to-end duration from submission to generation completion (from `job_metrics.duration_seconds`).
4. **Job Failure Rate:** Count of workflow job invocations that complete with `FAILED` status.

### SLOs

| SLI | Objective | Workflow Field | Default |
| --- | --- | --- | --- |
| Export Job Duration | 99% of export jobs complete within 45 minutes | `geniesimExportTimeoutSeconds` | 2700 seconds |
| Submission Job Duration | 99% of submission jobs complete within 60 minutes | `geniesimSubmissionTimeoutSeconds` | 3600 seconds |
| Generation Duration | 95% of Genie Sim generation jobs complete within 4 hours | `geniesimGenerationSlaSeconds` | 14400 seconds |
| Job Failure Rate | <1% failures per 7-day window | `bp_metric=job_invocation` | 1% error budget |

### Error Budgets

- **Export/Submission Duration:** 1% of runs per 7-day window may exceed SLOs before paging.
- **Generation Duration:** 5% of runs per 7-day window may exceed 4 hours before paging.
- **Failure Rate:** 1% of workflow invocations may fail per 7-day window.

### Monitoring & Alerting Wiring

The following log-based metrics and alert policies are wired into Cloud Monitoring:

- `blueprint_job_failure_events` → Alert policy: **[Blueprint] Workflow Job Failure Detected**
- `blueprint_geniesim_sla_violations` → Alert policy: **[Blueprint] Genie Sim SLA Violation**
- `blueprint_job_timeout_events`, `blueprint_job_timeout_usage_ratio`, `blueprint_job_retry_exhausted_total`

These metrics are emitted via structured workflow logs and should be deployed alongside existing monitoring infrastructure:

- Metrics definitions: `infrastructure/monitoring/metrics/*.yaml`
- Alert policies: `infrastructure/monitoring/alerts/alert-policies.yaml`

Add validation in `_convert_object_to_node`:
```python
def _convert_object_to_node(self, obj: Dict, usd_base_path: str) -> Optional[SceneNode]:
    # Validate inputs
    try:
        obj_id = validate_object_id(obj.get("id", "unknown"))
        category = validate_category(obj.get("category", "object"))
        description = validate_description(obj.get("description", ""))

        # Validate dimensions if present
        if "dimensions" in obj:
            dimensions = validate_dimensions(obj["dimensions"])

        # Sanitize paths
        if "asset_path" in obj:
            asset_path = sanitize_path(
                obj["asset_path"],
                allowed_parent=Path(usd_base_path),
            )

    except ValidationError as e:
        self.log(f"Invalid object data: {e}", "ERROR")
        return None

    # Continue with validated data
    ...
```

---

### Priority 4: Configuration Schema Validation (GAP-CM-001)

**Impact:** Catches configuration errors early
**Effort:** 1-2 hours
**Files to Update:**
- All job entry points
- `tools/scene_manifest/manifest_loader.py`

**Instructions:**

#### 4.1 Validate Scene Manifests

**File:** `tools/scene_manifest/manifest_loader.py` (or create if doesn't exist)

```python
from tools.validation.config_schemas import load_and_validate_manifest

def load_manifest(manifest_path: Path) -> dict:
    """Load and validate scene manifest."""
    try:
        # Validate with Pydantic schema
        manifest = load_and_validate_manifest(manifest_path)

        # Convert to dict for backward compatibility
        return manifest.model_dump()

    except ValidationError as e:
        logger.error(f"Manifest validation failed: {e}")
        raise ConfigurationError(f"Invalid manifest: {e}")
```

#### 4.2 Validate Environment Configuration

**File:** All job entry points (`main.py`, `run.py`, etc.)

```python
from tools.validation.config_schemas import load_and_validate_env_config

def main():
    # Validate environment configuration
    try:
        env_config = load_and_validate_env_config()
    except ValidationError as e:
        print(f"[ERROR] Invalid environment configuration: {e}")
        sys.exit(1)

    # Use validated config
    bucket = env_config.bucket
    scene_id = env_config.scene_id
    ...
```

---

### Priority 5: Secret Manager Integration (GAP-SEC-001)

**Impact:** Eliminates credential leakage risk
**Effort:** 2 hours
**Files to Update:**
- All files using API keys
- Workflow YAML files (if feasible)

**Instructions:**

#### 5.1 Replace Plain Text API Keys

**File:** `genie-sim-export-job/geniesim_client.py`

```python
from tools.secrets import get_secret_or_env, SecretIds

class GenieSimClient:
    def __init__(self, ...):
        # Local framework operation - no API key needed
        self.api_key = None
        self.grpc_host = os.getenv("GENIESIM_HOST", "localhost")
        self.grpc_port = int(os.getenv("GENIESIM_PORT", "50051"))
```

#### 5.2 Batch Load Secrets at Startup

**File:** Job entry points

```python
from tools.secrets import load_pipeline_secrets

def main():
    # Load all secrets at startup
    try:
        secrets = load_pipeline_secrets()
        print("[INFO] Loaded secrets from Secret Manager")
    except Exception as e:
        print(f"[WARNING] Could not load secrets: {e}")
        print("[INFO] Falling back to environment variables")

    ...
```

---

### Priority 6: Streaming JSON Parser (GAP-PERF-001)

**Impact:** Prevents OOM for large scenes (>1000 objects)
**Effort:** 2-3 hours
**Files to Update:**
- `tools/geniesim_adapter/scene_graph.py`
- Any code loading large manifests

**Instructions:**

#### 6.1 Use Streaming Parser for Large Scenes

**File:** `tools/geniesim_adapter/scene_graph.py`

```python
from tools.performance import StreamingManifestParser

def convert_manifest_to_scene_graph(manifest_path: Path, ...) -> SceneGraph:
    """Convert manifest with streaming for large scenes."""
    parser = StreamingManifestParser(manifest_path)

    # Get metadata without loading objects
    scene_id = parser.get_scene_id()
    version = parser.get_version()
    metadata = parser.get_metadata()

    # Check object count
    object_count = parser.count_objects()

    if object_count > 1000:
        print(f"[INFO] Large scene detected ({object_count} objects), using streaming parser")
        nodes = []

        # Process objects in batches
        for batch in parser.stream_objects(batch_size=100):
            for obj in batch:
                node = convert_object_to_node(obj)
                if node:
                    nodes.append(node)
    else:
        # Small scene - use standard loading
        with open(manifest_path) as f:
            manifest = json.load(f)
        nodes = [convert_object_to_node(obj) for obj in manifest["objects"]]

    return SceneGraph(nodes=nodes, metadata=metadata)
```

---

### Priority 7: Parallel Processing (GAP-PERF-002)

**Impact:** 10-50x speedup for independent operations
**Effort:** 2-3 hours
**Files to Update:**
- `simready-job/prepare_simready_assets.py`
- Any code processing objects sequentially

**Instructions:**

#### 7.1 Parallelize Physics Estimation

**File:** `simready-job/prepare_simready_assets.py`

```python
from tools.performance import process_parallel_threaded

# Before (sequential - slow):
physics_estimates = []
for obj in objects:
    physics = estimate_physics_with_gemini(obj)
    physics_estimates.append(physics)

# After (parallel - fast):
result = process_parallel_threaded(
    objects,
    process_fn=estimate_physics_with_gemini,
    max_workers=20,  # I/O-bound, can use many workers
    timeout=10.0,
)

physics_estimates = result.successful

# Handle failures
if result.failed:
    print(f"[WARNING] {len(result.failed)} objects failed physics estimation")
    for failure in result.failed:
        print(f"  - {failure['item_index']}: {failure['error']}")
```

---

### Priority 8: Partial Failure Handling (GAP-EH-004)

**Impact:** Saves successful episodes even when some fail
**Effort:** 2 hours
**Files to Update:**
- `episode-generation-job/generate_episodes.py`

**Instructions:**

#### 8.1 Add Partial Failure Handling to Episode Generation

**File:** `episode-generation-job/generate_episodes.py`

Add import:
```python
from tools.error_handling import PartialFailureHandler, save_successful_items
```

Modify `_generate_seed_episodes` method:
```python
def _generate_seed_episodes(
    self,
    tasks_with_specs: List[Tuple[Dict, TaskSpecification]],
    manifest: Dict,
) -> List[GeneratedEpisode]:
    """Generate seed episodes with partial failure handling."""

    # Create failure handler
    failure_handler = PartialFailureHandler(
        min_success_rate=0.5,  # Allow 50% failure rate
        save_successful=True,
        output_dir=self.config.output_dir / "partial_results",
        failure_report_path=self.config.output_dir / "failures" / "seed_episodes.json",
    )

    # Process tasks with partial failure handling
    def generate_single_episode(task_spec_tuple):
        task, spec = task_spec_tuple
        # ... existing episode generation logic ...
        return episode

    result = failure_handler.process_batch(
        tasks_with_specs,
        process_fn=generate_single_episode,
        item_id_fn=lambda ts: ts[0]["task_id"],
        batch_name="seed_episodes",
    )

    # Log results
    self.log(
        f"Generated {result.success_count}/{result.total_attempted} seed episodes "
        f"({result.success_rate:.1%} success rate)"
    )

    if result.failed:
        self.log(f"Failed episodes: {len(result.failed)}", "WARNING")
        for failure in result.failed[:5]:  # Show first 5
            self.log(f"  - {failure['item_id']}: {failure['error']}", "WARNING")

    return result.successful
```

---

### Priority 9: Enhanced Failure Markers (GAP-EH-003)

**Impact:** 8x faster debugging
**Effort:** 1 hour
**Files to Update:**
- All job entry points

**Instructions:**

#### 9.1 Wrap Main Execution with Failure Marker

**File:** All job `main.py` or `run.py` files

```python
from tools.workflow import write_failure_marker

def main():
    # Get environment variables
    bucket = os.getenv("BUCKET")
    scene_id = os.getenv("SCENE_ID")
    job_name = "genie-sim-export-job"  # Change per job

    # Collect input params
    input_params = {
        "robot_type": os.getenv("ROBOT_TYPE", "franka"),
        "max_tasks": int(os.getenv("MAX_TASKS", "10")),
        "enable_premium_analytics": os.getenv("ENABLE_PREMIUM_ANALYTICS", "true"),
    }

    try:
        # Main job logic
        result = process_scene(scene_id, ...)

        # Success marker (optional)
        write_success_marker(bucket, scene_id, result)

    except Exception as e:
        # Write enhanced failure marker
        write_failure_marker(
            bucket=bucket,
            scene_id=scene_id,
            job_name=job_name,
            exception=e,
            failed_step="process_scene",  # Update per failure location
            input_params=input_params,
            partial_results={
                "objects_processed": getattr(e, 'objects_processed', 0),
            },
            recommendations=[
                "Check manifest format matches schema",
                "Verify all required assets are available",
            ],
        )

        # Re-raise to fail the job
        raise

if __name__ == "__main__":
    main()
```

---

## Testing Checklist

After applying fixes, test each component:

### Unit Tests

```bash
# Test retry logic
pytest tests/test_retry.py -v

# Test input validation
pytest tests/test_validation.py -v

# Test streaming parser
pytest tests/test_streaming_json.py -v

# Test parallel processing
pytest tests/test_parallel_processing.py -v
```

### Integration Tests

```bash
# Test with small scene (< 100 objects)
python tools/run_local_pipeline.py --scene-id test_small

# Test with large scene (> 1000 objects)
python tools/run_local_pipeline.py --scene-id test_large

# Test failure scenarios
python tests/test_failure_handling.py
```

---

## Rollout Strategy

### Phase 1: Non-Critical Jobs (Week 1)
- Apply fixes to `simready-job`
- Monitor for issues
- Adjust configurations based on feedback

### Phase 2: Critical Jobs (Week 2)
- Apply to `genie-sim-export-job`
- Apply to `episode-generation-job`
- Monitor production metrics

### Phase 3: Workflows (Week 3)
- Update workflow YAML files with enhanced error handling
- Deploy to production

---

## Success Metrics

Track these KPIs before and after implementation:

| Metric | Before | Target | Measurement |
|--------|--------|--------|-------------|
| Pipeline success rate | 85% | 99% | Cloud Logging |
| Mean time to resolution (MTTR) | 2 hours | 15 min | Incident tracking |
| Cost per scene (reruns) | $10 | $1 | Billing dashboard |
| P95 processing time (100 obj) | 200s | 20s | Custom metrics |
| Credential leaks | Unknown | 0 | Security audit |

---

## Rollback Plan

If issues arise:

1. **Revert Priority 1 (External Services)**
   - Restore original `geniesim_client.py`
   - Remove service client imports

2. **Disable New Features**
   ```python
   USE_STREAMING_PARSER = False
   USE_PARALLEL_PROCESSING = False
   USE_PARTIAL_FAILURE = False
   ```

3. **Monitor Logs**
   - Check for increased error rates
   - Verify no new exception types

---

## Appendix: Quick Reference

### Import Cheatsheet

```python
# External services (retry + circuit breaker + timeout)
from tools.external_services import (
    create_genie_sim_client,
    create_gemini_client,
    create_gcs_client,
    create_particulate_client,
)

# Timeout handling
from tools.error_handling import timeout, timeout_thread, TimeoutManager

# Input validation
from tools.validation import (
    validate_object_id,
    validate_scene_id,
    validate_category,
    validate_description,
    validate_dimensions,
    sanitize_path,
)

# Configuration schemas
from tools.validation.config_schemas import (
    load_and_validate_manifest,
    load_and_validate_env_config,
)

# Secrets
from tools.secrets import get_secret, get_secret_or_env, SecretIds

# Performance
from tools.performance import (
    StreamingManifestParser,
    process_parallel_threaded,
    process_parallel_multiprocess,
)

# Partial failure
from tools.error_handling import (
    PartialFailureHandler,
    process_with_partial_failure,
)

# Failure markers
from tools.workflow import write_failure_marker
```

---

## Next Steps

1. ✅ Review this implementation guide
2. 🔄 Create unit tests for new utilities
3. 🔄 Apply fixes to codebase (follow priority order)
4. 🔄 Run integration tests
5. 🔄 Deploy to staging environment
6. 🔄 Monitor metrics
7. 🔄 Deploy to production

---

**Questions?** Check the detailed audit spec: `GENIE_SIM_3_PIPELINE_AUDIT_SPEC_2026.md`
