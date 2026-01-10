# BlueprintPipeline Improvement Specification

**Date:** January 10, 2026
**Related:** `PRODUCTION_READINESS_AUDIT_2026.md`
**Purpose:** Detailed specifications for each identified gap

---

## Table of Contents

1. [P0: Critical Fixes](#p0-critical-fixes)
2. [P1: High Priority Fixes](#p1-high-priority-fixes)
3. [P2: Medium Priority Improvements](#p2-medium-priority-improvements)
4. [P3: Low Priority Enhancements](#p3-low-priority-enhancements)

---

## P0: Critical Fixes

### SPEC-001: Fix tarfile Path Traversal Vulnerability

**Gap Reference:** GAP-SEC-003
**File:** `genie-sim-import-job/import_from_geniesim.py`
**Line:** ~873-874

**Current Code (Vulnerable):**
```python
with tarfile.open(archive_path, "r:gz") as tar:
    tar.extractall(output_dir)
```

**Required Fix:**
```python
import os
import tarfile

def safe_extract_tar(archive_path: Path, output_dir: Path) -> None:
    """
    Safely extract tarfile with path traversal protection.

    Args:
        archive_path: Path to tar.gz archive
        output_dir: Directory to extract to

    Raises:
        ValueError: If path traversal is detected
    """
    output_dir = Path(output_dir).resolve()

    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            # Normalize the member path
            member_path = (output_dir / member.name).resolve()

            # Check if extracted path would be outside output_dir
            try:
                member_path.relative_to(output_dir)
            except ValueError:
                raise ValueError(
                    f"Path traversal detected in archive: {member.name}"
                )

            # Check for absolute paths
            if member.name.startswith('/') or member.name.startswith('..'):
                raise ValueError(
                    f"Suspicious path in archive: {member.name}"
                )

        # Safe to extract
        tar.extractall(output_dir)
```

**Testing:**
```python
def test_safe_extract_blocks_path_traversal():
    # Create malicious archive with ../../../etc/passwd
    # Verify ValueError is raised
    pass

def test_safe_extract_allows_normal_paths():
    # Create normal archive
    # Verify extraction succeeds
    pass
```

**Acceptance Criteria:**
- [ ] Path traversal attempts raise `ValueError`
- [ ] Normal archives extract successfully
- [ ] Unit tests added and passing

---

### SPEC-002: Secret Manager Integration for Remaining API Keys

**Gap Reference:** GAP-SEC-001 (partially fixed)

**Files to Update:**

#### File 1: `simready-job/prepare_simready_assets.py`

**Current:**
```python
gemini_api_key = os.getenv("GEMINI_API_KEY")
```

**Required:**
```python
from tools.secrets import get_secret_or_env, SecretIds

gemini_api_key = get_secret_or_env(
    SecretIds.GEMINI_API_KEY,
    env_var="GEMINI_API_KEY"
)
```

#### File 2: `episode-generation-job/task_specifier.py`

**Current:**
```python
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
```

**Required:**
```python
from tools.secrets import get_secret_or_env, SecretIds

gemini_key = get_secret_or_env(SecretIds.GEMINI_API_KEY, env_var="GEMINI_API_KEY")
openai_key = get_secret_or_env(SecretIds.OPENAI_API_KEY, env_var="OPENAI_API_KEY")
api_key = gemini_key or openai_key
```

#### File 3: `tools/llm_client/client.py` (if exists)

Apply same pattern for any LLM client initialization.

**Acceptance Criteria:**
- [ ] All API keys fetched via `get_secret_or_env`
- [ ] Fallback to env vars works for local development
- [ ] Secret Manager access verified in production

---

### SPEC-003: Isaac Sim Container Setup Documentation

**Gap Reference:** External dependency on Isaac Sim

**Deliverable:** `docs/ISAAC_SIM_SETUP.md`

**Required Content:**

```markdown
# Isaac Sim Environment Setup

## Prerequisites
- NVIDIA GPU with RTX support
- Docker with NVIDIA Container Toolkit
- 32GB+ RAM recommended

## Option 1: NVIDIA NGC Container

```bash
# Pull Isaac Sim container
docker pull nvcr.io/nvidia/isaac-sim:2024.1.0

# Run with BlueprintPipeline mounted
docker run --gpus all -it \
  -v $(pwd):/workspace/BlueprintPipeline \
  -e DISPLAY=$DISPLAY \
  nvcr.io/nvidia/isaac-sim:2024.1.0 \
  bash

# Inside container
cd /workspace/BlueprintPipeline
/isaac-sim/python.sh tools/run_local_pipeline.py --help
```

## Option 2: Custom Dockerfile

```dockerfile
FROM nvcr.io/nvidia/isaac-sim:2024.1.0

WORKDIR /workspace
COPY . /workspace/BlueprintPipeline

# Install additional dependencies
RUN /isaac-sim/python.sh -m pip install -r /workspace/BlueprintPipeline/requirements.txt

ENTRYPOINT ["/isaac-sim/python.sh"]
CMD ["/workspace/BlueprintPipeline/tools/run_local_pipeline.py"]
```

## Verifying Isaac Sim Integration

```bash
# Test Isaac Sim availability
/isaac-sim/python.sh -c "from episode_generation_job.isaac_sim_integration import is_isaac_sim_available; print(is_isaac_sim_available())"

# Test Replicator availability
/isaac-sim/python.sh -c "from episode_generation_job.isaac_sim_integration import is_replicator_available; print(is_replicator_available())"
```

## Production Mode

For production episode generation:
```bash
PRODUCTION_MODE=true /isaac-sim/python.sh episode-generation-job/generate_episodes.py
```

This enforces Isaac Sim usage and prevents mock fallback.
```

**Acceptance Criteria:**
- [ ] Documentation covers container setup
- [ ] Verification commands work
- [ ] Production mode documented

---

## P1: High Priority Fixes

### SPEC-004: Integrate Failure Markers into Jobs

**Gap Reference:** GAP-EH-003
**Module:** `tools/workflow/failure_markers.py` (exists)

**Files to Update:**

#### Template for Integration:

```python
# At the top of each job file
from tools.workflow.failure_markers import write_failure_marker

# Wrap main execution
def main():
    bucket = os.getenv("BUCKET")
    scene_id = os.getenv("SCENE_ID")

    try:
        # Existing job logic
        result = run_job_logic()
        return result

    except Exception as e:
        # Write failure marker with context
        write_failure_marker(
            bucket=bucket,
            scene_id=scene_id,
            job_name="<job-name>",
            error=e,
            input_params={
                "robot_type": os.getenv("ROBOT_TYPE"),
                # Add other relevant params
            },
            partial_results={
                # Any successful intermediate results
            }
        )
        raise
```

**Files to Apply Pattern:**
1. `genie-sim-export-job/export_to_geniesim.py`
2. `simready-job/prepare_simready_assets.py`
3. `episode-generation-job/generate_episodes.py`
4. `usd-assembly-job/build_scene_usd.py`
5. `replicator-job/generate_replicator_bundle.py`

**Acceptance Criteria:**
- [ ] Failure markers written on job failure
- [ ] Markers include error context and stack trace
- [ ] Markers include input parameters
- [ ] Partial results captured when available

---

### SPEC-005: Integrate Partial Failure Handling

**Gap Reference:** GAP-EH-004
**Module:** `tools/error_handling/partial_failure.py` (exists)
**Target:** `episode-generation-job/generate_episodes.py`

**Current Pattern (Problematic):**
```python
for episode_idx in range(num_episodes):
    try:
        episode = generate_single_episode(...)
        episodes.append(episode)
    except Exception as e:
        logger.error(f"Episode {episode_idx} failed: {e}")
        # Episode lost, but loop continues
```

**Required Pattern:**
```python
from tools.error_handling import PartialFailureHandler, save_successful_items

def generate_episodes_with_partial_failure(
    config: EpisodeGenerationConfig,
    min_success_rate: float = 0.5
) -> List[Episode]:
    """
    Generate episodes with partial failure handling.

    Saves successful episodes even if some fail.
    Raises if success rate falls below threshold.
    """
    handler = PartialFailureHandler(
        min_success_rate=min_success_rate,
        save_successful_fn=lambda items: save_episodes_to_disk(items, config.output_dir)
    )

    for episode_idx in range(config.num_episodes):
        result = handler.execute(
            fn=generate_single_episode,
            args=(config, episode_idx),
            item_id=f"episode_{episode_idx}"
        )

        if result.success:
            handler.add_successful(result.value)

        # Periodic save (every 10 episodes)
        if (episode_idx + 1) % 10 == 0:
            handler.save_checkpoint()

    # Final validation
    if handler.success_rate < min_success_rate:
        # Save what we have before raising
        handler.save_checkpoint()
        raise PartialFailureError(
            f"Success rate {handler.success_rate:.2%} below threshold {min_success_rate:.2%}",
            successful_items=handler.successful_items,
            failed_items=handler.failed_items
        )

    return handler.successful_items
```

**Acceptance Criteria:**
- [ ] Successful episodes saved even when others fail
- [ ] Periodic checkpoints prevent data loss
- [ ] Minimum success rate enforced
- [ ] Failure details logged for debugging

---

### SPEC-006: Integrate Config Validation at Job Startup

**Gap Reference:** GAP-CM-001
**Module:** `tools/validation/config_schemas.py` (exists)

**Pattern for Each Job:**

```python
# At the top of main()
from tools.validation.config_schemas import (
    validate_environment_config,
    EnvironmentConfigError
)

def main():
    # Validate environment before proceeding
    try:
        config = validate_environment_config(
            required_vars=[
                "BUCKET",
                "SCENE_ID",
            ],
            optional_vars=[
                ("ROBOT_TYPE", "franka"),
                ("NUM_EPISODES", "100"),
            ],
            validated_patterns={
                "SCENE_ID": r"^[a-zA-Z0-9_-]+$",
            }
        )
    except EnvironmentConfigError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)

    # Proceed with validated config
    bucket = config["BUCKET"]
    scene_id = config["SCENE_ID"]
    # ...
```

**Files to Apply:**
1. All job entry points (`*-job/*.py`)
2. `tools/run_local_pipeline.py`

**Acceptance Criteria:**
- [ ] Jobs fail fast with clear error on missing config
- [ ] Config patterns validated (no injection attacks)
- [ ] Defaults applied for optional configs

---

### SPEC-007: Add CI/CD Pipeline

**Gap Reference:** GAP-TEST-001

**Deliverable:** `.github/workflows/ci.yaml`

```yaml
name: CI

on:
  push:
    branches: [main, claude/*]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install ruff black mypy
      - name: Lint with ruff
        run: ruff check .
      - name: Check formatting
        run: black --check .
      - name: Type check (optional failures)
        run: mypy --ignore-missing-imports . || true

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-critical-gaps.txt
          pip install pytest pytest-cov
      - name: Run tests
        run: |
          pytest tests/ -v --cov=tools --cov=genie-sim-export-job \
            --cov-report=xml --cov-report=term
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: coverage.xml

  e2e-mock:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Generate mock data
        run: |
          python fixtures/generate_mock_regen3d.py \
            --scene-id ci_test \
            --output-dir ./test_output
      - name: Run local pipeline (mock mode)
        run: |
          python tools/run_local_pipeline.py \
            --scene-dir ./test_output/ci_test \
            --steps regen3d,simready,usd \
            --validate
```

**Acceptance Criteria:**
- [ ] CI runs on all PRs
- [ ] Linting with ruff
- [ ] Unit tests with pytest
- [ ] E2E test with mock data
- [ ] Coverage reporting

---

## P2: Medium Priority Improvements

### SPEC-008: Increase Test Coverage

**Gap Reference:** GAP-TEST-002
**Current:** 4 test files (~1.5% ratio)
**Target:** 20+ test files (30%+ coverage)

**Test Files to Create:**

```
tests/
├── unit/
│   ├── test_error_handling/
│   │   ├── test_retry.py
│   │   ├── test_circuit_breaker.py
│   │   ├── test_timeout.py
│   │   └── test_partial_failure.py
│   │
│   ├── test_validation/
│   │   ├── test_input_validation.py
│   │   ├── test_config_schemas.py
│   │   └── test_path_sanitization.py
│   │
│   ├── test_secrets/
│   │   └── test_secret_manager.py
│   │
│   ├── test_geniesim_adapter/
│   │   ├── test_scene_graph.py
│   │   ├── test_asset_index.py
│   │   └── test_task_config.py
│   │
│   └── test_performance/
│       ├── test_streaming_json.py
│       └── test_parallel_processing.py
│
├── integration/
│   ├── test_simready_job.py
│   ├── test_usd_assembly_job.py
│   └── test_replicator_job.py
│
└── e2e/
    ├── test_pipeline_e2e.py (exists)
    └── test_genie_sim_mode.py
```

**Example Test Template:**
```python
"""Tests for tools/error_handling/retry.py"""
import pytest
from unittest.mock import Mock, patch
from tools.error_handling import retry_with_backoff, RetryableError

class TestRetryWithBackoff:
    def test_succeeds_on_first_attempt(self):
        mock_fn = Mock(return_value="success")

        @retry_with_backoff(max_retries=3)
        def test_fn():
            return mock_fn()

        result = test_fn()

        assert result == "success"
        assert mock_fn.call_count == 1

    def test_retries_on_failure_then_succeeds(self):
        mock_fn = Mock(side_effect=[Exception(), Exception(), "success"])

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def test_fn():
            return mock_fn()

        result = test_fn()

        assert result == "success"
        assert mock_fn.call_count == 3

    def test_raises_after_max_retries(self):
        mock_fn = Mock(side_effect=Exception("always fails"))

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def test_fn():
            return mock_fn()

        with pytest.raises(Exception, match="always fails"):
            test_fn()

        assert mock_fn.call_count == 3
```

**Acceptance Criteria:**
- [ ] 20+ test files created
- [ ] 30%+ code coverage achieved
- [ ] All P0 modules have >80% coverage
- [ ] Tests run in CI

---

### SPEC-009: Add Structured Logging

**Gap Reference:** GAP-OBS-001

**Pattern to Apply:**

```python
import logging
import json
import sys
from typing import Any, Dict

class StructuredLogger:
    """JSON-structured logger for Cloud Logging compatibility."""

    def __init__(self, name: str, level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # JSON formatter for Cloud Logging
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(handler)

    def info(self, message: str, **extra: Any) -> None:
        self.logger.info(message, extra={"structured": extra})

    def error(self, message: str, **extra: Any) -> None:
        self.logger.error(message, extra={"structured": extra})

    def warning(self, message: str, **extra: Any) -> None:
        self.logger.warning(message, extra={"structured": extra})


class StructuredFormatter(logging.Formatter):
    """Formatter that outputs JSON for Cloud Logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "severity": record.levelname,
            "message": record.getMessage(),
            "timestamp": self.formatTime(record),
            "logger": record.name,
        }

        # Add structured extras
        if hasattr(record, "structured"):
            log_entry.update(record.structured)

        # Add exception info
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


# Usage in jobs:
logger = StructuredLogger("genie-sim-export-job")

logger.info(
    "Exporting scene to Genie Sim",
    scene_id=scene_id,
    num_objects=len(objects),
    robot_type=config.robot_type
)
```

**Files to Update:**
All job entry points and tools modules.

**Acceptance Criteria:**
- [ ] All logs output JSON when `STRUCTURED_LOGGING=true`
- [ ] Logs include scene_id, job_name for filtering
- [ ] Exception traces included in error logs

---

### SPEC-010: Add Alerting Rules

**Gap Reference:** GAP-MON-001

**Deliverable:** `monitoring/alerting_policies.yaml`

```yaml
# Cloud Monitoring Alerting Policies
alertPolicies:
  - displayName: "High Job Failure Rate"
    conditions:
      - displayName: "Job failures > 5%"
        conditionThreshold:
          filter: |
            resource.type = "cloud_run_job"
            metric.type = "run.googleapis.com/job/completed_task_count"
            metric.labels.result = "failed"
          aggregations:
            - alignmentPeriod: 300s
              perSeriesAligner: ALIGN_RATE
              crossSeriesReducer: REDUCE_SUM
          comparison: COMPARISON_GT
          thresholdValue: 0.05
          duration: 300s
    notificationChannels:
      - projects/PROJECT_ID/notificationChannels/CHANNEL_ID
    documentation:
      content: |
        Job failure rate exceeded 5% threshold.

        **Action Required:**
        1. Check Cloud Run logs for error details
        2. Check for external service outages (Genie Sim, Gemini)
        3. Review recent deployments

  - displayName: "High Latency"
    conditions:
      - displayName: "P95 latency > 10 minutes"
        conditionThreshold:
          filter: |
            resource.type = "cloud_run_job"
            metric.type = "run.googleapis.com/job/execution_time"
          aggregations:
            - alignmentPeriod: 300s
              perSeriesAligner: ALIGN_PERCENTILE_95
          comparison: COMPARISON_GT
          thresholdValue: 600  # 10 minutes in seconds
          duration: 600s
    notificationChannels:
      - projects/PROJECT_ID/notificationChannels/CHANNEL_ID

  - displayName: "API Rate Limiting"
    conditions:
      - displayName: "429 errors detected"
        conditionThreshold:
          filter: |
            resource.type = "cloud_run_job"
            log.level = "ERROR"
            textPayload =~ "rate limit"
          aggregations:
            - alignmentPeriod: 60s
              perSeriesAligner: ALIGN_COUNT
          comparison: COMPARISON_GT
          thresholdValue: 10
          duration: 60s
```

**Acceptance Criteria:**
- [ ] Alerts defined in YAML
- [ ] Alerts deployed via Terraform or gcloud
- [ ] Notification channel configured (email/Slack)
- [ ] Runbook linked in alert documentation

---

## P3: Low Priority Enhancements

### SPEC-011: Add Type Hints Across Codebase

**Gap Reference:** GAP-CQ-003

**Approach:**
1. Add `mypy` to CI (initially allow errors)
2. Gradually add type hints starting with:
   - `tools/` modules
   - Job entry points
   - Public interfaces

**Example Transformation:**

```python
# Before
def generate_scene_graph(manifest_path, output_dir, include_premium=True):
    manifest = load_manifest(manifest_path)
    ...

# After
from pathlib import Path
from typing import Optional
from tools.geniesim_adapter.types import SceneGraph

def generate_scene_graph(
    manifest_path: Path,
    output_dir: Path,
    include_premium: bool = True
) -> SceneGraph:
    """
    Generate scene graph from manifest.

    Args:
        manifest_path: Path to scene_manifest.json
        output_dir: Directory to write outputs
        include_premium: Whether to include premium features

    Returns:
        Generated scene graph
    """
    manifest = load_manifest(manifest_path)
    ...
```

**Acceptance Criteria:**
- [ ] `mypy` runs in CI (warnings only initially)
- [ ] Public interfaces have type hints
- [ ] Type stubs added where needed

---

### SPEC-012: Add OpenTelemetry Tracing

**Gap Reference:** GAP-OBS-003

**Implementation:**

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

# Initialize tracing
def init_tracing():
    provider = TracerProvider()
    processor = BatchSpanProcessor(CloudTraceSpanExporter())
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

tracer = trace.get_tracer(__name__)

# Usage in jobs
@tracer.start_as_current_span("generate_scene_graph")
def generate_scene_graph(manifest_path: Path) -> SceneGraph:
    span = trace.get_current_span()
    span.set_attribute("scene_id", scene_id)
    span.set_attribute("object_count", len(objects))

    with tracer.start_as_current_span("load_manifest"):
        manifest = load_manifest(manifest_path)

    with tracer.start_as_current_span("build_nodes"):
        nodes = build_nodes(manifest)

    return SceneGraph(nodes=nodes)
```

**Acceptance Criteria:**
- [ ] Traces visible in Cloud Trace
- [ ] End-to-end trace spans entire pipeline
- [ ] Slow operations identifiable

---

## Implementation Timeline

| Phase | Duration | Specs | Outcome |
|-------|----------|-------|---------|
| **Week 1** | 5 days | SPEC-001 to SPEC-003 | Security fixes, Isaac Sim docs |
| **Week 2** | 5 days | SPEC-004 to SPEC-007 | Error handling, CI/CD |
| **Week 3** | 5 days | SPEC-008 to SPEC-010 | Test coverage, observability |
| **Week 4+** | Ongoing | SPEC-011 to SPEC-012 | Type hints, tracing |

---

## Verification Checklist

### After P0 Fixes:
- [ ] tarfile extraction is secure
- [ ] All API keys use Secret Manager (with env fallback)
- [ ] Isaac Sim setup documented and tested

### After P1 Fixes:
- [ ] Failure markers written on job errors
- [ ] Partial failures save successful items
- [ ] Config validation at startup
- [ ] CI/CD pipeline running

### After P2 Fixes:
- [ ] 30%+ test coverage
- [ ] Structured JSON logging
- [ ] Alerting rules active

### After P3 Fixes:
- [ ] Type hints in public interfaces
- [ ] End-to-end tracing working

---

*Specification document for BlueprintPipeline improvements*
*Generated: 2026-01-10*
