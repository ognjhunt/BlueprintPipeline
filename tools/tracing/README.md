# Distributed Tracing for BlueprintPipeline

OpenTelemetry-based distributed tracing for cross-service observability.

## Overview

This module provides distributed tracing capabilities to track requests across services and understand:
- Cross-service dependencies
- Performance bottlenecks
- Error propagation
- Request flow through the pipeline

## Quick Start

### Installation

```bash
# Core dependencies
pip install -r tools/tracing/requirements.txt

# Or install minimal set
pip install opentelemetry-api opentelemetry-sdk
```

### Basic Usage

```python
from tools.tracing import trace_job, set_trace_attribute, init_tracing

# Initialize tracing (auto-detects exporter)
init_tracing(service_name="blueprint-pipeline")

# Trace a job
with trace_job("regen3d-job", scene_id="scene_001"):
    # Your processing code
    process_scene()

    # Add custom attributes
    set_trace_attribute("objects.count", 42)
    set_trace_attribute("physics.mode", "deterministic")
```

### Function Decorator

```python
from tools.tracing import trace_function

@trace_function(attributes={"component": "physics"})
def compute_physics_properties(obj_id: str):
    # Function is automatically traced
    pass
```

## Configuration

Tracing is **disabled by default** and can be enabled in several ways:

### 1. Environment Variables

```bash
# Enable tracing explicitly
export ENABLE_TRACING=true

# Or enable via production mode
export PIPELINE_ENV=production

# Configure exporter
export GCP_PROJECT_ID=your-project-id  # Uses Cloud Trace
# OR
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317  # Uses OTLP
# OR
export JAEGER_ENDPOINT=localhost:6831  # Uses Jaeger
```

### 2. Programmatic Initialization

```python
from tools.tracing import init_tracing

# Enable tracing
init_tracing(service_name="my-service", enabled=True)
```

## Supported Backends

### Cloud Trace (Google Cloud)

**Recommended for production GCP deployments**

```bash
# Install
pip install opentelemetry-exporter-gcp-trace

# Configure
export GCP_PROJECT_ID=your-project-id
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json

# Traces appear in: https://console.cloud.google.com/traces
```

### OTLP (OpenTelemetry Protocol)

**Works with Jaeger, Zipkin, and other OTLP-compatible backends**

```bash
# Install
pip install opentelemetry-exporter-otlp

# Configure
export OTEL_EXPORTER_OTLP_ENDPOINT=http://collector:4317
```

### Jaeger (Local Development)

```bash
# Install
pip install opentelemetry-exporter-jaeger

# Run Jaeger (Docker)
docker run -d --name jaeger \
  -p 16686:16686 \
  -p 6831:6831/udp \
  jaegertracing/all-in-one:latest

# Configure
export JAEGER_ENDPOINT=localhost:6831

# View traces: http://localhost:16686
```

### Console (Debugging)

```bash
# Enable console output for local debugging
export PIPELINE_ENV=local
export DEBUG_TRACING=true
```

## Integration with Job Wrapper

The `run_job_with_dead_letter_queue` wrapper **automatically traces all jobs** when tracing is enabled:

```python
from tools.error_handling.job_wrapper import run_job_with_dead_letter_queue

def main():
    # Job is automatically traced
    pass

if __name__ == "__main__":
    run_job_with_dead_letter_queue(
        main,
        scene_id="scene_001",
        job_type="regen3d-job",
        step="regen3d",
    )
```

## Advanced Usage

### Manual Span Management

```python
from tools.tracing import get_tracer

tracer = get_tracer()

with tracer.start_span("custom-operation", attributes={"key": "value"}):
    # Your code
    pass
```

### Cross-Service Propagation

```python
from tools.tracing import inject_trace_context, extract_trace_context
import requests

# Service A: Inject context
headers = inject_trace_context()
response = requests.post(url, headers=headers, json=data)

# Service B: Extract context
context = extract_trace_context(request.headers)
# Context is automatically propagated to child spans
```

### Error Tracking

```python
from tools.tracing import trace_job, set_trace_error

with trace_job("my-job"):
    try:
        risky_operation()
    except Exception as e:
        set_trace_error(e)
        raise
```

## Performance Impact

### FREE Path (Tracing Disabled)
- **Zero overhead**: No-op context managers when disabled
- No imports of OpenTelemetry libraries
- Safe for performance-critical paths

### Tracing Enabled
- **Minimal overhead**: ~1-5ms per span
- Batched export (background thread)
- Sampling supported (can be configured)

## Best Practices

1. **Use descriptive span names**: `job:regen3d-job` not `process`
2. **Add meaningful attributes**: Scene IDs, object counts, modes
3. **Trace at job boundaries**: Not individual function calls (too granular)
4. **Propagate context**: Use inject/extract for cross-service calls
5. **Don't trace high-frequency operations**: Avoid tracing tight loops

## Integration with Metrics

Tracing complements existing metrics (`tools/metrics/pipeline_metrics.py`):

- **Metrics**: Aggregate statistics (counts, durations, rates)
- **Tracing**: Individual request flow and timing breakdown

Use both together for comprehensive observability:

```python
from tools.metrics.pipeline_metrics import get_metrics
from tools.tracing import trace_job

metrics = get_metrics()

with trace_job("regen3d-job", scene_id="scene_001"):
    with metrics.track_job("regen3d-job", "scene_001"):
        # Both metrics and traces are collected
        process_scene()
```

## Viewing Traces

### Cloud Trace (GCP)
1. Open Google Cloud Console
2. Navigate to **Operations > Trace**
3. Filter by service name: `blueprint-pipeline`
4. View waterfall diagrams of request flow

### Jaeger
1. Open http://localhost:16686
2. Select service: `blueprint-pipeline`
3. Search for traces by operation or tags

## Troubleshooting

### Tracing Not Working

1. **Check if enabled**:
   ```bash
   # Must be true or in production mode
   echo $ENABLE_TRACING
   echo $PIPELINE_ENV
   ```

2. **Check exporter configuration**:
   ```bash
   # At least one must be set
   echo $GCP_PROJECT_ID
   echo $OTEL_EXPORTER_OTLP_ENDPOINT
   echo $JAEGER_ENDPOINT
   ```

3. **Check installation**:
   ```bash
   python -c "import opentelemetry; print('OK')"
   ```

4. **Enable debug logging**:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

### No Traces in Backend

1. **Verify exporter is initialized**:
   - Check logs for "OpenTelemetry tracing initialized"
   - Check for "Using [exporter] exporter"

2. **Check network connectivity**:
   - Test connection to exporter endpoint
   - Verify firewall rules

3. **Check sampling**:
   - Default is 100% sampling
   - Can be reduced via environment variable

## Examples

See test files for complete examples:
- `tests/test_tracing_integration.py`
- `examples/tracing_example.py`

## References

- [OpenTelemetry Documentation](https://opentelemetry.io/docs/)
- [Cloud Trace](https://cloud.google.com/trace/docs)
- [Jaeger Documentation](https://www.jaegertracing.io/docs/)
