# PhysX Service Integration Improvements

## Summary

This document describes improvements made to the interactive job's PhysX service integration to address timeout and reliability issues.

## Problem Analysis

### What Was Happening

Based on logs from the interactive job execution:

1. **PhysX service timeouts** - All initial requests to the PhysX service were timing out after 60 seconds:
   ```
   [PHYSX] WARNING: failed to query endpoint https://physx-service-744608654760.us-central1.run.app: The read operation timed out
   ```

2. **Placeholder assets generated** - Because all requests timed out, only placeholder assets were created (54-byte `part.glb` files instead of actual 3D meshes)

3. **No detailed error information** - Logs didn't show HTTP status codes, response bodies, or distinguish between different failure types

### Root Causes

1. **Heavy ML service cold starts** - The PhysX service uses a massive Docker image with:
   - NVIDIA CUDA 12.6
   - PyTorch 2.5.0 with GPU support
   - Kaolin, TRELLIS, PhysX-Anything ML models
   - Multiple CUDA extensions (xformers, flash-attn, spconv, etc.)

   Cold start time: **2-5+ minutes** (but timeout was only 60 seconds)

2. **No retry logic** - When requests failed, the job immediately fell back to placeholders

3. **Insufficient error diagnostics** - Couldn't tell if the service was down, cold starting, or failing during processing

## Improvements Implemented

### 1. Increased Timeout to 5 Minutes

**File**: `interactive-job/run_interactive_assets.py:67`

**Change**:
```python
# Before
with urllib.request.urlopen(req, timeout=60) as resp:

# After
with urllib.request.urlopen(req, timeout=300) as resp:  # 5 min for cold start
```

**Rationale**: Allows sufficient time for the PhysX service to cold start and process requests.

### 2. Enhanced Error Logging

**File**: `interactive-job/run_interactive_assets.py:78-110`

**Changes**:
- Separate exception handlers for HTTP errors, network errors, and unexpected errors
- Log HTTP status codes (4xx vs 5xx)
- Include error response bodies (first 500 chars)
- Log exception types and details
- Add success confirmation logging

**Example output**:
```
[PHYSX] ERROR: HTTP 503 from https://...: Service Unavailable - cold starting
[PHYSX] ERROR: Network error for https://...: timed out
[PHYSX] Response status: 200 for /mnt/gcs/scenes/.../crop.png
[PHYSX] Success on attempt 1 for /mnt/gcs/scenes/.../crop.png
```

### 3. Retry Logic with Exponential Backoff

**File**: `interactive-job/run_interactive_assets.py:50-113`

**Changes**:
- Added `max_retries` parameter (default: 2 retries = 3 total attempts)
- Exponential backoff: 1s, 2s, 4s between retries
- Smart retry strategy:
  - **Retry**: 5xx errors (server errors), network timeouts, connection errors
  - **Don't retry**: 4xx errors (client errors like bad request, auth failure)
  - **Retry**: Unexpected errors (defensive programming)

**Benefits**:
- Handles transient failures (network blips, temporary service unavailability)
- Gives cold-starting services time to become ready
- Doesn't waste time retrying permanent failures (4xx errors)

### 4. Service Health Check

**File**: `interactive-job/run_interactive_assets.py:26-47, 454-457`

**Changes**:
- Added `check_service_health()` function
- Calls GET endpoint before processing to verify service is reachable
- Logs warning if service is unavailable (helps distinguish cold start from misconfiguration)

**Example output**:
```
[PHYSX] Checking service health...
[PHYSX] Service health check OK: https://physx-service-744608654760.us-central1.run.app
```

Or:
```
[PHYSX] Service health check FAILED for https://...: HTTP Error 503: Service Unavailable
[PHYSX] WARNING: PhysX service may be cold starting or unavailable. Requests may timeout or fail.
```

## Expected Behavior After Changes

### Successful Case (Service Available)
```
[PHYSX] Checking service health...
[PHYSX] Service health check OK: https://physx-service-...
[PHYSX] Processing 8 objects with 8 workers
[PHYSX] POST https://physx-service-... with crop /mnt/gcs/.../view_0.png
[PHYSX] Response status: 200 for /mnt/gcs/.../view_0.png
[PHYSX] Success on attempt 1 for /mnt/gcs/.../view_0.png
[PHYSX] Wrote interactive bundle for obj_upper_cabinet_left -> ...
```

### Cold Start Case (Service Warming Up)
```
[PHYSX] Checking service health...
[PHYSX] Service health check FAILED for https://...: timed out
[PHYSX] WARNING: PhysX service may be cold starting or unavailable...
[PHYSX] POST https://physx-service-... with crop /mnt/gcs/.../view_0.png
[PHYSX] ERROR: Network error for https://...: timed out
[PHYSX] Retrying after 1s due to network error...
[PHYSX] Retry 1/2: POST https://physx-service-... with crop /mnt/gcs/.../view_0.png
[PHYSX] Response status: 200 for /mnt/gcs/.../view_0.png
[PHYSX] Success on attempt 2 for /mnt/gcs/.../view_0.png
```

### Persistent Failure Case
```
[PHYSX] POST https://physx-service-... with crop /mnt/gcs/.../view_0.png
[PHYSX] ERROR: HTTP 500 from https://...: Internal Server Error - Pipeline failed
[PHYSX] Retrying after 1s due to HTTP 500...
[PHYSX] Retry 1/2: POST ...
[PHYSX] ERROR: HTTP 500 from https://...: Internal Server Error - Pipeline failed
[PHYSX] Retrying after 2s due to HTTP 500...
[PHYSX] Retry 2/2: POST ...
[PHYSX] ERROR: HTTP 500 from https://...: Internal Server Error - Pipeline failed
[PHYSX] INFO: using placeholder assets for obj_upper_cabinet_left
```

## Additional Recommendations

### For Production Deployment

1. **Increase Cloud Run Job timeout** - Ensure the interactive-job has sufficient timeout in its Cloud Run configuration:
   ```bash
   gcloud run jobs update interactive-job \
     --region=us-central1 \
     --task-timeout=30m
   ```

2. **Configure PhysX service minimum instances** - Reduce cold starts by keeping at least 1 instance warm:
   ```bash
   gcloud run services update physx-service \
     --region=us-central1 \
     --min-instances=1 \
     --max-instances=10
   ```

   **Note**: This increases costs but eliminates cold start delays.

3. **Monitor service metrics** - Track:
   - Request latency (should be under 2-3 minutes for warm instances)
   - Cold start frequency
   - 5xx error rate
   - Request timeout rate

4. **Consider request queuing** - For high-volume scenarios, implement a queue (Cloud Tasks, Pub/Sub) to:
   - Batch requests to the PhysX service
   - Retry failed requests with backoff
   - Track processing status

### For Development/Testing

1. **Test with actual service** - Run a test job to verify:
   - Service responds within timeout
   - Actual GLB/URDF files are generated (not 54-byte placeholders)
   - Files are valid and loadable

2. **Monitor logs** - Check for:
   - "Success on attempt N" messages (indicates successful processing)
   - Actual file sizes in GCS (GLB files should be KB-MB, not 54 bytes)
   - HTTP status codes (200 = success, 503 = cold start, 500 = pipeline error)

3. **Test cold start resilience** - Scale PhysX service to 0, trigger job, verify it succeeds after retries

## Files Modified

- `interactive-job/run_interactive_assets.py` - All improvements implemented in this file

## Next Steps

1. Deploy updated interactive-job Docker image
2. Trigger a test job execution
3. Monitor logs for improved diagnostics
4. Verify actual 3D assets are generated (check file sizes in GCS)
5. Consider implementing production recommendations if issues persist
