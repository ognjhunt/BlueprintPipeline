# Regen3D Scaling Design for 1000 Scenes/Day (GKE)

## Objective
Scale Regen3D from single-scene execution on one VM to a queue-driven, autoscaled GPU worker fleet on GKE capable of sustained **1000 completed scenes/day** with controlled cost and operational safety.

## Baseline and Capacity Math

### Current baseline
- Average scene runtime (end-to-end): **~35 minutes**
- Current execution model: single VM, one scene at a time

### Throughput target
- Target: **1000 scenes/day**
- Required average throughput: `1000 / 24 = 41.67 scenes/hour`

### Concurrency requirement
Using average runtime `T = 35 min = 0.583 hr`:
- Required concurrent workers (ideal):
  - `41.67 * 0.583 = 24.3`
- Add production headroom for retries, stragglers, startup delays, and noisy neighbors:
  - +30% buffer -> `~31.6`

### Recommended sizing
- **Design concurrency target: 32 active GPU workers**
- **Initial autoscaler bounds:**
  - `min workers = 8`
  - `max workers = 48`
- One scene per worker, one GPU per pod.

## Target Architecture

### Components
1. **Ingress API / Event handler**
- Receives scene reconstruction requests.
- Validates payload and writes canonical job record to Firestore (or Cloud SQL).
- Publishes job message to Pub/Sub.

2. **Pub/Sub queue**
- Source of truth for pending work.
- Message includes: `scene_id`, `generation_id`, input asset URI, requested options, priority.

3. **Controller service (stateless)**
- Pulls messages from Pub/Sub.
- Performs idempotence check (`scene_id + generation_id`).
- Creates Kubernetes Job targeting GPU worker image.
- Applies retry metadata and dead-letter policy.

4. **GPU worker Jobs on GKE**
- One scene per Job.
- Node pool: GKE GPU nodes (L4 initially).
- Worker runs Regen3D pipeline steps and writes outputs directly to GCS.
- Worker emits structured status and metrics before exit.

5. **Result finalizer**
- Marks scene complete/failed in metadata store.
- Writes marker files in GCS (success/failure).
- Publishes completion event for downstream consumers.

## Scheduling and Execution Policy

### Scene scheduling
- **Policy:** one scene per GPU worker.
- No co-scheduling of multiple scenes on one GPU.
- Priority classes optional (normal vs urgent) if queue latency needs tiering.

### Idempotence
- Primary idempotence key: `scene_id:generation_id`.
- Controller enforces "create-once" semantics for active/recent jobs.
- Worker startup checks for completion marker; if exists, exits success (no duplicate processing).

### Retry and dead-letter
- Retry only for transient failures (VM/GPU init, network, dependency fetch, temporary quota).
- Suggested retry policy:
  - max attempts: 3
  - exponential backoff: 2m, 10m, 30m
- Non-transient errors (invalid input, unsupported format) route directly to dead-letter topic.
- Dead-letter processor captures diagnostics and opens triage issue.

### Failure isolation
- Job-level timeout (e.g. 90 minutes) to prevent stuck GPU pods.
- Kubernetes `backoffLimit: 0`; retries are orchestrated by controller for better reason visibility.

## GKE and Autoscaling Configuration

### Cluster layout
- Regional control plane.
- Dedicated GPU node pool for workers.
- Separate CPU node pool for controller/finalizer/ops tools.

### Autoscaling
- Use KEDA or custom metrics-driven autoscaling based on Pub/Sub queue depth + in-flight jobs.
- Scale-to-zero disabled for GPU pool in first rollout; maintain warm minimum (`min=8`) to reduce cold-start penalty.
- Node auto-provisioning allowed only for approved GPU types.

### Container/runtime
- Immutable worker image with pinned dependency versions.
- Read-only root FS where possible.
- Work dir on ephemeral disk; persistent artifacts only in GCS.

## Observability and SLOs

### Core metrics
- Queue depth (oldest message age + total backlog)
- Scene latency p50/p90/p95 (enqueue -> completed)
- Step-level failure rates (segmentation, inpainting, pose, assembly)
- Retry rate and dead-letter volume
- GPU utilization and GPU memory saturation
- Worker startup latency (job scheduled -> pipeline start)

### SLO targets (initial)
- Completion success rate: **>= 97% per day**
- End-to-end latency p95: **<= 120 minutes**
- Dead-letter ratio: **<= 1.5%**
- Queue staleness (oldest pending): **<= 30 minutes** during steady state

### Alerts
- Queue age breach >30m for 10m
- Success rate <97% over rolling 2h
- Dead-letter spike >2x baseline
- GPU pool at max capacity with growing backlog

## Security and Secrets
- Use Workload Identity; do not inject long-lived service account keys.
- Scope access via IAM roles per workload.
- Keep API keys in Secret Manager and mount at runtime.
- Sanitize command and exception logs to prevent secret leakage.

## Cost and Efficiency Controls
- Autoscaler max bounds enforced by budget guardrails.
- Preemptible/spot GPU pool can be introduced for non-urgent jobs after stability.
- Time-of-day bounds can cap spend during low-priority windows.
- Track cost per completed scene and cost per successful retry.

## Rollout Plan

### Phase 0: Readiness hardening (current)
- Runner defaults, secrets, VM scope, A/B segmentation validation.
- Lock dependency versions for deterministic worker image.

### Phase 1: Pilot
- Target: 100 scenes/day equivalent load.
- Fixed worker cap: 8-12 GPUs.
- Validate idempotence, retries, observability, and DLQ handling.

### Phase 2: Canary autoscale
- Route 20% of production traffic to GKE queue path.
- Autoscaler bounds: 8-24 GPUs.
- Compare quality and latency vs baseline path.

### Phase 3: Full rollout
- Route 100% traffic.
- Autoscaler bounds: 8-48 GPUs.
- Enforce SLO-based rollback trigger.

## Rollback Plan
- Keep legacy single-VM path available behind feature flag.
- If canary violates SLO for 30 minutes:
  - stop new job dispatch to GKE
  - drain in-flight jobs
  - reroute ingress to legacy path
- Preserve idempotence keys so re-dispatch does not duplicate completed scenes.

## Open Decisions (tracked, not blockers)
- Final metadata store (Firestore vs Cloud SQL) based on reporting/query needs.
- GPU type mix (L4 only vs mixed L4/A100 tiers) after pilot profiling.
- Whether to split long-running steps into separate queue stages for finer retry granularity.
