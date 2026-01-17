# Operations Handbook

This handbook consolidates operational guidance for the BlueprintPipeline and links to deeper runbooks. Use it as the
starting point for incident response, on-call routines, and runtime troubleshooting.

## Core runbooks and references

- Deployment runbook: [`docs/deployment_runbook.md`](../deployment_runbook.md)
- Rollback procedures: [`docs/rollback.md`](../rollback.md)
- Troubleshooting guide: [`docs/troubleshooting.md`](../troubleshooting.md)
- Runtime report template: [`docs/OPS_RUNTIME_REPORT.md`](../OPS_RUNTIME_REPORT.md)
- Scripts overview: [`scripts/README.md`](../../scripts/README.md)
- Workflow timeouts: [`workflows/TIMEOUT_AND_RETRY_POLICY.md`](../../workflows/TIMEOUT_AND_RETRY_POLICY.md)

## Incident response

1. **Declare and triage**
   - Open an incident channel, assign an incident commander, and start a timeline.
   - Identify impacted workflows, jobs, and regions (Cloud Run, GKE, workflows).
2. **Stabilize**
   - Pause new workflow triggers if needed (disable Eventarc triggers or upstream job submissions).
   - Capture current state: recent workflow run IDs, job revisions, image tags, and error logs.
3. **Diagnose**
   - Use [`docs/troubleshooting.md`](../troubleshooting.md) to map symptoms to likely causes.
   - Check pipeline health locally where relevant with:
     - `python tools/run_local_pipeline.py --validate`
     - `python -m tools.geniesim_adapter.geniesim_healthcheck --json`
4. **Mitigate or rollback**
   - Apply targeted fixes or follow [`docs/rollback.md`](../rollback.md) to revert jobs or workflows.
   - For infrastructure changes, follow Terraform rollback guidance in the rollback doc.
5. **Communicate and resolve**
   - Post status updates with scope, ETA, and mitigation status.
   - Log remediation steps and update the incident timeline.
6. **Post-incident**
   - File follow-up actions, update runbooks, and capture metrics in [`docs/OPS_RUNTIME_REPORT.md`](../OPS_RUNTIME_REPORT.md).

## Dataset export failure handling

Use this workflow when export artifacts (e.g., `scene.usda`, replicator outputs) are missing or corrupt.

1. **Confirm upstream inputs**
   - Validate that `scene_manifest.json` and layout files exist in the expected GCS prefixes.
   - If needed, regenerate sample outputs using `fixtures/generate_mock_regen3d.py`.
2. **Re-run validation locally**
   - `python tools/run_local_pipeline.py --validate` to catch missing assets or invalid transforms.
3. **Reprocess the scene**
   - Remove completion markers (e.g., `.regen3d_complete`) or partial outputs, then rerun the workflow.
   - See cleanup commands in [`docs/rollback.md`](../rollback.md).
4. **Escalate if exporter is unhealthy**
   - For Genie Sim exports, run `python -m tools.geniesim_adapter.geniesim_healthcheck --json` and follow the
     Genie Sim troubleshooting steps in [`docs/troubleshooting.md`](../troubleshooting.md).

## Quality-gate override audit process

When a pipeline gate is overridden (manual approval to proceed despite validation failures), capture and audit:

1. **Record the override**
   - Capture requester, approver, timestamp, and affected `SCENE_ID`s.
   - Document which validations failed and why the override was granted.
2. **Capture evidence**
   - Attach logs, screenshots, or summaries of the failing checks.
   - Link the workflow run ID and job logs.
3. **Post-override monitoring**
   - Monitor downstream jobs for regressions or follow-up failures.
   - Update [`docs/OPS_RUNTIME_REPORT.md`](../OPS_RUNTIME_REPORT.md) if timeouts or retries changed.
4. **Retrospective**
   - Decide whether to codify the exception (fix the gate) or tighten validation rules.

## Backup and restore steps

> Note: Update these steps with environment-specific bucket names and retention policies.

1. **Identify critical data**
   - GCS prefixes for pipeline outputs (scene assets, USD, replicator, logs).
   - Terraform state buckets and workflow definitions.
2. **Back up GCS data**
   - Use `gsutil -m rsync -r` to a backup bucket or `gsutil cp` for snapshots.
3. **Back up Terraform state**
   - Follow the Terraform state rollback guidance in [`docs/rollback.md`](../rollback.md).
4. **Restore**
   - Restore GCS artifacts before rerunning workflows.
   - Reapply Terraform, then redeploy workflow YAML and Cloud Run jobs using
     [`docs/deployment_runbook.md`](../deployment_runbook.md).

## On-call escalation

1. **Primary on-call**
   - Triage using this handbook and [`docs/troubleshooting.md`](../troubleshooting.md).
2. **Secondary escalation**
   - Escalate to platform or infra owners if the issue is in Terraform, GKE, or networking.
3. **Vendor escalation**
   - For Isaac Sim/Genie Sim runtime issues, involve the simulation platform owners.
4. **Communication**
   - Keep stakeholders updated with impact, mitigation status, and next update time.

## GPU / Isaac Sim outage playbooks

### GPU capacity exhaustion (Cloud Run/GKE)

1. **Confirm shortage**
   - Check GKE node pool status or Cloud Run GPU quota errors.
2. **Mitigate**
   - Scale down non-critical workloads.
   - Requeue or pause new pipeline submissions.
3. **Recover**
   - Scale node pools or request quota increases.
   - Re-run failed stages once GPUs are available.

### Isaac Sim service outage

1. **Health checks**
   - `python -m tools.geniesim_adapter.geniesim_healthcheck --json`
   - `python -m tools.geniesim_adapter.geniesim_server --health-check --host <host> --port 50051`
2. **Restart or redeploy**
   - Use `scripts/run-isaacsim-local.sh` for local restarts.
   - Redeploy Genie Sim jobs via `scripts/deploy-genie-sim-gpu-job.sh` or the runbook.
3. **Validate recovery**
   - Re-run a known-good scene or a small smoke test through the pipeline.

## Operational entrypoints and scripts

- Local pipeline validation: [`tools/run_local_pipeline.py`](../../tools/run_local_pipeline.py)
- Regen3d mock export generator: [`fixtures/generate_mock_regen3d.py`](../../fixtures/generate_mock_regen3d.py)
- Genie Sim health check: [`tools/geniesim_adapter/geniesim_healthcheck.py`](../../tools/geniesim_adapter/geniesim_healthcheck.py)
- Genie Sim server/health check: [`tools/geniesim_adapter/geniesim_server.py`](../../tools/geniesim_adapter/geniesim_server.py)
- Production validation harness: [`scripts/run_production_e2e_validation.py`](../../scripts/run_production_e2e_validation.py)
- Isaac Sim local runner: [`scripts/run-isaacsim-local.sh`](../../scripts/run-isaacsim-local.sh)
- Genie Sim GPU job deployer: [`scripts/deploy-genie-sim-gpu-job.sh`](../../scripts/deploy-genie-sim-gpu-job.sh)
- Pipeline enable script: [`scripts/enable_pipeline.sh`](../../scripts/enable_pipeline.sh)
