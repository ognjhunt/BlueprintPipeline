# Secret rotation runbook

This runbook covers how scheduled secret rotation works, how to validate that it succeeded, and how to respond if a
rotation causes downstream impact.

## Scheduling and cadence

Secret rotation is triggered by a Cloud Scheduler job that invokes a Cloud Run job.

- Scheduler resource: `google_cloud_scheduler_job.secret_rotation`
- Cloud Run job: `google_cloud_run_v2_job.secret_rotation`
- Default cadence: `0 3 * * *` (03:00 UTC daily)
- Time zone: `Etc/UTC`

Update cadence by changing Terraform variables:

- `secret_rotation_schedule` (cron format)
- `secret_rotation_time_zone`
- `secret_rotation_secret_ids` (the list of Secret Manager IDs to rotate)

These settings live in `infrastructure/terraform/secret-rotation.tf` and `infrastructure/terraform/variables.tf`.

## Required IAM roles and service accounts

Two service accounts are provisioned in Terraform:

1. **Secret rotation job SA** (`secret-rotation-job`)
   - Used by the Cloud Run job that creates new Secret Manager versions.
   - Required roles:
     - `roles/secretmanager.secretVersionAdder`
     - `roles/secretmanager.secretAccessor`
     - `roles/logging.logWriter`

2. **Secret rotation scheduler SA** (`secret-rotation-scheduler`)
   - Used by Cloud Scheduler to invoke the Cloud Run job with an OIDC token.
   - Required role:
     - `roles/run.invoker` on the Cloud Run job

If rotation fails with permission errors, confirm these bindings in Terraform and verify the deployed IAM policy in the
GCP console or with `gcloud projects get-iam-policy`.

## Validate rotation success

Use the following checks after each rotation window:

1. **Cloud Scheduler execution**
   - Confirm the scheduler ran successfully (no `FAILED` entries) in Cloud Scheduler logs.

2. **Cloud Run job logs**
   - Open Cloud Logging for the `secret-rotation-job` run and confirm the job exited without errors.
   - The job logs include the rotation actor and reason for audit (`ROTATION_ACTOR`, `ROTATION_REASON`).

3. **Secret Manager versions**
   - For each rotated secret, confirm a new **enabled** version was added during the run window.
   - Verify the latest version is enabled and not destroyed.

4. **Dependent service health checks**
   - Validate that downstream services can still authenticate (API calls, database connections, or workflow runs).
   - Run any service-specific health checks or smoke tests tied to the secrets.

If any of these checks fail, move to rollback or incident response steps below.

## Rollback guidance

If a rotation introduces failures, roll back quickly to a known-good secret version:

1. **Identify the last good version**
   - Use Secret Manager version history to select the most recent version that pre-dates the rotation.

2. **Pin consumers to the previous version**
   - Update the consuming service (Cloud Run, GKE, Workflow, etc.) to reference the specific secret version.
   - Avoid using `latest` until confidence is restored.

3. **Redeploy or restart consumers**
   - Restart workloads so they read the pinned version.

4. **Coordinate with downstream owners**
   - Communicate the rollback and timeline to service owners so they can validate recovery.

5. **Disable or pause the scheduler**
   - Temporarily disable the Cloud Scheduler job if repeated rotations would reintroduce the bad secret.

## Incident response if rotation breaks services

1. **Declare impact**
   - Open an incident channel and assign an incident commander.
   - List the affected services and which secrets were rotated.

2. **Stabilize**
   - Disable the Cloud Scheduler job to prevent further rotations.
   - Pin or roll back to the previous secret versions (see rollback guidance).

3. **Diagnose**
   - Review Cloud Run job logs for errors during rotation.
   - Check Secret Manager audit logs for failed access or version creation.
   - Verify that secret consumers are correctly configured (version pins, environment variable bindings).

4. **Recover**
   - Redeploy or restart services after pinning the correct secret version.
   - Run service health checks and validate pipelines/workflows.

5. **Follow-up**
   - Document root cause and update this runbook or Terraform configuration as needed.
   - Consider adding additional validation or canary testing before the next scheduled rotation.
