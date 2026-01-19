# Secret rotation job

This directory contains the Cloud Run job used by the automated secret rotation scheduler.

## Runbook

Operational guidance for scheduling, validation, rollback, and incident response lives in
[`docs/operations/secret-rotation.md`](../../docs/operations/secret-rotation.md).

## Build + push

```bash
# From repo root

docker build -f infrastructure/secret-rotation/Dockerfile -t us-docker.pkg.dev/<project>/<repo>/secret-rotation:latest .

docker push us-docker.pkg.dev/<project>/<repo>/secret-rotation:latest
```

## Runtime configuration

The Cloud Run job expects the following environment variables:

- `ROTATION_SECRET_IDS`: Comma-separated Secret Manager IDs to rotate.
- `ROTATION_BYTE_LENGTH`: Byte length for generated values (default: 32).
- `ROTATION_REASON`: Optional reason string for audit logging (default: `scheduled`).
- `ROTATION_ACTOR`: Optional actor string for audit logging (default: `cloud-scheduler`).
- `GCP_PROJECT`/`GOOGLE_CLOUD_PROJECT`: Project ID (required).

The job generates a new random value for each secret and adds a new version.
