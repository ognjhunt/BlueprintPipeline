# Secondary Region Jobs

Use this kustomize overlay to deploy critical GPU jobs to the secondary GKE
cluster for regional failover.

## Included jobs

- Episode generation
- Genie Sim export
- Genie Sim GPU execution
- Genie Sim import
- Genie Sim import webhook
- Genie Sim local submit
- Genie Sim submit
- Genie Sim training
- Firebase cleanup
- Dream2Flow preparation
- DWM preparation

## Apply

```bash
kubectl config use-context <secondary-gke-context>
kustomize build k8s/secondary-region | kubectl apply -f -
```

> Note: Configure the `BUCKET` environment variable and GCS service account
> secret in the secondary cluster before running these jobs.

## Required config and secrets

The secondary region overlay creates the base ConfigMaps defined in each
manifest, but you must provide the following shared secrets and any optional
override ConfigMaps before deploying:

- `gcs-service-account` Secret (used by Genie Sim export, submit, local, import,
  and training jobs).
- `firebase-service-account` Secret (used by Genie Sim import).
- `genie-sim-import-webhook-secrets` Secret (optional HMAC/OIDC values for the
  webhook deployment).
- Optional override ConfigMaps:
  - `genie-sim-export-overrides`
  - `genie-sim-import-overrides`
  - `genie-sim-local-overrides`
  - `genie-sim-submit-overrides`
