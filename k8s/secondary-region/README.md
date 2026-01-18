# Secondary Region Jobs

Use this kustomize overlay to deploy critical GPU jobs to the secondary GKE
cluster for regional failover.

## Included jobs

- Episode generation
- Genie Sim GPU execution
- Dream2Flow preparation
- DWM preparation

## Apply

```bash
kubectl config use-context <secondary-gke-context>
kustomize build k8s/secondary-region | kubectl apply -f -
```

> Note: Configure the `BUCKET` environment variable and GCS service account
> secret in the secondary cluster before running these jobs.
