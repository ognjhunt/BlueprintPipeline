# k8s

## Purpose / scope
Kubernetes manifests for deploying pipeline jobs and supporting infrastructure.

## Primary entrypoints
- `*.yaml` Kubernetes manifests (e.g., `genie-sim-gpu-job.yaml`, `namespace-setup.yaml`).

## Required inputs / outputs
- **Inputs:** Kubernetes cluster credentials and any referenced container images.
- **Outputs:** Kubernetes resources created in the cluster.

## Key environment variables
- `KUBECONFIG` or other cluster credentials used by `kubectl`.

## How to run locally
- Apply manifests with `kubectl apply -f <file>.yaml` after configuring cluster access.

