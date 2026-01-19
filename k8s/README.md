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

## Episode generation via Kustomize
The episode generation job uses Kustomize variables for `PROJECT_ID`, `SCENE_ID`, and
`BUCKET`, plus an image override for the Isaac Sim episode generator.

1. Update defaults in `kustomization.yaml`:
   - `configMapGenerator` literals for `PROJECT_ID`, `SCENE_ID`, and `BUCKET`.
   - `images` to point at the correct GCR image name/tag.
2. Render or apply:
   - Render: `kustomize build k8s`
   - Apply: `kubectl apply -k k8s`

Example override:
```
kustomize edit set image gcr.io/blueprint-project/blueprint-episode-gen=gcr.io/my-project/blueprint-episode-gen:isaacsim
```
