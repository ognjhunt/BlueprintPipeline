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
The episode generation job reads `SCENE_ID` and `BUCKET` from the `episode-gen-runtime`
ConfigMap, plus an image override for the Isaac Sim episode generator.

1. Supply runtime values:
   - With Kustomize: update `configMapGenerator` literals for `SCENE_ID` and `BUCKET`
     in `kustomization.yaml`.
   - With kubectl: `kubectl create configmap episode-gen-runtime --from-literal=SCENE_ID=example-scene --from-literal=BUCKET=example-bucket -n blueprint`
2. Render or apply:
   - Render: `kustomize build k8s`
   - Apply: `kubectl apply -k k8s`

Example override:
```
kustomize edit set image gcr.io/blueprint-project/blueprint-episode-gen=gcr.io/my-project/blueprint-episode-gen:isaacsim
```
