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

## Image registry validation
All Kubernetes manifests in this folder should reference images in `ghcr.io`. When updating
image names or tags, ensure every `image:` entry (including Kustomize overrides) points to
the same GHCR registry so production deployments stay consistent.

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
kustomize edit set image ghcr.io/blueprint-project/blueprint-episode-gen=ghcr.io/my-project/blueprint-episode-gen:isaacsim
```

## Firebase cleanup CronJob
The Firebase cleanup CronJob deletes orphaned blobs under a configured prefix by age
and/or by a manifest of known-good uploads.

### Required environment variables
- `FIREBASE_STORAGE_BUCKET`: Firebase Storage bucket name.
- `FIREBASE_SERVICE_ACCOUNT_JSON` or `FIREBASE_SERVICE_ACCOUNT_PATH`: credentials for Firebase Admin.

### Cleanup-specific environment variables
- `FIREBASE_CLEANUP_PREFIX`: Blob prefix to scan (default `datasets`).
- `FIREBASE_CLEANUP_MAX_AGE_HOURS`: Age threshold; blobs newer than this are retained.
- `FIREBASE_CLEANUP_MANIFEST_PATH`: Optional path to a manifest file containing known-good paths or prefixes.

### Manifest format
The manifest file can be either:
- A JSON list of blob paths.
- A JSON object with `paths` and/or `prefixes` arrays.
- A newline-delimited text file of blob paths (lines starting with `#` are ignored).

### Operations
1. Customize `k8s/firebase-cleanup-cronjob.yaml` and apply it:
   - `envsubst < k8s/firebase-cleanup-cronjob.yaml | kubectl apply -f -`
2. If using `FIREBASE_SERVICE_ACCOUNT_PATH`, mount the credentials via a Secret and set
   the env var in `firebase-cleanup-overrides`.
3. Review logs for the JSON summary containing `requested`, `deleted`, and `failed`.
