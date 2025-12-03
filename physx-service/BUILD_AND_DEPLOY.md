# PhysX-Service Build and Deploy Guide

## Quick Start

### Prerequisites
- Docker with buildx
- HuggingFace account and token (if repo requires auth)
- Google Cloud SDK configured

### 1. Get HuggingFace Token

Visit https://huggingface.co/settings/tokens and create a token with read access.

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxx"
```

### 2. Build the Docker Image

```bash
cd /path/to/BlueprintPipeline

docker build \
  --build-arg HF_TOKEN="${HF_TOKEN}" \
  --platform linux/amd64 \
  -t us-central1-docker.pkg.dev/blueprint-8c1ca/blueprint/physx-service:latest \
  -f physx-service/Dockerfile \
  physx-service/
```

**Note**: Build will take 30-60 minutes due to:
- CUDA dependencies
- Model downloads (~10-20GB)
- Flash-attention compilation

### 3. Push to GCP Artifact Registry

```bash
docker push us-central1-docker.pkg.dev/blueprint-8c1ca/blueprint/physx-service:latest
```

### 4. Deploy to Cloud Run

```bash
gcloud run deploy physx-service \
  --image us-central1-docker.pkg.dev/blueprint-8c1ca/blueprint/physx-service:latest \
  --project blueprint-8c1ca \
  --region us-central1 \
  --platform managed \
  --memory 32Gi \
  --cpu 8 \
  --gpu 1 \
  --gpu-type nvidia-l4 \
  --timeout 900 \
  --concurrency 1 \
  --min-instances 0 \
  --max-instances 2 \
  --port 8080 \
  --no-cpu-throttling \
  --set-env-vars "PHYSX_DEBUG=1" \
  --allow-unauthenticated
```

**Critical settings**:
- `--concurrency 1`: GPU can only handle one request at a time
- `--gpu 1 --gpu-type nvidia-l4`: Required for CUDA inference
- `--memory 32Gi`: VLM model requires significant RAM
- `--timeout 900`: Pipeline takes 5-15 minutes per object
- `--min-instances 0`: Can scale to zero (but cold starts are slow)

### 5. Test the Service

```bash
# Health check
curl https://physx-service-744608654760.us-central1.run.app/

# Debug info
curl https://physx-service-744608654760.us-central1.run.app/debug

# Test with an image
python test_physx_service.py
```

## Automated Build with Cloud Build

### Setup Secret in Secret Manager

```bash
# Create secret with your HF token
echo -n "hf_xxxxxxxxxxxxxxxxxxxxx" | \
  gcloud secrets create HF_TOKEN \
  --project blueprint-8c1ca \
  --replication-policy automatic \
  --data-file=-

# Grant Cloud Build access
gcloud secrets add-iam-policy-binding HF_TOKEN \
  --member serviceAccount:744608654760@cloudbuild.gserviceaccount.com \
  --role roles/secretmanager.secretAccessor
```

### Create cloudbuild.yaml

```yaml
steps:
  # Build with secret
  - name: 'gcr.io/cloud-builders/docker'
    secretEnv: ['HF_TOKEN']
    args:
      - 'build'
      - '--build-arg'
      - 'HF_TOKEN=$$HF_TOKEN'
      - '-t'
      - 'us-central1-docker.pkg.dev/blueprint-8c1ca/blueprint/physx-service:$SHORT_SHA'
      - '-t'
      - 'us-central1-docker.pkg.dev/blueprint-8c1ca/blueprint/physx-service:latest'
      - '-f'
      - 'physx-service/Dockerfile'
      - 'physx-service/'

  # Push images
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', '--all-tags', 'us-central1-docker.pkg.dev/blueprint-8c1ca/blueprint/physx-service']

  # Deploy to Cloud Run
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    args:
      - 'gcloud'
      - 'run'
      - 'deploy'
      - 'physx-service'
      - '--image'
      - 'us-central1-docker.pkg.dev/blueprint-8c1ca/blueprint/physx-service:$SHORT_SHA'
      - '--region'
      - 'us-central1'
      - '--memory'
      - '32Gi'
      - '--cpu'
      - '8'
      - '--gpu'
      - '1'
      - '--gpu-type'
      - 'nvidia-l4'
      - '--timeout'
      - '900'
      - '--concurrency'
      - '1'

availableSecrets:
  secretManager:
    - versionName: projects/blueprint-8c1ca/secrets/HF_TOKEN/versions/latest
      env: 'HF_TOKEN'

timeout: 3600s
options:
  machineType: 'E2_HIGHCPU_8'
```

### Trigger Build

```bash
gcloud builds submit \
  --config physx-service/cloudbuild.yaml \
  --project blueprint-8c1ca
```

## Troubleshooting

### Build fails with "401 Unauthorized"
- Check your HF_TOKEN is valid
- Ensure the token has access to the Caoza/PhysX-Anything repo
- Visit the repo page and accept any terms if it's gated

### Build times out
- Increase Cloud Build timeout: `timeout: 7200s` (2 hours)
- Use a more powerful machine: `machineType: 'E2_HIGHCPU_32'`

### Model verification fails
- Check internet connectivity during build
- Ensure git-lfs is working: `git lfs version`
- Try using `--use-git-lfs` flag in download script

### Runtime: "Missing required model files"
- The build succeeded but models weren't downloaded
- Rebuild with proper HF_TOKEN
- Check build logs for download errors

### Service returns 503 immediately
- Models failed to load at startup
- Check logs: `gcloud run services logs read physx-service --limit 100`
- Use `/debug` endpoint to see validation details

## Monitoring

### View Logs
```bash
# All logs
gcloud run services logs read physx-service --limit 100

# Filter by severity
gcloud run services logs read physx-service --log-filter="severity>=ERROR"

# Follow logs
gcloud run services logs tail physx-service
```

### Check Metrics
```bash
# Request count
gcloud monitoring time-series list \
  --filter='metric.type="run.googleapis.com/request_count"' \
  --format=json

# Request latency
gcloud monitoring time-series list \
  --filter='metric.type="run.googleapis.com/request_latencies"' \
  --format=json
```

## Cost Optimization

**GPU costs on Cloud Run are expensive (~$0.70/hour for L4)!**

### Option 1: Scale to Zero (Default)
- Set `--min-instances 0`
- Only pay when processing requests
- **Downside**: 10-15 minute cold starts

### Option 2: Keep Warm
- Set `--min-instances 1`
- Always have one instance ready
- **Cost**: ~$500/month continuous
- **Benefit**: No cold starts

### Option 3: Scheduled Scaling
Use Cloud Scheduler to scale up/down:
```bash
# Scale up (e.g., 8am)
gcloud scheduler jobs create http scale-up-physx \
  --schedule="0 8 * * *" \
  --uri="https://run.googleapis.com/v2/projects/blueprint-8c1ca/locations/us-central1/services/physx-service" \
  --http-method=PATCH \
  --message-body='{"template":{"scaling":{"minInstanceCount":1}}}'

# Scale down (e.g., 8pm)
gcloud scheduler jobs create http scale-down-physx \
  --schedule="0 20 * * *" \
  --uri="https://run.googleapis.com/v2/projects/blueprint-8c1ca/locations/us-central1/services/physx-service" \
  --http-method=PATCH \
  --message-body='{"template":{"scaling":{"minInstanceCount":0}}}'
```

## Testing

### Test Script
```python
#!/usr/bin/env python3
import base64
import requests
from pathlib import Path

# Test image
image_path = Path("test_object.png")
if not image_path.exists():
    print("ERROR: test_object.png not found")
    exit(1)

# Read and encode
image_bytes = image_path.read_bytes()
image_b64 = base64.b64encode(image_bytes).decode()

# Call service
url = "https://physx-service-744608654760.us-central1.run.app/"
response = requests.post(
    url,
    json={"image_base64": image_b64},
    timeout=900,
)

print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")

if response.ok:
    result = response.json()
    # Save outputs
    if "mesh_base64" in result:
        mesh_bytes = base64.b64decode(result["mesh_base64"])
        Path("output.glb").write_bytes(mesh_bytes)
        print("✓ Mesh saved to output.glb")

    if "urdf_base64" in result:
        urdf_bytes = base64.b64decode(result["urdf_base64"])
        Path("output.urdf").write_bytes(urdf_bytes)
        print("✓ URDF saved to output.urdf")
```

## References

- PhysX-Anything repo: https://github.com/ziangcao0312/PhysX-Anything
- HuggingFace model: https://huggingface.co/Caoza/PhysX-Anything
- Cloud Run GPU docs: https://cloud.google.com/run/docs/configuring/services/gpu
