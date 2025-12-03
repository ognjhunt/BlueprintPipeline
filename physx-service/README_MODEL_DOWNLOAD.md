# PhysX-Service Model Download Fix

## Problem
The PhysX-Anything model download was failing with `401 Unauthorized` errors during Docker build, resulting in missing model files (`tokenizer.json` and weight files).

## Solution
We now use a robust download script with:
- HuggingFace authentication support
- Retry logic with exponential backoff
- Fallback to git-lfs if needed
- Comprehensive verification

## Building with Authentication

### Option 1: Using HuggingFace Token (Recommended)

1. Get a HuggingFace token:
   - Go to https://huggingface.co/settings/tokens
   - Create a new token with "read" permission
   - Copy the token

2. Build with the token:
```bash
# Set your HF token
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxx"

# Build the image
docker build \
  --build-arg HF_TOKEN="${HF_TOKEN}" \
  -t us-central1-docker.pkg.dev/blueprint-8c1ca/blueprint/physx-service:latest \
  -f physx-service/Dockerfile \
  physx-service/
```

### Option 2: For Cloud Build

Add the token as a Secret Manager secret, then reference it:

```bash
gcloud builds submit \
  --config cloudbuild.yaml \
  --substitutions _HF_TOKEN="$(gcloud secrets versions access latest --secret=HF_TOKEN)"
```

### Option 3: Public Access (if repo is public)

If the repo becomes public, you can build without a token:
```bash
docker build \
  -t us-central1-docker.pkg.dev/blueprint-8c1ca/blueprint/physx-service:latest \
  -f physx-service/Dockerfile \
  physx-service/
```

## Verification

The build will fail immediately if:
- Model download fails
- Required files are missing (`config.json`, `tokenizer.json`, `tokenizer_config.json`)
- Model weights (`.safetensors` or `.bin` files) are missing
- Total model size is suspiciously small (< 1GB)

Look for these success messages:
```
✓ All required files present
✓ Model weights found (safetensors)
✓ Model verification PASSED
```

## Troubleshooting

### Issue: 401 Unauthorized
**Solution**: Make sure your HF_TOKEN is valid and has read access to the repo.

### Issue: Download timeout
**Solution**: The script has retry logic. If it keeps timing out, check your network connection.

### Issue: Missing git-lfs
**Solution**: The Dockerfile now installs git-lfs automatically.

### Issue: Repo is gated/private
**Solution**:
1. Accept the model terms at https://huggingface.co/Caoza/PhysX-Anything
2. Use a token from an account that has access

## Files Added/Modified

- `download_models.py` - New robust download script
- `Dockerfile` - Updated to use new download method with authentication
- `README_MODEL_DOWNLOAD.md` - This file

## Security Notes

- The HF_TOKEN is only used during build and is unset before the final image
- Tokens are not persisted in the image layers (they're in ENV which gets cleared)
- For production, use Secret Manager and never commit tokens to git
