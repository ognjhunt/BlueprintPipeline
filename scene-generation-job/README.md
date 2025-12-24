# Scene Generation Job

Automated scene image generation for BlueprintPipeline using Gemini 3.0 Pro Image.

## Overview

This job generates realistic, wide-angle scene images for different environment archetypes. Images are designed for 3D reconstruction and robotics simulation training.

### Key Features

- **AI-Powered Image Generation**: Uses Gemini 3.0 Pro Image (Nano Banana Pro) with Google Search grounding
- **Intelligent Prompt Diversification**: Uses Gemini 3.0 Pro Preview to generate diverse, unique prompts
- **History Tracking**: Firestore integration ensures diversity by tracking past generations
- **Balanced Coverage**: Weighted distribution across all supported archetypes
- **Pipeline Integration**: Auto-triggers downstream 3D reconstruction pipeline

## Supported Archetypes

| Archetype | Description | Weight |
|-----------|-------------|--------|
| `kitchen` | Commercial prep lines, dish pits, quick-serve stations | 18% |
| `warehouse` | Fulfillment centers, pallet racking, tote picking | 17% |
| `grocery` | Retail aisles, refrigeration, checkout | 15% |
| `lab` | Scientific labs, cleanrooms, precision equipment | 12% |
| `office` | Workspaces, conference rooms, filing | 12% |
| `home_laundry` | Residential laundry rooms, folding areas | 10% |
| `loading_dock` | Shipping bays, dock levelers | 8% |
| `utility_room` | Electrical panels, HVAC, mechanical spaces | 8% |

## Image Specifications

- **Resolution**: 2K (high detail for 3D reconstruction)
- **Aspect Ratio**: 16:9 (wide-angle coverage)
- **Style**: Photorealistic architectural photography
- **Coverage**: Full room capture with no blind spots
- **Perspective**: Elevated corner position (~1.5-2m height)

## Usage

### Local Development

```bash
# Set required environment variables
export GEMINI_API_KEY="your-api-key"

# Run with defaults (10 scenes)
python generate_scene_images.py

# Dry run (no actual generation)
DRY_RUN=true python generate_scene_images.py

# Generate specific archetypes
ARCHETYPES="kitchen,warehouse" python generate_scene_images.py

# Generate fewer scenes
SCENES_PER_RUN=3 python generate_scene_images.py
```

### Cloud Run Job

```bash
# Create the job
gcloud run jobs create scene-generation-job \
  --image=gcr.io/PROJECT_ID/scene-generation-job:latest \
  --region=us-central1 \
  --set-env-vars=GEMINI_API_KEY=your-key,BUCKET=your-bucket \
  --memory=4Gi \
  --timeout=3600

# Run manually
gcloud run jobs execute scene-generation-job \
  --region=us-central1 \
  --update-env-vars=SCENES_PER_RUN=5
```

### Scheduled Execution

The job includes a Cloud Scheduler configuration that runs daily at 8:00 AM Pacific.

**IMPORTANT**: The scheduler is **DISABLED by default** (`paused: true`).

To enable automatic daily generation:

```bash
# Enable the scheduler
gcloud scheduler jobs resume scene-generation-daily \
  --location=us-central1
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GEMINI_API_KEY` | Yes | - | API key for Gemini models |
| `BUCKET` | No | - | GCS bucket for scene storage |
| `SCENES_PER_RUN` | No | 10 | Number of scenes to generate |
| `ARCHETYPES` | No | - | Comma-separated list of specific archetypes |
| `DRY_RUN` | No | false | Skip actual generation |
| `FIRESTORE_PROJECT` | No | default | GCP project for Firestore |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Scene Generation Pipeline                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │   Cloud     │───▶│  Workflow   │───▶│ scene-generation-job│  │
│  │  Scheduler  │    │  Trigger    │    │                     │  │
│  │ (8:00 AM)   │    │             │    │  ┌───────────────┐  │  │
│  │  DISABLED   │    └─────────────┘    │  │   Firestore   │  │  │
│  └─────────────┘                       │  │   History     │◀─┤  │
│                                        │  └───────────────┘  │  │
│                                        │          │          │  │
│                                        │          ▼          │  │
│                                        │  ┌───────────────┐  │  │
│                                        │  │ Gemini Pro    │  │  │
│                                        │  │ (Diversifier) │  │  │
│                                        │  └───────────────┘  │  │
│                                        │          │          │  │
│                                        │          ▼          │  │
│                                        │  ┌───────────────┐  │  │
│                                        │  │ Gemini Image  │  │  │
│                                        │  │ (Generator)   │  │  │
│                                        │  └───────────────┘  │  │
│                                        │          │          │  │
│                                        └──────────┼──────────┘  │
│                                                   │             │
│                                                   ▼             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    GCS Storage                           │   │
│  │  scenes/{scene_id}/                                      │   │
│  │    ├── source_image.png                                  │   │
│  │    └── .scene_generation_complete                        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │             Downstream Pipeline (3D-RE-GEN)              │   │
│  │                                                          │   │
│  │   image → regen3d → simready → USD → replicator → isaac │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Diversity Mechanism

The job ensures diverse scene generation through:

1. **Firestore History Tracking**
   - Records all generated prompts with hashes and tags
   - Tracks variation coverage per archetype
   - Enables lookback of 30 days

2. **Gemini Pro Diversification**
   - Analyzes recent generations to identify gaps
   - Generates unique prompts avoiding similarity
   - Suggests under-represented variations

3. **Weighted Archetype Selection**
   - Base weights per archetype (see table above)
   - Adjusts weights based on recent coverage
   - Balances distribution over time

4. **Variation Tags**
   - Each generation tagged with unique aspects
   - Examples: `morning_light`, `busy_state`, `compact_space`
   - Coverage analysis identifies gaps

## Output Structure

```
scenes/{scene_id}/
├── source_image.png              # Generated scene image (2K, 16:9)
└── .scene_generation_complete    # Trigger marker for downstream pipeline
```

## Firestore Collections

```
scene_generation_history/
├── {scene_id}/
│   ├── scene_id: string
│   ├── archetype: string
│   ├── prompt_hash: string
│   ├── prompt_summary: string (first 200 chars)
│   ├── variation_tags: string[]
│   ├── generated_at: timestamp
│   └── success: boolean
```

## Deployment

### Prerequisites

1. GCP project with:
   - Cloud Run enabled
   - Firestore enabled
   - Cloud Storage bucket
   - Cloud Scheduler enabled

2. Service accounts:
   - Cloud Run job service account
   - Cloud Scheduler service account

3. Gemini API key

### Deploy Steps

```bash
# 1. Build and push container
docker build -t gcr.io/PROJECT_ID/scene-generation-job .
docker push gcr.io/PROJECT_ID/scene-generation-job

# 2. Deploy Cloud Run job
gcloud run jobs deploy scene-generation-job \
  --image=gcr.io/PROJECT_ID/scene-generation-job:latest \
  --region=us-central1 \
  --set-secrets=GEMINI_API_KEY=gemini-api-key:latest \
  --set-env-vars=BUCKET=your-bucket \
  --memory=4Gi \
  --timeout=3600 \
  --max-retries=1

# 3. Deploy workflow
gcloud workflows deploy scene-generation-pipeline \
  --location=us-central1 \
  --source=../workflows/scene-generation-pipeline.yaml

# 4. Create scheduler (paused)
gcloud scheduler jobs create http scene-generation-daily \
  --schedule="0 8 * * *" \
  --time-zone="America/Los_Angeles" \
  --location=us-central1 \
  --uri="https://workflowexecutions.googleapis.com/v1/projects/PROJECT_ID/locations/us-central1/workflows/scene-generation-pipeline/executions" \
  --http-method=POST \
  --oauth-service-account-email=scheduler-sa@PROJECT_ID.iam.gserviceaccount.com \
  --message-body='{"argument": "{\"scenes_per_run\": 10}"}' \
  --paused

# 5. When ready to go live:
gcloud scheduler jobs resume scene-generation-daily --location=us-central1
```

## Testing

```bash
# Run tests
pytest tests/

# Local dry run
DRY_RUN=true python generate_scene_images.py

# Single archetype test
ARCHETYPES=kitchen SCENES_PER_RUN=1 python generate_scene_images.py
```

## Files

```
scene-generation-job/
├── generate_scene_images.py      # Main generation script
├── scheduler_config.yaml         # Cloud Scheduler config (DISABLED)
├── archetypes/
│   └── archetype_config.json     # Archetype definitions
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Container image
└── README.md                     # This file
```

## Related

- [3D-RE-GEN Pipeline](../workflows/usd-assembly-pipeline.yaml)
- [Replicator Bundle Generator](../replicator-job/)
- [Variation Asset Pipeline](../variation-asset-pipeline-job/)
