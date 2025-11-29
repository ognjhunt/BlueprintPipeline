# SAM3 Gemini Configuration Guide

## Overview

This document describes how to configure SAM3 (segmentation) to use higher confidence thresholds and enable advanced Gemini features like grounding and structured outputs.

## Configuration Changes Made

### 1. SAM3 Confidence Threshold

**File**: `workflows/ingest-single-image-pipeline.yaml`

**Change**: Added `SAM3_CONFIDENCE` environment variable to the seg-job step

```yaml
- name: SAM3_CONFIDENCE
  value: "0.60"
```

**Default**: 0.30 (set in `seg-job/run_sam3.sh:33`)
**New Value**: 0.60

**Impact**: This increases the minimum confidence threshold for SAM3 segmentation masks, resulting in higher quality but potentially fewer detected segments.

### 2. Gemini Grounding (Google Search)

**File**: `workflows/ingest-single-image-pipeline.yaml`

**Change**: Added `GEMINI_GROUNDING_ENABLED` environment variable

```yaml
- name: GEMINI_GROUNDING_ENABLED
  value: "true"
```

**Purpose**: Enables Gemini to use Google Search grounding to enhance scene understanding with real-world knowledge.

**Implementation Requirements**: The SAM3 Python code needs to be updated to respect this flag when calling Gemini. Based on the simready-job pattern, this would involve:

```python
# In the Gemini API call configuration
config_kwargs = {
    "response_mime_type": "application/json",
}

# Enable grounding if configured
if os.getenv("GEMINI_GROUNDING_ENABLED", "false").lower() in {"1", "true", "yes", "on"}:
    from google.genai import types
    config_kwargs["grounding"] = types.GroundingConfig(
        google_search=types.GoogleSearchGrounding()
    )
```

### 3. Gemini Structured Outputs

**File**: `workflows/ingest-single-image-pipeline.yaml`

**Change**: Added `GEMINI_STRUCTURED_OUTPUT` environment variable

```yaml
- name: GEMINI_STRUCTURED_OUTPUT
  value: "true"
```

**Purpose**: Enables structured JSON schema outputs from Gemini for consistent, parseable responses.

**Implementation Requirements**: Based on the simready-job pattern (lines 641-669 in `prepare_simready_assets.py`), this involves:

```python
from google.genai import types

# Define your schema for scene inventory
response_schema = {
    "type": "object",
    "properties": {
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "category": {"type": "string"},
                    "short_description": {"type": "string"},
                    "approx_location": {"type": "string"},
                    "relationships": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["id", "category"]
            }
        }
    },
    "required": ["objects"]
}

# Configure Gemini with structured output
config = types.GenerateContentConfig(
    response_mime_type="application/json",
    response_schema=response_schema
)
```

## Gemini Thinking Level Configuration

The logs show `thinking_level=high` being used. Based on the simready-job implementation (lines 646-660), you can configure thinking levels:

```python
from google.genai import types

# For Gemini 3.x models
if hasattr(types, "ThinkingConfig"):
    ThinkingConfig = getattr(types, "ThinkingConfig")
    ThinkingLevel = getattr(types, "ThinkingLevel", None)

    if model_name.startswith("gemini-3") and ThinkingLevel is not None:
        config_kwargs["thinking_config"] = ThinkingConfig(
            thinking_level=getattr(ThinkingLevel, "HIGH", "HIGH")
        )
```

## Environment Variables Reference

| Variable | Default | New Value | Description |
|----------|---------|-----------|-------------|
| `SAM3_CONFIDENCE` | 0.30 | 0.60 | Minimum confidence threshold for segmentation masks |
| `GEMINI_GROUNDING_ENABLED` | false | true | Enable Google Search grounding for enhanced context |
| `GEMINI_STRUCTURED_OUTPUT` | false | true | Use JSON schema for structured, consistent responses |
| `GEMINI_MODEL` | gemini-2.5-pro | gemini-3-pro-preview | Gemini model to use (configurable) |

## Implementation Checklist

To fully enable these features, the SAM3 segmentation Python code needs to be updated:

- [ ] **Confidence Threshold**: Already implemented in `seg-job/run_sam3.sh` - reads `SAM3_CONFIDENCE` env var
- [ ] **Grounding**: Update Gemini API calls to include grounding configuration
- [ ] **Structured Output**: Define JSON schema and configure response format
- [ ] **Thinking Level**: Already configured (seen in logs as `thinking_level=high`)

## Example: Complete Gemini Configuration

Here's a complete example of how to configure Gemini with all features:

```python
import os
from google import genai
from google.genai import types

# Initialize client
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# Model configuration
model_name = os.getenv("GEMINI_MODEL", "gemini-3-pro-preview")
grounding_enabled = os.getenv("GEMINI_GROUNDING_ENABLED", "false").lower() in {"1", "true", "yes"}
structured_output = os.getenv("GEMINI_STRUCTURED_OUTPUT", "false").lower() in {"1", "true", "yes"}

# Build config
config_kwargs = {}

# Structured output schema
if structured_output:
    config_kwargs["response_mime_type"] = "application/json"
    config_kwargs["response_schema"] = {
        # Your schema here
    }

# Grounding
if grounding_enabled:
    config_kwargs["grounding"] = types.GroundingConfig(
        google_search=types.GoogleSearchGrounding()
    )

# Thinking level
if hasattr(types, "ThinkingConfig"):
    ThinkingConfig = getattr(types, "ThinkingConfig")
    ThinkingLevel = getattr(types, "ThinkingLevel", None)
    if model_name.startswith("gemini-3") and ThinkingLevel is not None:
        config_kwargs["thinking_config"] = ThinkingConfig(
            thinking_level=getattr(ThinkingLevel, "HIGH", "HIGH")
        )

# Create config and call
config = types.GenerateContentConfig(**config_kwargs)
response = client.models.generate_content(
    model=model_name,
    contents=prompt,
    config=config,
)
```

## Testing

After deploying the workflow changes:

1. Upload a test image to `gs://your-bucket/scenes/test-scene/images/test.jpg`
2. Check Cloud Run logs for the seg-job to verify:
   - `[SAM3] SAM3_CONFIDENCE=0.60`
   - `[SAM3] Gemini Google Search grounding: ENABLED`
   - `[SAM3] Calling Gemini model 'gemini-3-pro-preview' to build scene inventory (thinking_level=high, search_enabled=True)`

## References

- SAM3 confidence configuration: `seg-job/run_sam3.sh:33`
- Gemini configuration pattern: `simready-job/prepare_simready_assets.py:636-676`
- Multiview Gemini usage: `multiview-job/run_multiview_from_layout.py:218-257`
