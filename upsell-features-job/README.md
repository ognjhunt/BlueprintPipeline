# upsell-features-job

## Purpose / scope
Generates upsell feature analytics, reports, and enrichment outputs for the pipeline.

## Primary entrypoints
- `generate_upsell_outputs.py` main job entrypoint.
- `pipeline_integration.py` pipeline integration helpers.
- `post_processor.py` post-processing utilities.

## Required inputs / outputs
- **Inputs:** simulation outputs, analytics configuration, and customer configuration.
- **Outputs:** upsell feature reports, metrics, and derived artifacts.

## Key environment variables
- Variables defining input/output locations, customer configuration, and external service credentials.

## How to run locally
- Run the job: `python generate_upsell_outputs.py` (set required env vars).

