# Dataset Delivery Job

This job delivers Genie Sim import bundles to lab-specific buckets, generates a dataset card, and sends webhook notifications.

## Environment Variables

Required:
- `BUCKET`: Source GCS bucket containing the import manifest.
- `SCENE_ID`: Scene identifier (used when `IMPORT_MANIFEST_PATH` is not provided).

Optional:
- `JOB_ID`: Job identifier to include in notifications.
- `IMPORT_MANIFEST_PATH`: Full `gs://` path to the import manifest. Defaults to `gs://$BUCKET/scenes/$SCENE_ID/geniesim/import_manifest.json`.
- `DELIVERY_PREFIX`: Destination prefix template. Defaults to `deliveries/{scene_id}/{job_id}`.
- `LAB_DELIVERY_BUCKETS`: Mapping of lab name to bucket (`lab-a=bucket-a,lab-b=bucket-b`) or JSON object.
- `DEFAULT_DELIVERY_BUCKET`: Fallback bucket for lab deliveries when no mapping is provided.
- `LAB_WEBHOOK_URLS`: Mapping of lab name to webhook URL (`lab-a=https://...,lab-b=https://...`) or JSON object.

The job writes a failure marker to `scenes/<scene_id>/geniesim/.dataset_delivery_failed` on error.
