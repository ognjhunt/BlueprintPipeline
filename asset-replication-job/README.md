# asset-replication-job

Processes async replication queue entries and copies asset files from GCS-mounted
storage (`/mnt/gcs`) to Backblaze B2 (S3-compatible API).

## Queue contract

Queue files are written by text-scene-adapter-job:

- `automation/asset_replication/queue/*.json`

Each queue item contains scene metadata and file lists to replicate.

## Required environment variables

- `BUCKET` (GCS bucket mounted at `/mnt/gcs`)
- `B2_S3_ENDPOINT`
- `B2_BUCKET`
- `B2_KEY_ID` (bind from Secret Manager)
- `B2_APPLICATION_KEY` (bind from Secret Manager)

## Optional environment variables

- `QUEUE_OBJECT` (process one queue file only)
- `TEXT_ASSET_REPLICATION_QUEUE_PREFIX` (default `automation/asset_replication/queue`)
- `TEXT_ASSET_REPLICATION_PROCESSED_PREFIX` (default `automation/asset_replication/processed`)
- `TEXT_ASSET_REPLICATION_FAILED_PREFIX` (default `automation/asset_replication/failed`)
- `TEXT_ASSET_REPLICATION_MAX_ITEMS` (default `10`)
- `TEXT_ASSET_REPLICATION_DRY_RUN` (default `false`)
- `TEXT_ASSET_REPLICATION_FAIL_ON_ERROR` (default `true`)
- `B2_REGION` (default `us-west-000`)

Recommended secure deployment:
- bind `B2_KEY_ID` and `B2_APPLICATION_KEY` via Cloud Run job `--update-secrets`
- keep `B2_S3_ENDPOINT` / `B2_BUCKET` as non-secret job env vars
- do not pass Backblaze credentials through workflow env overrides
